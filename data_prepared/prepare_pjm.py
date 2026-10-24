#!/usr/bin/env python3
"""Download and clean hourly PJM demand from the public EIA-930 API."""

from __future__ import annotations

import argparse
import csv
import getpass
import hashlib
import json
import math
import os
import subprocess
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path


START = datetime(2021, 1, 1, tzinfo=timezone.utc)
PROFILES = {
    "paper_20240613": {
        "end": datetime(2024, 6, 13, tzinfo=timezone.utc),
        "suffix": "",
    },
    "extended_20251231": {
        "end": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "suffix": "_2021_2025",
    },
}

API_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "eia930_pjm"
PROCESSED_DIR = DATA_DIR / "processed"
PAGE_SIZE = 5000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def redact(message: str, api_key: str) -> str:
    return message.replace(api_key, "<redacted>") if api_key else message


def request_page(
    api_key: str,
    start: datetime,
    end: datetime,
    offset: int,
) -> dict:
    params = [
        ("api_key", api_key),
        ("frequency", "hourly"),
        ("data[0]", "value"),
        ("facets[respondent][]", "PJM"),
        ("facets[type][]", "D"),
        ("start", start.strftime("%Y-%m-%dT%H")),
        # EIA's end bound is inclusive. Request the final desired hour.
        ("end", (end - timedelta(hours=1)).strftime("%Y-%m-%dT%H")),
        ("sort[0][column]", "period"),
        ("sort[0][direction]", "asc"),
        ("offset", str(offset)),
        ("length", str(PAGE_SIZE)),
    ]
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    try:
        completed = subprocess.run(
            ["curl", "--config", "-"],
            input=(
                'fail\nsilent\nshow-error\nlocation\nmax-time = 120\n'
                'user-agent = "AT-VMD dataset preparation/1.0"\n'
                f'url = "{url}"\n'
            ).encode("utf-8"),
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = redact(exc.stderr.decode("utf-8", errors="replace"), api_key)
        raise RuntimeError(f"EIA request failed: {detail.strip()}") from None

    payload = json.loads(completed.stdout)
    if "error" in payload:
        raise RuntimeError(redact(json.dumps(payload["error"]), api_key))
    response = payload.get("response")
    if not isinstance(response, dict) or not isinstance(response.get("data"), list):
        raise RuntimeError("Unexpected EIA API response")
    return payload


def download(api_key: str, start: datetime, end: datetime, force: bool) -> Path:
    destination = RAW_DIR / (
        f"pjm_demand_{start:%Y%m%dT%H%MZ}_{end:%Y%m%dT%H%MZ}.json"
    )
    if destination.exists() and destination.stat().st_size > 0 and not force:
        return destination

    rows: list[dict] = []
    offset = 0
    total = None
    while total is None or offset < total:
        payload = request_page(api_key, start, end, offset)
        response = payload["response"]
        page = response["data"]
        total = int(response.get("total", len(page)))
        rows.extend(page)
        offset += len(page)
        if not page:
            break
        print(f"Downloaded {len(rows):,}/{total:,} PJM hourly records")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.part")
    temporary.write_text(
        json.dumps({"source": API_URL, "response": {"data": rows}}, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def parse_rows(
    path: Path,
    start: datetime,
    end: datetime,
) -> tuple[dict[datetime, float], list[datetime]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["response"]["data"]
    values: dict[datetime, float] = {}
    for row in rows:
        if row.get("respondent") != "PJM" or row.get("type") != "D":
            raise ValueError(f"Unexpected EIA-930 series in {path}")
        timestamp = datetime.fromisoformat(str(row["period"]).replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        timestamp = timestamp.astimezone(timezone.utc)
        raw_value = row.get("value")
        if raw_value is not None:
            value = float(raw_value)
            # EIA-930 contains three integer-overflow sentinels in Oct. 2021.
            # PJM system demand is well below 500 GW throughout this profile.
            if not math.isfinite(value) or value <= 0 or value > 500_000:
                continue
            if timestamp in values and not math.isclose(values[timestamp], value):
                raise ValueError(f"Conflicting PJM values at {timestamp}")
            if start <= timestamp < end:
                values[timestamp] = value

    expected = [
        start + timedelta(hours=offset)
        for offset in range(int((end - start).total_seconds() // 3600))
    ]
    imputed = [timestamp for timestamp in expected if timestamp not in values]
    if len(imputed) / len(expected) > 0.005:
        raise ValueError(
            f"PJM coverage requires excessive imputation: {len(imputed)} hours"
        )
    for timestamp in imputed:
        previous_week = timestamp - timedelta(days=7)
        if previous_week not in values:
            raise ValueError(
                f"Cannot seasonally impute PJM demand at {timestamp}; "
                f"missing previous-week value {previous_week}"
            )
        values[timestamp] = values[previous_week]
    return (
        {timestamp: values[timestamp] for timestamp in expected},
        imputed,
    )


def write_outputs(
    raw_path: Path,
    values: dict[datetime, float],
    profile: str,
    start: datetime,
    end: datetime,
    imputed: list[datetime],
) -> tuple[Path, Path]:
    suffix = PROFILES[profile]["suffix"]
    csv_path = PROCESSED_DIR / f"pjm_hourly{suffix}.csv"
    metadata_path = PROCESSED_DIR / f"pjm_hourly{suffix}.metadata.json"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["date", "electricity_demand_pjm"])
        for timestamp, value in values.items():
            rendered = str(int(value)) if value.is_integer() else f"{value:.6f}"
            writer.writerow([timestamp.strftime("%Y-%m-%d %H:%M:%S"), rendered])

    metadata = {
        "source": "U.S. EIA Form EIA-930 API",
        "api_endpoint": API_URL,
        "respondent": "PJM",
        "series_type": "D",
        "measurement": "Actual demand",
        "unit": "megawatthours",
        "timestamp_timezone": "UTC",
        "period_start_inclusive": start.isoformat(),
        "period_end_exclusive": end.isoformat(),
        "rows": len(values),
        "imputation": "previous-week same UTC hour; causal",
        "imputed_hours": [timestamp.isoformat() for timestamp in imputed],
        "raw_file": str(raw_path.relative_to(DATA_DIR)),
        "raw_sha256": sha256(raw_path),
        "output_file": str(csv_path.relative_to(DATA_DIR)),
        "output_sha256": sha256(csv_path),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return csv_path, metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="paper_20240613")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = START
    end = PROFILES[args.profile]["end"]
    raw_path = RAW_DIR / f"pjm_demand_{start:%Y%m%dT%H%MZ}_{end:%Y%m%dT%H%MZ}.json"
    if args.skip_download:
        if not raw_path.exists():
            raise FileNotFoundError(raw_path)
    else:
        api_key = os.environ.get("EIA_API_KEY") or getpass.getpass("EIA API key: ")
        if not api_key.strip():
            raise ValueError("An EIA API key is required")
        raw_path = download(api_key.strip(), start, end, args.force_download)

    values, imputed = parse_rows(raw_path, start, end)
    csv_path, metadata_path = write_outputs(
        raw_path, values, args.profile, start, end, imputed
    )
    loads = list(values.values())
    print(
        f"Validated: {csv_path.relative_to(DATA_DIR)} "
        f"({len(values):,} rows; min={min(loads):.3f}, "
        f"max={max(loads):.3f}, mean={sum(loads) / len(loads):.3f}; "
        f"imputed={len(imputed)})"
    )
    print(f"Metadata: {metadata_path.relative_to(DATA_DIR)}")


if __name__ == "__main__":
    main()
