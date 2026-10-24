#!/usr/bin/env python3
"""Download and clean ENTSO-E actual-load series for AT-VMD."""

from __future__ import annotations

import argparse
import csv
import getpass
import hashlib
import json
import math
import os
import re
import subprocess
import time
import urllib.parse
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path


START = datetime(2021, 1, 1, tzinfo=timezone.utc)
END = datetime(2024, 6, 13, tzinfo=timezone.utc)
EXPECTED_HOURS = 30_216

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

API_URL = "https://web-api.tp.entsoe.eu/api"
AREAS = {
    "PL": {"slug": "poland", "domain": "10YPL-AREA-----S"},
    "BE": {"slug": "belgium", "domain": "10YBE----------2"},
    "DE_LU": {"slug": "germany_luxembourg", "domain": "10Y1001A1001A82H"},
}
DOCUMENT_TYPE = "A65"
PROCESS_TYPE = "A16"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data"
ACTIVE_AREA = "PL"
RAW_DIR = DATASET_DIR / "raw" / "entsoe_poland_actual_load"
OUTPUT_PATH = DATASET_DIR / "processed" / "entsoe_poland_hourly.csv"
METADATA_PATH = DATASET_DIR / "processed" / "entsoe_poland_hourly.metadata.json"
PARTIAL_HOURS: list[tuple[datetime, int]] = []
IMPUTED_HOURS: list[datetime] = []


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in element if local_name(child.tag) == name]


def child_text(element: ET.Element, name: str) -> str:
    matches = children(element, name)
    if len(matches) != 1 or matches[0].text is None:
        raise ValueError(f"Expected one {name} element")
    return matches[0].text.strip()


def parse_utc(text: str) -> datetime:
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"Timestamp has no UTC offset: {text}")
    return parsed.astimezone(timezone.utc)


def parse_resolution(text: str) -> timedelta:
    match = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?", text)
    if match is None:
        raise ValueError(f"Unsupported ENTSO-E resolution: {text}")
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    duration = timedelta(hours=hours, minutes=minutes)
    if duration not in {timedelta(minutes=15), timedelta(minutes=30), timedelta(hours=1)}:
        raise ValueError(f"Expected a 15, 30, or 60 minute resolution, got {text}")
    return duration


def query_chunks():
    current = START
    while current < END:
        next_year = current.replace(year=current.year + 1, month=1, day=1)
        chunk_end = min(next_year, END)
        yield current, chunk_end
        current = chunk_end


def chunk_path(start: datetime, end: datetime) -> Path:
    name = f"{start:%Y%m%dT%H%MZ}_{end:%Y%m%dT%H%MZ}.xml"
    return RAW_DIR / name


def redact(message: str, token: str) -> str:
    return message.replace(token, "<redacted>") if token else message


def api_error_message(payload: bytes) -> str:
    try:
        root = ET.fromstring(payload)
    except ET.ParseError:
        return payload.decode("utf-8", errors="replace")[:500]
    reasons = [
        element.text.strip()
        for element in root.iter()
        if local_name(element.tag) == "text" and element.text
    ]
    return "; ".join(reasons) or f"Unexpected API document: {local_name(root.tag)}"


def download_chunk(
    token: str,
    start: datetime,
    end: datetime,
    destination: Path,
    force: bool,
    retries: int = 4,
) -> Path:
    if destination.exists() and destination.stat().st_size > 0 and not force:
        return destination

    params = {
        "securityToken": token,
        "documentType": DOCUMENT_TYPE,
        "processType": PROCESS_TYPE,
        "outBiddingZone_Domain": AREAS[ACTIVE_AREA]["domain"],
        "periodStart": start.strftime("%Y%m%d%H%M"),
        "periodEnd": end.strftime("%Y%m%d%H%M"),
    }
    request_url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".xml.part")

    for attempt in range(1, retries + 1):
        try:
            completed = subprocess.run(
                [
                    "curl",
                    "--config",
                    "-",
                ],
                input=(
                    "fail\n"
                    "silent\n"
                    "show-error\n"
                    "location\n"
                    "max-time = 120\n"
                    'user-agent = "AT-VMD dataset preparation/1.0"\n'
                    f'url = "{request_url}"\n'
                ).encode("utf-8"),
                check=True,
                capture_output=True,
            )
            payload = completed.stdout
            root = ET.fromstring(payload)
            if local_name(root.tag) != "GL_MarketDocument":
                raise ValueError(api_error_message(payload))
            temporary.write_bytes(payload)
            temporary.replace(destination)
            return destination
        except subprocess.CalledProcessError as exc:
            detail = redact(exc.stderr.decode("utf-8", errors="replace"), token)
            error = RuntimeError(f"ENTSO-E request failed: {detail.strip()}")
        except Exception as exc:
            error = RuntimeError(redact(str(exc), token))

        temporary.unlink(missing_ok=True)
        if attempt == retries:
            raise error
        time.sleep(2**attempt)

    raise RuntimeError("ENTSO-E download failed")


def download_sources(token: str, force: bool) -> list[Path]:
    paths = []
    for start, end in query_chunks():
        path = download_chunk(token, start, end, chunk_path(start, end), force)
        paths.append(path)
        print(f"Ready: {path.relative_to(DATASET_DIR)}")
    return paths


def parse_xml(path: Path) -> list[tuple[datetime, timedelta, float]]:
    root = ET.parse(path).getroot()
    if local_name(root.tag) != "GL_MarketDocument":
        raise ValueError(f"Unexpected ENTSO-E document in {path}")

    intervals = []
    time_series = [node for node in root.iter() if local_name(node.tag) == "TimeSeries"]
    if not time_series:
        raise ValueError(f"No TimeSeries in {path}")

    for series in time_series:
        units = [
            node.text.strip()
            for node in series.iter()
            if local_name(node.tag) == "quantity_Measure_Unit.name" and node.text
        ]
        if units and any(unit not in {"MAW", "MW"} for unit in units):
            raise ValueError(f"Unexpected load unit in {path}: {sorted(set(units))}")

        for period in children(series, "Period"):
            time_intervals = children(period, "timeInterval")
            if len(time_intervals) != 1:
                raise ValueError(f"Expected one Period timeInterval in {path}")
            period_start = parse_utc(child_text(time_intervals[0], "start"))
            period_end = parse_utc(child_text(time_intervals[0], "end"))
            resolution = parse_resolution(child_text(period, "resolution"))
            points = children(period, "Point")

            seen_positions = set()
            for point in points:
                position = int(child_text(point, "position"))
                if position <= 0 or position in seen_positions:
                    raise ValueError(f"Invalid point position {position} in {path}")
                seen_positions.add(position)
                timestamp = period_start + (position - 1) * resolution
                value = float(child_text(point, "quantity"))
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"Invalid load {value} at {timestamp} in {path}")
                if timestamp + resolution > period_end:
                    raise ValueError(f"Point exceeds Period end in {path}: {timestamp}")
                intervals.append((timestamp, resolution, value))

    return intervals


def aggregate_hourly(paths: list[Path]) -> dict[datetime, float]:
    global PARTIAL_HOURS, IMPUTED_HOURS
    unique_intervals: dict[tuple[datetime, timedelta], float] = {}
    for path in paths:
        for timestamp, duration, value in parse_xml(path):
            if timestamp < START or timestamp >= END:
                continue
            key = (timestamp, duration)
            previous = unique_intervals.get(key)
            if previous is not None and not math.isclose(previous, value, abs_tol=1e-9):
                raise ValueError(
                    f"Conflicting ENTSO-E values at {timestamp}: {previous} and {value}"
                )
            unique_intervals[key] = value

    weighted_totals: dict[datetime, float] = defaultdict(float)
    coverage_minutes: dict[datetime, int] = defaultdict(int)
    for (timestamp, duration), value in unique_intervals.items():
        hour = timestamp.replace(minute=0, second=0, microsecond=0)
        if timestamp + duration > hour + timedelta(hours=1):
            raise ValueError(f"Interval crosses an hour boundary: {timestamp}, {duration}")
        minutes = int(duration.total_seconds() // 60)
        weighted_totals[hour] += value * minutes
        coverage_minutes[hour] += minutes

    expected = [START + timedelta(hours=offset) for offset in range(EXPECTED_HOURS)]
    PARTIAL_HOURS = [
        (timestamp, coverage_minutes[timestamp])
        for timestamp in expected
        if 0 < coverage_minutes[timestamp] != 60
    ]
    unsupported_partial = [item for item in PARTIAL_HOURS if item[1] < 45]
    if unsupported_partial:
        raise ValueError(f"Hours below 45-minute coverage: {unsupported_partial[:10]}")

    hourly = {
        timestamp: weighted_totals[timestamp] / coverage_minutes[timestamp]
        for timestamp in expected
        if coverage_minutes[timestamp] >= 45
    }
    missing = [timestamp for timestamp in expected if timestamp not in hourly]
    IMPUTED_HOURS = missing
    for timestamp in missing:
        left = timestamp - timedelta(hours=1)
        while left not in hourly and left >= START:
            left -= timedelta(hours=1)
        right = timestamp + timedelta(hours=1)
        while right not in hourly and right < END:
            right += timedelta(hours=1)
        if left < START or right >= END or right - left > timedelta(hours=8):
            raise ValueError(f"Cannot interpolate ENTSO-E gap around {timestamp}")
        fraction = (timestamp - left) / (right - left)
        hourly[timestamp] = hourly[left] + fraction * (hourly[right] - hourly[left])
    return {timestamp: hourly[timestamp] for timestamp in expected}


def write_csv(values: dict[datetime, float]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["date", f"electricity_demand_{AREAS[ACTIVE_AREA]['slug']}"])
        for timestamp, value in values.items():
            rendered = str(int(value)) if value.is_integer() else f"{value:.6f}"
            writer.writerow([timestamp.strftime("%Y-%m-%d %H:%M:%S"), rendered])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_metadata(paths: list[Path], values: dict[datetime, float]) -> None:
    raw_files = [
        {
            "file": str(path.relative_to(DATASET_DIR)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in paths
    ]
    metadata = {
        "source": "ENTSO-E Transparency Platform REST API",
        "api_endpoint": API_URL,
        "document_type": DOCUMENT_TYPE,
        "process_type": PROCESS_TYPE,
        "area": ACTIVE_AREA,
        "out_bidding_zone_domain": AREAS[ACTIVE_AREA]["domain"],
        "measurement": "Actual Total Load",
        "unit": "MW",
        "timestamp_timezone": "UTC",
        "period_start_inclusive": START.isoformat(),
        "period_end_exclusive": END.isoformat(),
        "rows": len(values),
        "hourly_aggregation": (
            "duration-weighted observed mean; internal gaps of at most seven "
            "hours are linearly interpolated and enumerated below"
        ),
        "partial_hours": [
            {"timestamp": timestamp.isoformat(), "observed_minutes": minutes}
            for timestamp, minutes in PARTIAL_HOURS
        ],
        "linearly_interpolated_hours": [
            timestamp.isoformat() for timestamp in IMPUTED_HOURS
        ],
        "raw_files": raw_files,
        "output_file": str(OUTPUT_PATH.relative_to(DATASET_DIR)),
        "output_sha256": sha256(OUTPUT_PATH),
    }
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def validate_output(values: dict[datetime, float]) -> None:
    if len(values) != EXPECTED_HOURS:
        raise ValueError(f"Expected {EXPECTED_HOURS} rows, got {len(values)}")
    timestamps = list(values)
    if timestamps[0] != START or timestamps[-1] != END - timedelta(hours=1):
        raise ValueError("Unexpected output time range")
    if any(
        right - left != timedelta(hours=1)
        for left, right in zip(timestamps, timestamps[1:])
    ):
        raise ValueError("Output timestamps are not continuous hourly UTC")

    loads = list(values.values())
    print(
        f"Validated: {OUTPUT_PATH.relative_to(DATASET_DIR)} "
        f"({len(loads):,} rows; min={min(loads):.3f}, "
        f"max={max(loads):.3f}, mean={sum(loads) / len(loads):.3f} MW)"
    )


def main() -> None:
    global END, EXPECTED_HOURS, ACTIVE_AREA, RAW_DIR, OUTPUT_PATH, METADATA_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--areas",
        nargs="+",
        choices=AREAS,
        default=list(AREAS),
        help="ENTSO-E bidding zones to build (default: all registered zones).",
    )
    parser.add_argument(
        "--profile",
        choices=PROFILES,
        default="paper_20240613",
        help="Select the immutable paper cutoff or the 2021-2025 extension.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Build from existing raw XML files without requesting a token.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Replace existing raw XML files.",
    )
    args = parser.parse_args()

    profile = PROFILES[args.profile]
    END = profile["end"]
    EXPECTED_HOURS = int((END - START).total_seconds() // 3600)
    token = ""
    if not args.skip_download:
        token = os.environ.get("ENTSOE_SECURITY_TOKEN") or getpass.getpass(
            "ENTSO-E security token: "
        )
        if not token.strip():
            raise ValueError("An ENTSO-E security token is required")

    for area in args.areas:
        ACTIVE_AREA = area
        slug = AREAS[area]["slug"]
        suffix = profile["suffix"]
        RAW_DIR = DATASET_DIR / "raw" / f"entsoe_{slug}_actual_load"
        OUTPUT_PATH = DATASET_DIR / "processed" / f"entsoe_{slug}_hourly{suffix}.csv"
        METADATA_PATH = (
            DATASET_DIR / "processed" / f"entsoe_{slug}_hourly{suffix}.metadata.json"
        )
        expected_paths = [chunk_path(start, end) for start, end in query_chunks()]
        if args.skip_download:
            missing = [path for path in expected_paths if not path.exists()]
            if missing:
                raise FileNotFoundError(f"Missing raw ENTSO-E files: {missing}")
            paths = expected_paths
        else:
            paths = download_sources(token.strip(), args.force_download)

        values = aggregate_hourly(paths)
        write_csv(values)
        validate_output(values)
        write_metadata(paths, values)
        print(f"Metadata: {METADATA_PATH.relative_to(DATASET_DIR)}")


if __name__ == "__main__":
    main()
