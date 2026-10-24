#!/usr/bin/env python3
"""Prepare hourly NSW load from monthly MMSDM dispatch archives.

Extracted from TaskVMD-public/scripts/prepare_iso_ne_aemo.py; the NSW parsing,
interval-start convention, and hourly aggregation are unchanged.
"""

import argparse
import csv
import hashlib
import io
import json
import urllib.request
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = datetime(2021, 1, 1)
END = datetime(2024, 6, 13)
RAW = ROOT / "data/raw/aemo_dispatchregionsum"
OUTPUT = ROOT / "data/processed/aemo_nsw_hourly.csv"
BASE = "https://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/"


def months():
    current = START
    while current < END:
        yield current.year, current.month
        current = (current.replace(day=28) + timedelta(days=4)).replace(day=1)


def archive_path(year, month):
    path = RAW / f"DISPATCHREGIONSUM_{year}{month:02d}.zip"
    legacy = RAW / f"PUBLIC_DVD_DISPATCHREGIONSUM_{year}{month:02d}010000.zip"
    return legacy if legacy.exists() and not path.exists() else path


def download(year, month, path):
    if path.exists():
        return
    directory = (f"{BASE}{year}/MMSDM_{year}_{month:02d}/"
                 "MMSDM_Historical_Data_SQLLoader/DATA/")
    names = [f"PUBLIC_ARCHIVE%23DISPATCHREGIONSUM%23FILE01%23{year}{month:02d}010000.zip",
             f"PUBLIC_DVD_DISPATCHREGIONSUM_{year}{month:02d}010000.zip"]
    path.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for name in names:
        try:
            request = urllib.request.Request(directory + name,
                headers={"User-Agent": "AT-VMD dataset preparation/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = response.read()
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                if archive.testzip() is not None:
                    raise ValueError("Corrupt archive")
            path.write_bytes(payload)
            return
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            errors.append(str(exc))
    raise RuntimeError(f"Cannot fetch {year}-{month:02d}: {'; '.join(errors)}")


def aggregate(paths):
    intervals = {}
    for path in paths:
        with zipfile.ZipFile(path) as archive:
            members = [n for n in archive.namelist() if n.upper().endswith(".CSV")]
            if len(members) != 1:
                raise ValueError(f"Expected one CSV in {path}")
            with archive.open(members[0]) as stream:
                reader = csv.reader(io.TextIOWrapper(stream, encoding="utf-8-sig", newline=""))
                columns = None
                for row in reader:
                    if row[:3] == ["I", "DISPATCH", "REGIONSUM"]:
                        columns = {name: row.index(name) for name in
                                   ("REGIONID", "INTERVENTION", "SETTLEMENTDATE", "TOTALDEMAND")}
                    if row[:3] != ["D", "DISPATCH", "REGIONSUM"]:
                        continue
                    if columns is None:
                        raise ValueError(f"Missing header in {path}")
                    if row[columns["REGIONID"]] != "NSW1" or row[columns["INTERVENTION"]] != "0":
                        continue
                    timestamp = datetime.strptime(row[columns["SETTLEMENTDATE"]],
                                                  "%Y/%m/%d %H:%M:%S") - timedelta(minutes=5)
                    if not START <= timestamp < END:
                        continue
                    value = float(row[columns["TOTALDEMAND"]])
                    if timestamp in intervals and abs(intervals[timestamp] - value) > 1e-9:
                        raise ValueError(f"Conflicting values at {timestamp}")
                    intervals[timestamp] = value
    hours = defaultdict(list)
    for timestamp, value in intervals.items():
        hours[timestamp.replace(minute=0, second=0, microsecond=0)].append(value)
    output = {}
    for offset in range(int((END - START).total_seconds() // 3600)):
        timestamp = START + timedelta(hours=offset)
        values = hours[timestamp]
        if len(values) != 12:
            raise ValueError(f"Expected 12 dispatch intervals at {timestamp}, got {len(values)}")
        output[timestamp] = sum(values) / 12
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    paths = []
    for year, month in months():
        path = archive_path(year, month)
        if not args.skip_download:
            download(year, month, path)
        paths.append(path)
    values = aggregate(paths)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["date", "electricity_demand_aemo_nsw"])
        for timestamp, value in values.items():
            rendered = str(int(value)) if value.is_integer() else f"{value:.6f}"
            writer.writerow([timestamp.strftime("%Y-%m-%d %H:%M:%S"), rendered])
    metadata = {"source": BASE, "rows": len(values), "timezone": "AEST",
                "aggregation": "12 non-intervention NSW1 five-minute TOTALDEMAND values per hour",
                "start_inclusive": str(START), "end_exclusive": str(END),
                "output_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest()}
    OUTPUT.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Prepared {len(values)} hours: {OUTPUT.name}")


if __name__ == "__main__":
    main()
