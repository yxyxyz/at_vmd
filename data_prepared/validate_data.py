"""Check the paper CSV schema and continuous hourly coverage without loading torch."""

import csv
import math
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLUGS = ("entsoe_belgium", "entsoe_germany_luxembourg", "entsoe_poland", "aemo_nsw", "pjm")
TARGETS = ("belgium", "germany_luxembourg", "poland", "aemo_nsw", "pjm")


def validate(path, target):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["date", target]:
            raise ValueError(f"Unexpected columns: {path.name}")
        count = 0
        for count, row in enumerate(reader, 1):
            expected = datetime(2021, 1, 1) + timedelta(hours=count - 1)
            if datetime.strptime(row["date"], "%Y-%m-%d %H:%M:%S") != expected:
                raise ValueError(f"Non-continuous timestamp: {path.name}, row {count}")
            value = float(row[target])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid load: {path.name}, row {count}")
        if count != 30216:
            raise ValueError(f"Expected 30216 rows: {path.name}, got {count}")


if __name__ == "__main__":
    found = 0
    for slug, target in zip(SLUGS, TARGETS):
        path = ROOT / "data/processed" / f"{slug}_hourly.csv"
        if path.exists():
            validate(path, f"electricity_demand_{target}")
            found += 1
            print(f"OK: {path.name}")
        else:
            print(f"Not supplied: {path.name}")
    if not found:
        raise SystemExit("No prepared datasets found.")
