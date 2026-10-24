#!/usr/bin/env python3
"""Persistence baseline: repeat the last observed value over the horizon.

Deterministic, no training. Uses the original 60/20/20 protocol and train-only
z-score standardization; only the output format differs from the research script.
"""

import argparse
import csv
import json
import math
from contextlib import nullcontext
from pathlib import Path

import torch

from at_vmd import DATASETS, PowerDataModule
from at_vmd.engine import PREDICTION_COLUMNS, write_csv, write_prediction_batch

ROOT = Path(__file__).resolve().parent


def evaluate_persistence(data: PowerDataModule, horizon: int, prediction_path=None):
    loader = data.loader("test", batch_size=64, seed=2021)
    se = ae = 0.0
    n = windows = 0
    output = (Path(prediction_path).open("w", newline="", encoding="utf-8")
              if prediction_path is not None else nullcontext())
    with torch.no_grad(), output as handle:
        writer = csv.writer(handle) if handle is not None else None
        if writer is not None:
            writer.writerow(PREDICTION_COLUMNS)
        for x, _marks, target in loader:
            pred = x[:, -1:, :].expand(-1, horizon, -1)
            err = pred - target
            se += err.square().sum().item()
            ae += err.abs().sum().item()
            n += target.numel()
            if writer is not None:
                write_prediction_batch(writer, pred, target, windows,
                                       data.dates[data.val_end:], [data.spec.target])
            windows += target.size(0)
    mse = se / n
    return {
        "model": "persistence", "dataset": data.dataset, "horizon": horizon,
        "mse": mse, "rmse": math.sqrt(mse), "mae": ae / n,
        "samples": n, "data": data.metadata(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS),
                        default=["entsoe", "entsoe_be", "entsoe_de_lu", "aemo", "pjm"])
    parser.add_argument("--horizons", nargs="+", type=int, choices=(96, 192, 336),
                        default=[96, 192, 336])
    parser.add_argument("--output-root", type=Path, default=ROOT / "results" / "persistence")
    args = parser.parse_args()
    rows = []
    for dataset in args.datasets:
        for horizon in args.horizons:
            data = PowerDataModule(dataset, ROOT / "data" / "processed", 336, horizon)
            output = args.output_root / dataset / f"h{horizon}"
            output.mkdir(parents=True, exist_ok=True)
            result = evaluate_persistence(data, horizon, output / "predictions.csv")
            config = {"model": "persistence", "dataset": dataset, "lookback": 336,
                      "horizon": horizon, "batch_size": 64, "clock": data.spec.clock,
                      "data": data.metadata(), "prediction_scale": "z-score",
                      "standardization": {"method": "z-score", "fit_split": "train",
                                          "mean": data.train_mean, "std": data.train_std}}
            (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
            write_csv(output / "history.csv", [], ["epoch", "train_mse", "val_mse", "val_mae"])
            row = {key: value for key, value in result.items() if key != "data"}
            row.update(split="test", samples=len(data.split("test")), elements=result["samples"])
            write_csv(output / "metrics.csv", [row])
            rows.append(row)
            print(f"{dataset:14s} h{horizon}: MSE={result['mse']:.4f}  MAE={result['mae']:.4f}")
    write_csv(args.output_root / "summary.csv", rows)


if __name__ == "__main__":
    main()
