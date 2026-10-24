"""Build a compact report from five-seed results (standard library only)."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev

SEEDS = (2021, 2022, 2023, 2024, 2025)
HORIZONS = (96, 192, 336)
DATASETS = {"entsoe_be": "ENTSO-E Belgium", "entsoe_de_lu": "ENTSO-E DE-LU",
            "aemo": "AEMO NSW", "pjm": "PJM / EIA-930"}
MODELS = {"atvmd": "AT-VMD", "vmdnet": "VMDNet", "du-vmd": "DU-VMD",
          "dlinear": "DLinear", "patchtst": "PatchTST", "itransformer": "iTransformer"}
VARIANTS = {"at_vmd": "AT-VMD", "stop_task_gradient": "w/o task gradient",
            "fixed_omega": "w/o learned initial frequencies",
            "global_omega": "w/o window-conditional initial frequencies",
            "no_reconstruction": "w/o reconstruction regularizer",
            "adaptive_alpha": "w/ adaptive bandwidth", "two_sweeps": "w/ two sweeps"}


def records(directory):
    return [read_result(directory / f"seed{s}") for s in SEEDS]


def read_result(directory):
    path = directory / "metrics.csv"
    if path.exists():
        with path.open(newline="") as handle:
            row = next(r for r in csv.DictReader(handle) if r["split"] == "test")
        return {"test": {key: float(row[key]) for key in ("mse", "mae")}}
    # Existing research runs remain readable without converting or rerunning them.
    result = json.loads((directory / "result.json").read_text())
    return result if "test" in result else {"test": result}


def read_profile(path):
    if path.suffix == ".json":
        return json.loads(path.read_text())
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    results = []
    for row in rows:
        results.append({
            "model": row["model"], "parameters": int(row["parameters"]),
            "latency": {"median_ms": float(row["latency.median_ms"])},
            "throughput_windows_per_second": float(row["throughput_windows_per_second"]),
            "incremental_forward_memory_mb": {
                name: (float(row[f"incremental_forward_memory_mb.{name}"])
                       if row[f"incremental_forward_memory_mb.{name}"] else None)
                for name in ("latency_batch", "throughput_batch")
            },
        })
    return {**rows[0], "results": results}


def metric(directory, name):
    return [r["test"][name] for r in records(directory)]


def format_metric(values):
    return f"{mean(values):.4f} +/- {pstdev(values):.4f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/report.md"))
    args = parser.parse_args()
    root = args.results_root
    lines = ["# AT-VMD Experiment Results", "",
             "Five seeds: 2021-2025. Values are mean +/- population standard deviation",
             "across seeds (ddof=0); lower MSE and MAE are better. Input length 336,",
             "chronological 60/20/20 split, train-only z-score standardization, batch 64.",
             "Training: 10 epochs maximum, patience 5; forecaster LR 1e-3 and encoder LR 1e-4.",
             "These are recorded experimental results, not new runs performed for this release.", "",
             "## Forecasting Comparison", ""]
    for dataset, label in {**DATASETS, "entsoe": "ENTSO-E Poland (supplementary)"}.items():
        lines += [f"### {label}", "", "Each cell contains MSE / MAE.", "",
                  "| Model | H=96 | H=192 | H=336 |", "|---|---|---|---|"]
        for model, name in MODELS.items():
            cells = []
            for h in HORIZONS:
                path = root / dataset / f"h{h}" / model
                cells.append(" / ".join(format_metric(metric(path, m)) for m in ("mse", "mae")))
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        persistence = [root / "persistence" / dataset / f"h{h}" for h in HORIZONS]
        if all((p / "metrics.csv").exists() or (p / "result.json").exists()
               for p in persistence):
            cells = [" / ".join(f"{read_result(p)['test'][m]:.4f}"
                                for m in ("mse", "mae")) for p in persistence]
            lines.append("| Persistence | " + " | ".join(cells) + " |")
        lines.append("")

    lines += ["Persistence repeats the last observed value across the forecast horizon.",
              "It is deterministic and has no seed standard deviation.", ""]

    lines += ["## Component Ablation", "", "ENTSO-E Belgium, H=336; same five seeds.", "",
              "| Variant | MSE | MAE |", "|---|---|---|"]
    for variant, label in VARIANTS.items():
        path = root / "ablation/core/entsoe_be/h336" / variant
        lines.append(f"| {label} | {format_metric(metric(path, 'mse'))} | "
                     f"{format_metric(metric(path, 'mae'))} |")
    lines += ["", "The fixed-frequency variant uses uniform initialization; the global variant",
              "learns one initialization shared across windows. The task-gradient variant",
              "detaches the modes and post-sweep frequencies on the prediction path only.", ""]

    lines += ["## Downstream Forecaster Compatibility", "",
              "Skill (%) = 100 * (1 - AT-VMD error / raw-forecaster error), computed from",
              "five-seed mean errors per dataset, then averaged equally over the four main",
              "datasets. Positive values favor AT-VMD with that forecaster; negative values",
              "favor the raw forecaster. Poland is excluded from this paper summary.", "",
              "| AT-VMD forecaster | Horizon | MSE skill (%) | MAE skill (%) |",
              "|---|---|---|---|"]
    for forecaster in ("tcn", "dlinear", "lstm"):
        for h in HORIZONS:
            skills = []
            for m in ("mse", "mae"):
                values = []
                for ds in DATASETS:
                    base = root / "ablation/backbone" / ds / f"h{h}"
                    values.append(100 * (1 - mean(metric(base / f"at_vmd_{forecaster}", m))
                                         / mean(metric(base / forecaster, m))))
                skills.append(f"{mean(values):+.2f}")
            lines.append(f"| {forecaster.upper() if forecaster != 'dlinear' else 'DLinear'} "
                         f"| {h} | {' | '.join(skills)} |")
    lines.append("")

    profile = next((root / "analysis" / name for name in (
        "efficiency_v2_be_h336.csv", "efficiency_v2_be_h336.json", "efficiency_v2.csv")
        if (root / "analysis" / name).exists()), None)
    if profile is not None:
        data = read_profile(profile)
        label = DATASETS.get(data["dataset"], data["dataset"])
        horizon = data.get("horizon", 336)
        lines += ["## Computational Efficiency", "",
                  f"{label}, H={horizon}; {data['device_name']}, PyTorch {data['torch_version']}.",
                  "Latency is the median at batch 1; throughput uses batch 64. Memory is peak",
                  "incremental CUDA allocation during forward, in MiB (bytes / 1024^2),",
                  "not total process/GPU memory. Host-to-device transfer is excluded.", "",
                  "| Model | Parameters | Latency (ms) | Throughput (windows/s) | Memory B=1 (MiB) | Memory B=64 (MiB) |",
                  "|---|---|---|---|---|---|"]
        for row in data["results"]:
            name = MODELS.get(row["model"], "TCN forecaster")
            memory = row["incremental_forward_memory_mb"]
            memory_cells = [f"{memory[key]:.3f}" if memory[key] is not None else "N/A"
                            for key in ("latency_batch", "throughput_batch")]
            lines.append(f"| {name} | {row['parameters']:,} | {row['latency']['median_ms']:.3f} "
                         f"| {row['throughput_windows_per_second']:.1f} "
                         f"| {memory_cells[0]} | {memory_cells[1]} |")
        lines.append("")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
