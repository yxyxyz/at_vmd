#!/usr/bin/env python3
"""Core & backbone ablation studies."""
import argparse, copy, csv, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent

from at_vmd import PAPER_SEEDS, train  # noqa: E402
from at_vmd.engine import write_csv  # noqa: E402

CORE_VARIANTS = {
    "at_vmd":             {},
    "fixed_omega":        {"adaptive_omega": False},
    "adaptive_alpha":     {"adaptive_alpha": True},
    "global_omega":       {"control_type": "global"},
    "stop_task_gradient": {"task_gradient": False},
    "no_reconstruction":  {"use_reconstruction_constraint": False},
    "two_sweeps":         {"vmd_steps": 2},
}

BACKBONE_VARIANTS = {
    "tcn":              ("tcn", {}),
    "at_vmd_tcn":       ("atvmd", {"forecaster": "tcn"}),
    "lstm":             ("lstm", {}),
    "at_vmd_lstm":      ("atvmd", {"forecaster": "lstm"}),
    "dlinear":          ("dlinear", {}),
    "at_vmd_dlinear":   ("atvmd", {"forecaster": "dlinear"}),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--study", choices=("core", "backbone"), default="core")
    p.add_argument("--dataset", default="entsoe_be")
    p.add_argument("--horizon", type=int, choices=(96, 192, 336), default=336)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seeds", nargs="+", type=int, choices=PAPER_SEEDS,
                   default=list(PAPER_SEEDS))
    p.add_argument("--keep-checkpoint", action="store_true")
    p.add_argument("--save-diagnostics", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if args.study == "core":
        variants = {n: ("atvmd", o) for n, o in CORE_VARIANTS.items()}
    else:
        variants = BACKBONE_VARIANTS

    config = json.loads(open(ROOT / "configs" / f"power_h{args.horizon}.json").read())
    config["data_dir"] = str(ROOT / "data" / "processed")
    if args.study == "backbone":
        config["baseline_epochs"] = config["epochs"]
        config["baseline_patience"] = config["patience"]

    for variant, (model_name, overrides) in variants.items():
        cfg = copy.deepcopy(config); cfg.update(overrides)
        for seed in args.seeds:
            d = ROOT / "results" / "ablation" / args.study / args.dataset / f"h{args.horizon}" / variant / f"seed{seed}"
            if (d / "metrics.csv").exists() and not args.force:
                continue
            print(f"{variant} seed{seed}", flush=True)
            train(model_name, cfg, args.dataset, seed, d, args.device,
                  keep_checkpoint=args.keep_checkpoint,
                  save_diagnostics=args.save_diagnostics and model_name == "atvmd")

    # Summary
    study_dir = ROOT / "results" / "ablation" / args.study / args.dataset / f"h{args.horizon}"
    rows = []
    for vdir in sorted(study_dir.iterdir()):
        if not vdir.is_dir(): continue
        mse, mae = [], []
        for seed in args.seeds:
            path = vdir / f"seed{seed}" / "metrics.csv"
            if not path.exists(): continue
            with path.open(newline="") as handle:
                row = next(r for r in csv.DictReader(handle) if r["split"] == "test")
            mse.append(float(row["mse"]))
            mae.append(float(row["mae"]))
        if mse:
            rows.append({"variant": vdir.name, "runs": len(mse),
                         "mse_mean": float(np.mean(mse)), "mse_std": float(np.std(mse)),
                         "mae_mean": float(np.mean(mae)), "mae_std": float(np.std(mae))})
    write_csv(study_dir / "summary.csv", rows)
    for r in sorted(rows, key=lambda x: x["mse_mean"]):
        print(f'{r["variant"]}: {r["mse_mean"]:.4f}±{r["mse_std"]:.4f}')


if __name__ == "__main__":
    main()
