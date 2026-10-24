#!/usr/bin/env python3
"""Five-seed load forecasting comparison."""
import argparse, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

from at_vmd import PAPER_SEEDS, train, DATASETS              # noqa: E402
from at_vmd.baselines import BENCHMARKS                      # noqa: E402

ALL_MODELS = ("atvmd",) + tuple(BENCHMARKS)
HORIZONS = (96, 192, 336)
CONFIGS = {h: ROOT / "configs" / f"power_h{h}.json" for h in HORIZONS}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", choices=ALL_MODELS, default=["atvmd"])
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=["entsoe_be"])
    p.add_argument("--horizons", nargs="+", type=int, choices=HORIZONS, default=[96])
    p.add_argument("--device", default="cuda")
    p.add_argument("--cache-dir", type=Path, default=ROOT / "cache")
    p.add_argument("--output-root", type=Path, default=ROOT / "results")
    p.add_argument("--seeds", nargs="+", type=int, choices=PAPER_SEEDS,
                   default=list(PAPER_SEEDS))
    p.add_argument("--keep-checkpoint", action="store_true")
    p.add_argument("--save-diagnostics", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    if args.save_diagnostics and any(model != "atvmd" for model in args.models):
        p.error("--save-diagnostics requires --models atvmd")
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = ROOT / output_root

    for ds in args.datasets:
        for h in args.horizons:
            cfg = json.loads(CONFIGS[h].read_text())
            cfg["data_dir"] = str(ROOT / "data" / "processed")
            for m in args.models:
                for seed in args.seeds:
                    d = output_root / ds / f"h{h}" / m / f"seed{seed}"
                    if (d / "metrics.csv").exists() and not args.force:
                        print(f"skip {m} {ds} h{h} s{seed}")
                        continue
                    print(f"run  {m} {ds} h{h} s{seed}", flush=True)
                    train(m, dict(cfg), ds, seed, d, args.device,
                          keep_checkpoint=args.keep_checkpoint,
                          cache_dir=args.cache_dir,
                          save_diagnostics=args.save_diagnostics)


if __name__ == "__main__":
    main()
