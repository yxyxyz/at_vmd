# AT-VMD

**Amortized Task-Guided Variational Mode Decomposition for Electricity Load Forecasting.**

Jointly learns input-conditional VMD centre frequencies ω(x) from forecasting
loss, with fixed bandwidth α and an adaptive reconstruction regularizer implemented by an
augmented penalty and projected dual updates. The encoder predicts per-sample ω;
one differentiable VMD sweep produces K modes and a reconstruction residual; a
pluggable forecaster head (TCN, LSTM, or DLinear) maps modes to predictions.

```text
Forward:
  x ──→ amortized encoder ──→ ω(x)   (α fixed at α_base = 2000)
  │                              │
  └──→ differentiable VMD ──→ modes (K=4; residual for ℒ_recon)
                                  │
                                  └──→ forecaster head ──→ ŷ

Backward (task-driven):
  ∇ℒ_pred ──→ forecaster ──→ VMD ──→ encoder ──→ update ω
  ∇ℒ_recon ──→ residual (dual ascent, λ updated per epoch)
```

## Data

| Dataset | Source | File | Rows | Frequency |
|---------|--------|------|------|-----------|
| ENTSO-E Belgium | ENTSO-E Transparency | `entsoe_belgium_hourly.csv` | 30,216 | 1h |
| ENTSO-E DE-LU | ENTSO-E Transparency | `entsoe_germany_luxembourg_hourly.csv` | 30,216 | 1h |
| ENTSO-E Poland | ENTSO-E Transparency | `entsoe_poland_hourly.csv` | 30,216 | 1h |
| AEMO NSW | AEMO | `aemo_nsw_hourly.csv` | 30,216 | 1h |
| PJM | EIA-930 (PJM demand) | `pjm_hourly.csv` | 30,216 | 1h |

Data are not distributed. Prepare the CSVs under `data/processed/` using the
scripts and instructions in [data_prepared/README.md](data_prepared/README.md).
The paper interval is 2021-01-01 through 2024-06-12 inclusive (June 13 exclusive).
Poland remains available as a supplementary dataset.

## Install

```bash
conda create -n atvmd python=3.10 -y
conda activate atvmd
pip install -e .
python -c "import at_vmd; print(at_vmd.__file__)"
python run.py --help
```

Run installation from this repository root. The editable installation provides
the `at_vmd` import name for both scripts and library use, regardless of the
checkout directory name. Reinstall it when moving to another checkout.
The repository, installation name, and Python package are all `at_vmd`;
`atvmd` remains the model identifier in experiment commands and result paths.

## Quick start

Run from the repository root after preparing the data. CUDA commands require a
compatible GPU. All training commands use five seeds unless `--seeds` is specified.

```bash
# Model comparison on Belgium power data (5 seeds each)
python run.py \
  --models atvmd vmdnet du-vmd dlinear tcn lstm \
  --datasets entsoe_be \
  --horizons 96 192 336 \
  --device cuda

# Core ablation — component switches
python ablation.py \
  --study core \
  --dataset entsoe_be --horizon 192 \
  --device cuda

# Backbone ablation — forecaster head comparison
python ablation.py \
  --study backbone \
  --dataset entsoe_be --horizon 192 \
  --device cuda

# Paper efficiency profile on real test windows
python profile_efficiency.py \
  --models atvmd du-vmd vmdnet tcn dlinear \
  --config configs/power_h336.json \
  --dataset entsoe_be --device cuda \
  --output results/analysis/efficiency_v2_be_h336.csv

# Persistence — repeats the last observation, no training or seed variation
python persistence_baseline.py --datasets entsoe_be --horizons 96 192 336

# Five runs with per-window control diagnostics
python run.py \
  --models atvmd --datasets entsoe_be --horizons 336 \
  --device cuda --save-diagnostics \
  --output-root results/diagnostics

# Paper control figure — 5 seeds, intervals taken across seeds
python plot_utils/plot_5seed_diagnostics.py \
  --diagnostics-dir results/diagnostics/entsoe_be/h336/atvmd

# Single-seed exploratory view (bin means, no uncertainty band)
python plot_utils/plot_control_diagnostics.py \
  --diagnostics results/diagnostics/entsoe_be/h336/atvmd/seed2023/diagnostics.npz \
  --output results/analysis/control_diagnostics.pdf
```

Use `plot_5seed_diagnostics.py` for anything published. The independent
replication unit is the **seed**, not the window: the ~5,700 test windows are
336-hour spans sliding by one hour, so consecutive windows share 335/336 of their
content and cannot be treated as independent samples. The five-seed script takes
one bin mean per seed and then a 95% t-interval across the five seeds (df = 4,
t = 2.776, ddof = 1). The single-seed script deliberately draws no band.

Ablation runs auto-generate `summary.csv` (MSE and MAE mean/std) under
`results/ablation/{study}/{dataset}/h{horizon}/`.
Runs with `metrics.csv` are skipped; `--force` reruns them. Each run overwrites its best
checkpoint, which is removed after evaluation unless `--keep-checkpoint` is used.
Only [results/report.md](results/report.md) is distributed, not individual run files.

Each training run saves four files:

| File | Contents |
|------|----------|
| `config.json` | Run configuration, seed, dataset hash/splits, clock, and training-set z-score mean/std |
| `metrics.csv` | Validation/test MSE and MAE, best epoch, window count, and AT-VMD reconstruction/control statistics |
| `history.csv` | Per-epoch learning rates, training/validation errors, reconstruction errors, and λ |
| `predictions.csv` | Test-set `window_id,forecast_time,horizon_step,channel,target,prediction` |

Predictions and targets are in **z-score units**, matching the reported metrics.
To recover original load units, multiply by the saved standard deviation and add
the saved mean. Window IDs are zero-based; horizon steps start at 1. Timestamps
use the dataset clock recorded in `config.json`. Overlapping forecasts are kept
as separate windows, not deduplicated by timestamp. CSV rows are streamed during
the final test evaluation without an additional model pass.

History learning rates are those used in that epoch; λ is recorded after its
update. Inapplicable fields are blank. Persistence has an empty, header-only
history because it is not trained. Optional `diagnostics.npz` and `checkpoint.pt`
remain separate. Efficiency measurements are saved as a CSV, with nested field
names expanded into dotted column names.

To rebuild the report after completing the experiment grid:

```bash
python summarize_results.py --results-root results --output results/report.md
```

The summarizer also reads archived `result.json` files, so existing experiments
do not need to be rerun to build the report.

## Models

Models take `[B, T, C]` inputs. Use `build_atvmd` for AT-VMD and
`at_vmd.baselines.build_benchmark` for the baselines.

| Name | Description | VMD |
|------|-------------|-----|
| `atvmd` | **Proposed**: amortized ω(x) encoder + differentiable VMD + forecaster | Differentiable |
| `vmdnet` | Convergent VMD (fixed α and initial ω) + TCN forecaster | Convergent |
| `du-vmd` | Deep-unrolled VMD (20 iterations) + TCN forecaster | Unrolled |
| `dlinear` | Trend-seasonal decomposition + linear projection | — |
| `tcn` | Raw TCN (no VMD) | — |
| `lstm` | Raw LSTM (no VMD) | — |
| `patchtst` | PatchTST (TSL vendored) | — |
| `itransformer` | iTransformer (TSL vendored) | — |

DU-VMD is our matched, jointly trained 20-sweep baseline, not the original
MAGN/UVMD implementation. See [THIRD_PARTY.md](at_vmd/baselines/THIRD_PARTY.md) for code provenance.

```python
from at_vmd import build_atvmd, build_benchmark
import json

config = json.load(open("configs/power_h96.json"))
model = build_atvmd(config)                       # AT-VMD with TCN head
model = build_atvmd({**config, "forecaster": "lstm"})  # AT-VMD with LSTM head
model = build_benchmark("dlinear", config)        # Plain DLinear
```

## Structure

```
at_vmd/
├── README.md
├── pyproject.toml         # Installs the at_vmd package
├── run.py                 # 5-seed model comparison CLI
├── ablation.py            # Core + backbone ablation studies
├── profile_efficiency.py  # Batch-1 latency, batch-64 throughput, and memory
├── persistence_baseline.py # Deterministic last-value baseline
├── summarize_results.py   # Report from CSV or archived JSON results
├── at_vmd/                # Importable source package
│   ├── __init__.py        # Public API exports
│   ├── model.py           # ATVMD, build_atvmd()
│   ├── encoder.py         # Window-conditional and global encoders
│   ├── vmd.py             # Convergent and differentiable VMD
│   ├── forecasters.py     # Mode-wise TCN, LSTM, and DLinear heads
│   ├── data.py            # PowerDataModule and window datasets
│   ├── engine.py          # Training, evaluation, CSV output, and VMD caching
│   ├── objectives.py      # Reconstruction regularizer and dual updates
│   └── baselines/
│       ├── __init__.py    # BENCHMARKS and build_benchmark()
│       ├── vmdnet.py
│       ├── du_vmd.py
│       ├── dlinear.py
│       ├── tcn.py
│       ├── lstm.py
│       ├── patchtst.py
│       ├── itransformer.py
│       ├── THIRD_PARTY.md # Code provenance and third-party notices
│       └── third_party/tsl/ # Minimal upstream files and license
├── configs/
│   ├── power_h96.json
│   ├── power_h192.json
│   └── power_h336.json
├── plot_utils/
│   ├── plot_5seed_diagnostics.py    # Paper figure — intervals across seeds
│   └── plot_control_diagnostics.py  # Single-seed exploratory view
├── data_prepared/         # Data preparation scripts and instructions; no data files
├── data/processed/        # User-prepared CSVs (not distributed)
└── results/               # Only report.md is distributed; generated outputs are ignored
```

## Protocol

| Parameter | Value |
|-----------|-------|
| **Seeds** | 2021, 2022, 2023, 2024, 2025 |
| **Split** | Chronological 60/20/20, train-only z-score standardization |
| **Lookback / Horizon** | 336 / 96, 192, 336 |
| **Batch size** | 64 |
| **Time features** | Hour, day-of-week, day-of-month, day-of-year (all [−0.5, 0.5]) |
| **VMD modes (K)** | 4 |
| **α (bandwidth)** | 2000 (fixed) |
| **VMD sweeps** | AT-VMD: 1; DU-VMD: 20 |
| **VMDNet convergence** | max 500 iterations, batch-mean tolerance 1e−7 |
| **Encoder** | Hidden dim 32, K window-conditioned centre frequencies |
| **Initial frequency range** | ω⁰_k = clip(ω̄_k + (0.25/K)·tanh(·), 0, 0.5−1e−6); maximum shift ±0.0625 at K=4 |
| **TCN head** | 2 layers × 64 channels, dropout 0.2, fusion hidden 128 |
| **LSTM head** | 2 layers × 64 hidden, dropout 0.1 |
| **DLinear head** | Kernel size 25 |

| Training | AT-VMD | Baselines (VMDNet, DU-VMD, etc.) |
|----------|--------|----------------------------------|
| **Epochs** | 10 | 10 |
| **Patience** | 5 | 5 |
| **LR (forecaster)** | 1e−3 | 1e−3 |
| **LR (encoder)** | 1e−4 | — |
| **LR schedule** | Forecaster fixed for epochs 1–2, then ×0.5/epoch; encoder fixed | Fixed for epochs 1–2, then ×0.5/epoch |
| **Gradient clipping** | 1.0 | 1.0 |

| Reconstruction regularizer (AT-VMD only) | Value |
|---------------------------|-------|
| **Prediction weight** | 10.0 |
| **Reconstruction set-point ε** | 0.025 (regularization set-point, not a guaranteed bound) |
| **Reconstruction penalty (ρ)** | 5.0 |
| **Initial λ / Dual LR** | 0 / 0.1 |
| **Dual max (λ_max)** | 100.0 |
| **Update interval** | Per epoch |

VMDNet caches chronological decompositions before shuffling training samples.
The cache key includes the decomposition batch size because convergence uses a
batch-mean criterion. Changing batch size creates a new cache; the default stays 64.
For exact repeat runs, keep the cache state consistent: creating a cold cache
iterates additional data loaders and consumes random numbers, so cold-cache and
warm-cache training can differ slightly even with the same seed. This release
preserves the original training and caching behavior.

## License

See [LICENSE](LICENSE) and [THIRD_PARTY.md](at_vmd/baselines/THIRD_PARTY.md) for third-party notices.
