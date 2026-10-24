# Dataset Preparation

This directory contains preparation scripts only, adapted from the earlier
TaskVMD-public workflow. No raw or processed observations, credentials, or
checkpoints are distributed. Outputs go to Git-ignored `data/raw/` and
`data/processed/`, not this directory. Scripts require Python's standard library;
ENTSO-E and PJM downloading also requires `curl` on PATH.

## Commands

Run from the repository root:

```bash
python data_prepared/prepare_entsoe.py --areas BE DE_LU PL
python data_prepared/prepare_pjm.py
python data_prepared/prepare_aemo.py
python data_prepared/validate_data.py
```

ENTSO-E and EIA scripts prompt privately for your own API credentials, or read
`ENTSOE_SECURITY_TOKEN` and `EIA_API_KEY` from the environment. Do not put secrets
in source files or commit `.env` files. All three scripts support
`--skip-download` to rebuild from locally obtained raw files. Nothing is
downloaded automatically when installing or training the model.

The paper interval is **2021-01-01 00:00 through 2024-06-12 23:00**, 30,216 hourly
rows. June 13 is the exclusive boundary. ENTSO-E and PJM use UTC; AEMO uses AEST.
The optional extended profile in the ENTSO-E/PJM scripts is not the paper data.

## Processing Rules

- **ENTSO-E BE, DE-LU, PL:** Actual Total Load (`A65`, `A16`), aggregated using
  duration-weighted hourly means. Hours with 45-60 observed minutes are retained;
  lower partial coverage is rejected. Interior gaps up to seven hours can be
  linearly interpolated; affected hours and hashes are recorded in metadata.
  This is the legacy processing rule, not a claim of causal imputation.
  Rebuilding the available BE and DE-LU snapshots produced zero interpolated
  hours and matched the SHA256 hashes recorded in the experiments.
- **PJM / EIA-930:** Hourly demand (`respondent=PJM`, `type=D`). Missing,
  nonpositive, nonfinite, or above-500,000 values are repaired using the same UTC
  hour seven days earlier. The script rejects more than 0.5% missing hours and
  records all repairs. The original snapshot required 73 repaired hours;
  providers may revise historical observations.
  The available original PJM snapshot also reproduces the experiment's SHA256.
- **AEMO NSW:** Monthly MMSDM `DISPATCHREGIONSUM`, `REGIONID=NSW1`,
  `INTERVENTION=0`, `TOTALDEMAND`. Shift interval-end timestamps back five minutes
  and average twelve intervals per hour. Incomplete hours are rejected, not
  interpolated. Only NSW is prepared; the old ISO-NE branch is omitted.

Preparation preserves the original measurement scale. Train-only z-score
standardization occurs later in `data.py`. File names and target columns are
listed in the root README. New downloads may differ from the historical snapshot;
retain your metadata and hashes alongside experimental results.

