"""Training loop with deterministic VMDNet decomposition caching."""

import csv
from contextlib import nullcontext
import hashlib
import inspect
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import PowerDataModule
from .model import build_atvmd
from .baselines import build_benchmark
from .objectives import ReconstructionConstraint, relative_reconstruction_error
from .vmd import ConvergentVMD


PAPER_SEEDS = (2021, 2022, 2023, 2024, 2025)

PREDICTION_COLUMNS = ("window_id", "forecast_time", "horizon_step", "channel",
                      "target", "prediction")


def _flatten(record, prefix=""):
    result = {}
    for key, value in record.items():
        name = f"{prefix}{key}"
        if isinstance(value, (list, tuple)):
            value = dict(enumerate(value))
        if isinstance(value, dict):
            result.update(_flatten(value, name + "."))
        else:
            result[name] = value
    return result


def write_csv(path, rows, fieldnames=None):
    """Write scalar columns; expand nested diagnostics without JSON in cells."""
    rows = [_flatten(row) for row in rows]
    columns = fieldnames or list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_prediction_batch(writer, pred, target, offset, forecast_times, channels):
    """Stream chronological test windows in z-score units, retaining overlaps."""
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    writer.writerows(
        (offset + i, forecast_times[offset + i + h], h + 1, channels[c],
         float(target[i, h, c]), float(pred[i, h, c]))
        for i in range(len(pred)) for h in range(pred.shape[1])
        for c in range(pred.shape[2])
    )


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def _move(batch, device):
    return tuple(t.to(device) for t in batch)


def _vmd_cache_key(data, config):
    solver_source = inspect.getsource(ConvergentVMD)
    payload = {
        "data_sha256": data.sha256,
        "lookback": int(config["lookback"]),
        "horizon": int(config["horizon"]),
        # Batch-mean stopping makes decomposition depend on the batch grouping.
        "batch_size": int(config["batch_size"]),
        "modes": int(config["modes"]),
        "alpha": float(config["alpha_base"]),
        "tau": float(config["vmd_tau"]),
        "max_iterations": int(config.get("vmdnet_max_iterations", 500)),
        "tolerance": float(config.get("vmdnet_tolerance", 1e-7)),
        "solver_sha256": hashlib.sha256(solver_source.encode()).hexdigest(),
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# Cached inputs must remain usable by the forecaster's backward pass.
@torch.no_grad()
def _vmd_cached_loaders(model, data, config, cache_dir, device, seed):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _vmd_cache_key(data, config)
    loaders = {}

    for split in ("train", "val", "test"):
        path = cache_dir / f"{data.dataset}-{key}-{split}.pt"
        if path.exists():
            tensors = torch.load(path, map_location="cpu", weights_only=True)
        else:
            source = DataLoader(
                data.split(split), batch_size=int(config["batch_size"]),
                shuffle=False, num_workers=0)
            parts = [[], [], [], []]
            for x, marks, target in source:
                modes, omega = model.decompose(x.to(device))
                values = modes.cpu(), marks, omega.cpu(), target
                for destination, value in zip(parts, values):
                    destination.append(value)
            tensors = tuple(torch.cat(part) for part in parts)
            torch.save(tensors, path)

        generator = torch.Generator().manual_seed(seed) if split == "train" else None
        loaders[split] = DataLoader(
            TensorDataset(*tensors), batch_size=int(config["batch_size"]),
            shuffle=split == "train", generator=generator, num_workers=0)
    return loaders


@torch.inference_mode()
def evaluate(model, loader, device, is_atvmd=False, control_statistics=False,
             prediction_path=None, forecast_times=None, channels=None):
    model.eval()
    se = ae = recon = 0.0
    el = samples = 0
    alpha_values, omega_control_values, omega_values = [], [], []
    output = (Path(prediction_path).open("w", newline="", encoding="utf-8")
              if prediction_path is not None else nullcontext())
    with output as handle:
        writer = csv.writer(handle) if handle is not None else None
        if writer is not None:
            writer.writerow(PREDICTION_COLUMNS)
        for batch in loader:
            if len(batch) == 4:
                modes, marks, omega, target = _move(batch, device)
                pred = model.forecast(modes, marks, omega)
            else:
                x, marks, target = _move(batch, device)
                out = model(x, marks)
                pred = out["prediction"] if is_atvmd else out
                if is_atvmd:
                    recon += relative_reconstruction_error(x, out["residual"]).sum().item()
                    if control_statistics:
                        alpha_values.append(out["alpha"].cpu())
                        omega_control_values.append(out["omega_control"].cpu())
                        omega_values.append(out["omega"].cpu())
            if writer is not None:
                write_prediction_batch(writer, pred, target, samples, forecast_times, channels)
            err = pred - target
            se += err.square().sum().item(); ae += err.abs().sum().item()
            el += target.numel(); samples += target.size(0)
    r = {"mse": se / el, "mae": ae / el, "samples": samples}
    if is_atvmd:
        r["reconstruction_error"] = recon / samples
        if control_statistics:
            for name, values in (
                ("alpha", alpha_values),
                ("omega_control", omega_control_values),
                ("omega", omega_values),
            ):
                tensor = torch.cat(values)
                r[f"{name}_mean"] = tensor.mean(dim=(0, 2)).tolist()
                r[f"{name}_std"] = (
                    tensor.permute(1, 0, 2).flatten(1).std(dim=1, unbiased=False).tolist()
                )
    return r


@torch.inference_mode()
def export_diagnostics(model, loader, device, path):
    """Export per-window AT-VMD controls and signal diagnostics."""
    model.eval()
    rows = {
        "input_window": [],
        "omega_control": [],
        "omega_after_sweep": [],
        "ramp_intensity": [],
        "high_frequency_ratio": [],
        "prediction_mse": [],
        "reconstruction_error": [],
    }
    for x, marks, target in loader:
        x, marks, target = _move((x, marks, target), device)
        out = model(x, marks)
        spectrum = torch.fft.rfft(x, dim=1)
        power = spectrum.abs().square()
        high = power[:, power.size(1) // 2:].sum(dim=1)
        total = power.sum(dim=1).clamp_min(1e-12)

        rows["input_window"].append(x.cpu())
        rows["omega_control"].append(out["omega_control"].cpu())
        rows["omega_after_sweep"].append(out["omega"].cpu())
        rows["ramp_intensity"].append(x.diff(dim=1).abs().mean(dim=1).cpu())
        rows["high_frequency_ratio"].append((high / total).cpu())
        rows["prediction_mse"].append(
            (out["prediction"] - target).square().mean(dim=1).cpu()
        )
        rows["reconstruction_error"].append(
            relative_reconstruction_error(x, out["residual"]).cpu()
        )

    arrays = {name: torch.cat(values).numpy() for name, values in rows.items()}
    arrays["window_index"] = np.arange(len(arrays["ramp_intensity"]))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def train(model_name, config, dataset, seed, output_dir, device_name="auto",
          keep_checkpoint=False, cache_dir=None, save_diagnostics=False):
    assert seed in PAPER_SEEDS
    set_seed(seed)
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device_name)
    out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    # metrics.csv is written last and serves as the completed-run marker.
    (out_dir / "metrics.csv").unlink(missing_ok=True)
    ckpt = out_dir / "checkpoint.pt"

    data = PowerDataModule(dataset, config["data_dir"], int(config["lookback"]), int(config["horizon"]))
    loaders = {s: data.loader(s, int(config["batch_size"]), seed) for s in ("train", "val", "test")}

    config = dict(config)
    is_atvmd = (model_name == "atvmd")
    if is_atvmd:
        model = build_atvmd(config).to(dev)
    else:
        model = build_benchmark(model_name, config).to(dev)
    if model_name == "vmdnet" and cache_dir is not None:
        loaders = _vmd_cached_loaders(
            model, data, config, cache_dir, dev, seed)

    if is_atvmd:
        opt = torch.optim.Adam([
            {"params": model.forecaster.parameters(), "lr": float(config["forecaster_lr"])},
            {"params": model.encoder.parameters(), "lr": float(config["encoder_lr"])},
        ])
    else:
        opt = torch.optim.Adam(model.parameters(), lr=float(config.get("baseline_lr", 1e-3)))

    use_constraint = is_atvmd and bool(config.get("use_reconstruction_constraint", True))
    constraint = ReconstructionConstraint(float(config["reconstruction_tolerance"]),
                                          float(config["dual_lr"]),
                                          max_value=float(config["dual_max"])) if use_constraint else None
    dual_interval = config.get("dual_update_interval", "epoch")

    epochs = int(config.get("epochs", 20) if is_atvmd else config.get("baseline_epochs", 10))
    patience = int(config.get("patience", 5) if is_atvmd else config.get("baseline_patience", 3))
    clip = float(config["grad_clip"])
    best_score, best_ep, stale = float("inf"), 0, 0
    history = []
    forecaster_lr = float(config["forecaster_lr"]) if is_atvmd else float(config["baseline_lr"])

    for epoch in range(1, epochs + 1):
        model.train()
        tr_se = tr_el = tr_rec = tr_samples = 0.0
        for batch in loaders["train"]:
            opt.zero_grad(set_to_none=True)
            if len(batch) == 4:
                # Cached VMDNet
                modes, marks, omega, target = _move(batch, dev)
                pred = model.forecast(modes, marks, omega)
                pred_loss = (pred - target).square().mean()
                loss = pred_loss
            elif is_atvmd:
                # ATVMD — prediction + reconstruction loss
                x, marks, target = _move(batch, dev)
                out = model(x, marks)
                pred = out["prediction"]
                pred_loss = (pred - target).square().mean()
                rec_loss = relative_reconstruction_error(x, out["residual"]).mean()
                tr_rec += float(rec_loss.detach()) * target.size(0)
                if use_constraint:
                    v = torch.relu(rec_loss - constraint.tolerance)
                    loss = float(config["prediction_weight"]) * pred_loss \
                           + constraint.value * rec_loss \
                           + 0.5 * float(config["reconstruction_penalty"]) * v.square()
                else:
                    loss = float(config["prediction_weight"]) * pred_loss
                if dual_interval == "batch" and use_constraint:
                    constraint.update(float(rec_loss.detach()))
            else:
                # Plain baseline
                x, marks, target = _move(batch, dev)
                out = model(x, marks)
                pred = out
                pred_loss = (pred - target).square().mean()
                loss = pred_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()

            tr_se += float(pred_loss.detach()) * target.numel()
            tr_el += target.numel()
            tr_samples += target.size(0)

        if is_atvmd and dual_interval == "epoch" and use_constraint:
            constraint.update(tr_rec / tr_samples)

        val = evaluate(model, loaders["val"], dev, is_atvmd)
        score = float(val["mse"])

        history.append({"epoch": epoch, "train_mse": tr_se / tr_el, "validation": val})
        history[-1]["learning_rate"] = opt.param_groups[0]["lr"]
        if is_atvmd:
            history[-1]["encoder_learning_rate"] = opt.param_groups[1]["lr"]
        if is_atvmd:
            history[-1]["train_reconstruction"] = tr_rec / tr_samples
            if use_constraint: history[-1]["dual"] = constraint.value

        if score <= best_score:
            best_score, best_ep, stale = score, epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt)
        else:
            stale += 1

        # Released schedule: epochs 1-2 use the base LR, then it halves each epoch.
        next_epoch_exponent = max(0, epoch - 1)
        opt.param_groups[0]["lr"] = forecaster_lr * (0.5 ** next_epoch_exponent)

        if stale >= patience: break

    sd = torch.load(ckpt, map_location=dev, weights_only=True)
    model.load_state_dict(sd["model"])
    result = {"model": model_name, "dataset": dataset, "seed": seed,
              "best_epoch": best_ep, "data": data.metadata(),
              "config": config,
              "validation": evaluate(
                  model, loaders["val"], dev, is_atvmd,
                  control_statistics=is_atvmd),
              "test": evaluate(
                  model, loaders["test"], dev, is_atvmd,
                  control_statistics=is_atvmd,
                  prediction_path=out_dir / "predictions.csv",
                  forecast_times=data.dates[data.val_end:],
                  channels=[data.column or data.spec.target]),
              "history": history}
    if save_diagnostics:
        if not is_atvmd:
            raise ValueError("Per-window diagnostics are available only for AT-VMD")
        diagnostics_path = out_dir / "diagnostics.npz"
        export_diagnostics(model, loaders["test"], dev, diagnostics_path)
        result["diagnostics"] = diagnostics_path.name
    saved_config = {
        **config, "model": model_name, "dataset": dataset, "seed": seed,
        "data": data.metadata(), "clock": data.spec.clock,
        "standardization": {"method": "z-score", "fit_split": "train",
                            "mean": data.train_mean, "std": data.train_std},
        "prediction_scale": "z-score",
    }
    (out_dir / "config.json").write_text(json.dumps(saved_config, indent=2) + "\n")
    history_rows = []
    for row in history:
        history_rows.append({
            "epoch": row["epoch"], "learning_rate": row["learning_rate"],
            "encoder_learning_rate": row.get("encoder_learning_rate"),
            "train_mse": row["train_mse"], "val_mse": row["validation"]["mse"],
            "val_mae": row["validation"]["mae"],
            "train_recon": row.get("train_reconstruction"),
            "val_recon": row["validation"].get("reconstruction_error"),
            "lambda": row.get("dual"),
        })
    write_csv(out_dir / "history.csv", history_rows)
    write_csv(out_dir / "metrics.csv", [
        {"model": model_name, "dataset": dataset, "seed": seed,
         "best_epoch": best_ep, "split": split, **result[split]}
        for split in ("validation", "test")
    ])
    if not keep_checkpoint: ckpt.unlink(missing_ok=True)
    return result
