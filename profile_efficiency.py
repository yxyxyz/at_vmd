#!/usr/bin/env python3
"""Measure inference efficiency on real standardized test windows."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent

from at_vmd import DATASETS, PowerDataModule, build_atvmd, build_benchmark  # noqa: E402
from at_vmd.engine import set_seed, write_csv  # noqa: E402


DEFAULT_MODELS = ("atvmd", "du-vmd", "vmdnet", "tcn", "dlinear")
VMD_MODELS = {"atvmd", "du-vmd", "vmdnet"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "power_h336.json",
    )
    parser.add_argument("--dataset", choices=list(DATASETS), default="entsoe")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--latency-batch-size", type=int, default=1)
    parser.add_argument("--throughput-batch-size", type=int, default=64)
    parser.add_argument("--latency-batches", type=int, default=50)
    parser.add_argument("--throughput-batches", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "analysis" / "efficiency_v2.csv",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_model(name: str, config: dict) -> torch.nn.Module:
    return build_atvmd(config) if name == "atvmd" else build_benchmark(name, config)


def fixed_batches(dataset, batch_size: int, count: int):
    """Select full, deterministic batches evenly across a split."""
    if batch_size > len(dataset):
        raise ValueError(f"batch size {batch_size} exceeds split size {len(dataset)}")
    starts = np.linspace(
        0,
        len(dataset) - batch_size,
        num=min(count, len(dataset) - batch_size + 1),
        dtype=int,
    )
    batches = []
    for first in np.unique(starts):
        items = [dataset[index] for index in range(int(first), int(first) + batch_size)]
        batches.append(
            (
                torch.stack([item[0] for item in items]),
                torch.stack([item[1] for item in items]),
            )
        )
    return batches


def summarize_ms(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(array.mean()),
        "std_ms": float(array.std(ddof=0)),
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95)),
    }


def time_forward(model, cpu_batches, device, warmup: int) -> list[float]:
    x0, marks0 = (tensor.to(device) for tensor in cpu_batches[0])
    with torch.inference_mode():
        for _ in range(warmup):
            model(x0, marks0)
        synchronize(device)

    durations = []
    with torch.inference_mode():
        for x_cpu, marks_cpu in cpu_batches:
            x = x_cpu.to(device)
            marks = marks_cpu.to(device)
            synchronize(device)
            started = time.perf_counter()
            output = model(x, marks)
            synchronize(device)
            durations.append((time.perf_counter() - started) * 1000.0)
            del output, x, marks
    return durations


def decompose(model_name: str, model, x: torch.Tensor):
    """Return ((modes, omega), iterations) for the three VMD models."""
    if x.ndim == 2:
        x = x.unsqueeze(-1)
    modes_all, omega_all, iterations = [], [], []

    for channel in range(x.size(-1)):
        signal = x[:, :, channel]
        if model_name == "atvmd":
            controls = model._encode(signal)
            modes, _, omega = model.vmd.differentiable_decompose(
                signal, controls["alpha"], controls["omega"], model.vmd_steps
            )
        elif model_name == "du-vmd":
            alpha = model.alpha_base * torch.exp(torch.tanh(model.alpha_delta))
            alpha = alpha.unsqueeze(0).expand(signal.size(0), -1).to(signal)
            omega_init = model.w0.unsqueeze(0).expand(signal.size(0), -1).to(signal)
            modes, _, omega = model.vmd(signal, alpha, omega_init)
        elif model_name == "vmdnet":
            alpha = torch.full(
                (signal.size(0), model.K),
                model.alpha,
                device=signal.device,
                dtype=signal.dtype,
            )
            omega_init = model.w0.unsqueeze(0).expand(signal.size(0), -1).to(signal)
            result = model.vmd.solve(signal, alpha, omega_init)
            modes, omega = result.modes, result.omega
            iterations.append(result.iterations)
        else:
            raise ValueError(f"{model_name!r} has no VMD stage")
        modes_all.append(modes)
        omega_all.append(omega)

    representation = (
        torch.stack(modes_all, dim=-1),
        torch.stack(omega_all, dim=-1),
    )
    return representation, iterations


def forecast(model_name: str, model, representation, marks: torch.Tensor):
    modes, omega = representation
    if model_name == "vmdnet":
        return model.forecast(modes, marks, omega)

    head = model.forecaster if model_name == "atvmd" else model.fc
    outputs = [
        head(modes[:, :, :, channel], marks, omega[:, :, channel])
        for channel in range(modes.size(-1))
    ]
    return torch.stack(outputs, dim=-1)


def time_vmd_stages(model_name, model, cpu_batches, device) -> dict:
    decomposition_ms, forecast_ms, iteration_counts = [], [], []
    with torch.inference_mode():
        for x_cpu, marks_cpu in cpu_batches:
            x = x_cpu.to(device)
            marks = marks_cpu.to(device)

            synchronize(device)
            started = time.perf_counter()
            representation, iterations = decompose(model_name, model, x)
            synchronize(device)
            decomposition_ms.append((time.perf_counter() - started) * 1000.0)

            synchronize(device)
            started = time.perf_counter()
            output = forecast(model_name, model, representation, marks)
            synchronize(device)
            forecast_ms.append((time.perf_counter() - started) * 1000.0)

            iteration_counts.extend(iterations)
            del output, representation, x, marks

    result = {
        "decomposition": summarize_ms(decomposition_ms),
        "forecaster": summarize_ms(forecast_ms),
    }
    if iteration_counts:
        result["convergent_vmd_iterations"] = {
            "mean": statistics.fmean(iteration_counts),
            "median": statistics.median(iteration_counts),
            "p95": float(np.percentile(iteration_counts, 95)),
            "min": min(iteration_counts),
            "max": max(iteration_counts),
        }
    return result


def incremental_memory(model, cpu_batch, device) -> float | None:
    if device.type != "cuda":
        return None
    x, marks = (tensor.to(device) for tensor in cpu_batch)
    synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)
    with torch.inference_mode():
        output = model(x, marks)
        synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    del output, x, marks
    return (peak - baseline) / (1024**2)


def profile_model(
    name: str,
    config: dict,
    device: torch.device,
    latency_inputs,
    throughput_inputs,
    warmup: int,
) -> dict:
    set_seed(2021)
    model = build_model(name, config).to(device).eval()

    latency = summarize_ms(time_forward(model, latency_inputs, device, warmup))
    throughput_timing = summarize_ms(
        time_forward(model, throughput_inputs, device, warmup)
    )
    batch_size = int(throughput_inputs[0][0].size(0))
    throughput = batch_size * 1000.0 / throughput_timing["median_ms"]

    result = {
        "model": name,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "latency_batch_size": int(latency_inputs[0][0].size(0)),
        "latency": latency,
        "throughput_batch_size": batch_size,
        "throughput_timing": throughput_timing,
        "throughput_windows_per_second": throughput,
        "incremental_forward_memory_mb": {
            "latency_batch": incremental_memory(model, latency_inputs[0], device),
            "throughput_batch": incremental_memory(
                model, throughput_inputs[0], device
            ),
        },
    }
    if name in VMD_MODELS:
        result["stages_batch1"] = time_vmd_stages(
            name, model, latency_inputs, device
        )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    numeric = (
        args.latency_batch_size,
        args.throughput_batch_size,
        args.latency_batches,
        args.throughput_batches,
    )
    if any(value < 1 for value in numeric) or args.warmup < 0:
        raise ValueError("batch sizes and batch counts must be positive")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    data_dir = Path(config["data_dir"])
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    data = PowerDataModule(
        args.dataset,
        data_dir,
        int(config["lookback"]),
        int(config["horizon"]),
    )
    test = data.split("test")
    latency_inputs = fixed_batches(
        test, args.latency_batch_size, args.latency_batches
    )
    throughput_inputs = fixed_batches(
        test, args.throughput_batch_size, args.throughput_batches
    )
    device = resolve_device(args.device)

    results = []
    for name in args.models:
        print(f"profile {name}", flush=True)
        results.append(
            profile_model(
                name,
                config,
                device,
                latency_inputs,
                throughput_inputs,
                args.warmup,
            )
        )

    payload = {
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        ),
        "torch_version": torch.__version__,
        "dataset": args.dataset,
        "lookback": int(config["lookback"]),
        "horizon": int(config["horizon"]),
        "split": "test",
        "config": str(args.config),
        "input_policy": (
            "Full standardized test batches sampled evenly in chronological order; "
            "host-to-device transfer excluded from timed inference."
        ),
        "metric_policy": (
            "Batch-1 wall-clock latency and batched throughput are reported separately. "
            "Memory is reported separately for the latency and throughput batches. "
            "Profiler-derived MACs are omitted because FFT, complex, and fused recurrent "
            "operations are not counted consistently."
        ),
        "data": {
            "rows": data.rows,
            "sha256": data.sha256,
            "train_end": data.train_end,
            "val_end": data.val_end,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload.pop("results")
    write_csv(args.output, [{**payload, **row} for row in results])
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
