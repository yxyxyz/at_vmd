#!/usr/bin/env python3
"""Export real AT-VMD waveforms for the architecture figure.

The script runs one standardized test window through a saved AT-VMD checkpoint
and writes editable SVG/PDF assets for the input, target, prediction, VMD modes,
and residual. It also stores the underlying arrays in ``waveforms.npz``.
"""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]  # repo root (at_vmd/)

from at_vmd.data import PowerDataModule  # noqa: E402
from at_vmd.model import build_atvmd  # noqa: E402


COLORS = {
    "input": "#222222",
    "target": "#174A7E",
    "prediction": "#B33A3A",
    "residual": "#6B6B6B",
}
MODE_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export real input, forecast, target, and VMD-mode waveforms."
    )
    parser.add_argument("--dataset", default="entsoe_be")
    parser.add_argument("--horizon", type=int, choices=(96, 192, 336), default=336)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--window-index", type=int)
    parser.add_argument(
        "--variability-quantile",
        type=float,
        default=0.8,
        help="Input-variability quantile used when --window-index is omitted.",
    )
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "analysis" / "architecture_waveforms",
    )
    return parser.parse_args()


def resolve_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def select_window(dataset, quantile):
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("--variability-quantile must lie in [0, 1]")
    scores = np.empty(len(dataset), dtype=np.float64)
    for index in range(len(dataset)):
        x, _, _ = dataset[index]
        scores[index] = x.diff(dim=0).abs().mean().item()
    target = np.quantile(scores, quantile)
    return int(np.argmin(np.abs(scores - target))), float(target)


def save_line(values, path, color, linewidth=1.6, dashed=False):
    fig, ax = plt.subplots(figsize=(3.2, 1.0))
    ax.plot(
        np.arange(len(values)),
        values,
        color=color,
        linewidth=linewidth,
        linestyle="--" if dashed else "-",
        solid_capstyle="round",
    )
    ax.margins(x=0.01, y=0.12)
    ax.axis("off")
    fig.savefig(path.with_suffix(".pdf"), transparent=True, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(path.with_suffix(".svg"), transparent=True, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(path.with_suffix(".png"), transparent=True, bbox_inches="tight",
                pad_inches=0.01, dpi=300)
    plt.close(fig)


def save_mode_stack(modes, path):
    fig, axes = plt.subplots(modes.shape[0], 1, figsize=(3.2, 2.4), sharex=True)
    axes = np.atleast_1d(axes)
    for mode_index, (axis, values) in enumerate(zip(axes, modes)):
        color = MODE_COLORS[mode_index % len(MODE_COLORS)]
        samples = np.arange(len(values))
        axis.plot(samples, values, color=color, linewidth=1.15)
        axis.set_xlim(-42, len(values) - 1)
        axis.text(-36, 0, f"u{mode_index + 1}", fontsize=9, color=color,
                  ha="left", va="center")
        axis.margins(y=0.15)
        axis.axis("off")
    fig.subplots_adjust(hspace=0.18)
    fig.savefig(path.with_suffix(".pdf"), transparent=True, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(path.with_suffix(".svg"), transparent=True, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(path.with_suffix(".png"), transparent=True, bbox_inches="tight",
                pad_inches=0.01, dpi=300)
    plt.close(fig)


def save_forecast_overlay(target, prediction, path):
    fig, ax = plt.subplots(figsize=(3.2, 1.0))
    time = np.arange(len(target))
    ax.plot(time, target, color=COLORS["target"], linewidth=1.5, label="Target")
    ax.plot(time, prediction, color=COLORS["prediction"], linewidth=1.5,
            linestyle="--", label="Prediction")
    ax.margins(x=0.01, y=0.12)
    ax.axis("off")
    fig.savefig(path.with_suffix(".pdf"), transparent=True, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(path.with_suffix(".svg"), transparent=True, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(path.with_suffix(".png"), transparent=True, bbox_inches="tight",
                pad_inches=0.01, dpi=300)
    plt.close(fig)


def load_model(config, checkpoint, device):
    model = build_atvmd(config).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    return model


def main():
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}. Run one representative "
            "AT-VMD seed with --keep-checkpoint before exporting the figure assets."
        )

    config_path = ROOT / "configs" / f"power_h{args.horizon}.json"
    config = json.loads(config_path.read_text())
    config["data_dir"] = str(ROOT / "data" / "processed")
    data = PowerDataModule(
        args.dataset,
        config["data_dir"],
        int(config["lookback"]),
        int(config["horizon"]),
    )
    test = data.split("test")

    if args.window_index is None:
        index, variability = select_window(test, args.variability_quantile)
    else:
        index = args.window_index
        if not 0 <= index < len(test):
            raise IndexError(f"window index {index} outside [0, {len(test) - 1}]")
        x0, _, _ = test[index]
        variability = x0.diff(dim=0).abs().mean().item()

    x, marks, target = test[index]
    if not 0 <= args.channel < x.shape[-1]:
        raise IndexError(f"channel {args.channel} outside [0, {x.shape[-1] - 1}]")

    device = resolve_device(args.device)
    model = load_model(config, args.checkpoint, device)
    with torch.inference_mode():
        output = model(x.unsqueeze(0).to(device), marks.unsqueeze(0).to(device))

    channel = args.channel
    input_signal = x[:, channel].numpy()
    target_signal = target[:, channel].numpy()
    prediction = output["prediction"][0, :, channel].cpu().numpy()
    controls = model._encode(x[:, channel].unsqueeze(0).to(device))
    mode_tensor, residual_tensor, omega = model.vmd.differentiable_decompose(
        x[:, channel].unsqueeze(0).to(device),
        controls["alpha"],
        controls["omega"],
        model.vmd_steps,
    )
    modes = mode_tensor[0].detach().cpu().numpy()
    residual = residual_tensor[0].detach().cpu().numpy()
    omega_control = controls["omega"][0].detach().cpu().numpy()
    omega_after = omega[0].detach().cpu().numpy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({
        "font.family": "Times New Roman",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    save_line(input_signal, args.output_dir / "input_signal", COLORS["input"])
    save_line(target_signal, args.output_dir / "target_signal", COLORS["target"])
    save_line(prediction, args.output_dir / "prediction_signal", COLORS["prediction"])
    save_line(residual, args.output_dir / "vmd_residual", COLORS["residual"], dashed=True)
    save_forecast_overlay(target_signal, prediction, args.output_dir / "target_prediction")
    save_mode_stack(modes, args.output_dir / "vmd_modes")
    for mode_index, values in enumerate(modes, start=1):
        save_line(
            values,
            args.output_dir / f"vmd_mode_{mode_index}",
            MODE_COLORS[(mode_index - 1) % len(MODE_COLORS)],
            linewidth=1.3,
        )

    np.savez_compressed(
        args.output_dir / "waveforms.npz",
        input=input_signal,
        target=target_signal,
        prediction=prediction,
        modes=modes,
        residual=residual,
        omega_control=omega_control,
        omega_after_sweep=omega_after,
    )
    metadata = {
        "dataset": args.dataset,
        "horizon": args.horizon,
        "lookback": int(config["lookback"]),
        "test_window_index": index,
        "channel": channel,
        "input_variability": variability,
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "standardized_values": True,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
