"""Export compact ENTSO-E BE waveforms for the architecture figure.

This script reconstructs the one-sweep AT-VMD modes from a saved diagnostics
archive. It does not require PyTorch and does not invent a forecast curve: the
current diagnostics store prediction MSE, but not point-wise predictions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams.update({
    "font.family": "Times New Roman",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


COLORS = {
    "input": "#174A7E",
    "target": "#B33A3A",
    "residual": "#555555",
    "modes": ("#0072B2", "#E69F00", "#009E73", "#CC79A7"),
}


def one_sweep_vmd(x: np.ndarray, omega: np.ndarray, alpha: float = 2000.0):
    """NumPy equivalent of ConvergentVMD.differentiable_decompose(..., steps=1)."""
    length = x.size
    half = length // 2
    mirrored = np.concatenate((x[:half][::-1], x, x[half:][::-1]))
    mirrored_length = mirrored.size

    frequencies = np.arange(mirrored_length) / mirrored_length - 0.5
    spectrum = np.roll(np.fft.fft(mirrored), mirrored_length // 2)
    positive = spectrum.copy()
    positive[: mirrored_length // 2] = 0

    modes_hat = np.zeros((omega.size, mirrored_length), dtype=np.complex128)
    mode_sum = modes_hat.sum(axis=0)
    omega_after = np.empty_like(omega, dtype=np.float64)

    for k in range(omega.size):
        denominator = 1.0 + alpha * (frequencies - omega[k]) ** 2
        mode_hat = (positive - mode_sum) / denominator
        modes_hat[k] = mode_hat
        mode_sum += mode_hat

        positive_half = mode_hat[mirrored_length // 2 :]
        power = np.abs(positive_half) ** 2
        power_sum = power.sum()
        omega_after[k] = (
            np.sum(frequencies[mirrored_length // 2 :] * power) / power_sum
            if power_sum > 1e-12
            else 0.0
        )

    positive_modes = modes_hat[:, mirrored_length // 2 :]
    full = np.zeros_like(modes_hat)
    full[:, mirrored_length // 2 :] = positive_modes
    indices = np.arange(1, mirrored_length // 2 + 1)[::-1]
    full[:, indices] = positive_modes.conj()
    full[:, 0] = full[:, -1].conj()
    mirrored_modes = np.fft.ifft(
        np.roll(full, -(mirrored_length // 2), axis=-1), axis=-1
    ).real
    modes = mirrored_modes[:, mirrored_length // 4 : 3 * mirrored_length // 4]
    residual = x - modes.sum(axis=0)
    return modes, residual, omega_after


def export_curve(path: Path, values: np.ndarray, color: str, dashed: bool = False):
    fig, ax = plt.subplots(figsize=(2.35, 0.72))
    ax.plot(
        np.arange(values.size),
        values,
        color=color,
        linewidth=1.7,
        linestyle="--" if dashed else "-",
        solid_capstyle="round",
    )
    ax.margins(x=0.01, y=0.13)
    ax.axis("off")
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    fig.savefig(path.with_suffix(".pdf"), transparent=True, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(path.with_suffix(".svg"), transparent=True, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(path.with_suffix(".png"), transparent=True, bbox_inches="tight", pad_inches=0.01, dpi=300)
    plt.close(fig)


def export_modes(path: Path, modes: np.ndarray):
    fig, axes = plt.subplots(modes.shape[0], 1, figsize=(2.35, 1.65), sharex=True)
    for k, (ax, mode, color) in enumerate(zip(axes, modes, COLORS["modes"]), start=1):
        scale = np.std(mode)
        display = mode / scale if scale > 1e-12 else mode
        samples = np.arange(display.size)
        ax.plot(samples, display, color=color, linewidth=1.35, solid_capstyle="round")
        ax.set_xlim(-42, display.size - 1)
        ax.text(-36, 0, f"u{k}", fontsize=9, color=color,
                ha="left", va="center")
        ax.margins(y=0.18)
        ax.axis("off")
    fig.patch.set_alpha(0)
    fig.subplots_adjust(hspace=0.10)
    fig.savefig(path.with_suffix(".pdf"), transparent=True, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(path.with_suffix(".svg"), transparent=True, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(path.with_suffix(".png"), transparent=True, bbox_inches="tight", pad_inches=0.01, dpi=300)
    plt.close(fig)


def export_preview(path: Path, x: np.ndarray, target: np.ndarray, modes: np.ndarray, residual: np.ndarray):
    fig, axes = plt.subplots(4, 1, figsize=(7.0, 4.2))
    axes[0].plot(x, color=COLORS["input"], linewidth=1.2)
    axes[0].set_title("Observed input window", loc="left")
    axes[1].plot(target, color=COLORS["target"], linewidth=1.2)
    axes[1].set_title("Future target", loc="left")
    for mode, color in zip(modes, COLORS["modes"]):
        axes[2].plot(mode, color=color, linewidth=0.9)
    axes[2].set_title("One-sweep AT-VMD modes", loc="left")
    axes[3].plot(residual, color=COLORS["residual"], linewidth=1.0, linestyle="--")
    axes[3].set_title("Reconstruction residual", loc="left")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quantile", type=float, default=0.80)
    parser.add_argument("--lookback", type=int, default=336)
    parser.add_argument("--horizon", type=int, default=336)
    args = parser.parse_args()

    diagnostics = np.load(args.diagnostics)
    ramp = diagnostics["ramp_intensity"].reshape(-1)
    desired = np.quantile(ramp, args.quantile)
    selected = int(np.argmin(np.abs(ramp - desired)))
    window_index = int(diagnostics["window_index"][selected])
    x = diagnostics["input_window"][selected, :, 0].astype(np.float64)
    omega = diagnostics["omega_control"][selected, :, 0].astype(np.float64)
    modes, residual, omega_after = one_sweep_vmd(x, omega)

    frame = pd.read_csv(args.csv)
    target_column = "electricity_demand_belgium"
    raw = frame[target_column].to_numpy(np.float64)
    train_end = int(raw.size * 0.6)
    val_end = int(raw.size * 0.8)
    mean = raw[:train_end].mean()
    std = raw[:train_end].std(ddof=0)
    normalized = (raw - mean) / std
    start = val_end - args.lookback + window_index
    target_start = start + args.lookback
    target = normalized[target_start : target_start + args.horizon]
    if target.size != args.horizon:
        raise ValueError("Selected target window is incomplete")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    export_curve(args.output_dir / "input_signal", x, COLORS["input"])
    export_curve(args.output_dir / "target_signal", target, COLORS["target"])
    export_curve(args.output_dir / "residual_signal", residual, COLORS["residual"], dashed=True)
    for k, (mode, color) in enumerate(zip(modes, COLORS["modes"]), start=1):
        export_curve(args.output_dir / f"vmd_mode_{k}", mode, color)
    export_modes(args.output_dir / "vmd_modes", modes)
    export_preview(args.output_dir / "preview.png", x, target, modes, residual)

    np.savez_compressed(
        args.output_dir / "architecture_waveforms.npz",
        input=x,
        target=target,
        modes=modes,
        residual=residual,
        omega_control=omega,
        omega_after_sweep=omega_after,
        diagnostics_row=selected,
        window_index=window_index,
        ramp_quantile=args.quantile,
    )
    print(f"Selected diagnostics row: {selected}")
    print(f"Test-window index: {window_index}")
    print(f"Ramp intensity: {ramp[selected]:.6f} (target P{args.quantile * 100:.0f})")
    print(f"omega control: {omega}")
    print(f"omega after sweep: {omega_after}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
