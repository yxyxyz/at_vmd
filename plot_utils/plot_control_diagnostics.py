#!/usr/bin/env python3
"""Single-seed exploratory plot of AT-VMD window-conditioned controls.

NOT the paper figure. This script reads one seed's ``diagnostics.npz`` and draws the
binned control response with no uncertainty band, because a single seed carries no
usable estimate of run-to-run variability.

It previously drew a band computed as ``1.96 * std / sqrt(n)`` over the windows in
each bin. That was invalid: the windows are 336-hour spans sliding by one hour, so
neighbours share 335/336 of their content and `n` (~570 per bin) is not a count of
independent observations. The band was understated by roughly an order of magnitude.
The independent replication unit is the seed, not the window.

For the publication figure with a defensible interval across seeds, use
``plot_utils/plot_5seed_diagnostics.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent.parent
MODE_COLORS = ("#2166AC", "#4D9221", "#E08214", "#762A83")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostics",
        type=Path,
        default=(
            ROOT / "results" / "diagnostics" / "entsoe_be" / "h336"
            / "atvmd" / "seed2023" / "diagnostics.npz"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "analysis" / "control_diagnostics.pdf",
    )
    parser.add_argument(
        "--regime-metric",
        choices=("ramp_intensity", "high_frequency_ratio"),
        default="ramp_intensity",
    )
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--low-quantile", type=float, default=0.2)
    parser.add_argument("--high-quantile", type=float, default=0.8)
    return parser.parse_args()


def representative_index(values, quantile):
    target = np.quantile(values, quantile)
    return int(np.argmin(np.abs(values - target)))


def normalized_psd(window):
    centered = window - window.mean()
    tapered = centered * np.hanning(len(centered))
    power = np.abs(np.fft.rfft(tapered)) ** 2
    frequency = np.fft.rfftfreq(len(centered))
    return frequency, power / max(power.max(), 1e-12)


def binned_response(x, y, bins):
    """Bin centres and bin means. Deliberately returns no interval — see module docstring."""
    edges = np.quantile(x, np.linspace(0.0, 1.0, bins + 1))
    centers, means = [], []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (x >= left) & (x <= right if right == edges[-1] else x < right)
        xb, yb = x[mask], y[mask]
        centers.append(xb.mean())
        means.append(yb.mean())
    return np.asarray(centers), np.asarray(means)


def main():
    args = parse_args()
    data = np.load(args.diagnostics)
    required = {
        "input_window",
        "omega_control",
        args.regime_metric,
    }
    missing = required.difference(data.files)
    if missing:
        raise ValueError(
            f"Missing {sorted(missing)}; regenerate diagnostics with the current code."
        )

    windows = data["input_window"][:, :, args.channel]
    controls = data["omega_control"][:, :, args.channel]
    regime = data[args.regime_metric][:, args.channel]
    modes = controls.shape[1]
    fixed = np.arange(modes) * (0.5 / modes)

    low_index = representative_index(regime, args.low_quantile)
    high_index = representative_index(regime, args.high_quantile)

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    figure = plt.figure(figsize=(7.05, 2.55))
    outer = figure.add_gridspec(
        1, 2, width_ratios=(1.35, 1.0), left=0.07, right=0.985,
        bottom=0.20, top=0.82, wspace=0.30,
    )
    examples = outer[0].subgridspec(2, 2, hspace=0.16, wspace=0.18)
    axes_load = [figure.add_subplot(examples[0, i]) for i in range(2)]
    axes_psd = [figure.add_subplot(examples[1, i]) for i in range(2)]
    axis_response = figure.add_subplot(outer[1])

    example_indices = (low_index, high_index)
    example_titles = (
        f"Low ramp (P{int(args.low_quantile * 100)})",
        f"High ramp (P{int(args.high_quantile * 100)})",
    )
    hours = np.arange(windows.shape[1])

    for column, index in enumerate(example_indices):
        load_axis = axes_load[column]
        psd_axis = axes_psd[column]
        load_axis.plot(hours, windows[index], color="#174A7E", linewidth=0.85)
        load_axis.set_title(example_titles[column], pad=2)
        load_axis.set_xlim(0, len(hours) - 1)
        load_axis.set_xticks([])
        load_axis.spines["top"].set_visible(False)
        load_axis.spines["right"].set_visible(False)

        frequency, power = normalized_psd(windows[index])
        psd_axis.plot(frequency, power, color="#222222", linewidth=0.75)
        for center in fixed:
            psd_axis.axvline(center, color="0.65", linestyle="--", linewidth=0.6)
        psd_axis.scatter(
            controls[index],
            np.full(modes, 0.035),
            c=MODE_COLORS[:modes],
            marker="^",
            s=24,
            edgecolors="black",
            linewidths=0.3,
            zorder=3,
        )
        psd_axis.set_xlim(0, 0.5)
        psd_axis.set_ylim(0, 1.04)
        psd_axis.set_xlabel("Normalized frequency")
        psd_axis.spines["top"].set_visible(False)
        psd_axis.spines["right"].set_visible(False)

    axes_load[0].set_ylabel("Load (z-score)")
    axes_psd[0].set_ylabel("PSD (norm.)")
    axes_load[1].set_yticklabels([])
    axes_psd[1].set_yticklabels([])

    shift = controls - fixed[None, :]
    # Bin means only: a single seed gives no basis for an uncertainty band, and the
    # windows within a bin are not independent. No fill_between here by design.
    for mode in range(modes):
        x, mean = binned_response(regime, shift[:, mode], args.bins)
        color = MODE_COLORS[mode]
        axis_response.plot(
            x, mean, color=color, marker="o", markersize=2.7,
            linewidth=1.0, label=f"Mode {mode + 1}",
        )
    axis_response.axhline(0, color="0.35", linestyle="--", linewidth=0.7)
    # The encoder can only move each control within +/- omega_half_bin = 0.25/K of its
    # uniform initialisation (encoder.py), so the reachable range is a property of the
    # parameterisation, not of the data. Draw it so saturation is visible.
    half_bin = 0.25 / modes
    for sign in (-1.0, 1.0):
        axis_response.axhline(
            sign * half_bin, color="0.35", linestyle=":", linewidth=0.7,
        )
    axis_response.set_ylim(-1.15 * half_bin, 1.15 * half_bin)
    metric_label = (
        "Ramp intensity"
        if args.regime_metric == "ramp_intensity"
        else "High-frequency energy ratio"
    )
    axis_response.set_xlabel(metric_label)
    axis_response.set_ylabel(
        r"Control shift $\omega_k^0(\mathbf{x})-\bar{\omega}_k$"
    )
    axis_response.legend(frameon=False, ncol=2, loc="best")
    axis_response.grid(axis="y", color="0.90", linewidth=0.5)
    axis_response.spines["top"].set_visible(False)
    axis_response.spines["right"].set_visible(False)

    fixed_handle = Line2D(
        [0], [0], color="0.6", linestyle="--", linewidth=0.7,
        label="Fixed initialization",
    )
    control_handle = Line2D(
        [0], [0], color=MODE_COLORS[0], marker="^", linestyle="none",
        markersize=4, label="AT-VMD control",
    )
    axes_psd[0].legend(
        handles=(fixed_handle, control_handle), frameon=False,
        loc="upper right",
    )

    figure.text(0.01, 0.97, "(a) Window-conditioned spectral control",
                fontsize=8.5, weight="bold", va="top")
    figure.text(0.62, 0.97, "(b) Control response, single seed (bin means)",
                fontsize=8.5, weight="bold", va="top")
    figure.text(0.62, 0.045,
                rf"Dotted lines: reachable range $\pm{half_bin:.4f}$ (parameterisation bound)",
                fontsize=6.2, color="0.35", va="bottom")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output)
    figure.savefig(args.output.with_suffix(".png"), dpi=300)
    plt.close(figure)

    correlations = [
        float(np.corrcoef(regime, shift[:, mode])[0, 1])
        for mode in range(modes)
    ]
    print(f"saved {args.output}")
    print(
        f"representative windows: low={low_index}, high={high_index}; "
        f"mode correlations={correlations}"
    )
    saturated = float(np.mean(np.abs(shift) >= 0.999 * half_bin))
    print(f"controls at the +/-{half_bin:.4f} parameterisation bound: {saturated:.1%}")
    print(
        "NOTE: single-seed exploratory figure, no uncertainty band. "
        "Use plot_5seed_diagnostics.py for the publication figure."
    )


if __name__ == "__main__":
    main()
