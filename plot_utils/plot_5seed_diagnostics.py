#!/usr/bin/env python3
"""AT-VMD 5-seed control diagnostics -- two figures, human-readable labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent.parent
MODE_COLORS = ("#E69F00", "#009E73", "#CC79A7", "#0072B2")
MODE_NAMES = ("Mode 1", "Mode 2", "Mode 3", "Mode 4")
SEEDS = (2021, 2022, 2023, 2024, 2025)

# Student-t 97.5th percentile by degrees of freedom, for a two-sided 95% interval.
# Hard-coded so the plotting scripts do not depend on SciPy. With five seeds,
# df = 4 and the multiplier is 2.776 -- the normal quantile 1.96 understates the
# interval by ~42% at this sample size.
T_CRITICAL_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics-dir", type=Path,
                        default=ROOT / "results" / "diagnostics" / "entsoe_be" / "h336" / "atvmd")
    parser.add_argument("--windows-output", type=Path,
                        default=ROOT / "results" / "analysis" / "window_controls_5seed.pdf")
    parser.add_argument("--response-output", type=Path,
                        default=ROOT / "results" / "analysis" / "control_response_5seed.pdf")
    parser.add_argument("--regime-metric", choices=("ramp_intensity", "high_frequency_ratio"),
                        default="ramp_intensity")
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--low-quantile", type=float, default=0.2)
    parser.add_argument("--high-quantile", type=float, default=0.8)
    return parser.parse_args()


def idx_at_quantile(values, quantile):
    target = np.quantile(values, quantile)
    return int(np.argmin(np.abs(values - target)))


def seed_interval(stacked):
    """Half-widths of the 95% interval across seeds, per bin.

    ``stacked`` is ``(n_seeds, n_bins)`` of per-seed bin means. The seed is the
    independent replication unit: the ~5700 test windows are 336-hour spans sliding
    by one hour, so they are not independent and must be collapsed to one number per
    seed before any interval is taken.

    Uses the sample standard deviation (``ddof=1``) and a t quantile on the number of
    seeds that actually contributed to each bin, which can be fewer than ``n_seeds``
    when a seed leaves a bin empty. Bins with fewer than two seeds get NaN rather
    than a fabricated interval.
    """
    n_bins = stacked.shape[1]
    valid = np.sum(~np.isnan(stacked), axis=0)
    half = np.full(n_bins, np.nan)
    for b in range(n_bins):
        n = int(valid[b])
        if n < 2:
            continue
        sd = float(np.nanstd(stacked[:, b], ddof=1))
        half[b] = T_CRITICAL_95.get(n - 1, 1.96) * sd / np.sqrt(n)
    return half, valid


def freq_spectrum(window):
    centred = window - window.mean()
    tapered = centred * np.hanning(len(centred))
    power = np.abs(np.fft.rfft(tapered)) ** 2
    freq = np.fft.rfftfreq(len(centred))
    return freq, power / max(power.max(), 1e-12)


def load_all_seeds(diagnostics_dir):
    results = []
    for seed in SEEDS:
        p = diagnostics_dir / f"seed{seed}" / "diagnostics.npz"
        if p.is_file():
            results.append((seed, dict(np.load(p))))
    if not results:
        raise FileNotFoundError(f"No diagnostics.npz in {diagnostics_dir}")
    return results


def main():
    args = parse_args()
    all_data = load_all_seeds(args.diagnostics_dir)

    ref = None
    for seed, d in all_data:
        if seed == 2023:
            ref = d; break
    if ref is None:
        ref = all_data[0][1]

    ch = args.channel
    ref_windows = ref["input_window"][:, :, ch]
    ref_regime = ref[args.regime_metric][:, ch]
    modes = ref["omega_control"].shape[1]
    fixed = np.arange(modes) * (0.5 / modes)

    lo_idx = idx_at_quantile(ref_regime, args.low_quantile)
    hi_idx = idx_at_quantile(ref_regime, args.high_quantile)

    if args.regime_metric == "ramp_intensity":
        regime_label = "Load volatility\n(mean absolute hour-to-hour change)"
    else:
        regime_label = "High-frequency energy ratio"

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    # ================================================================
    # Figure 1: regimes in columns, time/frequency views in rows
    # ================================================================
    column_titles = (f"Low Volatility (P{int(args.low_quantile * 100)})",
                     f"High Volatility (P{int(args.high_quantile * 100)})")
    windows_figure = plt.figure(figsize=(10, 6.2))
    windows_grid = windows_figure.add_gridspec(
        2, 2, left=0.07, right=0.97, bottom=0.11, top=0.90,
        hspace=0.42, wspace=0.28,
    )

    axes_load = [windows_figure.add_subplot(windows_grid[0, col]) for col in range(2)]
    axes_spec = [windows_figure.add_subplot(windows_grid[1, col]) for col in range(2)]
    for ax, title in zip(axes_load, column_titles):
        position = ax.get_position()
        windows_figure.text(
            (position.x0 + position.x1) / 2, 0.935, title,
            fontsize=17, fontweight="bold", va="center", ha="center",
        )

    example_indices = (lo_idx, hi_idx)
    hours = np.arange(ref_windows.shape[1])

    # Unify y-axis limits for load rows so ylabels align
    load_min = min(ref_windows[lo_idx].min(), ref_windows[hi_idx].min())
    load_max = max(ref_windows[lo_idx].max(), ref_windows[hi_idx].max())
    for col, idx in enumerate(example_indices):
        al = axes_load[col]
        al.plot(hours, ref_windows[idx], color="#174A7E", linewidth=0.85)
        al.set_xlim(0, len(hours) - 1)
        al.set_ylim(load_min - 0.2, load_max + 0.2)
        al.spines["top"].set_visible(False); al.spines["right"].set_visible(False)

        asp = axes_spec[col]
        freq, power = freq_spectrum(ref["input_window"][idx, :, ch])
        power_db = 10.0 * np.log10(np.maximum(power, 1e-5))
        asp.plot(freq, power_db, color="#222222", linewidth=0.85)

        for centre in fixed:
            asp.axvline(centre, color="0.55", linestyle="--", linewidth=0.55)

        all_ctrl = np.stack([d["omega_control"][idx, :, ch] for _, d in all_data])
        for si in range(len(all_data)):
            asp.scatter(all_ctrl[si], np.full(modes, -47.0), c=MODE_COLORS[:modes],
                        marker="D", s=12, alpha=0.25,
                        edgecolors="none", linewidths=0, zorder=2)
        mean_ctrl = all_ctrl.mean(axis=0)
        asp.scatter(mean_ctrl, np.full(modes, -47.0), c=MODE_COLORS[:modes],
                    marker="D", s=28, alpha=0.95,
                    edgecolors="black", linewidths=0.4, zorder=5)
        asp.set_xlim(0, 0.5); asp.set_ylim(-50, 1.5)
        # xlabel handled by shared supxlabel at figure bottom
        asp.spines["top"].set_visible(False); asp.spines["right"].set_visible(False)

    # Column labels
    axes_load[0].set_ylabel("Load (z-score)")
    axes_load[1].set_ylabel("Load (z-score)")
    axes_spec[0].set_ylabel("Spectral energy (dB)")
    axes_spec[1].set_ylabel("Spectral energy (dB)")
    # xlabels
    for al in axes_load:
        al.set_xlabel("Time (hours)")
    for asp in axes_spec:
        asp.set_xlabel("Normalized frequency (cycles/hour)")

    leg_spec = Line2D([0], [0], color="#222222", linewidth=0.85,
                      label="Window spectrum")
    leg_fixed = Line2D([0], [0], color="0.55", linestyle="--", linewidth=0.7,
                       label="Default (uniform spacing)")
    leg_ctrl = Line2D([0], [0], color="0.2", marker="D", linestyle="none",
                      markersize=4, label="AT-VMD (5-seed mean)")
    for asp in axes_spec:
        asp.legend(
            handles=(leg_spec, leg_fixed, leg_ctrl), frameon=False,
            loc="upper right", fontsize=13, borderaxespad=0.4,
            handlelength=2.2, handletextpad=0.6,
        )

    args.windows_output.parent.mkdir(parents=True, exist_ok=True)
    windows_figure.savefig(args.windows_output)
    windows_figure.savefig(args.windows_output.with_suffix(".png"), dpi=300)
    plt.close(windows_figure)

    # ================================================================
    # Figure 2: Control response -- 2x2 grid (folded)
    # ================================================================
    response_figure, axes_response = plt.subplots(
        2, 2, figsize=(10.0, 6.0), sharex=True,
    )
    response_figure.subplots_adjust(
        left=0.12, right=0.97, bottom=0.14, top=0.93, hspace=0.45, wspace=0.12,
    )
    axes_flat = axes_response.flatten()

    common_edges = np.quantile(ref_regime, np.linspace(0.0, 1.0, args.bins + 1))

    for mode in range(modes):
        ax = axes_flat[mode]
        color = MODE_COLORS[mode]
        all_means = []
        for seed, d in all_data:
            regime = d[args.regime_metric][:, ch]
            shift = d["omega_control"][:, mode, ch] - fixed[mode]
            bc, bm = [], []
            for left, right in zip(common_edges[:-1], common_edges[1:]):
                mask = (regime >= left) & (regime < right)
                if right == common_edges[-1]:
                    mask = (regime >= left) & (regime <= right)
                if mask.sum() > 0:
                    bc.append(regime[mask].mean()); bm.append(shift[mask].mean())
                else:
                    bc.append((left + right) / 2); bm.append(np.nan)
            bc_arr = np.array(bc); bm_arr = np.array(bm)
            ok = ~np.isnan(bm_arr)
            ax.plot(bc_arr[ok], bm_arr[ok], color=color, linewidth=0.35, alpha=0.25)
            all_means.append(bm_arr)

        # all_means holds one bin-mean curve per seed, so the interval below is taken
        # over seeds (ddof=1, t quantile on the per-bin seed count), not over windows.
        stacked = np.stack(all_means)
        cross_mean = np.nanmean(stacked, axis=0)
        cross_ci, valid_seeds = seed_interval(stacked)

        bc_ref = np.array([
            ref_regime[(ref_regime >= l) & (ref_regime < (r if r < common_edges[-1] else r + 1e-9))].mean()
            if ((ref_regime >= l) & (ref_regime < (r if r < common_edges[-1] else r + 1e-9))).sum() > 0
            else (l + r) / 2
            for l, r in zip(common_edges[:-1], common_edges[1:])
        ])

        ax.plot(bc_ref, cross_mean, color=color, linewidth=1.3, zorder=6)
        ax.fill_between(bc_ref, cross_mean - cross_ci, cross_mean + cross_ci,
                        color=color, alpha=0.20, linewidth=0, zorder=5)
        ax.axhline(0, color="0.45", linestyle="--", linewidth=0.55)
        # The encoder parameterisation (encoder.py) allows each control to move only
        # +/- 0.25/K around its uniform initialisation, so this range is a property of
        # the model, not of the data. Draw it so saturation is legible.
        half_bin = 0.25 / modes
        for sign in (-1.0, 1.0):
            ax.axhline(sign * half_bin, color="0.45", linestyle=":", linewidth=0.7)
        ax.set_ylim(-1.15 * half_bin, 1.15 * half_bin)
        if int(valid_seeds.min()) < len(all_data):
            print(f"  Mode {mode + 1}: bins with fewer than {len(all_data)} seeds: "
                  f"{np.flatnonzero(valid_seeds < len(all_data)).tolist()}")

        seed_corrs = []
        for seed, d in all_data:
            r_reg = d[args.regime_metric][:, ch]
            r_shift = d["omega_control"][:, mode, ch] - fixed[mode]
            ok = ~np.isnan(r_reg) & ~np.isnan(r_shift)
            if ok.sum() > 2:
                seed_corrs.append(np.corrcoef(r_reg[ok], r_shift[ok])[0, 1])
        r_mean = np.mean(seed_corrs)

        ax.set_title(f"{MODE_NAMES[mode]}  (r = {r_mean:+.2f})",
                     color=color, pad=3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="0.92", linewidth=0.4)

    # Shared axis labels
    for ax in axes_response[1, :]:
        ax.set_xlabel(regime_label)
    for ax in axes_response[0, :]:
        ax.tick_params(axis="x", labelbottom=True)
    for ax in axes_response[:, 0]:
        ax.set_ylabel("Frequency shift\n(+ higher, − lower)")

    args.response_output.parent.mkdir(parents=True, exist_ok=True)
    response_figure.savefig(args.response_output)
    response_figure.savefig(args.response_output.with_suffix(".png"), dpi=300)
    plt.close(response_figure)

    print(f"saved {args.windows_output}")
    print(f"saved {args.response_output}")
    for mode in range(modes):
        seed_corrs = []
        for seed, d in all_data:
            r_reg = d[args.regime_metric][:, ch]
            r_shift = d["omega_control"][:, mode, ch] - fixed[mode]
            ok = ~np.isnan(r_reg) & ~np.isnan(r_shift)
            if ok.sum() > 2:
                seed_corrs.append(np.corrcoef(r_reg[ok], r_shift[ok])[0, 1])
        print(f"  Mode {mode + 1}: r = {np.mean(seed_corrs):+.3f} +- {np.std(seed_corrs):.3f}  [{', '.join(f'{c:+.2f}' for c in seed_corrs)}]")
    print(f"\n  windows: low={lo_idx}, high={hi_idx}")

    half_bin = 0.25 / modes
    print(f"  control reachable range: +-{half_bin:.4f} (encoder parameterisation)")
    for mode in range(modes):
        at_bound = np.mean([
            np.mean(np.abs(d["omega_control"][:, mode, ch] - fixed[mode])
                    >= 0.999 * half_bin)
            for _, d in all_data
        ])
        print(f"    Mode {mode + 1}: {at_bound:.1%} of windows at the bound")
    print("  intervals: 95% t-interval across seeds (ddof=1), seeds as replication unit")


if __name__ == "__main__":
    main()
