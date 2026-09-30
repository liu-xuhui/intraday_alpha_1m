"""Plot the alpha decay of factors: how their cross-sectional IC changes as the
forward-return horizon grows.

For each factor, the script collects its summary row from every
results/target_<H>min/<period>/ic_summary.csv whose period folder starts with
--period (written by src/eval_ic.py), and draws one figure:

    top     mean IC by horizon, with a 95% band (mean +- 1.96 * std / sqrt(n)),
            and the mean RankIC
    bottom  ICIR by horizon

    python src/plot_alpha_decay.py
    python src/plot_alpha_decay.py --factors ati_volume_8s --period 20260907-20260915_7d_1030

Figures are saved as results/other/alpha_decay_<factor>.png.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline_utils import RESULTS_DIR

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_FACTORS = ["ati_volume_8s", "lob_near_vs_deep_composite"]
DEFAULT_PERIOD = "20260907-20260915_7d_1030"   # out-of-sample, 10:30 start
OUT_DIR = RESULTS_DIR / "other"

# Light-surface chart palette (validated: categorical slots 1-2).
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
IC_COLOR = "#2a78d6"        # series 1: mean IC
RANK_IC_COLOR = "#eb6834"   # series 2: mean RankIC

TARGET_DIR_PATTERN = re.compile(r"^target_(\d+)min$")


# =============================================================================
# Loading
# =============================================================================

def collect_decay(factor: str, period_prefix: str) -> pd.DataFrame:
    """One row per horizon (minutes) with the factor's IC summary for the
    period folder that starts with period_prefix."""
    rows = []
    for target_dir in RESULTS_DIR.iterdir():
        match = TARGET_DIR_PATTERN.match(target_dir.name)
        if not (match and target_dir.is_dir()):
            continue
        candidates = sorted(target_dir.glob(f"{period_prefix}_*/ic_summary.csv"))
        if len(candidates) > 1:
            raise SystemExit(f"Several period folders in {target_dir} match {period_prefix}: {candidates}")
        if not candidates:
            continue
        summary = pd.read_csv(candidates[0], index_col=0)
        if factor not in summary.index:
            continue
        row = summary.loc[factor].to_dict()
        row["horizon"] = int(match.group(1))
        row["source"] = str(candidates[0].parent)
        rows.append(row)

    if not rows:
        raise SystemExit(f"No evaluated result for {factor} under {RESULTS_DIR}/target_*min/{period_prefix}_*")
    decay = pd.DataFrame(rows).set_index("horizon").sort_index()
    decay["ci95"] = 1.96 * decay["std_ic"] / np.sqrt(decay["n_ic"])
    return decay


# =============================================================================
# Plotting
# =============================================================================

def style_axis(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY, length=0, labelsize=9)


def plot_decay(factor: str, decay: pd.DataFrame, subtitle: str, out_path: Path) -> None:
    horizons = decay.index.to_numpy()
    mean_ic = decay["mean_ic"].to_numpy()

    fig, (ax_ic, ax_ir) = plt.subplots(
        2, 1, figsize=(8, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.28},
    )
    fig.patch.set_facecolor(SURFACE)

    # Top: mean IC with its 95% band, and mean RankIC.
    style_axis(ax_ic)
    ax_ic.fill_between(
        horizons, mean_ic - decay["ci95"], mean_ic + decay["ci95"],
        color=IC_COLOR, alpha=0.14, linewidth=0, label="95% band of mean IC",
    )
    ax_ic.plot(horizons, mean_ic, color=IC_COLOR, linewidth=2, marker="o", markersize=6,
               markeredgecolor=SURFACE, markeredgewidth=1.5, label="Mean IC", zorder=3)
    ax_ic.plot(horizons, decay["mean_rank_ic"], color=RANK_IC_COLOR, linewidth=2, linestyle="--",
               marker="s", markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=1.5,
               label="Mean RankIC", zorder=3)
    ax_ic.axhline(0.0, color=AXIS, linewidth=1)
    # Headroom above the band keeps the legend clear of the data.
    ax_ic.set_ylim(0.0, float((mean_ic + decay["ci95"]).max()) * 1.22)
    ax_ic.set_ylabel("Cross-sectional IC", color=INK_SECONDARY, fontsize=10)

    # Direct labels on the first and last IC points only, on the side away
    # from the RankIC line so the two never collide.
    rank_ic = decay["mean_rank_ic"].to_numpy()
    for idx, align in ((0, "left"), (-1, "right")):
        above = mean_ic[idx] >= rank_ic[idx]
        ax_ic.annotate(
            f"{mean_ic[idx]:+.4f}", (horizons[idx], mean_ic[idx]),
            xytext=(0, 10 if above else -10), textcoords="offset points", ha=align,
            va="bottom" if above else "top", fontsize=9, color=INK_PRIMARY,
        )
    retained = mean_ic[-1] / mean_ic[0]
    ax_ic.text(
        0.99, 0.04, f"{horizons[-1]}-min IC = {retained:.0%} of {horizons[0]}-min IC",
        transform=ax_ic.transAxes, ha="right", va="bottom", fontsize=9, color=INK_SECONDARY,
    )
    ax_ic.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK_SECONDARY, ncol=3)

    # Bottom: ICIR (mean / std of the per-minute IC).
    style_axis(ax_ir)
    ax_ir.bar(horizons, decay["icir"], width=0.62, color=IC_COLOR, edgecolor=SURFACE, linewidth=2)
    ax_ir.axhline(0.0, color=AXIS, linewidth=1)
    for idx in (0, -1):
        ax_ir.annotate(
            f"{decay['icir'].iloc[idx]:.2f}", (horizons[idx], decay["icir"].iloc[idx]),
            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
            fontsize=9, color=INK_PRIMARY,
        )
    ax_ir.set_ylim(0, decay["icir"].max() * 1.18)
    ax_ir.set_ylabel("ICIR", color=INK_SECONDARY, fontsize=10)
    ax_ir.set_xticks(horizons)
    ax_ir.set_xticklabels([f"{h} min\nn={int(n)}" for h, n in zip(horizons, decay["n_ic"])])
    ax_ir.set_xlabel("Forward-return horizon", color=INK_SECONDARY, fontsize=10)

    fig.suptitle(f"Alpha decay: {factor}", x=0.07, ha="left", fontsize=13, color=INK_PRIMARY, y=0.985)
    fig.text(0.07, 0.925, subtitle, ha="left", fontsize=9.5, color=INK_SECONDARY)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.88, bottom=0.12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--factors", nargs="+", default=DEFAULT_FACTORS)
    parser.add_argument(
        "--period", default=DEFAULT_PERIOD,
        help="Period-folder prefix under each results/target_<H>min/ "
             f"(default: {DEFAULT_PERIOD}); the window-length suffix differs by horizon.",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    for factor in args.factors:
        decay = collect_decay(factor, args.period)
        print(f"{factor}:")
        print(decay[["n_ic", "mean_ic", "ci95", "mean_rank_ic", "icir", "t_stat"]].round(4).to_string())
        subtitle = f"Out-of-sample ({args.period}): per-minute IC vs the H-minute forward return"
        out_path = args.out_dir / f"alpha_decay_{factor}.png"
        plot_decay(factor, decay, subtitle, out_path)
        print(f"Saved {out_path}\n")


if __name__ == "__main__":
    main()
