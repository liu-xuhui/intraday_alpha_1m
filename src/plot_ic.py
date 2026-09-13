"""Evaluate every factor in data/factors/ against the 1-minute forward-return
target and write the reportable results to results/.

Outputs:
    results/ic_summary.txt          per-factor IC / RankIC summary table
    results/ic_timeseries.csv       the per-timestamp IC series for all factors
    results/ic_<family>.png         per-minute IC and cumulative IC by window
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path("data")
FACTOR_DIR = DATA_DIR / "factors"
RESULTS_DIR = Path("results")
TARGET_FILE = DATA_DIR / "target_return_1min.pkl"

MIN_STOCKS_PER_TIMESTAMP = 2

# Factor families plotted together, in window order. Each entry maps a figure
# title to the factor names (one line/colour per lookback window).
FACTOR_FAMILIES = {
    "burst_volume_imbalance": {
        "title": "Burst Volume Imbalance",
        "panels": {
            "Burst volume imbalance": [
                "burst_volume_imbalance_8s",
                "burst_volume_imbalance_15s",
                "burst_volume_imbalance_30s",
            ],
        },
    },
    "ati": {
        "title": "Aggressive Trade Imbalance",
        "panels": {
            "Count-based (ATI count)": [
                "ati_count_30s",
                "ati_count_60s",
                "ati_count_90s",
            ],
            "Volume-based (ATI volume)": [
                "ati_volume_30s",
                "ati_volume_60s",
                "ati_volume_90s",
            ],
        },
    },
    "order_recency_imbalance": {
        "title": "Order Recency Imbalance",
        "panels": {
            "Order recency imbalance": [
                "order_recency_imbalance_15s",
                "order_recency_imbalance_30s",
            ],
        },
    },
}

WINDOW_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]


# =============================================================================
# Cross-sectional IC
# =============================================================================

def cross_sectional_ic(factor_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.Series:
    """Per-timestamp cross-sectional Pearson correlation between factor and
    forward return, using only stocks where both values are present."""
    x = factor_df.to_numpy(dtype=float)
    y = target_df.to_numpy(dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    n = valid.sum(axis=1)

    x0 = np.where(valid, x, 0.0)
    y0 = np.where(valid, y, 0.0)
    x_mean = np.divide(x0.sum(axis=1), n, out=np.full(n.shape, np.nan), where=n > 0)
    y_mean = np.divide(y0.sum(axis=1), n, out=np.full(n.shape, np.nan), where=n > 0)

    xc = np.where(valid, x - x_mean[:, None], 0.0)
    yc = np.where(valid, y - y_mean[:, None], 0.0)

    numerator = np.sum(xc * yc, axis=1)
    denominator = np.sqrt(np.sum(xc ** 2, axis=1) * np.sum(yc ** 2, axis=1))

    ic = np.full(x.shape[0], np.nan)
    good = (n >= MIN_STOCKS_PER_TIMESTAMP) & (denominator > 0)
    ic[good] = numerator[good] / denominator[good]
    return pd.Series(ic, index=factor_df.index)


def rank_ic(factor_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.Series:
    """Spearman rank IC: Pearson IC computed on cross-sectional ranks. Ranking
    each row independently keeps NaNs as NaNs, so the pairing is unchanged."""
    both_valid = factor_df.notna() & target_df.notna()
    factor_ranks = factor_df.where(both_valid).rank(axis=1)
    target_ranks = target_df.where(both_valid).rank(axis=1)
    return cross_sectional_ic(factor_ranks, target_ranks)


def coverage(factor_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.Series:
    return (factor_df.notna() & target_df.notna()).sum(axis=1)


def summarize(ic: pd.Series, rank_ic_series: pd.Series, n_stocks: pd.Series) -> dict:
    valid_ic = ic.dropna()
    mean_ic = valid_ic.mean()
    std_ic = valid_ic.std()
    # ICIR here is the per-minute mean/std ratio; t-stat scales it by sqrt(n).
    icir = mean_ic / std_ic if std_ic and std_ic > 0 else np.nan
    t_stat = icir * np.sqrt(len(valid_ic)) if np.isfinite(icir) else np.nan
    return {
        "n_ic": len(valid_ic),
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "icir": icir,
        "t_stat": t_stat,
        "pct_positive": 100.0 * (valid_ic > 0).mean(),
        "mean_rank_ic": rank_ic_series.dropna().mean(),
        "mean_stocks": n_stocks.mean(),
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_family(
    family_key: str,
    family: dict,
    ic_table: pd.DataFrame,
    summaries: dict,
) -> Path:
    panels = family["panels"]
    n_panels = len(panels)
    fig, axes = plt.subplots(
        n_panels + 1, 1,
        figsize=(13, 3.6 * (n_panels + 1)),
        sharex=True,
    )
    axes = np.atleast_1d(axes)

    index = ic_table.index
    x = np.arange(len(index))
    day_starts = [i for i, ts in enumerate(index) if i == 0 or index[i - 1].date() != ts.date()]
    day_labels = [index[i].strftime("%m-%d") for i in day_starts]

    for panel_idx, (panel_title, factor_names) in enumerate(panels.items()):
        ax = axes[panel_idx]
        for color, name in zip(WINDOW_COLORS, factor_names):
            series = ic_table[name]
            window_label = name.split("_")[-1]
            summary = summaries[name]
            ax.plot(
                x, series.to_numpy(), color=color, linewidth=1.0, alpha=0.85,
                marker="o", markersize=2.5,
                label=f"{window_label}  (mean IC {summary['mean_ic']:+.4f}, ICIR {summary['icir']:+.3f})",
            )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylabel("Cross-sectional IC")
        ax.set_title(panel_title, fontsize=11)
        ax.legend(fontsize=8, loc="upper left", ncol=len(factor_names))
        ax.grid(alpha=0.25)
        for day_x in day_starts[1:]:
            ax.axvline(day_x - 0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)

    # Bottom panel: cumulative IC across all factors in the family.
    ax = axes[-1]
    for panel_idx, (panel_title, factor_names) in enumerate(panels.items()):
        linestyle = "-" if panel_idx == 0 else "--"
        for color, name in zip(WINDOW_COLORS, factor_names):
            cumulative = ic_table[name].fillna(0.0).cumsum()
            window_label = name.split("_")[-1]
            suffix = "" if n_panels == 1 else f" ({panel_title.split()[0].lower()})"
            ax.plot(
                x, cumulative.to_numpy(), color=color, linewidth=1.4,
                linestyle=linestyle, label=f"{window_label}{suffix}",
            )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("Cumulative IC")
    ax.set_title("Cumulative IC", fontsize=11)
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.grid(alpha=0.25)
    for day_x in day_starts[1:]:
        ax.axvline(day_x - 0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)

    ax.set_xticks(day_starts)
    ax.set_xticklabels(day_labels)
    ax.set_xlabel("Prediction minute (09:30-09:58 each day, 7 trading days, 203 points)")

    fig.suptitle(
        f"{family['title']}: 1-minute forward-return IC, CSI 500, 09:30-10:00",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    out_path = RESULTS_DIR / f"ic_{family_key}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# =============================================================================
# Report
# =============================================================================

def write_summary(summaries: dict, target_df: pd.DataFrame) -> Path:
    header = (
        f"{'factor':<32}{'n_ic':>6}{'mean_IC':>10}{'std_IC':>9}"
        f"{'ICIR':>8}{'t_stat':>8}{'%IC>0':>8}{'mean_RankIC':>13}{'avg_stocks':>12}"
    )
    lines = [
        "Cross-sectional IC report",
        "=" * len(header),
        f"Target: {TARGET_FILE}  shape={target_df.shape}",
        f"Prediction timestamps: {target_df.index.min()} .. {target_df.index.max()}",
        "IC = per-minute cross-sectional Pearson correlation between factor and",
        "the next-minute forward mid-price return. ICIR = mean(IC)/std(IC),",
        "t_stat = ICIR * sqrt(n_ic). RankIC = Spearman equivalent.",
        "=" * len(header),
        header,
        "-" * len(header),
    ]
    for name, s in summaries.items():
        lines.append(
            f"{name:<32}{s['n_ic']:>6}{s['mean_ic']:>10.4f}{s['std_ic']:>9.4f}"
            f"{s['icir']:>8.3f}{s['t_stat']:>8.2f}{s['pct_positive']:>8.1f}"
            f"{s['mean_rank_ic']:>13.4f}{s['mean_stocks']:>12.1f}"
        )
    lines.append("-" * len(header))

    best = max(summaries.items(), key=lambda kv: abs(kv[1]["mean_ic"]))
    lines.append(f"Largest |mean IC|: {best[0]} ({best[1]['mean_ic']:+.4f})")

    out_path = RESULTS_DIR / "ic_summary.txt"
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    target_df = pd.read_pickle(TARGET_FILE)

    factor_paths = sorted(FACTOR_DIR.glob("*.pkl"))
    ic_table = {}
    summaries = {}
    for path in factor_paths:
        factor_df = pd.read_pickle(path)
        assert factor_df.shape == target_df.shape
        assert factor_df.index.equals(target_df.index)
        assert factor_df.columns.equals(target_df.columns)

        ic = cross_sectional_ic(factor_df, target_df)
        ic_table[path.stem] = ic
        summaries[path.stem] = summarize(
            ic, rank_ic(factor_df, target_df), coverage(factor_df, target_df)
        )

    ic_table = pd.DataFrame(ic_table, index=target_df.index)
    ic_csv = RESULTS_DIR / "ic_timeseries.csv"
    ic_table.to_csv(ic_csv)
    print(f"Saved {ic_csv}")

    summary_path = write_summary(summaries, target_df)
    print(f"Saved {summary_path}")
    print(summary_path.read_text())

    for family_key, family in FACTOR_FAMILIES.items():
        missing = [
            name
            for names in family["panels"].values()
            for name in names
            if name not in ic_table.columns
        ]
        if missing:
            print(f"Skipping {family_key}, missing factors: {missing}")
            continue
        out_path = plot_family(family_key, family, ic_table, summaries)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
