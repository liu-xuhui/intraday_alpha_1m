"""Evaluate factors against the forward-return target over a chosen period and
write the reportable results to a per-target, per-period folder under results/.

The period is given with the same three lists every builder takes (see
src/pipeline_utils.py):

    python src/eval_ic.py --days 20260820 20260821 20260824 --start-times 09:30 --window-mins 29
    python src/eval_ic.py --days 20260820 20260820 --start-times 09:30 13:00 --window-mins 30 30 \
        --factors "ati_*" lob_near_vs_deep_composite
    python src/eval_ic.py --days 20260820 --start-times 09:30 --window-mins 25 \
        --target data/target_return_5min.pkl

Factors default to every pickle in data/factors/; --factors takes names or glob
patterns. The target and each factor are sliced to exactly the requested
timestamps, so every pickle must already contain them (run the builders with
the same period first); factors missing any timestamp are skipped and listed.

Results for different targets never mix: each target gets its own folder.
Re-running on a target and period whose folder already exists merges into it: factors
evaluated in this run are added (or replace their earlier result) and every
other factor already in the folder is kept. The folder's recorded timestamps
and target must match this run's; --fresh discards the old results instead.

Outputs, in results/<target tag>/<period tag>/
(e.g. results/target_1min/20260820-20260828_7d_0930_29m/, where target_<H>min
comes from data/target_return_<H>min.pkl):
    eval_period.json        the exact segments and every timestamp evaluated
    ic_summary.txt          per-factor IC / RankIC summary table
    ic_summary.csv          the same table at full precision (used for merging)
    ic_timeseries.csv       the per-timestamp IC series for all factors
    factors/ic_<name>.png   per-minute IC and cumulative IC for each factor
    ic_<family>.png         per-family comparison figure (FACTOR_FAMILIES), when
                            every member of the family was evaluated
"""
from __future__ import annotations

import argparse
import fnmatch
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from pipeline_utils import (
    DATA_DIR,
    FACTOR_DIR,
    RESULTS_DIR,
    Segment,
    add_period_arguments,
    describe_segments,
    period_tag,
    segments_from_args,
    segments_timestamps,
)

# =============================================================================
# Configuration
# =============================================================================

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
                "ati_count_15s",
                "ati_count_30s",
                "ati_count_60s",
                "ati_count_90s",
            ],
            "Volume-based (ATI volume)": [
                "ati_volume_15s",
                "ati_volume_30s",
                "ati_volume_60s",
                "ati_volume_90s",
            ],
        },
    },
    "lob_imbalance": {
        "title": "Limit-Order-Book Depth Imbalance",
        "panels": {
            "Best bid/ask only (N=1)": [
                "lob_imbalance_n1_mean_3s",
                "lob_imbalance_n1_mean_15s",
                "lob_imbalance_n1_mean_30s",
                "lob_imbalance_n1_mean_60s",
            ],
            "Top 3 levels (N=3)": [
                "lob_imbalance_n3_mean_3s",
                "lob_imbalance_n3_mean_15s",
                "lob_imbalance_n3_mean_30s",
                "lob_imbalance_n3_mean_60s",
            ],
            "Top 5 levels (N=5)": [
                "lob_imbalance_n5_mean_3s",
                "lob_imbalance_n5_mean_15s",
                "lob_imbalance_n5_mean_30s",
                "lob_imbalance_n5_mean_60s",
            ],
            "Top 10 levels (N=10)": [
                "lob_imbalance_n10_mean_3s",
                "lob_imbalance_n10_mean_15s",
                "lob_imbalance_n10_mean_30s",
                "lob_imbalance_n10_mean_60s",
            ],
        },
    },
    "lob_band_imbalance": {
        "title": "Limit-Order-Book Depth-Band Imbalance",
        "panels": {
            "Levels 2-3 (N=2:3)": [
                "lob_band_imbalance_l2_3_mean_3s",
                "lob_band_imbalance_l2_3_mean_15s",
                "lob_band_imbalance_l2_3_mean_30s",
                "lob_band_imbalance_l2_3_mean_60s",
            ],
            "Levels 4-5 (N=4:5)": [
                "lob_band_imbalance_l4_5_mean_3s",
                "lob_band_imbalance_l4_5_mean_15s",
                "lob_band_imbalance_l4_5_mean_30s",
                "lob_band_imbalance_l4_5_mean_60s",
            ],
            "Levels 6-10 (N=6:10)": [
                "lob_band_imbalance_l6_10_mean_3s",
                "lob_band_imbalance_l6_10_mean_15s",
                "lob_band_imbalance_l6_10_mean_30s",
                "lob_band_imbalance_l6_10_mean_60s",
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
PANEL_LINESTYLES = ["-", "--", ":", "-."]


def panel_tag(panel_title: str) -> str:
    """Short label distinguishing panels in the shared cumulative legend."""
    if "(" in panel_title and ")" in panel_title:
        return panel_title[panel_title.index("(") + 1 : panel_title.index(")")]
    return panel_title.split()[0]


# =============================================================================
# Loading
# =============================================================================

def select_factor_paths(factor_dir: Path, patterns: list[str] | None) -> list[Path]:
    paths = sorted(factor_dir.glob("*.pkl"))
    if not patterns:
        return paths
    selected = [p for p in paths if any(fnmatch.fnmatch(p.stem, pat) for pat in patterns)]
    unmatched = [pat for pat in patterns if not any(fnmatch.fnmatch(p.stem, pat) for p in paths)]
    if unmatched:
        print(f"Warning: no factor in {factor_dir} matches {unmatched}")
    return selected


def slice_period(frame: pd.DataFrame, timestamps: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Rows of frame at exactly the requested timestamps, and the timestamps
    the frame does not contain."""
    missing = timestamps.difference(frame.index)
    return frame.reindex(timestamps), missing


def target_tag(target_path: Path) -> str:
    """Result-folder name for a target: 'target_5min' for
    target_return_5min.pkl, else the file's stem."""
    stem = target_path.stem
    prefix = "target_return_"
    if stem.startswith(prefix) and stem.endswith("min") and stem[len(prefix):-len("min")].isdigit():
        return f"target_{stem[len(prefix):-len('min')]}min"
    return stem


def forward_label(target_path: Path) -> str:
    """'1-minute' for target_return_1min.pkl, else a generic label."""
    stem = target_path.stem
    if stem.startswith("target_return_") and stem.endswith("min"):
        return f"{stem[len('target_return_'):-len('min')]}-minute"
    return "forward"


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
        "pct_positive": 100.0 * (valid_ic > 0).mean() if len(valid_ic) else np.nan,
        "mean_rank_ic": rank_ic_series.dropna().mean(),
        "mean_stocks": n_stocks.mean(),
    }


# =============================================================================
# Plotting
# =============================================================================

def segment_axis(index: pd.DatetimeIndex) -> tuple[np.ndarray, list[int], list[str]]:
    """x positions plus the start position and label of every contiguous run of
    one-minute timestamps (a new run begins at each gap, e.g. a new day)."""
    x = np.arange(len(index))
    gaps = np.diff(index.to_numpy()) != np.timedelta64(1, "m")
    starts = [0] + [i + 1 for i in np.nonzero(gaps)[0]]
    days_with_several_runs = len({index[i].date() for i in starts}) < len(starts)
    fmt = "%m-%d %H:%M" if days_with_several_runs else "%m-%d"
    return x, starts, [index[i].strftime(fmt) for i in starts]


def decorate_time_axis(ax, starts: list[int], labels: list[str]) -> None:
    for pos in starts[1:]:
        ax.axvline(pos - 0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xticks(starts)
    ax.set_xticklabels(labels, rotation=45 if len(starts) > 12 else 0, fontsize=8)


def x_axis_label(index: pd.DatetimeIndex, segments: list[Segment]) -> str:
    n_days = len({s.day for s in segments})
    return f"Prediction minute ({len(segments)} segments, {n_days} trading days, {len(index)} points)"


def plot_factor(
    name: str,
    ic: pd.Series,
    summary: dict,
    segments: list[Segment],
    title_suffix: str,
    out_path: Path,
) -> None:
    """Per-minute IC (bars) and cumulative IC (line) for a single factor."""
    x, starts, labels = segment_axis(ic.index)
    fig, (ax_ic, ax_cum) = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)

    values = ic.to_numpy()
    colors = np.where(values >= 0, "#1f77b4", "#d62728")
    ax_ic.bar(x, np.nan_to_num(values), color=colors, width=0.85, alpha=0.8)
    ax_ic.axhline(summary["mean_ic"] if np.isfinite(summary["mean_ic"]) else 0.0,
                  color="black", linestyle=":", linewidth=1.0, label=f"mean IC {summary['mean_ic']:+.4f}")
    ax_ic.axhline(0.0, color="black", linewidth=0.8)
    ax_ic.set_ylabel("Cross-sectional IC")
    ax_ic.set_title(
        f"Per-minute IC   (n={summary['n_ic']}, ICIR {summary['icir']:+.3f}, "
        f"t {summary['t_stat']:+.2f}, RankIC {summary['mean_rank_ic']:+.4f})",
        fontsize=10,
    )
    ax_ic.legend(fontsize=8, loc="upper left")
    ax_ic.grid(alpha=0.25)
    for pos in starts[1:]:
        ax_ic.axvline(pos - 0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)

    ax_cum.plot(x, ic.fillna(0.0).cumsum().to_numpy(), color="#1f77b4", linewidth=1.5)
    ax_cum.axhline(0.0, color="black", linewidth=0.8)
    ax_cum.set_ylabel("Cumulative IC")
    ax_cum.set_title("Cumulative IC", fontsize=10)
    ax_cum.grid(alpha=0.25)
    decorate_time_axis(ax_cum, starts, labels)
    ax_cum.set_xlabel(x_axis_label(ic.index, segments))

    fig.suptitle(f"{name}: {title_suffix}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_family(
    family_key: str,
    family: dict,
    ic_table: pd.DataFrame,
    summaries: dict,
    segments: list[Segment],
    title_suffix: str,
    out_dir: Path,
) -> Path:
    panels = family["panels"]
    n_panels = len(panels)
    fig, axes = plt.subplots(
        n_panels + 1, 1,
        figsize=(13, 3.6 * (n_panels + 1)),
        sharex=True,
    )
    axes = np.atleast_1d(axes)

    x, starts, labels = segment_axis(ic_table.index)

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
        for pos in starts[1:]:
            ax.axvline(pos - 0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)

    # Bottom panel: cumulative IC across all factors in the family.
    ax = axes[-1]
    for panel_idx, (panel_title, factor_names) in enumerate(panels.items()):
        linestyle = PANEL_LINESTYLES[panel_idx % len(PANEL_LINESTYLES)]
        for color, name in zip(WINDOW_COLORS, factor_names):
            cumulative = ic_table[name].fillna(0.0).cumsum()
            window_label = name.split("_")[-1]
            suffix = "" if n_panels == 1 else f" ({panel_tag(panel_title)})"
            ax.plot(
                x, cumulative.to_numpy(), color=color, linewidth=1.4,
                linestyle=linestyle, label=f"{window_label}{suffix}",
            )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("Cumulative IC")
    ax.set_title("Cumulative IC", fontsize=11)
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.grid(alpha=0.25)
    decorate_time_axis(ax, starts, labels)
    ax.set_xlabel(x_axis_label(ic_table.index, segments))

    fig.suptitle(f"{family['title']}: {title_suffix}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    out_path = out_dir / f"ic_{family_key}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# =============================================================================
# Report
# =============================================================================

def write_summary(
    summaries: dict,
    skipped: dict[str, str],
    target_path: Path,
    target_df: pd.DataFrame,
    segments: list[Segment],
    out_dir: Path,
) -> Path:
    name_width = max([len("factor"), *map(len, summaries), *map(len, skipped)]) + 2
    header = (
        f"{'factor':<{name_width}}{'n_ic':>6}{'mean_IC':>10}{'std_IC':>9}"
        f"{'ICIR':>8}{'t_stat':>8}{'%IC>0':>8}{'mean_RankIC':>13}{'avg_stocks':>12}"
    )
    horizon = forward_label(target_path)
    lines = [
        "Cross-sectional IC report",
        "=" * len(header),
        f"Target: {target_path}  evaluated shape={target_df.shape}",
        f"Prediction timestamps: {target_df.index.min()} .. {target_df.index.max()}",
        f"Evaluation period ({len(segments)} segments; exact timestamps in eval_period.json):",
        describe_segments(segments),
        f"IC = per-minute cross-sectional Pearson correlation between factor and",
        f"the {horizon} forward mid-price return. ICIR = mean(IC)/std(IC),",
        "t_stat = ICIR * sqrt(n_ic). RankIC = Spearman equivalent.",
        "=" * len(header),
        header,
        "-" * len(header),
    ]
    for name, s in summaries.items():
        lines.append(
            f"{name:<{name_width}}{s['n_ic']:>6}{s['mean_ic']:>10.4f}{s['std_ic']:>9.4f}"
            f"{s['icir']:>8.3f}{s['t_stat']:>8.2f}{s['pct_positive']:>8.1f}"
            f"{s['mean_rank_ic']:>13.4f}{s['mean_stocks']:>12.1f}"
        )
    lines.append("-" * len(header))

    finite = {k: v for k, v in summaries.items() if np.isfinite(v["mean_ic"])}
    if finite:
        best = max(finite.items(), key=lambda kv: abs(kv[1]["mean_ic"]))
        lines.append(f"Largest |mean IC|: {best[0]} ({best[1]['mean_ic']:+.4f})")
    if skipped:
        lines.append("")
        lines.append("Skipped factors:")
        lines.extend(f"  {name}: {reason}" for name, reason in skipped.items())

    out_path = out_dir / "ic_summary.txt"
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


SUMMARY_FIELDS = [
    "n_ic", "mean_ic", "std_ic", "icir", "t_stat", "pct_positive", "mean_rank_ic", "mean_stocks",
]


def write_summary_table(summaries: dict, out_dir: Path) -> Path:
    """Full-precision, machine-readable copy of the summary, used to merge
    later runs into this folder without loss."""
    table = pd.DataFrame.from_dict(summaries, orient="index", columns=SUMMARY_FIELDS)
    table.index.name = "factor"
    out_path = out_dir / "ic_summary.csv"
    table.to_csv(out_path)
    return out_path


def write_period(
    segments: list[Segment],
    timestamps: pd.DatetimeIndex,
    target_path: Path,
    evaluated: list[str],
    skipped: dict[str, str],
    out_dir: Path,
    created_utc: str | None,
) -> Path:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload = {
        "created_utc": created_utc or now,
        "updated_utc": now,
        "target": str(target_path),
        "n_timestamps": len(timestamps),
        "segments": [s.as_dict() for s in segments],
        "timestamps": [str(t) for t in timestamps],
        "factors_evaluated": evaluated,
        "factors_skipped": skipped,
    }
    out_path = out_dir / "eval_period.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    return out_path


# =============================================================================
# Merging with an existing result folder
# =============================================================================

def parse_summary_text(path: Path) -> dict:
    """Read the table rows back out of an ic_summary.txt (used only for folders
    written before ic_summary.csv existed; values carry the printed precision)."""
    lines = path.read_text().splitlines()
    header_pos = next(i for i, line in enumerate(lines) if line.split()[:2] == ["factor", "n_ic"])
    summaries = {}
    for line in lines[header_pos + 2:]:
        if line.startswith("-"):
            break
        name, *values = line.split()
        summaries[name] = {
            field: (int(v) if field == "n_ic" else float(v))
            for field, v in zip(SUMMARY_FIELDS, values)
        }
    return summaries


def load_existing(
    out_dir: Path, timestamps: pd.DatetimeIndex, target_path: Path
) -> tuple[pd.DataFrame, dict, dict, str | None]:
    """Previous results in out_dir: (IC table, summaries, skipped, created_utc).
    Refuses to merge a folder that was evaluated on a different period or
    target, since its ICs would not be comparable."""
    period_path = out_dir / "eval_period.json"
    ic_csv = out_dir / "ic_timeseries.csv"
    if not (period_path.exists() and ic_csv.exists()):
        return pd.DataFrame(index=timestamps), {}, {}, None

    period = json.loads(period_path.read_text())
    previous_stamps = pd.DatetimeIndex(pd.to_datetime(period["timestamps"]))
    if not previous_stamps.equals(timestamps) or period["target"] != str(target_path):
        raise SystemExit(
            f"{out_dir} holds results for a different period or target "
            f"({period['n_timestamps']} timestamps, target {period['target']}). "
            "Pass --out-name to write elsewhere, or --fresh to replace it."
        )

    ic_table = pd.read_csv(ic_csv, index_col=0, parse_dates=True)
    ic_table.index = timestamps

    summary_csv = out_dir / "ic_summary.csv"
    if summary_csv.exists():
        table = pd.read_csv(summary_csv, index_col=0)
        summaries = {
            name: {f: (int(row[f]) if f == "n_ic" else float(row[f])) for f in SUMMARY_FIELDS}
            for name, row in table.iterrows()
        }
    else:
        summaries = parse_summary_text(out_dir / "ic_summary.txt")

    missing_summary = set(ic_table.columns) - set(summaries)
    if missing_summary:
        raise SystemExit(f"{out_dir}: no summary row for {sorted(missing_summary)}; rerun with --fresh.")
    return ic_table, summaries, period.get("factors_skipped", {}), period.get("created_utc") or period.get("generated_utc")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_period_arguments(parser)
    parser.add_argument(
        "--factors", nargs="+", default=None,
        help="Factor names or glob patterns (default: every pickle in --factor-dir).",
    )
    parser.add_argument("--factor-dir", type=Path, default=FACTOR_DIR)
    parser.add_argument("--target", type=Path, default=TARGET_FILE)
    parser.add_argument(
        "--out-name", default=None,
        help="Result folder name under results/<target tag>/ (default: a short tag derived from the period).",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Discard results already in the folder instead of merging into them.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip every figure.")
    args = parser.parse_args()

    segments = segments_from_args(args)
    timestamps = segments_timestamps(segments)
    out_dir = RESULTS_DIR / target_tag(args.target) / (args.out_name or period_tag(segments))
    plot_dir = out_dir / "factors"
    print(f"Evaluating {len(timestamps)} timestamps -> {out_dir}\n{describe_segments(segments)}")

    target_df, missing = slice_period(pd.read_pickle(args.target), timestamps)
    if len(missing):
        raise SystemExit(
            f"{args.target} lacks {len(missing)} of the requested timestamps "
            f"(e.g. {list(map(str, missing[:3]))}). Build them first with "
            "src/build_targets.py and the same --days/--start-times/--window-mins."
        )

    factor_paths = select_factor_paths(args.factor_dir, args.factors)
    if not factor_paths:
        raise SystemExit(f"No factors selected from {args.factor_dir}.")

    if args.fresh:
        previous_table, previous_summaries, previous_skipped, created_utc = pd.DataFrame(index=timestamps), {}, {}, None
    else:
        previous_table, previous_summaries, previous_skipped, created_utc = load_existing(
            out_dir, timestamps, args.target
        )
    if previous_summaries:
        print(f"Merging into {len(previous_summaries)} factors already evaluated in {out_dir}")
    plot_dir.mkdir(parents=True, exist_ok=True)

    title_suffix = f"{forward_label(args.target)} forward-return IC, CSI 500"
    new_ics = {}
    new_summaries = {}
    new_skipped = {}
    for path in tqdm(factor_paths, desc="Evaluating factors", unit="factor"):
        name = path.stem
        factor_df, missing = slice_period(pd.read_pickle(path), timestamps)
        if len(missing):
            new_skipped[name] = (
                f"missing {len(missing)} of {len(timestamps)} timestamps "
                f"(first: {missing[0]}); rebuild it for this period"
            )
            continue
        factor_df = factor_df.reindex(columns=target_df.columns)

        ic = cross_sectional_ic(factor_df, target_df)
        new_ics[name] = ic
        new_summaries[name] = summarize(
            ic, rank_ic(factor_df, target_df), coverage(factor_df, target_df)
        )
        if not args.no_plots:
            plot_factor(name, ic, new_summaries[name], segments, title_suffix, plot_dir / f"ic_{name}.png")

    for name, reason in new_skipped.items():
        kept = " (keeping its earlier result)" if name in previous_summaries else ""
        tqdm.write(f"Skipped {name}: {reason}{kept}")

    # Factors evaluated now replace their earlier rows; every other factor
    # already in the folder is kept as it was.
    summaries = {**previous_summaries, **new_summaries}
    summaries = {name: summaries[name] for name in sorted(summaries)}
    ic_table = previous_table.drop(columns=list(new_ics), errors="ignore")
    ic_table = pd.concat([ic_table, pd.DataFrame(new_ics, index=timestamps)], axis=1)
    ic_table = ic_table[sorted(ic_table.columns)]
    ic_table.index.name = "datetime"
    skipped = {
        name: reason
        for name, reason in {**previous_skipped, **new_skipped}.items()
        if name not in summaries
    }

    ic_csv = out_dir / "ic_timeseries.csv"
    ic_table.to_csv(ic_csv)
    print(f"Saved {ic_csv}")

    print(f"Saved {write_summary_table(summaries, out_dir)}")

    period_path = write_period(
        segments, timestamps, args.target, list(summaries), skipped, out_dir, created_utc
    )
    print(f"Saved {period_path}")

    summary_path = write_summary(summaries, skipped, args.target, target_df, segments, out_dir)
    print(f"Saved {summary_path}")
    print(summary_path.read_text())
    print(
        f"This run: {len(new_summaries)} evaluated, {len(new_skipped)} skipped; "
        f"folder now holds {len(summaries)} factors."
    )

    if not args.no_plots:
        print(f"Saved {len(new_summaries)} per-factor figures to {plot_dir}")
        for family_key, family in FACTOR_FAMILIES.items():
            members = [name for names in family["panels"].values() for name in names]
            if not all(name in ic_table.columns for name in members):
                continue
            if not any(name in new_summaries for name in members):
                continue  # figure already up to date
            out_path = plot_family(
                family_key, family, ic_table, summaries, segments, title_suffix, out_dir
            )
            print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
