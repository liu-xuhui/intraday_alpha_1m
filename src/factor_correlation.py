"""Mean cross-sectional correlation matrix for a set of factors.

For every prediction timestamp the correlation between two factors is computed
across stocks (pairwise-complete), and those per-timestamp correlations are
then averaged. This measures how similarly two factors *rank the cross-section
at a point in time*, which is the quantity that matters when deciding whether
factors are redundant -- unlike a pooled correlation over all cells, which is
contaminated by differences in level between timestamps.

Correlation math matches src/eval_ic.py so the numbers are comparable with the
IC report in results/ic_summary.txt.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# Configuration
# =============================================================================

FACTOR_NAMES = [
    "lob_imbalance_n1_mean_3s",
    "lob_band_imbalance_l2_3_mean_30s",
    "lob_band_imbalance_l4_5_mean_30s",
    "lob_band_imbalance_l6_10_mean_30s",
]

DATA_DIR = Path("data")
FACTOR_DIR = DATA_DIR / "factors"
TARGET_FILE = DATA_DIR / "target_return_1min.pkl"
RESULTS_DIR = Path("results")
OUTPUT_FILE = RESULTS_DIR / "factor_correlation.txt"

MIN_STOCKS_PER_TIMESTAMP = 5


# =============================================================================
# Cross-sectional correlation
# =============================================================================

def cross_sectional_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-timestamp Pearson correlation across stocks, using only the stocks
    where both inputs are present at that timestamp."""
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

    result = np.full(x.shape[0], np.nan)
    good = (n >= MIN_STOCKS_PER_TIMESTAMP) & (denominator > 0)
    result[good] = numerator[good] / denominator[good]
    return result


def rank_frame(frame: pd.DataFrame, both_valid: pd.DataFrame) -> np.ndarray:
    """Cross-sectional ranks, restricted to the stocks valid in both factors so
    the two rankings cover exactly the same names."""
    return frame.where(both_valid).rank(axis=1).to_numpy(dtype=float)


def pair_statistics(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    x = a.to_numpy(dtype=float)
    y = b.to_numpy(dtype=float)

    pearson = cross_sectional_correlation(x, y)

    both_valid = a.notna() & b.notna()
    spearman = cross_sectional_correlation(rank_frame(a, both_valid), rank_frame(b, both_valid))

    finite = pearson[np.isfinite(pearson)]
    counts = both_valid.sum(axis=1)
    return {
        "pearson_series": pearson,
        "mean_pearson": float(np.nanmean(pearson)),
        "std_pearson": float(np.nanstd(pearson)),
        "min_pearson": float(finite.min()) if finite.size else np.nan,
        "max_pearson": float(finite.max()) if finite.size else np.nan,
        "mean_spearman": float(np.nanmean(spearman)),
        "n_timestamps": int(np.isfinite(pearson).sum()),
        "mean_stocks": float(counts[counts > 0].mean()),
    }


def mean_ic(factor_df: pd.DataFrame, target_df: pd.DataFrame) -> float:
    ic = cross_sectional_correlation(
        factor_df.to_numpy(dtype=float), target_df.to_numpy(dtype=float)
    )
    return float(np.nanmean(ic))


# =============================================================================
# Report
# =============================================================================

def format_matrix(title: str, labels: list[str], matrix: np.ndarray) -> list[str]:
    lines = [title, "-" * len(title)]
    lines.append(" " * 6 + "".join(f"{label:>9}" for label in labels))
    for i, label in enumerate(labels):
        row = "".join(f"{matrix[i, j]:>9.3f}" for j in range(len(labels)))
        lines.append(f"{label:<6}{row}")
    return lines


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    target_df = pd.read_pickle(TARGET_FILE)

    factors = {}
    for name in FACTOR_NAMES:
        factor_df = pd.read_pickle(FACTOR_DIR / f"{name}.pkl")
        assert factor_df.shape == target_df.shape
        assert factor_df.index.equals(target_df.index)
        assert factor_df.columns.equals(target_df.columns)
        factors[name] = factor_df

    labels = [f"F{i + 1}" for i in range(len(FACTOR_NAMES))]
    n = len(FACTOR_NAMES)
    pearson_matrix = np.eye(n)
    spearman_matrix = np.eye(n)
    pair_rows = []

    for i in range(n):
        for j in range(i + 1, n):
            stats = pair_statistics(factors[FACTOR_NAMES[i]], factors[FACTOR_NAMES[j]])
            pearson_matrix[i, j] = pearson_matrix[j, i] = stats["mean_pearson"]
            spearman_matrix[i, j] = spearman_matrix[j, i] = stats["mean_spearman"]
            pair_rows.append((labels[i], labels[j], stats))

    width = 78
    lines = [
        "Mean cross-sectional factor correlation",
        "=" * width,
        f"Grid: {TARGET_FILE}  shape={target_df.shape}",
        f"Prediction timestamps: {target_df.index.min()} .. {target_df.index.max()}",
        "",
        "Method: at each prediction timestamp the correlation between two factors is",
        "computed across stocks (pairwise-complete, minimum "
        f"{MIN_STOCKS_PER_TIMESTAMP} stocks); those",
        "per-timestamp correlations are then averaged. This is the same correlation",
        "routine used for IC in src/eval_ic.py, so values are comparable with",
        "results/ic_summary.txt.",
        "=" * width,
        "",
        "Factors",
        "-" * width,
    ]
    for label, name in zip(labels, FACTOR_NAMES):
        lines.append(f"{label}  {name:<36}mean IC {mean_ic(factors[name], target_df):+.4f}")

    lines.append("")
    lines.extend(format_matrix("Mean cross-sectional Pearson correlation", labels, pearson_matrix))
    lines.append("")
    lines.extend(
        format_matrix("Mean cross-sectional Spearman (rank) correlation", labels, spearman_matrix)
    )

    lines.append("")
    lines.append("Pair detail (Pearson)")
    lines.append("-" * width)
    header = (
        f"{'pair':<10}{'mean':>9}{'std':>9}{'min':>9}{'max':>9}"
        f"{'n_ts':>7}{'avg_stocks':>12}"
    )
    lines.append(header)
    for left, right, stats in pair_rows:
        lines.append(
            f"{left + '-' + right:<10}{stats['mean_pearson']:>9.3f}{stats['std_pearson']:>9.3f}"
            f"{stats['min_pearson']:>9.3f}{stats['max_pearson']:>9.3f}"
            f"{stats['n_timestamps']:>7}{stats['mean_stocks']:>12.1f}"
        )

    report = "\n".join(lines) + "\n"
    OUTPUT_FILE.write_text(report)
    print(report)
    print(f"Saved {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
