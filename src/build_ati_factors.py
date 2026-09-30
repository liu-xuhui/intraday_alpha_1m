"""Construct aggressive-trade-imbalance (ATI) factors from CSI 500 Level-2
trade data.

Count-based and volume-based aggressive buy/sell imbalance over each trailing
window in WINDOW_SECONDS. Factors are computed for every prediction minute in
the requested period (see src/pipeline_utils.py) and merged into
data/factors/<name>.pkl (see data/data_reference.txt for the raw trade schema).

    python src/build_ati_factors.py --days 20260820 20260821 --start-times 09:30 --window-mins 30
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from pipeline_utils import (
    RAW_DIR,
    day_clock_bounds,
    describe_segments,
    factor_period_parser,
    iter_days,
    load_universe,
    lookback_available,
    raw_timestamps,
    save_panel,
    segments_from_args,
    segments_timestamps,
)

# =============================================================================
# Configuration
# =============================================================================

WINDOW_SECONDS = [8]


# =============================================================================
# Data loading
# =============================================================================

def load_day_trades(day: str, universe: list[str], clock_lo: int, clock_hi: int) -> pd.DataFrame:
    """Load aggressive (bs_flag in {'B','S'}) trades for one day with raw
    clock in [clock_lo, clock_hi], restricted to the stock universe. Returns
    wind_code, timestamp, bs_flag, volume, sorted by (wind_code, timestamp)."""
    path = RAW_DIR / day / "trades.parquet"
    con = duckdb.connect()
    df = con.execute(
        f"""
        SELECT wind_code, date, time, bs_flag, volume
        FROM '{path.as_posix()}'
        WHERE bs_flag IN ('B', 'S') AND wind_code = ANY(?)
          AND time >= ? AND time <= ?
        """,
        [universe, clock_lo, clock_hi],
    ).df()
    con.close()

    df["timestamp"] = raw_timestamps(df["date"], df["time"])

    df = df[["wind_code", "timestamp", "bs_flag", "volume"]]
    df = df.sort_values(["wind_code", "timestamp"]).reset_index(drop=True)
    return df


# =============================================================================
# Per-stock cumulative buy/sell arrays
# =============================================================================

def build_stock_cumulatives(group: pd.DataFrame) -> dict:
    """Prefix sums of buy/sell counts and volumes for one stock's trades,
    sorted ascending by timestamp, so that a [start, end) window's totals
    can be read off with two searchsorted lookups and a subtraction."""
    is_buy = (group["bs_flag"].to_numpy() == "B")
    is_sell = (group["bs_flag"].to_numpy() == "S")
    volume = group["volume"].to_numpy()

    return {
        "timestamps": group["timestamp"].to_numpy(),
        "cum_buy_count": np.concatenate(([0], np.cumsum(is_buy))),
        "cum_sell_count": np.concatenate(([0], np.cumsum(is_sell))),
        "cum_buy_volume": np.concatenate(([0.0], np.cumsum(is_buy * volume))),
        "cum_sell_volume": np.concatenate(([0.0], np.cumsum(is_sell * volume))),
    }


def window_totals(cum: dict, window_start_times: np.ndarray, times: np.ndarray) -> tuple:
    """Return (N_B, N_S, V_B, V_S) arrays for the half-open window
    [window_start_times[k], times[k]) at each k, using prefix sums."""
    ts = cum["timestamps"]
    idx_start = np.searchsorted(ts, window_start_times, side="left")
    idx_end = np.searchsorted(ts, times, side="left")

    n_b = cum["cum_buy_count"][idx_end] - cum["cum_buy_count"][idx_start]
    n_s = cum["cum_sell_count"][idx_end] - cum["cum_sell_count"][idx_start]
    v_b = cum["cum_buy_volume"][idx_end] - cum["cum_buy_volume"][idx_start]
    v_s = cum["cum_sell_volume"][idx_end] - cum["cum_sell_volume"][idx_start]
    return n_b, n_s, v_b, v_s


def safe_imbalance(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """(pos - neg) / (pos + neg), NaN where pos + neg == 0."""
    denom = pos + neg
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (pos - neg) / denom
    result = np.where(denom == 0, np.nan, result)
    return result


# =============================================================================
# Factor construction
# =============================================================================

def build_factors(timestamps: pd.DatetimeIndex, universe: list[str]) -> dict[str, pd.DataFrame]:
    stock_to_col = {code: i for i, code in enumerate(universe)}
    n_rows, n_cols = len(timestamps), len(universe)

    factor_names = [f"ati_count_{w}s" for w in WINDOW_SECONDS] + [
        f"ati_volume_{w}s" for w in WINDOW_SECONDS
    ]
    factor_values = {name: np.full((n_rows, n_cols), np.nan) for name in factor_names}

    for day, day_row_pos in iter_days(timestamps):
        day_times = timestamps[day_row_pos]

        trades_path = RAW_DIR / day / "trades.parquet"
        if not trades_path.exists():
            print(f"Warning: missing trades for {day}, leaving factors as NaN.")
            continue

        clock_lo, clock_hi = day_clock_bounds(day_times, lookback_seconds=max(WINDOW_SECONDS))
        trades = load_day_trades(day, universe, clock_lo, clock_hi)
        grouped = trades.groupby("wind_code", sort=False)
        # Build each stock's cumulative buy/sell count and volume arrays once
        # per day, then reuse them for every lookback window below.
        stock_cumulatives = {
            wind_code: build_stock_cumulatives(group)
            for wind_code, group in grouped
            if wind_code in stock_to_col
        }
        day_times_arr = day_times.to_numpy()

        for window_seconds in WINDOW_SECONDS:
            window_start_times = (day_times - pd.Timedelta(seconds=window_seconds)).to_numpy()
            # A window reaching back before the session open is unavailable
            # rather than merely empty; blank the whole row.
            row_valid = lookback_available(day_times, window_seconds)
            if not row_valid.any():
                continue  # entire window unavailable for this day; leave as NaN

            count_name = f"ati_count_{window_seconds}s"
            volume_name = f"ati_volume_{window_seconds}s"

            for wind_code, cum in stock_cumulatives.items():
                col = stock_to_col[wind_code]
                n_b, n_s, v_b, v_s = window_totals(cum, window_start_times, day_times_arr)

                count_val = safe_imbalance(n_b, n_s)
                volume_val = safe_imbalance(v_b, v_s)

                count_val = np.where(row_valid, count_val, np.nan)
                volume_val = np.where(row_valid, volume_val, np.nan)

                factor_values[count_name][day_row_pos, col] = count_val
                factor_values[volume_name][day_row_pos, col] = volume_val

    factors = {
        name: pd.DataFrame(values, index=timestamps, columns=universe)
        for name, values in factor_values.items()
    }
    return factors


# =============================================================================
# Sanity checks
# =============================================================================

def run_sanity_checks(factors: dict[str, pd.DataFrame], timestamps: pd.DatetimeIndex) -> None:
    print("=" * 78)
    print("SANITY CHECKS (rows built in this run)")
    print("=" * 78)

    for name, factor_df in factors.items():
        values = factor_df.to_numpy()
        n_total = values.size
        n_nan = int(np.isnan(values).sum())
        n_non_missing = n_total - n_nan
        flat = values[~np.isnan(values)]

        index_ok = factor_df.index.equals(timestamps)
        in_range = bool(((flat >= -1 - 1e-9) & (flat <= 1 + 1e-9)).all()) if flat.size else True

        print(f"--- {name} ---")
        print(f"  shape: {factor_df.shape}")
        print(f"  index matches requested timestamps: {index_ok}")
        print(f"  NaN: {n_nan} / {n_total} ({100 * n_nan / n_total:.2f}%)")
        print(f"  non-missing: {n_non_missing}")
        if flat.size:
            print(f"  min: {flat.min():.6f}  max: {flat.max():.6f}")
            print(f"  mean: {flat.mean():.6f}  std: {flat.std():.6f}")
            for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99):
                print(f"  q{q:.2f}: {np.quantile(flat, q):.6f}")
        print(f"  all non-missing in [-1, 1]: {in_range}")

    print()
    print("Early-window behavior check (rows whose lookback reaches before the session open):")
    for w in WINDOW_SECONDS:
        unavailable = ~lookback_available(timestamps, w)
        for kind in ("count", "volume"):
            name = f"ati_{kind}_{w}s"
            frame = factors[name]
            all_nan = bool(frame[unavailable].isna().all(axis=None))
            some_valid = bool(frame[~unavailable].notna().any(axis=1).all())
            print(
                f"  {name:<18} unavailable rows: {int(unavailable.sum()):>4} all NaN: {all_nan}   "
                f"every available row has some non-NaN value: {some_valid}"
            )

    print()
    print("Sample rows (first 5 rows, first 5 stocks):")
    sample_times = timestamps[:5]
    for name, factor_df in factors.items():
        print(f"--- {name} ---")
        print(factor_df.loc[sample_times, factor_df.columns[:5]])


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = factor_period_parser(__doc__).parse_args()
    segments = segments_from_args(args)
    timestamps = segments_timestamps(segments)
    print(f"Building ATI factors for {len(timestamps)} timestamps:\n{describe_segments(segments)}")

    factors = build_factors(timestamps, load_universe())

    for name, factor_df in factors.items():
        save_panel(factor_df, args.out_dir / f"{name}.pkl")

    run_sanity_checks(factors, timestamps)


if __name__ == "__main__":
    main()
