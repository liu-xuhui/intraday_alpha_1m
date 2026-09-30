"""Construct order-arrival recency imbalance factors from CSI 500 Level-2
order data.

Two factors are produced: RecencyImbalance = (Age_S - Age_B) / (Age_S + Age_B)
over 15s / 30s trailing windows, where Age_X is the elapsed time since the
most recent qualifying buy/sell order-submission event before t (capped at
the window length W when no such event exists). Factors are computed for every
prediction minute in the requested period (see src/pipeline_utils.py) and
merged into data/factors/<name>.pkl (see data/data_reference.txt for the raw
order schema).

    python src/build_order_recency_factors.py --days 20260820 --start-times 09:30 --window-mins 30
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

WINDOW_SECONDS = [15, 30]


# =============================================================================
# Data loading
# =============================================================================

def load_day_orders(day: str, universe: list[str], clock_lo: int, clock_hi: int) -> pd.DataFrame:
    """Load qualifying order-submission events for one day: excludes
    cancellation/delete messages (order_type == 'D') and keeps only
    order_code in {'B', 'S'}, with raw clock in [clock_lo, clock_hi],
    restricted to the stock universe.
    Returns wind_code, timestamp, order_code, sorted by (wind_code,
    order_code, timestamp)."""
    path = RAW_DIR / day / "orders.parquet"
    con = duckdb.connect()
    df = con.execute(
        f"""
        SELECT wind_code, date, time, order_code
        FROM '{path.as_posix()}'
        WHERE order_type != 'D'
          AND order_code IN ('B', 'S')
          AND wind_code = ANY(?)
          AND time >= ? AND time <= ?
        """,
        [universe, clock_lo, clock_hi],
    ).df()
    con.close()

    df["timestamp"] = raw_timestamps(df["date"], df["time"])

    df = df[["wind_code", "order_code", "timestamp"]]
    df = df.sort_values(["wind_code", "order_code", "timestamp"]).reset_index(drop=True)
    return df


# =============================================================================
# Recency (age) computation
# =============================================================================

def compute_age(
    sorted_event_times: np.ndarray,
    factor_times: np.ndarray,
    window_seconds: int,
) -> np.ndarray:
    """For each factor timestamp t in factor_times, elapsed seconds since the
    most recent event strictly before t among sorted_event_times, capped at
    window_seconds if no qualifying event exists within [t-W, t)."""
    age = np.full(factor_times.shape[0], float(window_seconds))
    if sorted_event_times.size == 0:
        return age

    # Last index with event_time < t (searchsorted 'left' finds the first
    # index >= t, so one before that is the most recent strictly-earlier event).
    idx = np.searchsorted(sorted_event_times, factor_times, side="left") - 1
    has_prior = idx >= 0
    safe_idx = np.where(has_prior, idx, 0)
    candidate_time = sorted_event_times[safe_idx]

    elapsed_seconds = (factor_times - candidate_time) / np.timedelta64(1, "s")
    within_window = has_prior & (elapsed_seconds <= window_seconds)
    age = np.where(within_window, elapsed_seconds, float(window_seconds))
    return age


# =============================================================================
# Factor construction
# =============================================================================

def build_factors(timestamps: pd.DatetimeIndex, universe: list[str]) -> dict[str, pd.DataFrame]:
    stock_to_col = {code: i for i, code in enumerate(universe)}
    n_rows, n_cols = len(timestamps), len(universe)

    factor_names = [f"order_recency_imbalance_{w}s" for w in WINDOW_SECONDS]
    factor_values = {name: np.full((n_rows, n_cols), np.nan) for name in factor_names}

    for day, day_row_pos in iter_days(timestamps):
        day_times = timestamps[day_row_pos]
        day_times_arr = day_times.to_numpy()

        orders_path = RAW_DIR / day / "orders.parquet"
        if not orders_path.exists():
            print(f"Warning: missing orders for {day}, leaving factors as NaN.")
            continue

        clock_lo, clock_hi = day_clock_bounds(day_times, lookback_seconds=max(WINDOW_SECONDS))
        orders = load_day_orders(day, universe, clock_lo, clock_hi)
        # Per-stock sorted event-time arrays for each side, built once per
        # day and reused across both lookback windows.
        buy_times = {
            wind_code: group["timestamp"].to_numpy()
            for wind_code, group in orders[orders["order_code"] == "B"].groupby("wind_code", sort=False)
        }
        sell_times = {
            wind_code: group["timestamp"].to_numpy()
            for wind_code, group in orders[orders["order_code"] == "S"].groupby("wind_code", sort=False)
        }
        empty_times = np.array([], dtype="datetime64[ns]")

        for window_seconds in WINDOW_SECONDS:
            row_valid = lookback_available(day_times, window_seconds)
            if not row_valid.any():
                continue  # entire window unavailable for this day; leave as NaN

            factor_name = f"order_recency_imbalance_{window_seconds}s"

            for wind_code in universe:
                col = stock_to_col[wind_code]
                age_b = compute_age(buy_times.get(wind_code, empty_times), day_times_arr, window_seconds)
                age_s = compute_age(sell_times.get(wind_code, empty_times), day_times_arr, window_seconds)

                denom = age_s + age_b
                with np.errstate(divide="ignore", invalid="ignore"):
                    factor_val = (age_s - age_b) / denom
                factor_val = np.where(denom == 0, np.nan, factor_val)
                factor_val = np.where(row_valid, factor_val, np.nan)

                factor_values[factor_name][day_row_pos, col] = factor_val

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
        name = f"order_recency_imbalance_{w}s"
        unavailable = ~lookback_available(timestamps, w)
        all_nan = bool(factors[name][unavailable].isna().all(axis=None))
        print(f"  {name:<32} unavailable rows: {int(unavailable.sum()):>4}  all NaN: {all_nan}")

    print()
    print("Sample rows (first 6 rows, first 5 stocks):")
    sample_times = timestamps[:6]
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
    print(f"Building order-recency factors for {len(timestamps)} timestamps:\n{describe_segments(segments)}")

    factors = build_factors(timestamps, load_universe())

    for name, factor_df in factors.items():
        save_panel(factor_df, args.out_dir / f"{name}.pkl")

    run_sanity_checks(factors, timestamps)


if __name__ == "__main__":
    main()
