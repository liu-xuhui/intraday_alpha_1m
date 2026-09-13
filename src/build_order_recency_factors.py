"""Construct order-arrival recency imbalance factors from CSI 500 Level-2
order data.

Two factors are produced: RecencyImbalance = (Age_S - Age_B) / (Age_S + Age_B)
over 15s / 30s trailing windows, where Age_X is the elapsed time since the
most recent qualifying buy/sell order-submission event before t (capped at
the window length W when no such event exists). Every factor DataFrame is
aligned exactly to the index and columns of data/target_return_1min.pkl (see
data/data_reference.txt for the raw order schema).
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# =============================================================================
# Configuration
# =============================================================================

WINDOW_SECONDS = [15, 30]

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
FACTOR_DIR = DATA_DIR / "factors"
TARGET_FILE = DATA_DIR / "target_return_1min.pkl"

TRADING_DAYS = [
    "20260820",
    "20260821",
    "20260824",
    "20260825",
    "20260826",
    "20260827",
    "20260828",
]

WINDOW_START_HOUR = 9
WINDOW_START_MINUTE = 30


# =============================================================================
# Data loading
# =============================================================================

def load_day_orders(day: str, universe: list[str]) -> pd.DataFrame:
    """Load qualifying order-submission events for one day: excludes
    cancellation/delete messages (order_type == 'D') and keeps only
    order_code in {'B', 'S'}, restricted to the target's stock universe.
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
        """,
        [universe],
    ).df()
    con.close()

    time_str = df["time"].astype(np.int64).astype(str).str.zfill(9)
    date_str = df["date"].astype(np.int64).astype(str)
    # HHMMSSmmm (9 digits) -> pad milliseconds to microseconds for strptime.
    timestamp_str = date_str + time_str + "000"
    df["timestamp"] = pd.to_datetime(timestamp_str, format="%Y%m%d%H%M%S%f")

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

def build_factors(target_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    universe = list(target_df.columns)
    stock_to_col = {code: i for i, code in enumerate(universe)}
    n_rows, n_cols = target_df.shape

    factor_names = [f"order_recency_imbalance_{w}s" for w in WINDOW_SECONDS]
    factor_values = {name: np.full((n_rows, n_cols), np.nan) for name in factor_names}

    index_dates = target_df.index.normalize()

    for day in TRADING_DAYS:
        day_date = pd.Timestamp(day)
        day_row_mask = np.asarray(index_dates == day_date)
        day_row_pos = np.nonzero(day_row_mask)[0]
        if day_row_pos.size == 0:
            print(f"Warning: {day} has no rows in the target index, skipping.")
            continue

        day_times = target_df.index[day_row_pos]
        day_times_arr = day_times.to_numpy()
        day_open = day_date + pd.Timedelta(hours=WINDOW_START_HOUR, minutes=WINDOW_START_MINUTE)

        orders_path = RAW_DIR / day / "orders.parquet"
        if not orders_path.exists():
            print(f"Warning: missing orders for {day}, leaving factors as NaN.")
            continue

        orders = load_day_orders(day, universe)
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
            row_valid = day_times_arr >= (day_open + pd.Timedelta(seconds=window_seconds)).to_datetime64()
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
        name: pd.DataFrame(values, index=target_df.index, columns=target_df.columns)
        for name, values in factor_values.items()
    }
    return factors


# =============================================================================
# Sanity checks
# =============================================================================

def run_sanity_checks(factors: dict[str, pd.DataFrame], target_df: pd.DataFrame) -> None:
    print("=" * 78)
    print("SANITY CHECKS")
    print("=" * 78)

    for name, factor_df in factors.items():
        values = factor_df.to_numpy()
        n_total = values.size
        n_nan = int(np.isnan(values).sum())
        n_non_missing = n_total - n_nan
        flat = values[~np.isnan(values)]

        shape_ok = factor_df.shape == target_df.shape
        index_ok = factor_df.index.equals(target_df.index)
        columns_ok = factor_df.columns.equals(target_df.columns)
        in_range = bool(((flat >= -1 - 1e-9) & (flat <= 1 + 1e-9)).all()) if flat.size else True

        print(f"--- data/factors/{name}.pkl ---")
        print(f"  shape: {factor_df.shape} (matches target: {shape_ok})")
        print(f"  index matches target: {index_ok}")
        print(f"  columns match target: {columns_ok}")
        print(f"  NaN: {n_nan} / {n_total} ({100 * n_nan / n_total:.2f}%)")
        print(f"  non-missing: {n_non_missing}")
        if flat.size:
            print(f"  min: {flat.min():.6f}  max: {flat.max():.6f}")
            print(f"  mean: {flat.mean():.6f}  std: {flat.std():.6f}")
            for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99):
                print(f"  q{q:.2f}: {np.quantile(flat, q):.6f}")
        print(f"  all non-missing in [-1, 1]: {in_range}")

    print()
    print("Early-window behavior check (per trading day):")
    index_dates = target_df.index.normalize()
    for day in TRADING_DAYS:
        day_date = pd.Timestamp(day)
        day_times = target_df.index[np.asarray(index_dates == day_date)]
        if len(day_times) == 0:
            continue
        t_0930 = day_date + pd.Timedelta(hours=9, minutes=30)
        if t_0930 in day_times:
            all_nan_0930 = all(factors[name].loc[t_0930].isna().all() for name in factors)
            print(f"  {day} 09:30 -> both factors entirely NaN: {all_nan_0930}")

    print()
    print("Sample rows for the first trading day:")
    first_day = pd.Timestamp(TRADING_DAYS[0])
    first_day_times = target_df.index[np.asarray(index_dates == first_day)]
    sample_times = first_day_times[:6]
    for name, factor_df in factors.items():
        print(f"--- {name} (first 6 rows, first 5 stocks) ---")
        print(factor_df.loc[sample_times, factor_df.columns[:5]])


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    target_df = pd.read_pickle(TARGET_FILE)
    FACTOR_DIR.mkdir(parents=True, exist_ok=True)

    factors = build_factors(target_df)

    for name, factor_df in factors.items():
        assert factor_df.shape == target_df.shape
        assert factor_df.index.equals(target_df.index)
        assert factor_df.columns.equals(target_df.columns)
        out_path = FACTOR_DIR / f"{name}.pkl"
        factor_df.to_pickle(out_path)
        print(f"Saved {out_path}")

    run_sanity_checks(factors, target_df)


if __name__ == "__main__":
    main()
