"""Construct Burst Volume Imbalance factors from CSI 500 Level-2 order data.

A "burst" is >=2 qualifying (non-cancellation) orders on the same stock and
side landing at exactly the same raw timestamp. BurstVolumeImbalance is the
buy/sell imbalance of burst volume over 8s / 15s / 30s trailing windows.
Every factor DataFrame is aligned exactly to the index and columns of
data/target_return_1min.pkl (see data/data_reference.txt for the raw order
schema).
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# =============================================================================
# Configuration
# =============================================================================

WINDOW_SECONDS = [8, 15, 30]
MIN_BURST_ORDERS = 2

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
    order_code in {'B', 'S'}, restricted to the target's stock universe."""
    path = RAW_DIR / day / "orders.parquet"
    con = duckdb.connect()
    df = con.execute(
        f"""
        SELECT wind_code, date, time, order_code, volume
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

    return df[["wind_code", "order_code", "timestamp", "volume"]]


def detect_bursts(orders: pd.DataFrame) -> pd.DataFrame:
    """Group orders by (wind_code, order_code, exact timestamp); a group is a
    burst when it has >= MIN_BURST_ORDERS orders. Returns one row per burst:
    wind_code, order_code, timestamp, burst_volume, order_count."""
    grouped = orders.groupby(["wind_code", "order_code", "timestamp"], sort=False).agg(
        order_count=("volume", "size"),
        burst_volume=("volume", "sum"),
    )
    bursts = grouped[grouped["order_count"] >= MIN_BURST_ORDERS].reset_index()
    return bursts


# =============================================================================
# Rolling burst-volume lookup
# =============================================================================

def build_cumulative_volume(sorted_burst_times: np.ndarray, sorted_burst_volumes: np.ndarray) -> dict:
    """Prefix sum of burst volume, sorted ascending by timestamp, so that any
    [start, end) window's total burst volume can be read off with two
    searchsorted lookups and a subtraction."""
    return {
        "timestamps": sorted_burst_times,
        "cum_volume": np.concatenate(([0.0], np.cumsum(sorted_burst_volumes))),
    }


def window_burst_volume(cum: dict, window_start_times: np.ndarray, times: np.ndarray) -> np.ndarray:
    ts = cum["timestamps"]
    idx_start = np.searchsorted(ts, window_start_times, side="left")
    idx_end = np.searchsorted(ts, times, side="left")
    return cum["cum_volume"][idx_end] - cum["cum_volume"][idx_start]


# =============================================================================
# Factor construction
# =============================================================================

def build_factors(target_df: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict]:
    universe = list(target_df.columns)
    stock_to_col = {code: i for i, code in enumerate(universe)}
    n_rows, n_cols = target_df.shape

    factor_names = [f"burst_volume_imbalance_{w}s" for w in WINDOW_SECONDS]
    factor_values = {name: np.full((n_rows, n_cols), np.nan) for name in factor_names}

    index_dates = target_df.index.normalize()
    empty_times = np.array([], dtype="datetime64[ns]")
    empty_volumes = np.array([], dtype=float)

    day_diagnostics = {}

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
        bursts = detect_bursts(orders)

        buy_bursts = bursts[bursts["order_code"] == "B"]
        sell_bursts = bursts[bursts["order_code"] == "S"]

        day_diagnostics[day] = {
            "n_buy_bursts": int(len(buy_bursts)),
            "n_sell_bursts": int(len(sell_bursts)),
            "total_buy_burst_volume": float(buy_bursts["burst_volume"].sum()),
            "total_sell_burst_volume": float(sell_bursts["burst_volume"].sum()),
        }

        # Per-stock sorted burst-time / cumulative-volume arrays for each
        # side, built once per day and reused across all lookback windows.
        buy_cum = {
            wind_code: build_cumulative_volume(
                group["timestamp"].to_numpy(), group["burst_volume"].to_numpy()
            )
            for wind_code, group in buy_bursts.sort_values(["wind_code", "timestamp"]).groupby(
                "wind_code", sort=False
            )
        }
        sell_cum = {
            wind_code: build_cumulative_volume(
                group["timestamp"].to_numpy(), group["burst_volume"].to_numpy()
            )
            for wind_code, group in sell_bursts.sort_values(["wind_code", "timestamp"]).groupby(
                "wind_code", sort=False
            )
        }
        empty_cum = build_cumulative_volume(empty_times, empty_volumes)

        for window_seconds in WINDOW_SECONDS:
            window_start_times = (day_times - pd.Timedelta(seconds=window_seconds)).to_numpy()
            row_valid = day_times_arr >= (day_open + pd.Timedelta(seconds=window_seconds)).to_datetime64()
            if not row_valid.any():
                continue  # entire window unavailable for this day; leave as NaN

            factor_name = f"burst_volume_imbalance_{window_seconds}s"

            for wind_code in universe:
                col = stock_to_col[wind_code]
                bv_b = window_burst_volume(
                    buy_cum.get(wind_code, empty_cum), window_start_times, day_times_arr
                )
                bv_s = window_burst_volume(
                    sell_cum.get(wind_code, empty_cum), window_start_times, day_times_arr
                )

                denom = bv_b + bv_s
                with np.errstate(divide="ignore", invalid="ignore"):
                    factor_val = (bv_b - bv_s) / denom
                factor_val = np.where(denom == 0, np.nan, factor_val)
                factor_val = np.where(row_valid, factor_val, np.nan)

                factor_values[factor_name][day_row_pos, col] = factor_val

    factors = {
        name: pd.DataFrame(values, index=target_df.index, columns=target_df.columns)
        for name, values in factor_values.items()
    }
    return factors, day_diagnostics


# =============================================================================
# Sanity checks
# =============================================================================

def run_sanity_checks(
    factors: dict[str, pd.DataFrame],
    target_df: pd.DataFrame,
    day_diagnostics: dict,
) -> None:
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
    print("Burst diagnostics per trading day (window-independent, from raw burst detection):")
    for day, diag in day_diagnostics.items():
        print(
            f"  {day}: buy_bursts={diag['n_buy_bursts']} "
            f"sell_bursts={diag['n_sell_bursts']} "
            f"total_buy_burst_volume={diag['total_buy_burst_volume']:.0f} "
            f"total_sell_burst_volume={diag['total_sell_burst_volume']:.0f}"
        )

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
            print(f"  {day} 09:30 -> all three factors entirely NaN: {all_nan_0930}")

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

    factors, day_diagnostics = build_factors(target_df)

    for name, factor_df in factors.items():
        assert factor_df.shape == target_df.shape
        assert factor_df.index.equals(target_df.index)
        assert factor_df.columns.equals(target_df.columns)
        out_path = FACTOR_DIR / f"{name}.pkl"
        factor_df.to_pickle(out_path)
        print(f"Saved {out_path}")

    run_sanity_checks(factors, target_df, day_diagnostics)


if __name__ == "__main__":
    main()
