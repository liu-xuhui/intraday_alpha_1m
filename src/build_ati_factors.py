"""Construct aggressive-trade-imbalance (ATI) factors from CSI 500 Level-2
trade data.

Six factors are produced: count-based and volume-based aggressive buy/sell
imbalance over 30s / 60s / 90s trailing windows. Every factor DataFrame is
aligned exactly to the index and columns of data/target_return_1min.pkl (see
data/data_reference.txt for the raw trade schema).
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# =============================================================================
# Configuration
# =============================================================================

WINDOW_SECONDS = [30, 60, 90]

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

def load_day_trades(day: str, universe: list[str]) -> pd.DataFrame:
    """Load aggressive (bs_flag in {'B','S'}) trades for one day, restricted
    to the target's stock universe. Returns wind_code, timestamp, bs_flag,
    volume, sorted by (wind_code, timestamp)."""
    path = RAW_DIR / day / "trades.parquet"
    con = duckdb.connect()
    df = con.execute(
        f"""
        SELECT wind_code, date, time, bs_flag, volume
        FROM '{path.as_posix()}'
        WHERE bs_flag IN ('B', 'S') AND wind_code = ANY(?)
        """,
        [universe],
    ).df()
    con.close()

    time_str = df["time"].astype(np.int64).astype(str).str.zfill(9)
    date_str = df["date"].astype(np.int64).astype(str)
    # HHMMSSmmm (9 digits) -> pad milliseconds to microseconds for strptime.
    timestamp_str = date_str + time_str + "000"
    df["timestamp"] = pd.to_datetime(timestamp_str, format="%Y%m%d%H%M%S%f")

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

def build_factors(target_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    universe = list(target_df.columns)
    stock_to_col = {code: i for i, code in enumerate(universe)}
    n_rows, n_cols = target_df.shape

    factor_names = [f"ati_count_{w}s" for w in WINDOW_SECONDS] + [
        f"ati_volume_{w}s" for w in WINDOW_SECONDS
    ]
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
        day_open = day_date + pd.Timedelta(hours=WINDOW_START_HOUR, minutes=WINDOW_START_MINUTE)

        trades_path = RAW_DIR / day / "trades.parquet"
        if not trades_path.exists():
            print(f"Warning: missing trades for {day}, leaving factors as NaN.")
            continue

        trades = load_day_trades(day, universe)
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
            row_valid = day_times_arr >= (day_open + pd.Timedelta(seconds=window_seconds)).to_datetime64()
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
        day_times = target_df.index[index_dates == day_date]
        if len(day_times) == 0:
            continue
        t_0930 = day_date + pd.Timedelta(hours=9, minutes=30)
        t_0931 = day_date + pd.Timedelta(hours=9, minutes=31)

        if t_0930 in day_times:
            row = pd.concat([factors[n].loc[t_0930] for n in factors], axis=0)
            all_nan_0930 = row.isna().all()
            print(f"  {day} 09:30 -> all six factors entirely NaN: {all_nan_0930}")

        if t_0931 in day_times:
            for w in WINDOW_SECONDS:
                for kind in ("count", "volume"):
                    name = f"ati_{kind}_{w}s"
                    row = factors[name].loc[t_0931]
                    if w == 90:
                        print(f"  {day} 09:31 {name} entirely NaN (expected True): {row.isna().all()}")
                    else:
                        print(f"  {day} 09:31 {name} has some non-NaN values (expected True): {row.notna().any()}")

    print()
    print("Sample rows for the first trading day:")
    first_day = pd.Timestamp(TRADING_DAYS[0])
    first_day_times = target_df.index[index_dates == first_day]
    sample_times = first_day_times[:5]
    for name, factor_df in factors.items():
        print(f"--- {name} (first 5 rows, first 5 stocks) ---")
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
