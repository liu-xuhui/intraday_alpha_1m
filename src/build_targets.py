"""Construct configurable forward intraday return targets from CSI 500
Level-2 quote data.

For each stock and each minute boundary t in the 09:30-10:00 research
window, the target is the forward return of the mid-price starting at the
first valid quote at/after t and ending at the first valid quote at/after
t + HORIZON_MINUTES. See data/data_reference.txt for the raw quote schema.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# =============================================================================
# Configuration
# =============================================================================

HORIZON_MINUTES = 1
MAX_QUOTE_DELAY_SECONDS = 5

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
UNIVERSE_FILE = DATA_DIR / "csi500_wind_codes_20260820.txt"

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
WINDOW_MINUTES = 30  # research window is 09:30-10:00


# =============================================================================
# Data loading
# =============================================================================

def load_universe() -> list[str]:
    codes = [line.strip() for line in UNIVERSE_FILE.read_text().splitlines() if line.strip()]
    return codes


def load_day_quotes(day: str, universe: set[str]) -> pd.DataFrame:
    """Load valid mid-price quotes for one trading day, restricted to the
    fixed stock universe. Returns columns: wind_code, timestamp, mid_price,
    sorted by timestamp."""
    path = RAW_DIR / day / "quotes.parquet"
    con = duckdb.connect()
    df = con.execute(
        f"""
        SELECT wind_code, date, time, bid_px1, ask_px1
        FROM '{path.as_posix()}'
        WHERE bid_px1 > 0 AND ask_px1 > 0
        """
    ).df()
    con.close()

    df = df[df["wind_code"].isin(universe)]

    time_str = df["time"].astype(np.int64).astype(str).str.zfill(9)
    date_str = df["date"].astype(np.int64).astype(str)
    # HHMMSSmmm (9 digits) -> pad milliseconds to microseconds for strptime.
    timestamp_str = date_str + time_str + "000"
    df["timestamp"] = pd.to_datetime(timestamp_str, format="%Y%m%d%H%M%S%f")
    df["mid_price"] = (df["bid_px1"] + df["ask_px1"]) / 2.0

    df = df[["wind_code", "timestamp", "mid_price"]]
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# =============================================================================
# Forward-quote lookup
# =============================================================================

def lookup_boundary_prices(
    quotes: pd.DataFrame,
    universe: list[str],
    boundary_times: list[pd.Timestamp],
    max_delay_seconds: int,
) -> tuple[pd.DataFrame, int]:
    """For every (stock, boundary) pair, find the earliest quote at/after the
    boundary time, subject to max_delay_seconds. Returns a stock x boundary
    price matrix (NaN where no acceptable quote exists) and the count of
    lookups that found a forward quote which was rejected for being too
    stale (i.e. later than the tolerance, as opposed to no quote at all)."""
    n_boundaries = len(boundary_times)
    left = pd.DataFrame(
        {
            "wind_code": np.repeat(universe, n_boundaries),
            "boundary_time": np.tile(pd.to_datetime(boundary_times), len(universe)),
        }
    ).sort_values("boundary_time").reset_index(drop=True)

    merged = pd.merge_asof(
        left,
        quotes,
        left_on="boundary_time",
        right_on="timestamp",
        by="wind_code",
        direction="forward",
    )

    delay_seconds = (merged["timestamp"] - merged["boundary_time"]).dt.total_seconds()
    found = merged["timestamp"].notna()
    within_tolerance = found & (delay_seconds <= max_delay_seconds)
    rejected_for_delay = int((found & ~within_tolerance).sum())

    merged.loc[~within_tolerance, "mid_price"] = np.nan

    price_matrix = merged.pivot(index="wind_code", columns="boundary_time", values="mid_price")
    price_matrix = price_matrix.reindex(index=universe, columns=boundary_times)
    return price_matrix, rejected_for_delay


# =============================================================================
# Target construction
# =============================================================================

def build_day_target(
    day: str,
    universe: list[str],
    horizon_minutes: int,
    max_delay_seconds: int,
) -> tuple[pd.DataFrame, int]:
    """Build the forward-return target matrix for a single trading day.
    Returns (target_df indexed by factor timestamp with universe columns,
    count of rejected-for-delay boundary lookups)."""
    n_boundary_offsets = WINDOW_MINUTES  # every minute mark in 09:30-10:00
    if horizon_minutes >= n_boundary_offsets:
        raise ValueError(
            f"HORIZON_MINUTES={horizon_minutes} is too large for the "
            f"{WINDOW_MINUTES}-minute research window."
        )

    year, month, day_num = int(day[:4]), int(day[4:6]), int(day[6:8])
    day_start = pd.Timestamp(
        year=year, month=month, day=day_num,
        hour=WINDOW_START_HOUR, minute=WINDOW_START_MINUTE,
    )
    boundary_times = [day_start + pd.Timedelta(minutes=m) for m in range(n_boundary_offsets)]

    universe_set = set(universe)
    quotes = load_day_quotes(day, universe_set)
    price_matrix, rejected_for_delay = lookup_boundary_prices(
        quotes, universe, boundary_times, max_delay_seconds
    )

    # price_matrix: rows = universe (stock order), columns = boundary_times.
    prices = price_matrix.to_numpy()  # shape (n_stocks, n_boundaries)
    n_valid_t = n_boundary_offsets - horizon_minutes

    start_prices = prices[:, :n_valid_t]
    end_prices = prices[:, horizon_minutes:horizon_minutes + n_valid_t]
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = end_prices / start_prices - 1.0

    target_df = pd.DataFrame(
        returns.T,
        index=pd.DatetimeIndex(boundary_times[:n_valid_t], name="datetime"),
        columns=universe,
    )
    return target_df, rejected_for_delay


def build_target(
    universe: list[str],
    horizon_minutes: int,
    max_delay_seconds: int,
) -> tuple[pd.DataFrame, int]:
    day_frames = []
    total_rejected_for_delay = 0
    for day in TRADING_DAYS:
        path = RAW_DIR / day / "quotes.parquet"
        if not path.exists():
            print(f"Warning: missing quotes for {day}, skipping.")
            continue
        day_target, rejected_for_delay = build_day_target(
            day, universe, horizon_minutes, max_delay_seconds
        )
        day_frames.append(day_target)
        total_rejected_for_delay += rejected_for_delay

    target_df = pd.concat(day_frames, axis=0)
    target_df = target_df.sort_index()
    return target_df, total_rejected_for_delay


# =============================================================================
# Sanity checks
# =============================================================================

def run_sanity_checks(
    target_df: pd.DataFrame,
    universe: list[str],
    horizon_minutes: int,
    rejected_for_delay: int,
) -> None:
    values = target_df.to_numpy()
    n_rows, n_cols = values.shape
    n_days = len(set(target_df.index.date))
    n_nan = int(np.isnan(values).sum())
    n_total = values.size
    columns_match = list(target_df.columns) == universe

    print("=" * 78)
    print("SANITY CHECKS")
    print("=" * 78)
    print(f"1. Target matrix shape: {target_df.shape}")
    print(f"2. Number of trading days: {n_days}")
    print(f"3. Number of datetime rows: {n_rows}")
    print(f"4. Number of stock columns: {n_cols}")
    print(f"5. Columns match universe file exactly (order included): {columns_match}")
    print(f"6. First prediction datetime: {target_df.index.min()}")
    print(f"   Last prediction datetime:  {target_df.index.max()}")
    print(f"7. Datetime index is unique: {target_df.index.is_unique}")
    print(f"8. NaN count: {n_nan} / {n_total} ({100 * n_nan / n_total:.2f}%)")

    non_missing_per_row = target_df.notna().sum(axis=1)
    print("9. Non-missing stocks per prediction timestamp (summary):")
    print(non_missing_per_row.describe())

    print(
        f"10. Boundary lookups rejected for exceeding "
        f"MAX_QUOTE_DELAY_SECONDS={MAX_QUOTE_DELAY_SECONDS}s: {rejected_for_delay}"
    )

    flat = values[~np.isnan(values)]
    print("11. Descriptive statistics of non-missing returns:")
    print(f"    mean:   {flat.mean():.6f}")
    print(f"    std:    {flat.std():.6f}")
    print(f"    min:    {flat.min():.6f}")
    print(f"    max:    {flat.max():.6f}")
    for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99):
        print(f"    q{q:.2f}:  {np.quantile(flat, q):.6f}")

    print("12. First rows:")
    print(target_df.head())
    print("    Last rows:")
    print(target_df.tail())

    if horizon_minutes == 1:
        expected_rows = 7 * 29
        print(
            f"Expected row count check (HORIZON_MINUTES=1, 7 days): "
            f"expected {expected_rows}, got {n_rows} "
            f"({'OK' if n_rows == expected_rows else 'MISMATCH'})"
        )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    universe = load_universe()
    target_df, rejected_for_delay = build_target(
        universe, HORIZON_MINUTES, MAX_QUOTE_DELAY_SECONDS
    )

    out_path = DATA_DIR / f"target_return_{HORIZON_MINUTES}min.pkl"
    target_df.to_pickle(out_path)
    print(f"Saved target matrix to {out_path}")

    run_sanity_checks(target_df, universe, HORIZON_MINUTES, rejected_for_delay)


if __name__ == "__main__":
    main()
