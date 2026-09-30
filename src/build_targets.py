"""Construct configurable forward intraday return targets from CSI 500
Level-2 quote data.

For each stock and each prediction minute t in the requested period (see
src/pipeline_utils.py for the --days / --start-times / --window-mins
convention), the target is the forward return of the mid-price starting at the
first valid quote at/after t and ending at the first valid quote at/after
t + HORIZON_MINUTES. See data/data_reference.txt for the raw quote schema.

The horizon H is set with --horizon-mins (default HORIZON_MINUTES). Timestamps
whose end price t + H would fall at or past the end of the raw data downloaded
for that day (data/raw/<day>/window.json) are dropped, so longer horizons give
fewer rows. The result is merged into data/target_return_<H>min.pkl: rows for
the requested timestamps are added or overwritten, all other rows are kept.

    python src/build_targets.py --days 20260820 20260821 --start-times 09:30 --window-mins 30
    python src/build_targets.py --days 20260820 --start-times 09:30 --window-mins 29 --horizon-mins 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from pipeline_utils import (
    DATA_DIR,
    RAW_DIR,
    add_period_arguments,
    day_clock_bounds,
    describe_segments,
    iter_days,
    load_universe,
    raw_data_window,
    raw_timestamps,
    save_panel,
    segments_from_args,
    segments_timestamps,
)

# =============================================================================
# Configuration
# =============================================================================

HORIZON_MINUTES = 1
MAX_QUOTE_DELAY_SECONDS = 5



# =============================================================================
# Data loading
# =============================================================================

def load_day_quotes(day: str, universe: set[str], clock_lo: int, clock_hi: int) -> pd.DataFrame:
    """Load valid mid-price quotes for one trading day with raw clock in
    [clock_lo, clock_hi], restricted to the fixed stock universe. Returns
    columns: wind_code, timestamp, mid_price, sorted by timestamp."""
    path = RAW_DIR / day / "quotes.parquet"
    con = duckdb.connect()
    df = con.execute(
        f"""
        SELECT wind_code, date, time, bid_px1, ask_px1
        FROM '{path.as_posix()}'
        WHERE bid_px1 > 0 AND ask_px1 > 0
          AND time >= ? AND time <= ?
        """,
        [clock_lo, clock_hi],
    ).df()
    con.close()

    df = df[df["wind_code"].isin(universe)]

    df["timestamp"] = raw_timestamps(df["date"], df["time"])
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
    day_times: pd.DatetimeIndex,
    universe: list[str],
    horizon_minutes: int,
    max_delay_seconds: int,
) -> tuple[pd.DataFrame, int]:
    """Build the forward-return target rows for the prediction timestamps
    day_times of a single trading day. Returns (target_df indexed by
    prediction timestamp with universe columns, count of rejected-for-delay
    boundary lookups)."""
    horizon = pd.Timedelta(minutes=horizon_minutes)
    # Every price boundary needed: each t and each t + horizon.
    boundary_times = list(day_times.append(day_times + horizon).unique().sort_values())

    clock_lo, clock_hi = day_clock_bounds(
        day_times, forward_seconds=horizon.total_seconds() + max_delay_seconds
    )
    quotes = load_day_quotes(day, set(universe), clock_lo, clock_hi)
    price_matrix, rejected_for_delay = lookup_boundary_prices(
        quotes, universe, boundary_times, max_delay_seconds
    )

    # price_matrix: rows = universe (stock order), columns = boundary_times.
    start_prices = price_matrix[list(day_times)].to_numpy()
    end_prices = price_matrix[list(day_times + horizon)].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = end_prices / start_prices - 1.0

    target_df = pd.DataFrame(
        returns.T,
        index=pd.DatetimeIndex(day_times, name="datetime"),
        columns=universe,
    )
    return target_df, rejected_for_delay


def drop_unreachable_horizons(
    timestamps: pd.DatetimeIndex, horizon_minutes: int
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Split timestamps into (kept, dropped): a timestamp is dropped when
    t + horizon is at or past the end of that day's downloaded raw window, so
    no end-price quote can exist for it."""
    horizon = pd.Timedelta(minutes=horizon_minutes)
    keep = np.ones(len(timestamps), dtype=bool)
    for day, rows in iter_days(timestamps):
        window = raw_data_window(day)
        if window is not None:
            keep[rows] = np.asarray(timestamps[rows] + horizon < window[1])
    return timestamps[keep], timestamps[~keep]


def build_target(
    timestamps: pd.DatetimeIndex,
    universe: list[str],
    horizon_minutes: int,
    max_delay_seconds: int,
) -> tuple[pd.DataFrame, int]:
    day_frames = []
    total_rejected_for_delay = 0
    for day, row_pos in iter_days(timestamps):
        path = RAW_DIR / day / "quotes.parquet"
        if not path.exists():
            print(f"Warning: missing quotes for {day}, leaving its target rows as NaN.")
            day_frames.append(pd.DataFrame(np.nan, index=timestamps[row_pos], columns=universe))
            continue
        day_target, rejected_for_delay = build_day_target(
            day, timestamps[row_pos], universe, horizon_minutes, max_delay_seconds
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
    expected_timestamps: pd.DatetimeIndex,
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
    print("SANITY CHECKS (rows built in this run)")
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

    expected_rows = len(expected_timestamps)
    print(
        f"13. Expected row count (H={horizon_minutes}min): "
        f"expected {expected_rows}, got {n_rows} "
        f"({'OK' if n_rows == expected_rows else 'MISMATCH'})"
    )
    all_nan_rows = target_df.index[target_df.isna().all(axis=1)]
    if len(all_nan_rows):
        print(
            f"14. Rows with no valid target at all: {len(all_nan_rows)} "
            f"(e.g. {list(map(str, all_nan_rows[:3]))}); t + {horizon_minutes}min likely "
            "falls outside the downloaded raw data."
        )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_period_arguments(parser)
    parser.add_argument(
        "--horizon-mins", type=int, default=HORIZON_MINUTES,
        help=f"Forward-return horizon in minutes (default: {HORIZON_MINUTES}).",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Target pickle to create or update (default: data/target_return_<H>min.pkl).",
    )
    args = parser.parse_args()
    if args.horizon_mins <= 0:
        parser.error("--horizon-mins must be positive.")
    out_path = args.out or DATA_DIR / f"target_return_{args.horizon_mins}min.pkl"

    segments = segments_from_args(args)
    timestamps, dropped = drop_unreachable_horizons(segments_timestamps(segments), args.horizon_mins)
    print(
        f"Building {args.horizon_mins}-minute target for {len(timestamps)} timestamps "
        f"({len(dropped)} dropped: t + {args.horizon_mins}min is past the downloaded data):\n"
        f"{describe_segments(segments)}"
    )
    if len(timestamps) == 0:
        raise SystemExit("No timestamp has its end price inside the downloaded data.")

    universe = load_universe()
    target_df, rejected_for_delay = build_target(
        timestamps, universe, args.horizon_mins, MAX_QUOTE_DELAY_SECONDS
    )

    save_panel(target_df, out_path)

    run_sanity_checks(target_df, universe, timestamps, args.horizon_mins, rejected_for_delay)


if __name__ == "__main__":
    main()
