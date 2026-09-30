"""Construct depth-band limit-order-book imbalance factors from CSI 500
Level-2 quote snapshots.

Twelve factors are produced: for each depth band [a:b] in {2:3, 4:5, 6:10} the
snapshot imbalance (B_ab - A_ab) / (B_ab + A_ab) of displayed bid/ask volume
summed over *only* those levels, averaged over trailing {3s, 15s, 30s, 60s}
windows.

Unlike the cumulative 1:N factors in build_lob_imbalance_factors.py, a band
excludes the levels below it, so it isolates what a specific slice of the book
is doing. A band can legitimately be empty on both sides (a thin book may not
populate levels 6-10 at all), in which case the snapshot imbalance is NaN.

These are order-book *state* factors: the imbalance is computed per snapshot
first and only then averaged over time, so resting volume is never summed
across snapshots as if it were new orders.

Factors are computed for every prediction minute in the requested period (see
src/pipeline_utils.py) and merged into data/factors/<name>.pkl (see
data/data_reference.txt for the raw quote schema).

    python src/build_lob_band_imbalance_factors.py --days 20260820 --start-times 09:30 --window-mins 30
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

# Inclusive, 1-indexed (first_level, last_level) book slices.
DEPTH_BANDS = [(2, 3), (4, 5), (6, 10)]
WINDOW_SECONDS = [3, 15, 30, 60]

MAX_LEVEL = 10  # the quote schema publishes ten levels per side


def factor_name(band: tuple[int, int], window_seconds: int) -> str:
    first, last = band
    return f"lob_band_imbalance_l{first}_{last}_mean_{window_seconds}s"


def band_key(band: tuple[int, int]) -> str:
    first, last = band
    return f"l{first}_{last}"


# =============================================================================
# Data loading
# =============================================================================

def load_day_quotes(day: str, universe: list[str], clock_lo: int, clock_hi: int) -> pd.DataFrame:
    """Load quote snapshots for one day with raw clock in [clock_lo,
    clock_hi], restricted to the stock universe. Returns wind_code, timestamp
    and the twenty displayed-volume columns, sorted by (wind_code, timestamp)."""
    path = RAW_DIR / day / "quotes.parquet"
    volume_columns = [f"bid_vol{k}" for k in range(1, MAX_LEVEL + 1)] + [
        f"ask_vol{k}" for k in range(1, MAX_LEVEL + 1)
    ]
    con = duckdb.connect()
    df = con.execute(
        f"""
        SELECT wind_code, date, time, {", ".join(volume_columns)}
        FROM '{path.as_posix()}'
        WHERE wind_code = ANY(?)
          AND time >= ? AND time <= ?
        """,
        [universe, clock_lo, clock_hi],
    ).df()
    con.close()

    df["timestamp"] = raw_timestamps(df["date"], df["time"])

    df = df[["wind_code", "timestamp", *volume_columns]]
    # Stable sort keeps file order within a (stock, timestamp) tie, so the
    # de-duplication below is deterministic.
    df = df.sort_values(["wind_code", "timestamp"], kind="stable").reset_index(drop=True)

    # Two snapshots for the same stock at the same instant would double-weight
    # one book state in the mean; keep the last, which is the freshest record
    # for that timestamp. (None occur in the 2026-08-20..28 sample.)
    duplicate_mask = df.duplicated(subset=["wind_code", "timestamp"], keep="last")
    n_duplicates = int(duplicate_mask.sum())
    if n_duplicates:
        print(f"  {day}: dropped {n_duplicates} duplicate (stock, timestamp) snapshots")
        df = df[~duplicate_mask].reset_index(drop=True)

    return df


# =============================================================================
# Snapshot-level band imbalance
# =============================================================================

def snapshot_band_imbalances(
    quotes: pd.DataFrame,
    bands: list[tuple[int, int]] = DEPTH_BANDS,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, int]]]:
    """Per-snapshot imbalance (B_ab - A_ab) / (B_ab + A_ab) for every band in
    bands, where B_ab and A_ab sum displayed volume over levels a..b
    only. NaN where the band's levels are unusable or where the band holds no
    depth on either side. Also returns per-band counts for diagnostics."""
    bid = quotes[[f"bid_vol{k}" for k in range(1, MAX_LEVEL + 1)]].to_numpy(dtype=float)
    ask = quotes[[f"ask_vol{k}" for k in range(1, MAX_LEVEL + 1)]].to_numpy(dtype=float)

    # A level is usable only if both sides are present and non-negative.
    level_ok = np.isfinite(bid) & np.isfinite(ask) & (bid >= 0) & (ask >= 0)

    n_snapshots = bid.shape[0]
    zero_column = np.zeros((n_snapshots, 1))
    bid_prefix = np.concatenate([zero_column, np.cumsum(np.where(np.isfinite(bid), bid, 0.0), axis=1)], axis=1)
    ask_prefix = np.concatenate([zero_column, np.cumsum(np.where(np.isfinite(ask), ask, 0.0), axis=1)], axis=1)

    imbalances = {}
    diagnostics = {}
    for band in bands:
        first, last = band
        # Prefix difference gives the sum over levels first..last inclusive.
        b = bid_prefix[:, last] - bid_prefix[:, first - 1]
        a = ask_prefix[:, last] - ask_prefix[:, first - 1]
        band_ok = np.all(level_ok[:, first - 1 : last], axis=1)

        denom = b + a
        with np.errstate(divide="ignore", invalid="ignore"):
            imbalance = (b - a) / denom
        usable = band_ok & (denom > 0)
        imbalances[band_key(band)] = np.where(usable, imbalance, np.nan)

        diagnostics[band_key(band)] = {
            "snapshots": n_snapshots,
            "empty_band": int((band_ok & (denom == 0)).sum()),
            "unusable_levels": int((~band_ok).sum()),
            "empty_bid_side": int((usable & (b == 0)).sum()),
            "empty_ask_side": int((usable & (a == 0)).sum()),
        }
    return imbalances, diagnostics


# =============================================================================
# Per-stock cumulative arrays
# =============================================================================

def build_stock_cumulatives(group: pd.DataFrame) -> dict:
    """Prefix sums of the snapshot imbalance and of the valid-snapshot count,
    one pair per band, sorted ascending by timestamp, so that a [start, end)
    window mean is two searchsorted lookups and a subtraction."""
    cumulatives = {"timestamps": group["timestamp"].to_numpy()}
    for band in DEPTH_BANDS:
        key = band_key(band)
        imbalance = group[f"imbalance_{key}"].to_numpy()
        valid = np.isfinite(imbalance)
        cumulatives[f"cum_sum_{key}"] = np.concatenate(
            ([0.0], np.cumsum(np.where(valid, imbalance, 0.0)))
        )
        cumulatives[f"cum_count_{key}"] = np.concatenate(([0], np.cumsum(valid)))
    return cumulatives


def window_mean(
    cumulatives: dict,
    key: str,
    window_start_times: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Mean snapshot imbalance over the half-open window
    [window_start_times[k], times[k]) at each k. NaN where the window holds no
    valid snapshot. The 'left' side on the upper bound makes it strict: a
    snapshot stamped exactly at t is never included."""
    ts = cumulatives["timestamps"]
    idx_start = np.searchsorted(ts, window_start_times, side="left")
    idx_end = np.searchsorted(ts, times, side="left")

    total = cumulatives[f"cum_sum_{key}"][idx_end] - cumulatives[f"cum_sum_{key}"][idx_start]
    count = cumulatives[f"cum_count_{key}"][idx_end] - cumulatives[f"cum_count_{key}"][idx_start]

    with np.errstate(divide="ignore", invalid="ignore"):
        mean = total / count
    return np.where(count == 0, np.nan, mean)


# =============================================================================
# Factor construction
# =============================================================================

def build_factors(
    timestamps: pd.DatetimeIndex, universe: list[str]
) -> tuple[dict[str, pd.DataFrame], dict]:
    stock_to_col = {code: i for i, code in enumerate(universe)}
    n_rows, n_cols = len(timestamps), len(universe)

    factor_values = {
        factor_name(band, window_seconds): np.full((n_rows, n_cols), np.nan)
        for band in DEPTH_BANDS
        for window_seconds in WINDOW_SECONDS
    }

    totals = {
        band_key(band): {
            "snapshots": 0, "empty_band": 0, "unusable_levels": 0,
            "empty_bid_side": 0, "empty_ask_side": 0,
        }
        for band in DEPTH_BANDS
    }

    for day, day_row_pos in iter_days(timestamps):
        day_times = timestamps[day_row_pos]
        day_times_arr = day_times.to_numpy()

        quotes_path = RAW_DIR / day / "quotes.parquet"
        if not quotes_path.exists():
            print(f"Warning: missing quotes for {day}, leaving factors as NaN.")
            continue

        clock_lo, clock_hi = day_clock_bounds(day_times, lookback_seconds=max(WINDOW_SECONDS))
        quotes = load_day_quotes(day, universe, clock_lo, clock_hi)
        imbalances, diagnostics = snapshot_band_imbalances(quotes)
        for key, values in imbalances.items():
            quotes[f"imbalance_{key}"] = values
        for key, counts in diagnostics.items():
            for field, value in counts.items():
                totals[key][field] += value

        # Per-stock prefix sums, built once per day and reused by every
        # (band, window) combination.
        stock_cumulatives = {
            wind_code: build_stock_cumulatives(group)
            for wind_code, group in quotes.groupby("wind_code", sort=False)
            if wind_code in stock_to_col
        }

        for window_seconds in WINDOW_SECONDS:
            window_start_times = (day_times - pd.Timedelta(seconds=window_seconds)).to_numpy()
            # A window reaching back before the session open is unavailable
            # rather than merely empty; blank the whole row.
            row_valid = lookback_available(day_times, window_seconds)
            if not row_valid.any():
                continue

            for wind_code, cumulatives in stock_cumulatives.items():
                col = stock_to_col[wind_code]
                for band in DEPTH_BANDS:
                    values = window_mean(
                        cumulatives, band_key(band), window_start_times, day_times_arr
                    )
                    values = np.where(row_valid, values, np.nan)
                    factor_values[factor_name(band, window_seconds)][day_row_pos, col] = values

    # A mean of snapshot imbalances, each of which lies in [-1, 1], is
    # mathematically in [-1, 1]. The prefix-sum subtraction can still overshoot
    # the bound by ~1e-15 of floating-point noise (e.g. a one-sided band whose
    # every snapshot is exactly +1), so clip that away and report how large it
    # ever got, rather than repairing it silently.
    max_excess = 0.0
    for values in factor_values.values():
        finite = np.isfinite(values)
        if finite.any():
            max_excess = max(max_excess, float(np.abs(values[finite]).max()) - 1.0)
        np.clip(values, -1.0, 1.0, out=values)
    print(f"Max float-error excess beyond [-1, 1] before clipping: {max(max_excess, 0.0):.3e}")

    factors = {
        name: pd.DataFrame(values, index=timestamps, columns=universe)
        for name, values in factor_values.items()
    }
    return factors, totals


# =============================================================================
# Sanity checks
# =============================================================================

def run_sanity_checks(
    factors: dict[str, pd.DataFrame],
    timestamps: pd.DatetimeIndex,
    n_stocks: int,
    totals: dict,
) -> None:
    print("=" * 110)
    print("SANITY CHECKS (rows built in this run)")
    print("=" * 110)

    header = (
        f"{'file':<46}{'shape':>12}{'non_NaN':>10}{'frac':>8}"
        f"{'mean':>10}{'std':>9}{'min':>9}{'max':>9}{'out_of_range':>14}"
    )
    print(header)
    print("-" * len(header))

    all_aligned = True
    any_out_of_range = False
    for name, factor_df in factors.items():
        values = factor_df.to_numpy()
        finite = values[np.isfinite(values)]
        out_of_range = int(((finite < -1.0) | (finite > 1.0)).sum())
        any_out_of_range = any_out_of_range or out_of_range > 0

        aligned = factor_df.index.equals(timestamps) and factor_df.shape[1] == n_stocks
        all_aligned = all_aligned and aligned

        print(
            f"{name + '.pkl':<46}{str(factor_df.shape):>12}{finite.size:>10}"
            f"{finite.size / values.size:>8.3f}{finite.mean():>10.4f}{finite.std():>9.4f}"
            f"{finite.min():>9.4f}{finite.max():>9.4f}{out_of_range:>14}"
        )

    print("-" * len(header))
    print(f"All {len(factors)} factors match the requested timestamps and universe: {all_aligned}")
    print(f"Any finite value outside [-1, 1]: {any_out_of_range}")

    print()
    print("Snapshot-level band diagnostics (all requested days):")
    print(f"  {'band':<10}{'snapshots':>12}{'empty_band':>13}{'empty_band_%':>14}"
          f"{'one_sided_bid':>15}{'one_sided_ask':>15}")
    for band in DEPTH_BANDS:
        key = band_key(band)
        t = totals[key]
        pct = 100.0 * t["empty_band"] / t["snapshots"] if t["snapshots"] else 0.0
        print(f"  {key:<10}{t['snapshots']:>12,}{t['empty_band']:>13,}{pct:>13.2f}%"
              f"{t['empty_ask_side']:>15,}{t['empty_bid_side']:>15,}")
    print("  empty_band = no depth on either side in that slice -> snapshot imbalance NaN")
    print("  one_sided  = depth on one side only -> imbalance saturates at +1 / -1")

    print()
    print("Rows whose lookback reaches before the session open are entirely NaN:")
    for w in WINDOW_SECONDS:
        unavailable = ~lookback_available(timestamps, w)
        names = [n for n in factors if n.endswith(f"_mean_{w}s")]
        all_nan = all(bool(factors[n][unavailable].isna().all(axis=None)) for n in names)
        print(f"  {w:>3}s window: {int(unavailable.sum())} unavailable rows, all NaN: {all_nan}")

    sample_name = factor_name(DEPTH_BANDS[0], 30)
    print()
    print(f"Sample rows ({sample_name}, first 6 rows, first 5 stocks):")
    sample = factors[sample_name]
    print(sample.loc[timestamps[:6], sample.columns[:5]])


# =============================================================================
# Manual verification against a brute-force recomputation
# =============================================================================

def verify_examples(
    factors: dict[str, pd.DataFrame], timestamps: pd.DatetimeIndex, universe: list[str]
) -> None:
    """Recompute a handful of cells directly from the raw Parquet, the slow and
    obvious way, and print every intermediate quantity so the window logic can
    be checked by eye."""
    print()
    print("=" * 110)
    print("MANUAL VERIFICATION (independent brute-force recomputation from raw quotes)")
    print("=" * 110)

    day, first_rows = next(iter_days(timestamps))
    day_times = timestamps[first_rows]
    path = (RAW_DIR / day / "quotes.parquet").as_posix()
    if not (RAW_DIR / day / "quotes.parquet").exists():
        print(f"Skipped: no raw quotes for {day}.")
        return

    # The earliest timestamp whose longest lookback is available, and one from
    # the middle of the day's grid.
    usable = day_times[lookback_available(day_times, max(WINDOW_SECONDS))]
    if len(usable) == 0:
        print("Skipped: no timestamp on the first day has a full lookback window.")
        return
    t_early, t_mid = usable[0], usable[len(usable) // 2]

    con = duckdb.connect()
    # Prefer a stock that actually has a snapshot stamped exactly at t_early,
    # so the strict upper bound is visibly exercised.
    exact = con.execute(
        f"SELECT wind_code FROM '{path}' WHERE time = ? AND wind_code = ANY(?) "
        "ORDER BY wind_code LIMIT 1",
        [int(t_early.strftime("%H%M%S%f")[:9]), universe],
    ).fetchone()
    stock = exact[0] if exact else universe[0]

    checks = [
        (stock, t_early, (2, 3), 3),
        (stock, t_early, (4, 5), 15),
        (stock, t_mid, (6, 10), 30),
    ]
    checks = [c for c in checks if factor_name(c[2], c[3]) in factors]

    for wind_code, t, band, window_seconds in checks:
        first, last = band
        window_start = t - pd.Timedelta(seconds=window_seconds)

        bid_sum = " + ".join(f"bid_vol{k}" for k in range(first, last + 1))
        ask_sum = " + ".join(f"ask_vol{k}" for k in range(first, last + 1))
        rows = con.execute(
            f"""
            SELECT time, ({bid_sum}) AS b_ab, ({ask_sum}) AS a_ab
            FROM '{path}'
            WHERE wind_code = ? AND time >= ? AND time <= ?
            ORDER BY time
            """,
            [
                wind_code,
                int(window_start.strftime("%H%M%S%f")[:9]),
                int(t.strftime("%H%M%S%f")[:9]),
            ],
        ).df()

        print()
        print(f"{wind_code}  t={t}  band=levels {first}:{last}  L={window_seconds}s  "
              f"window=[{window_start.time()}, {t.time()})")
        print(f"  {'quote_time':<14}{'B_ab':>14}{'A_ab':>14}{'imbalance':>12}   used?")

        used = []
        for _, row in rows.iterrows():
            clock_str = f"{int(row['time']):09d}"
            pretty = f"{clock_str[0:2]}:{clock_str[2:4]}:{clock_str[4:6]}.{clock_str[6:9]}"
            b, a = float(row["b_ab"]), float(row["a_ab"])
            imbalance = (b - a) / (b + a) if (b + a) > 0 else float("nan")
            # The window is half-open: t itself is excluded.
            in_window = int(row["time"]) < int(t.strftime("%H%M%S%f")[:9])
            if in_window and np.isfinite(imbalance):
                used.append(imbalance)
            print(f"  {pretty:<14}{b:>14,.0f}{a:>14,.0f}{imbalance:>12.6f}   "
                  f"{'yes' if in_window else 'NO  <- excluded: stamped at t'}")

        expected = float(np.mean(used)) if used else float("nan")
        actual = factors[factor_name(band, window_seconds)].loc[t, wind_code]
        match = (np.isnan(expected) and np.isnan(actual)) or np.isclose(expected, actual, atol=1e-12)
        print(f"  brute-force mean over {len(used)} snapshot(s): {expected:.10f}")
        print(f"  pipeline factor value:                        {actual:.10f}")
        print(f"  MATCH: {match}")

    con.close()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = factor_period_parser(__doc__).parse_args()
    segments = segments_from_args(args)
    timestamps = segments_timestamps(segments)
    print(f"Building LOB band-imbalance factors for {len(timestamps)} timestamps:\n{describe_segments(segments)}")

    universe = load_universe()
    factors, totals = build_factors(timestamps, universe)

    for name, factor_df in factors.items():
        save_panel(factor_df, args.out_dir / f"{name}.pkl")

    run_sanity_checks(factors, timestamps, len(universe), totals)
    verify_examples(factors, timestamps, universe)


if __name__ == "__main__":
    main()
