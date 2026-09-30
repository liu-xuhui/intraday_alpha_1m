"""Shared helpers for the target / factor / IC scripts.

Every builder and the IC evaluator describe *which* prediction timestamps to
work on the same way: three parallel lists

    --days         20260820 20260821 ...   trading days (YYYYMMDD)
    --start-times  09:30 ...               first prediction minute on that day
    --window-mins  30 ...                  number of one-minute timestamps

Entry i defines the segment start_times[i] + {0, 1, ..., window_mins[i] - 1}
minutes on days[i]. A --start-times or --window-mins list of length one is
broadcast to every day, so "--days A B C --start-times 09:30 --window-mins 30"
means 09:30..09:59 on each of the three days. Repeat a day to give it several
segments (e.g. a morning and an afternoon window).

Every saved matrix (target or factor) is a pickled DataFrame indexed by a
sorted, unique DatetimeIndex named "datetime" with one column per universe
stock. Re-running a builder over a new period merges the new rows into the
existing pickle: timestamps already present are overwritten, new ones added.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# Paths
# =============================================================================

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
FACTOR_DIR = DATA_DIR / "factors"
RESULTS_DIR = Path("results")
UNIVERSE_FILE = DATA_DIR / "csi500_wind_codes_20260820.txt"

# Continuous-trading sessions (Asia/Shanghai). A trailing lookback window is
# available only if it lies entirely inside the session containing t; a window
# that reaches back across the open or the lunch break is unavailable rather
# than merely empty.
SESSIONS = [("09:30", "11:30"), ("13:00", "15:00")]

DEFAULT_START_TIME = "09:30"
DEFAULT_WINDOW_MINS = 30


# =============================================================================
# Universe
# =============================================================================

def load_universe() -> list[str]:
    return [line.strip() for line in UNIVERSE_FILE.read_text().splitlines() if line.strip()]


def available_days() -> list[str]:
    return sorted(p.name for p in RAW_DIR.iterdir() if p.is_dir() and p.name.isdigit())


# =============================================================================
# Evaluation / build period
# =============================================================================

@dataclass(frozen=True)
class Segment:
    day: str        # YYYYMMDD
    start: str      # HH:MM
    minutes: int    # number of one-minute prediction timestamps

    @property
    def timestamps(self) -> pd.DatetimeIndex:
        first = pd.Timestamp(f"{self.day} {self.start}")
        return pd.DatetimeIndex(
            [first + pd.Timedelta(minutes=m) for m in range(self.minutes)], name="datetime"
        )

    def as_dict(self) -> dict:
        ts = self.timestamps
        return {
            "day": self.day,
            "start": self.start,
            "window_mins": self.minutes,
            "first_timestamp": str(ts[0]),
            "last_timestamp": str(ts[-1]),
        }


def normalize_start_time(value: str) -> str:
    """Accept 09:30, 9:30 or 0930 and return HH:MM."""
    text = value.strip().replace(":", "")
    if not text.isdigit() or len(text) not in (3, 4):
        raise ValueError(f"Cannot parse start time {value!r}; expected HH:MM or HHMM.")
    hour, minute = int(text[:-2]), int(text[-2:])
    if not (0 <= hour < 24 and 0 <= minute < 60):
        raise ValueError(f"Start time {value!r} is out of range.")
    return f"{hour:02d}:{minute:02d}"


def build_segments(days: list[str], start_times: list[str], window_mins: list[int]) -> list[Segment]:
    n = len(days)
    if n == 0:
        raise ValueError("At least one trading day is required.")
    for label, values in (("--start-times", start_times), ("--window-mins", window_mins)):
        if len(values) not in (1, n):
            raise ValueError(
                f"{label} has {len(values)} entries; expected 1 (broadcast) or {n} (one per day)."
            )
    starts = start_times * n if len(start_times) == 1 else start_times
    windows = window_mins * n if len(window_mins) == 1 else window_mins

    segments = []
    for day, start, minutes in zip(days, starts, windows):
        pd.Timestamp(day)  # validates YYYYMMDD
        if int(minutes) <= 0:
            raise ValueError(f"Window length must be positive, got {minutes} for {day}.")
        segments.append(Segment(day=day, start=normalize_start_time(start), minutes=int(minutes)))
    return segments


def segments_timestamps(segments: list[Segment]) -> pd.DatetimeIndex:
    """Sorted, de-duplicated union of every segment's timestamps."""
    stamps = pd.DatetimeIndex([], name="datetime")
    for segment in segments:
        stamps = stamps.append(segment.timestamps)
    return pd.DatetimeIndex(stamps.unique(), name="datetime").sort_values()


def add_period_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--days", nargs="+", default=None,
        help="Trading days as YYYYMMDD (default: every day under data/raw).",
    )
    parser.add_argument(
        "--start-times", nargs="+", default=[DEFAULT_START_TIME],
        help=f"First prediction minute per day, HH:MM (default: {DEFAULT_START_TIME}). "
             "One value is broadcast to all days.",
    )
    parser.add_argument(
        "--window-mins", nargs="+", type=int, default=[DEFAULT_WINDOW_MINS],
        help=f"Number of one-minute timestamps per day (default: {DEFAULT_WINDOW_MINS}). "
             "One value is broadcast to all days.",
    )


def segments_from_args(args: argparse.Namespace) -> list[Segment]:
    days = args.days if args.days else available_days()
    return build_segments(days, args.start_times, args.window_mins)


def describe_segments(segments: list[Segment]) -> str:
    return "\n".join(
        f"  {s.day}  {s.start}  +{s.minutes}m  ({s.timestamps[0].time()} .. {s.timestamps[-1].time()})"
        for s in segments
    )


def period_tag(segments: list[Segment]) -> str:
    """Short, filesystem-safe name for a period, e.g.
    20260820-20260828_7d_0930_30m, or 20260820-20260828_9seg_1a2b3c when the
    segments do not share one start time and window length."""
    days = sorted({s.day for s in segments})
    day_part = days[0] if len(days) == 1 else f"{days[0]}-{days[-1]}"
    shapes = {(s.start, s.minutes) for s in segments}
    one_per_day = len(segments) == len(days)
    if len(shapes) == 1 and one_per_day:
        start, minutes = shapes.pop()
        count = f"_{len(days)}d" if len(days) > 1 else ""
        return f"{day_part}{count}_{start.replace(':', '')}_{minutes}m"
    key = ";".join(f"{s.day},{s.start},{s.minutes}" for s in sorted(segments, key=lambda s: (s.day, s.start)))
    digest = hashlib.sha1(key.encode()).hexdigest()[:6]
    return f"{day_part}_{len(segments)}seg_{digest}"


# =============================================================================
# Time helpers
# =============================================================================

def iter_days(timestamps: pd.DatetimeIndex):
    """Yield (YYYYMMDD, row positions into timestamps) for each calendar day."""
    dates = timestamps.normalize()
    for day_date in dates.unique():
        yield day_date.strftime("%Y%m%d"), np.nonzero(np.asarray(dates == day_date))[0]


def clock_int(ts: pd.Timestamp) -> int:
    """Timestamp -> the raw HHMMSSmmm integer clock used by the Parquet files."""
    return ts.hour * 10_000_000 + ts.minute * 100_000 + ts.second * 1_000 + ts.microsecond // 1_000


def raw_timestamps(date: pd.Series, time: pd.Series) -> pd.Series:
    """Raw integer date (YYYYMMDD) + time (HHMMSSmmm) columns -> datetimes."""
    time_str = time.astype(np.int64).astype(str).str.zfill(9)
    date_str = date.astype(np.int64).astype(str)
    # HHMMSSmmm (9 digits) -> pad milliseconds to microseconds for strptime.
    return pd.to_datetime(date_str + time_str + "000", format="%Y%m%d%H%M%S%f")


@lru_cache(maxsize=None)
def raw_data_window(day: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """[start, end) clock window held by data/raw/<day>/, as recorded in
    window.json by download_data.py; None if the day has no record."""
    path = RAW_DIR / day / "window.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    return (
        pd.Timestamp(f"{day} {record['start_time']}"),
        pd.Timestamp(f"{day} {record['end_time']}"),
    )


def lookback_available(times: pd.DatetimeIndex, lookback_seconds: float) -> np.ndarray:
    """True where the trailing window [t - lookback, t) lies entirely inside the
    continuous-trading session containing t *and* inside the window of raw data
    downloaded for that day (so a window reaching before the first downloaded
    message is unavailable, not merely empty)."""
    result = np.zeros(len(times), dtype=bool)
    day = times.normalize()
    lookback = pd.Timedelta(seconds=lookback_seconds)
    for open_clock, close_clock in SESSIONS:
        session_open = day + pd.Timedelta(open_clock + ":00")
        session_close = day + pd.Timedelta(close_clock + ":00")
        in_session = (times >= session_open) & (times <= session_close)
        result |= np.asarray(in_session & (times - lookback >= session_open))

    for day_str, rows in iter_days(times):
        window = raw_data_window(day_str)
        if window is None:
            continue
        start, end = window
        # The window is half-open and excludes t itself, so t == end is fine.
        covered = (times[rows] - lookback >= start) & (times[rows] <= end)
        result[rows] &= np.asarray(covered)
    return result


def day_clock_bounds(
    day_times: pd.DatetimeIndex,
    lookback_seconds: float = 0.0,
    forward_seconds: float = 0.0,
) -> tuple[int, int]:
    """Inclusive (lo, hi) raw HHMMSSmmm bounds covering every row a builder
    might touch for these prediction timestamps, for filtering the Parquet read."""
    lo = day_times.min() - pd.Timedelta(seconds=lookback_seconds)
    hi = day_times.max() + pd.Timedelta(seconds=forward_seconds)
    day_start = day_times.min().normalize()
    lo = max(lo, day_start)
    hi = min(hi, day_start + pd.Timedelta(hours=23, minutes=59, seconds=59, milliseconds=999))
    return clock_int(lo), clock_int(hi)


# =============================================================================
# Panel persistence
# =============================================================================

def save_panel(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Merge frame's rows into the pickle at path (overwriting timestamps that
    already exist, adding the rest), save it sorted, and return the result."""
    frame = frame.copy()
    frame.index = pd.DatetimeIndex(frame.index, name="datetime")
    if not frame.index.is_unique:
        raise ValueError(f"{path}: new rows have duplicate timestamps.")

    n_replaced = 0
    if path.exists():
        existing = pd.read_pickle(path)
        n_replaced = int(existing.index.isin(frame.index).sum())
        kept = existing[~existing.index.isin(frame.index)]
        extra_columns = kept.columns.difference(frame.columns, sort=False)
        columns = frame.columns.append(extra_columns)
        combined = pd.concat([kept.reindex(columns=columns), frame.reindex(columns=columns)])
    else:
        combined = frame

    combined = combined.sort_index()
    combined.index.name = "datetime"
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_pickle(path)
    print(
        f"Saved {path}: {len(frame) - n_replaced} rows added, {n_replaced} replaced, "
        f"{len(combined)} total ({combined.index.min()} .. {combined.index.max()})"
    )
    return combined


def factor_period_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    add_period_arguments(parser)
    parser.add_argument(
        "--out-dir", type=Path, default=FACTOR_DIR,
        help=f"Directory holding the factor pickles (default: {FACTOR_DIR}).",
    )
    return parser
