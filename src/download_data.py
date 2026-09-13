from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
from huggingface_hub import HfApi


# =============================================================================
# Configuration
# =============================================================================

REPO = "venvoo/china-a-share-l2-level2-limit-order-book-tick-data"

# Project-relative paths.
# If you literally want filesystem-root /data instead, change this to:
# DATA_DIR = Path("/data")
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
UNIVERSE_FILE = DATA_DIR / "csi500_wind_codes_20260820.txt"
REPORT_FILE = DATA_DIR / "download_summary_20260820_20260828.txt"

# 2026-08-20 through 2026-08-28 contains these seven A-share trading days.
TRADING_DAYS = [
    "20260820",
    "20260821",
    "20260824",
    "20260825",
    "20260826",
    "20260827",
    "20260828",
]

START_TIME = 93000000   # 09:30:00.000 Asia/Shanghai
END_TIME = 100000000    # 10:00:00.000, exclusive

# Retry transient HTTP/Hugging Face failures.
MAX_RETRIES = 3
RETRY_SECONDS = 5

STREAMS = {
    "quotes": {
        "remote_name": "行情.parquet",
        "local_name": "quotes.parquet",
        "expected_columns": [
            "wind_code", "ex_code", "date", "time", "price", "volume", "amount",
            "num_trades", "iopv", "trade_flag", "bs_flag", "cum_volume",
            "cum_amount", "high", "low", "open", "prev_close",
            *[f"ask_px{i}" for i in range(1, 11)],
            *[f"ask_vol{i}" for i in range(1, 11)],
            *[f"bid_px{i}" for i in range(1, 11)],
            *[f"bid_vol{i}" for i in range(1, 11)],
            "wavg_ask_px", "wavg_bid_px", "tot_ask_vol", "tot_bid_vol",
            "idx_unweighted", "n_sym", "n_up", "n_down", "n_flat",
        ],
    },
    "orders": {
        "remote_name": "逐笔委托.parquet",
        "local_name": "orders.parquet",
        "expected_columns": [
            "wind_code", "ex_code", "date", "time", "order_id", "ex_order_id",
            "order_type", "order_code", "price", "volume",
        ],
    },
    "trades": {
        "remote_name": "逐笔成交.parquet",
        "local_name": "trades.parquet",
        "expected_columns": [
            "wind_code", "ex_code", "date", "time", "trade_id", "trade_code",
            "order_code", "bs_flag", "price", "volume",
            "ask_order_id", "bid_order_id",
        ],
    },
}


# =============================================================================
# Helpers
# =============================================================================

def sql_quote(value: str) -> str:
    """Quote a string literal for DuckDB SQL."""
    return "'" + value.replace("'", "''") + "'"


def human_size(n_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(n_bytes)
    for unit in units:
        if x < 1024.0 or unit == units[-1]:
            return f"{x:.2f} {unit}"
        x /= 1024.0
    raise AssertionError("unreachable")


def hhmmssmmm(x: int | None) -> str:
    if x is None:
        return "NA"
    s = f"{int(x):09d}"
    return f"{s[0:2]}:{s[2:4]}:{s[4:6]}.{s[6:9]}"


def load_universe(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Universe file not found: {path}")

    codes = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if len(codes) != 500:
        raise ValueError(
            f"Expected exactly 500 CSI 500 codes, found {len(codes)} in {path}"
        )

    if len(set(codes)) != len(codes):
        raise ValueError("Universe file contains duplicate Wind codes.")

    pattern = re.compile(r"^\d{6}\.(SH|SZ)$")
    bad = [x for x in codes if not pattern.match(x)]
    if bad:
        raise ValueError(f"Invalid Wind-code format, examples: {bad[:10]}")

    return codes


def write_report(lines: list[str]) -> None:
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def hf_path(revision: str, day: str, filename: str) -> str:
    return (
        f"hf://datasets/{REPO}@{revision}/"
        f"{day}/{filename}"
    )


def local_stats(con: duckdb.DuckDBPyConnection, path: Path) -> dict:
    p = sql_quote(str(path))

    row = con.execute(
        f"""
        SELECT
            COUNT(*) AS n_rows,
            COUNT(DISTINCT wind_code) AS n_symbols,
            MIN(time) AS min_time,
            MAX(time) AS max_time,
            MIN(date) AS min_date,
            MAX(date) AS max_date
        FROM {p}
        """
    ).fetchone()

    columns = [
        r[0]
        for r in con.execute(
            f"DESCRIBE SELECT * FROM {p}"
        ).fetchall()
    ]

    return {
        "n_rows": int(row[0]),
        "n_symbols": int(row[1]),
        "min_time": None if row[2] is None else int(row[2]),
        "max_time": None if row[3] is None else int(row[3]),
        "min_date": None if row[4] is None else int(row[4]),
        "max_date": None if row[5] is None else int(row[5]),
        "columns": columns,
    }


def universe_checks(
    con: duckdb.DuckDBPyConnection,
    path: Path,
    universe: list[str],
) -> tuple[list[str], list[str]]:
    """
    Return:
      - symbols present in local file but not in the universe (should be empty)
      - universe symbols with no rows in this stream/window
    """
    p = sql_quote(str(path))
    present = {
        r[0]
        for r in con.execute(
            f"SELECT DISTINCT wind_code FROM {p}"
        ).fetchall()
    }

    universe_set = set(universe)
    extra = sorted(present - universe_set)
    missing = sorted(universe_set - present)
    return extra, missing


def quote_coverage_stats(
    con: duckdb.DuckDBPyConnection,
    path: Path,
) -> dict:
    """
    Coverage diagnostics for active quote symbols only.

    Quotes are roughly every 3 seconds, but the archive does not guarantee an
    exact 600 rows per symbol. These are warning diagnostics, not hard failures.
    """
    p = sql_quote(str(path))

    rows = con.execute(
        f"""
        WITH s AS (
            SELECT
                wind_code,
                COUNT(*) AS n,
                MIN(time) AS first_time,
                MAX(time) AS last_time
            FROM {p}
            GROUP BY wind_code
        )
        SELECT
            COUNT(*) AS active_symbols,
            MIN(n) AS min_rows,
            MEDIAN(n) AS median_rows,
            MAX(n) AS max_rows,
            SUM(CASE WHEN first_time > 93015000 THEN 1 ELSE 0 END)
                AS late_start_symbols,
            SUM(CASE WHEN last_time < 95945000 THEN 1 ELSE 0 END)
                AS early_end_symbols,
            SUM(CASE WHEN n < 400 THEN 1 ELSE 0 END)
                AS low_row_symbols
        FROM s
        """
    ).fetchone()

    return {
        "active_symbols": int(rows[0] or 0),
        "min_rows": int(rows[1] or 0),
        "median_rows": float(rows[2] or 0),
        "max_rows": int(rows[3] or 0),
        "late_start_symbols": int(rows[4] or 0),
        "early_end_symbols": int(rows[5] or 0),
        "low_row_symbols": int(rows[6] or 0),
    }


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    universe = load_universe(UNIVERSE_FILE)
    universe_sql = ", ".join(sql_quote(x) for x in universe)

    api = HfApi()

    # Pin one repository revision for the entire experiment.
    info = api.dataset_info(REPO)
    revision = info.sha

    # -------------------------------------------------------------------------
    # Preflight: verify all required remote files exist at the pinned revision.
    # -------------------------------------------------------------------------
    repo_files = set(
        api.list_repo_files(
            repo_id=REPO,
            repo_type="dataset",
            revision=revision,
        )
    )

    missing_remote = []
    for day in TRADING_DAYS:
        for cfg in STREAMS.values():
            remote_rel = f"{day}/{cfg['remote_name']}"
            if remote_rel not in repo_files:
                missing_remote.append(remote_rel)

    if missing_remote:
        raise RuntimeError(
            "Required Hugging Face files are missing at pinned revision "
            f"{revision}:\n  " + "\n  ".join(missing_remote)
        )

    report: list[str] = [
        "CSI 500 Level-2 Raw Data Download Summary",
        "=" * 78,
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        f"Hugging Face repo: {REPO}",
        f"Pinned revision: {revision}",
        f"Universe file: {UNIVERSE_FILE}",
        f"Universe size: {len(universe)} unique Wind codes",
        "Date range: 2026-08-20 to 2026-08-28",
        f"Trading days requested: {', '.join(TRADING_DAYS)}",
        "Window: 09:30:00.000 <= time < 10:00:00.000 Asia/Shanghai",
        f"Output directory: {RAW_DIR}",
        "",
        "Preflight remote-file check: PASS",
        f"  All {len(TRADING_DAYS) * len(STREAMS)} required source files exist.",
        "",
        "Important interpretation:",
        "  Missing symbols in an individual stream are reported, not automatically",
        "  treated as extraction failure. A CSI 500 constituent can be suspended,",
        "  or a symbol can have no order/trade events in a particular window.",
        "  Hard PASS/FAIL checks concern file existence, row-copy integrity, schema,",
        "  requested date/time bounds, and universe contamination.",
        "",
    ]
    write_report(report)

    con = duckdb.connect()

    # -------------------------------------------------------------------------
    # Memory-safe settings for WSL (~8 GB RAM)
    # -------------------------------------------------------------------------

    con.execute("SET memory_limit = '3GB'")
    con.execute("SET threads = 2")
    con.execute("SET preserve_insertion_order = false")

    tmp_dir = DATA_DIR / "duckdb_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    con.execute(f"SET temp_directory = '{tmp_dir}'")
    con.execute("SET max_temp_directory_size = '20GB'")

    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")

    # Reads token saved by: hf auth login
    con.execute(
        """
        CREATE SECRET hf_intraday_alpha (
            TYPE HUGGINGFACE,
            PROVIDER credential_chain
        )
        """
    )

    hard_failures: list[str] = []
    warnings: list[str] = []
    total_bytes = 0
    total_rows = 0

    for day in TRADING_DAYS:
        day_dir = RAW_DIR / day
        day_dir.mkdir(parents=True, exist_ok=True)

        report.extend(
            [
                "",
                "=" * 78,
                f"DATE {day}",
                "=" * 78,
            ]
        )

        for stream_name, cfg in STREAMS.items():
            remote = hf_path(revision, day, cfg["remote_name"])
            local = day_dir / cfg["local_name"]

            # Always rebuild the file on rerun so a prior partial file cannot
            # silently survive.
            if local.exists():
                local.unlink()

            query = f"""
                SELECT *
                FROM {sql_quote(remote)}
                WHERE
                    wind_code IN ({universe_sql})
                    AND time >= {START_TIME}
                    AND time < {END_TIME}
            """

            copy_count = None
            last_error = None

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    result = con.execute(
                        f"""
                        COPY (
                            {query}
                        )
                        TO {sql_quote(str(local))}
                        (
                            FORMAT PARQUET,
                            COMPRESSION ZSTD
                        )
                        """
                    ).fetchone()

                    # DuckDB COPY returns the number of copied rows.
                    if result is not None and len(result) > 0:
                        copy_count = int(result[0])
                    break

                except Exception as exc:
                    last_error = exc
                    if local.exists():
                        local.unlink()

                    if attempt == MAX_RETRIES:
                        raise

                    print(
                        f"[{day} {stream_name}] attempt {attempt} failed: {exc}\n"
                        f"Retrying in {RETRY_SECONDS}s..."
                    )
                    time.sleep(RETRY_SECONDS)

            if not local.exists():
                raise RuntimeError(
                    f"Local output was not created for {day} {stream_name}. "
                    f"Last error: {last_error}"
                )

            stats = local_stats(con, local)
            extra_symbols, missing_symbols = universe_checks(con, local, universe)

            size_bytes = local.stat().st_size
            total_bytes += size_bytes
            total_rows += stats["n_rows"]

            checks = {}

            checks["nonempty"] = stats["n_rows"] > 0
            checks["copy_count_match"] = (
                copy_count is not None and copy_count == stats["n_rows"]
            )
            checks["schema_match"] = stats["columns"] == cfg["expected_columns"]
            checks["date_match"] = (
                stats["min_date"] == int(day)
                and stats["max_date"] == int(day)
            )
            checks["time_window"] = (
                stats["min_time"] is not None
                and stats["max_time"] is not None
                and stats["min_time"] >= START_TIME
                and stats["max_time"] < END_TIME
            )
            checks["universe_only"] = len(extra_symbols) == 0

            failed = [name for name, ok in checks.items() if not ok]
            status = "PASS" if not failed else "FAIL"

            if failed:
                hard_failures.append(
                    f"{day} {stream_name}: failed checks {failed}"
                )

            if missing_symbols:
                warnings.append(
                    f"{day} {stream_name}: "
                    f"{len(missing_symbols)} universe symbols have no rows"
                )

            report.extend(
                [
                    "",
                    f"[{stream_name.upper()}] {status}",
                    f"  source: {day}/{cfg['remote_name']}",
                    f"  local: {local}",
                    f"  size: {human_size(size_bytes)} ({size_bytes:,} bytes)",
                    f"  rows: {stats['n_rows']:,}",
                    f"  DuckDB COPY rows: "
                    f"{copy_count:,}" if copy_count is not None
                    else "  DuckDB COPY rows: unavailable",
                    f"  symbols with rows: {stats['n_symbols']}/{len(universe)}",
                    f"  missing universe symbols: {len(missing_symbols)}",
                    (
                        "  missing list: "
                        + ", ".join(missing_symbols)
                        if missing_symbols
                        else "  missing list: none"
                    ),
                    f"  first time: {hhmmssmmm(stats['min_time'])}",
                    f"  last time:  {hhmmssmmm(stats['max_time'])}",
                    f"  columns: {len(stats['columns'])}",
                    "  checks:",
                    *[
                        f"    {name}: {'PASS' if ok else 'FAIL'}"
                        for name, ok in checks.items()
                    ],
                ]
            )

            if stream_name == "quotes" and stats["n_rows"] > 0:
                q = quote_coverage_stats(con, local)

                report.extend(
                    [
                        "  quote coverage diagnostics (warning-only):",
                        f"    active symbols: {q['active_symbols']}",
                        f"    rows/symbol min: {q['min_rows']}",
                        f"    rows/symbol median: {q['median_rows']:.1f}",
                        f"    rows/symbol max: {q['max_rows']}",
                        (
                            "    symbols starting after 09:30:15: "
                            f"{q['late_start_symbols']}"
                        ),
                        (
                            "    symbols ending before 09:59:45: "
                            f"{q['early_end_symbols']}"
                        ),
                        (
                            "    symbols with fewer than 400 quote snapshots: "
                            f"{q['low_row_symbols']}"
                        ),
                    ]
                )

                if (
                    q["late_start_symbols"] > 0
                    or q["early_end_symbols"] > 0
                    or q["low_row_symbols"] > 0
                ):
                    warnings.append(
                        f"{day} quotes: coverage diagnostics contain warnings"
                    )

            write_report(report)

            print(
                f"{day} {stream_name:6s} | "
                f"{stats['n_rows']:,} rows | "
                f"{stats['n_symbols']} symbols | "
                f"{human_size(size_bytes)} | {status}"
            )

    con.close()

    overall_status = "PASS" if not hard_failures else "FAIL"

    report.extend(
        [
            "",
            "=" * 78,
            "OVERALL SUMMARY",
            "=" * 78,
            f"Overall hard sanity status: {overall_status}",
            f"Total local rows: {total_rows:,}",
            f"Total local storage: {human_size(total_bytes)} "
            f"({total_bytes:,} bytes)",
            f"Files written: {len(TRADING_DAYS) * len(STREAMS)}",
            f"Hard failures: {len(hard_failures)}",
            f"Warnings: {len(warnings)}",
            "",
            "Hard failures:",
            *(
                [f"  - {x}" for x in hard_failures]
                if hard_failures
                else ["  none"]
            ),
            "",
            "Warnings:",
            *(
                [f"  - {x}" for x in warnings]
                if warnings
                else ["  none"]
            ),
            "",
            "Sanity-check interpretation:",
            "  PASS means every requested source file existed at the pinned",
            "  repository revision, every local Parquet was nonempty, DuckDB's",
            "  copied-row count matched the locally readable row count, schemas",
            "  matched the known dataset schemas, all rows stayed inside the",
            "  requested date/time window, and no out-of-universe symbols appeared.",
            "",
            "  Warnings about missing symbols or quote-session coverage should be",
            "  reviewed before factor construction. They can reflect suspension or",
            "  genuine no-event periods, but can also reveal source-session gaps.",
            "",
        ]
    )

    write_report(report)

    print()
    print("=" * 78)
    print(f"Overall hard sanity status: {overall_status}")
    print(f"Total storage: {human_size(total_bytes)}")
    print(f"Summary report: {REPORT_FILE}")
    print("=" * 78)

    if hard_failures:
        raise RuntimeError(
            "Download finished with hard sanity-check failures. "
            f"See {REPORT_FILE}"
        )


if __name__ == "__main__":
    main()
