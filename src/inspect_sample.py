from huggingface_hub import HfApi
import duckdb
import os
import sys


# ============================================================
# Config
# ============================================================

REPO = "venvoo/china-a-share-l2-level2-limit-order-book-tick-data"
DAY = "20260828"

START_TIME = 93000000    # 09:30:00.000
END_TIME = 100000000     # 10:00:00.000

N_SAMPLE_ROWS = 10

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"inspect_sample_{DAY}.txt")


# ============================================================
# Tee stdout to console + file
# ============================================================

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = open(OUTPUT_PATH, "w", encoding="utf-8")
sys.stdout = Tee(sys.stdout, log_file)


# ============================================================
# Get current repo revision
# ============================================================

api = HfApi()
REV = api.dataset_info(REPO).sha

print(f"Repository revision: {REV}")
print(f"Trading day: {DAY}")


def hf_path(filename):
    return (
        f"hf://datasets/{REPO}@{REV}/"
        f"{DAY}/{filename}"
    )


QUOTE_PATH = hf_path("行情.parquet")
ORDER_PATH = hf_path("逐笔委托.parquet")
TRADE_PATH = hf_path("逐笔成交.parquet")


# ============================================================
# DuckDB setup
# ============================================================

con = duckdb.connect()

con.execute("INSTALL httpfs")
con.execute("LOAD httpfs")

con.execute("""
    CREATE SECRET hf_token (
        TYPE HUGGINGFACE,
        PROVIDER credential_chain
    )
""")


# ============================================================
# 1. Print real schemas
# ============================================================

for name, path in [
    ("QUOTE", QUOTE_PATH),
    ("ORDER", ORDER_PATH),
    ("TRADE", TRADE_PATH),
]:
    print("\n" + "=" * 80)
    print(f"{name} SCHEMA")
    print("=" * 80)

    schema = con.sql(
        f"""
        DESCRIBE
        SELECT *
        FROM '{path}'
        """
    ).df()

    print(schema.to_string(index=False))


# ============================================================
# 2. Automatically select 10 ordinary A-share stocks
#
# Shanghai main board:
# 600 / 601 / 603 / 605
#
# Shenzhen main board:
# 000 / 001 / 002 / 003
#
# Require valid bid/ask during 09:30-10:00.
# ============================================================

sh_stocks = con.sql(
    f"""
    SELECT wind_code
    FROM '{QUOTE_PATH}'
    WHERE
        time >= {START_TIME}
        AND time < {END_TIME}

        AND substr(wind_code, 1, 3)
            IN ('600', '601', '603', '605')

        AND bid_px1 > 0
        AND ask_px1 > 0

    GROUP BY wind_code

    HAVING COUNT(*) >= 100

    ORDER BY wind_code
    LIMIT 5
    """
).df()["wind_code"].tolist()


sz_stocks = con.sql(
    f"""
    SELECT wind_code
    FROM '{QUOTE_PATH}'
    WHERE
        time >= {START_TIME}
        AND time < {END_TIME}

        AND substr(wind_code, 1, 3)
            IN ('000', '001', '002', '003')

        AND bid_px1 > 0
        AND ask_px1 > 0

    GROUP BY wind_code

    HAVING COUNT(*) >= 100

    ORDER BY wind_code
    LIMIT 5
    """
).df()["wind_code"].tolist()


symbols = sh_stocks + sz_stocks

print("\n" + "=" * 80)
print("SELECTED STOCKS")
print("=" * 80)

for symbol in symbols:
    print(symbol)


symbols_sql = ", ".join(f"'{x}'" for x in symbols)


# ============================================================
# 3. Print quote samples
# ============================================================

print("\n" + "=" * 80)
print("QUOTE SAMPLE")
print("=" * 80)

quotes = con.sql(
    f"""
    SELECT *
    FROM '{QUOTE_PATH}'

    WHERE
        wind_code IN ({symbols_sql})
        AND time >= {START_TIME}
        AND time < {END_TIME}

    ORDER BY wind_code, time

    LIMIT {N_SAMPLE_ROWS}
    """
).df()

print(quotes.to_string(index=False))


# ============================================================
# 4. Print order samples
# ============================================================

print("\n" + "=" * 80)
print("ORDER SAMPLE")
print("=" * 80)

orders = con.sql(
    f"""
    SELECT *
    FROM '{ORDER_PATH}'

    WHERE
        wind_code IN ({symbols_sql})
        AND time >= {START_TIME}
        AND time < {END_TIME}

    ORDER BY wind_code, time

    LIMIT {N_SAMPLE_ROWS}
    """
).df()

print(orders.to_string(index=False))


# ============================================================
# 5. Print trade samples
# ============================================================

print("\n" + "=" * 80)
print("TRADE SAMPLE")
print("=" * 80)

trades = con.sql(
    f"""
    SELECT *
    FROM '{TRADE_PATH}'

    WHERE
        wind_code IN ({symbols_sql})
        AND time >= {START_TIME}
        AND time < {END_TIME}

    ORDER BY wind_code, time

    LIMIT {N_SAMPLE_ROWS}
    """
).df()

print(trades.to_string(index=False))


# ============================================================
# 6. Row counts for each selected stock
# ============================================================

for name, path in [
    ("QUOTE", QUOTE_PATH),
    ("ORDER", ORDER_PATH),
    ("TRADE", TRADE_PATH),
]:
    print("\n" + "=" * 80)
    print(f"{name} ROW COUNTS: 09:30-10:00")
    print("=" * 80)

    counts = con.sql(
        f"""
        SELECT
            wind_code,
            COUNT(*) AS n_rows,
            MIN(time) AS first_time,
            MAX(time) AS last_time

        FROM '{path}'

        WHERE
            wind_code IN ({symbols_sql})
            AND time >= {START_TIME}
            AND time < {END_TIME}

        GROUP BY wind_code
        ORDER BY wind_code
        """
    ).df()

    print(counts.to_string(index=False))


# ============================================================
# 7. Inspect categorical values by market
# ============================================================

OUT_FILE = "data/inspect_sample_20260828.txt"


def append_table(title, query):
    df = con.sql(query).df()

    text = (
        "\n\n"
        + "=" * 80
        + f"\n{title}\n"
        + "=" * 80
        + "\n"
        + df.to_string(index=False)
        + "\n"
    )

    print(text)

    with open(OUT_FILE, "a", encoding="utf-8") as f:
        f.write(text)


# ------------------------------------------------------------
# Quote categorical fields
# ------------------------------------------------------------

append_table(
    "QUOTE CATEGORICAL VALUES BY MARKET",
    f"""
    SELECT
        CASE
            WHEN substr(wind_code, 1, 1) = '6' THEN 'SH'
            ELSE 'SZ'
        END AS market,

        COALESCE(trade_flag, '<NULL>') AS trade_flag,
        COALESCE(bs_flag, '<NULL>') AS bs_flag,

        COUNT(*) AS n

    FROM '{QUOTE_PATH}'

    WHERE
        wind_code IN ({symbols_sql})
        AND time >= {START_TIME}
        AND time < {END_TIME}

    GROUP BY
        market,
        trade_flag,
        bs_flag

    ORDER BY
        market,
        n DESC
    """
)


# ------------------------------------------------------------
# Order categorical fields
# ------------------------------------------------------------

append_table(
    "ORDER CATEGORICAL VALUES BY MARKET",
    f"""
    SELECT
        CASE
            WHEN substr(wind_code, 1, 1) = '6' THEN 'SH'
            ELSE 'SZ'
        END AS market,

        COALESCE(order_type, '<NULL>') AS order_type,
        COALESCE(order_code, '<NULL>') AS order_code,

        COUNT(*) AS n

    FROM '{ORDER_PATH}'

    WHERE
        wind_code IN ({symbols_sql})
        AND time >= {START_TIME}
        AND time < {END_TIME}

    GROUP BY
        market,
        order_type,
        order_code

    ORDER BY
        market,
        n DESC
    """
)


# ------------------------------------------------------------
# Trade categorical fields
# ------------------------------------------------------------

append_table(
    "TRADE CATEGORICAL VALUES BY MARKET",
    f"""
    SELECT
        CASE
            WHEN substr(wind_code, 1, 1) = '6' THEN 'SH'
            ELSE 'SZ'
        END AS market,

        COALESCE(trade_code, '<NULL>') AS trade_code,
        COALESCE(order_code, '<NULL>') AS order_code,
        COALESCE(bs_flag, '<NULL>') AS bs_flag,

        COUNT(*) AS n

    FROM '{TRADE_PATH}'

    WHERE
        wind_code IN ({symbols_sql})
        AND time >= {START_TIME}
        AND time < {END_TIME}

    GROUP BY
        market,
        trade_code,
        order_code,
        bs_flag

    ORDER BY
        market,
        n DESC
    """
)

con.close()

print(f"\nSaved full output to: {OUTPUT_PATH}")

sys.stdout = sys.stdout.streams[0]
log_file.close()