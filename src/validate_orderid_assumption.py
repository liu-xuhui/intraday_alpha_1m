import pandas as pd

path = "data/raw/20260820/orders.parquet"

df = pd.read_parquet(path)

cols = [
    "time",
    "wind_code",
    "order_id",
    "ex_order_id",
    "order_type",
    "order_code",
    "price",
    "volume",
]


sample = (
    df.loc[
        (df["time"] >= 93500000) &
        (df["time"] < 93501000)
    ]
    .sort_values("time", kind="stable")
    .head(100)[cols]
)

print(sample.to_string(index=False))