from pathlib import Path

import numpy as np
import pandas as pd


factor_path = "data/factors/burst_volume_imbalance_8s.pkl"
target_path = "data/target_return_1min.pkl"
factor_name = Path(factor_path).stem

factor_df = pd.read_pickle(factor_path)
target_df = pd.read_pickle(target_path)

# Make sure factor and target are perfectly aligned
assert factor_df.shape == target_df.shape
assert factor_df.index.equals(target_df.index)
assert factor_df.columns.equals(target_df.columns)

x = factor_df.to_numpy(dtype=float)
y = target_df.to_numpy(dtype=float)

# Valid observations require both factor and target to be non-NaN
valid = np.isfinite(x) & np.isfinite(y)

# Number of valid stocks at each timestamp
n = valid.sum(axis=1)

# Replace invalid entries with 0 only for vectorized summation
x0 = np.where(valid, x, 0.0)
y0 = np.where(valid, y, 0.0)

# Cross-sectional means using only valid pairs
x_mean = np.divide(
    x0.sum(axis=1),
    n,
    out=np.full(n.shape, np.nan, dtype=float),
    where=n > 0,
)

y_mean = np.divide(
    y0.sum(axis=1),
    n,
    out=np.full(n.shape, np.nan, dtype=float),
    where=n > 0,
)

# Center values only where observations are valid
xc = np.where(valid, x - x_mean[:, None], 0.0)
yc = np.where(valid, y - y_mean[:, None], 0.0)

# Pearson correlation:
# sum((x-xbar)(y-ybar)) /
# sqrt(sum((x-xbar)^2) * sum((y-ybar)^2))
numerator = np.sum(xc * yc, axis=1)

x_ss = np.sum(xc ** 2, axis=1)
y_ss = np.sum(yc ** 2, axis=1)

denominator = np.sqrt(x_ss * y_ss)

ic = np.full(x.shape[0], np.nan)

good = (
    (n >= 2)
    & (denominator > 0)
)

ic[good] = numerator[good] / denominator[good]

# Keep timestamps
ic_series = pd.Series(
    ic,
    index=factor_df.index,
    name=f"{factor_name}_ic",
)

print(ic_series)

print("\nIC summary")
print(ic_series.describe())

print(f"\nValid IC rows: {ic_series.notna().sum()} / {len(ic_series)}")
print(f"Mean IC: {ic_series.mean():.6f}")