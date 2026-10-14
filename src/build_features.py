"""
Feature table: the cleaned series plus all candidate lags and calendar features,
all backward shifts or timestamp functions, so no leakage. Leading rows with
undefined lags are kept here and excluded from train by core.data.split_series.

Output: output/build_features/whw_v100_with_features.parquet (+ .csv twin)
"""

import numpy as np
import pandas as pd

from core.config import CFG

CALENDAR_COLS = ["hour", "hour_sin", "hour_cos", "dow_local", "dow_sin", "dow_cos", "weekend"]


def load_v100() -> pd.DataFrame:
    """Load the cleaned V100-period series and check it is a gap-free 1-minute grid."""
    df = pd.read_parquet(CFG.v100_parquet)
    checks = {
        "index_name_is_unix_ts": df.index.name == "unix_ts",
        "index_unique": bool(df.index.is_unique),
        "index_monotone_increasing": bool(df.index.is_monotonic_increasing),
        "regular_1min_grid": bool((df.index.to_series().diff().dropna() == 60).all()),
        "no_missing_values": bool(df[["avg_rate", "counter", "datetime_local"]].notna().all().all()),
    }
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"Input artifact failed integrity checks: {failed}")
    return df


def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-shifted target values: lag_k at row t is avg_rate at t - k minutes."""
    out = df.copy()
    for k in CFG.lags:
        out[f"lag_{k}"] = out[CFG.target].shift(k)  # shift only looks at earlier rows
    return out


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calendar features from datetime_local (America/Vancouver); the UTC hour would
    rotate the diurnal profile by 7-8 hours. Raw integers suit tree models,
    cyclic sin/cos pairs suit linear models, so both are stored.
    """
    out = df.copy()
    local = out["datetime_local"]
    hour = local.dt.hour
    dow = local.dt.dayofweek  # Monday=0 .. Sunday=6

    out["hour"] = hour
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow_local"] = dow
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out["weekend"] = dow >= 5
    return out


def check_features(src: pd.DataFrame, df: pd.DataFrame) -> dict[str, bool]:
    """Integrity checks of the built table against its source series (target unchanged, exact lags)."""
    checks = {
        "target_identical_to_input": bool(df[CFG.target].equals(src[CFG.target])),
        "row_count_identical_to_input": len(df) == len(src),
    }
    expected = src[CFG.target].to_numpy()
    for k in CFG.lags:
        values = df[f"lag_{k}"].to_numpy()
        values_match = np.array_equal(values[k:], expected[:-k], equal_nan=True)
        nan_count_ok = int(df[f"lag_{k}"].isna().sum()) == k
        checks[f"lag_{k}_correct"] = bool(values_match and nan_count_ok)
    return checks


def main() -> None:
    src = load_v100()
    df = add_calendar(add_lags(src))

    failed = [k for k, ok in check_features(src, df).items() if not ok]
    if failed:
        raise ValueError(f"Feature-table self-checks failed: {failed}")

    df = df[["datetime_local", "counter", "avg_rate", *CFG.lag_cols, *CALENDAR_COLS]]
    CFG.features_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CFG.features_parquet)
    df.to_csv(CFG.features_parquet.with_suffix(".csv"))
    print(f"Saved {CFG.features_parquet} ({len(df)} rows, {df.shape[1]} columns)")


if __name__ == "__main__":
    main()
