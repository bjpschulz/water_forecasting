"""Paths, feature sets and model settings shared by all pipeline scripts, as one read-only `CFG` object."""

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType


@dataclass(frozen=True)
class Config:
    """Project settings. Later fields are derived from earlier ones in the class body."""

    repo_root: Path = Path(__file__).resolve().parents[2]
    output_dir: Path = repo_root / "output"
    raw_csv: Path = repo_root / "data" / "Water_WHW.csv"
    v100_parquet: Path = output_dir / "eda" / "whw_v100.parquet"
    features_parquet: Path = output_dir / "build_features" / "whw_v100_with_features.parquet"

    local_tz: str = "America/Vancouver"
    target: str = "avg_rate"

    # Candidate lags (1-minute units): dense 1-15 (ACF is highest there), then
    # hour-scale, daily, weekly. Shared by the ACF analyses, feature table and models.
    lags: tuple[int, ...] = (*range(1, 16), 30, 45, 60, 1440, 10080)
    lag_cols: tuple[str, ...] = tuple(f"lag_{k}" for k in lags)
    immediate: tuple[str, ...] = tuple(f"lag_{k}" for k in range(1, 16))
    hour_scale: tuple[str, ...] = ("lag_30", "lag_45", "lag_60")
    seasonal: tuple[str, ...] = ("lag_1440", "lag_10080")

    # Feature sets of the diagnostic models: cyclic calendar for the linear model, raw for LightGBM.
    features_linear: tuple[str, ...] = ("hour_sin", "hour_cos", "dow_sin", "dow_cos", "weekend", *lag_cols)
    features_tree: tuple[str, ...] = ("hour", "dow_local", "weekend", *lag_cols)
    # Final feature set, chosen by the feature ablation; used by every final model.
    final_features: tuple[str, ...] = ("hour", *immediate)

    # LightGBM defaults written out explicitly (untuned by design), plus a fixed seed.
    lgbm_params: MappingProxyType = MappingProxyType(
        {"n_estimators": 100, "num_leaves": 31, "learning_rate": 0.1, "random_state": 42, "verbosity": -1}
    )
    # Training losses compared for every model class; the names double as LightGBM objectives.
    losses: tuple[str, ...] = ("l2", "l1")


CFG = Config()
