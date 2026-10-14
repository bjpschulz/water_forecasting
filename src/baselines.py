"""
The five naive baselines on the validation split (always-mean uses the training
mean). Also records the split boundaries and the validation zero share.

Output: output/baselines/summary.json and validation_predictions.parquet
"""

import numpy as np
import pandas as pd

from core.config import CFG
from core.data import describe_splits, load_features, split_series
from core.evaluation import score, write_summary

OUT_DIR = CFG.output_dir / "baselines"
DEFINITIONS = {
    "always_zero": "Predict 0 L/min for every row",
    "always_mean": "Predict the mean avg_rate of the fitting data",
    "persistence": "Predict avg_rate from 1 UTC-minute earlier",
    "seasonal_naive_daily": "Predict avg_rate from 1,440 UTC-minute rows earlier",
    "seasonal_naive_weekly": "Predict avg_rate from 10,080 UTC-minute rows earlier",
}


def run(splits: dict[str, pd.DataFrame]) -> tuple[dict, pd.DataFrame]:
    """Score every baseline on the validation split; return the summary and the predictions."""
    valid = splits["valid"]
    y_valid = valid[CFG.target].to_numpy(dtype=np.float64)
    train_mean = float(splits["train"][CFG.target].mean())
    preds = {
        "always_zero": np.zeros(len(valid)),
        "always_mean": np.full(len(valid), train_mean),
        # lag_k[t] == avg_rate[t-k] is guaranteed by build_features, so the lag baselines are read from it.
        "persistence": valid["lag_1"].to_numpy(dtype=np.float64),
        "seasonal_naive_daily": valid["lag_1440"].to_numpy(dtype=np.float64),
        "seasonal_naive_weekly": valid["lag_10080"].to_numpy(dtype=np.float64),
    }

    summary = {
        "split": "validation",
        "splits": describe_splits(splits),
        "train_target_mean": train_mean,
        "valid_target_zero_share": float(np.mean(y_valid == 0)),
        "definitions": DEFINITIONS,
        "baselines": {name: score(y_valid, p) for name, p in preds.items()},
    }
    pred_frame = pd.DataFrame({"unix_ts": valid.index.to_numpy(dtype=np.int64), "actual": y_valid, **preds})
    return summary, pred_frame


def main() -> None:
    summary, predictions = run(split_series(load_features()))
    write_summary(OUT_DIR / "summary.json", summary)
    predictions.to_parquet(OUT_DIR / "validation_predictions.parquet", index=False)
    for name, m in summary["baselines"].items():
        print(f"{name:24s} MAE={m['mae']:.4f} RMSE={m['rmse']:.4f}")


if __name__ == "__main__":
    main()
