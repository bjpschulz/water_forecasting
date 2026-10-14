"""
Diagnostic models: linear regression and LightGBM, each under an L2 and an L1
loss, on calendar + all lags, fitted on train and scored on validation.
Linear L1 is median regression; only the loss differs within a model class.

Output: output/compare_models/summary.json and validation_predictions.parquet
"""

import lightgbm as lgb
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

from core.config import FEATURES_LINEAR, FEATURES_TREE, LGBM_OBJECTIVES, LGBM_PARAMS, LOSSES, OUTPUT_DIR
from core.data import load_features, split_series, xy
from core.evaluation import score, write_summary

OUT_DIR = OUTPUT_DIR / "compare_models"


def run(splits: dict[str, pd.DataFrame]) -> tuple[dict, pd.DataFrame]:
    """Fit every (model, loss) pair on train and score it on validation."""
    train, valid = splits["train"], splits["valid"]
    preds = {}

    X_train, y_train = xy(train, FEATURES_LINEAR)
    X_valid, y_valid = xy(valid, FEATURES_LINEAR)
    ols = LinearRegression().fit(X_train, y_train)
    median = sm.QuantReg(y_train, sm.add_constant(X_train)).fit(q=0.5, max_iter=5000)
    preds["linear_l2"] = ols.predict(X_valid)
    preds["linear_l1"] = median.predict(sm.add_constant(X_valid, has_constant="add"))

    X_train, y_train = xy(train, FEATURES_TREE)
    X_valid, y_valid = xy(valid, FEATURES_TREE)
    for loss in LOSSES:
        lgbm = lgb.LGBMRegressor(objective=LGBM_OBJECTIVES[loss], **LGBM_PARAMS).fit(X_train, y_train)
        preds[f"lightgbm_{loss}"] = lgbm.predict(X_valid)

    summary = {
        "split": "validation",
        "models": {name: {"loss": name.rsplit("_", 1)[1], **score(y_valid, p)} for name, p in preds.items()},
    }
    pred_frame = pd.DataFrame({"unix_ts": valid.index.to_numpy(), "actual": y_valid, **preds})
    return summary, pred_frame


def main() -> None:
    summary, predictions = run(split_series(load_features()))
    write_summary(OUT_DIR / "summary.json", summary)
    predictions.to_parquet(OUT_DIR / "validation_predictions.parquet", index=False)
    for name, m in summary["models"].items():
        print(f"{name:12s} MAE={m['mae']:.4f} RMSE={m['rmse']:.4f} zero_pred={m['zero_prediction_share']:.3f}")


if __name__ == "__main__":
    main()
