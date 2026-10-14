"""
Single evaluation on the test split: the final models (LightGBM and linear
regression, L2 and L1, on hour + lags 1-15) are fitted on train + valid and
scored next to the five baselines; always-mean uses the train + valid mean.

Output: output/test_evaluation/summary.json and test_predictions.parquet
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

from core.config import CFG
from core.data import load_features, split_series, xy
from core.evaluation import score, write_summary

OUT_DIR = CFG.output_dir / "test_evaluation"


def run(splits: dict[str, pd.DataFrame]) -> tuple[dict, pd.DataFrame]:
    """Fit the final models on train + valid; score them and the baselines on test."""
    fit_frame = pd.concat([splits["train"], splits["valid"]])  # contiguous, chronological
    test = splits["test"]
    X_fit, y_fit = xy(fit_frame, CFG.final_features)
    X_test, y_test = xy(test, CFG.final_features)
    fit_mean = float(y_fit.mean())

    preds = {
        "always_zero": np.zeros(len(test)),
        "always_mean": np.full(len(test), fit_mean),
        "persistence": test["lag_1"].to_numpy(dtype=np.float64),
        "seasonal_naive_daily": test["lag_1440"].to_numpy(dtype=np.float64),
        "seasonal_naive_weekly": test["lag_10080"].to_numpy(dtype=np.float64),
    }
    kinds = dict.fromkeys(preds, "baseline")  # every later entry is a model
    ols = LinearRegression().fit(X_fit, y_fit)
    median = sm.QuantReg(y_fit, sm.add_constant(X_fit)).fit(q=0.5, max_iter=5000)
    preds["linear_l2"] = ols.predict(X_test)
    preds["linear_l1"] = median.predict(sm.add_constant(X_test, has_constant="add"))
    for loss in CFG.losses:
        lgbm = lgb.LGBMRegressor(objective=loss, **CFG.lgbm_params).fit(X_fit, y_fit)
        preds[f"lightgbm_{loss}"] = lgbm.predict(X_test)

    summary = {
        "split": "test",
        "features": CFG.final_features,
        "fit_rows": len(fit_frame),
        "n": len(test),
        "fit_target_mean": fit_mean,
        "test_target_zero_share": float(np.mean(y_test == 0)),
        "results": {name: {"kind": kinds.get(name, "model"), **score(y_test, p)} for name, p in preds.items()},
    }
    pred_frame = pd.DataFrame({"unix_ts": test.index.to_numpy(), "actual": y_test, **preds})
    return summary, pred_frame


def main() -> None:
    summary, predictions = run(split_series(load_features()))
    write_summary(OUT_DIR / "summary.json", summary)
    predictions.to_parquet(OUT_DIR / "test_predictions.parquet", index=False)
    print(f"{'predictor':24s} {'kind':9s} {'MAE':>8s} {'RMSE':>8s} {'zero_pred':>10s}")
    for name, r in summary["results"].items():
        print(f"{name:24s} {r['kind']:9s} {r['mae']:8.4f} {r['rmse']:8.4f} {r['zero_prediction_share']:10.3f}")


if __name__ == "__main__":
    main()
