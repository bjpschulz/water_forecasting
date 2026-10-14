"""
Two-step feature ablation with LightGBM (L2 and L1), scored on the validation split.

Step 1 fits every calendar subset alone and on top of all lags; step 2 compares
lag subsets with the calendar fixed to `hour`. Only the feature set varies.

Output: output/feature_ablation/summary.json
"""

import lightgbm as lgb
import pandas as pd

from core.config import CFG
from core.data import load_features, split_series, xy
from core.evaluation import score, write_summary

OUT_DIR = CFG.output_dir / "feature_ablation"

CALENDAR_SETS = {
    "all_calendar": ["hour", "dow_local", "weekend"],
    "hour": ["hour"],
    "dow": ["dow_local"],
    "weekend": ["weekend"],
    "hour_dow": ["hour", "dow_local"],
    "hour_weekend": ["hour", "weekend"],
    "dow_weekend": ["dow_local", "weekend"],
}
LAG_SETS = {
    "all": CFG.lag_cols,
    "no_seasonal": [*CFG.immediate, *CFG.hour_scale],
    "immediate": CFG.immediate,
    "sparse_hour": ["lag_1", "lag_15", *CFG.hour_scale],
    "seasonal": CFG.seasonal,
    "persistence": ["lag_1"],
}
ABLATIONS = {
    "calendar_only": CALENDAR_SETS,
    "calendar_plus_all_lags": {f"{n} + all_lags": [*c, *CFG.lag_cols] for n, c in CALENDAR_SETS.items()},
    "lags_with_hour": {
        **{n: ["hour", *c] for n, c in LAG_SETS.items()},
        "all_lags_without_hour": CFG.lag_cols,  # reference
    },
}


def run(splits: dict[str, pd.DataFrame]) -> dict:
    """Fit every feature set of every ablation under both losses on train; score on validation."""
    summary = {"split": "validation"}
    for ablation, sets in ABLATIONS.items():
        summary[ablation] = {}
        for loss in CFG.losses:
            results = {}
            for name, columns in sets.items():
                X_train, y_train = xy(splits["train"], columns)
                X_valid, y_valid = xy(splits["valid"], columns)
                lgbm = lgb.LGBMRegressor(objective=loss, **CFG.lgbm_params).fit(X_train, y_train)
                preds = lgbm.predict(X_valid)
                results[name] = {"n_features": len(columns), **score(y_valid, preds)}
            summary[ablation][f"lightgbm_{loss}"] = results
    return summary


def print_tables(summary: dict) -> None:
    """One table per ablation and loss."""
    for ablation in ABLATIONS:
        for model, results in summary[ablation].items():
            print(f"\n{ablation} | {model}")
            print(f"{'feature set':28s} {'n_feat':>6s} {'MAE':>8s} {'RMSE':>8s} {'zero_pred':>10s}")
            for name, r in results.items():
                print(
                    f"{name:28s} {r['n_features']:6d} {r['mae']:8.4f} {r['rmse']:8.4f} "
                    f"{r['zero_prediction_share']:10.3f}"
                )


def main() -> None:
    summary = run(split_series(load_features()))
    write_summary(OUT_DIR / "summary.json", summary)
    print_tables(summary)


if __name__ == "__main__":
    main()
