"""
Grouped TreeSHAP importance of LightGBM (L2 and L1), fitted on train and
explained on the validation split, for all features and for the final set.
A group's importance is the mean |row-wise sum of its features' contributions|.

Output: output/feature_importance/summary.json
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from core.config import CFG
from core.data import load_features, split_series, xy
from core.evaluation import score, write_summary

OUT_DIR = CFG.output_dir / "feature_importance"

# Feature set -> (model columns, {group: member columns}); groups partition the columns.
FEATURE_SETS = {
    "all_features": (
        CFG.features_tree,
        {
            "hour": ["hour"],
            "day_of_week": ["dow_local", "weekend"],
            "immediate_lags": CFG.immediate,
            "hour_scale_lags": CFG.hour_scale,
            "seasonal_lags": CFG.seasonal,
        },
    ),
    "final": (CFG.final_features, {"hour": ["hour"], "immediate_lags": CFG.immediate}),
}


def shap_groups(model, X, columns, groups) -> dict:
    """Mean |TreeSHAP| per group and its share of the total."""
    # pred_contrib=True returns per-row TreeSHAP values, one column per feature; the last column is the expected value.
    contrib = model.predict(X, pred_contrib=True)
    assert np.allclose(contrib.sum(axis=1), model.predict(X, raw_score=True)), "SHAP not additive"
    mean_abs = {
        name: np.abs(contrib[:, [columns.index(c) for c in members]].sum(axis=1)).mean()
        for name, members in groups.items()
    }
    total = sum(mean_abs.values())
    return {name: {"mean_abs_shap": v, "share": v / total} for name, v in mean_abs.items()}


def run(splits: dict[str, pd.DataFrame]) -> dict:
    """Fit each (loss, feature set) on train; score it and compute group importances on validation."""
    summary = {"split": "validation", "models": {}}
    for loss in CFG.losses:
        results = {}
        for set_name, (columns, groups) in FEATURE_SETS.items():
            X_train, y_train = xy(splits["train"], columns)
            X_valid, y_valid = xy(splits["valid"], columns)
            model = lgb.LGBMRegressor(objective=loss, **CFG.lgbm_params).fit(X_train, y_train)
            results[set_name] = {
                "valid": score(y_valid, model.predict(X_valid)),
                "groups": shap_groups(model, X_valid, columns, groups),
            }
        summary["models"][f"lightgbm_{loss}"] = results
    return summary


def print_tables(summary: dict) -> None:
    """One table per model and feature set, sorted by mean |SHAP|."""
    for model, results in summary["models"].items():
        for set_name, result in results.items():
            print(f"\n{model} | {set_name}")
            print(f"{'group':18s} {'mean|SHAP|':>11s} {'share':>7s}")
            for name, g in sorted(result["groups"].items(), key=lambda kv: -kv[1]["mean_abs_shap"]):
                print(f"{name:18s} {g['mean_abs_shap']:11.4f} {g['share']:7.3f}")


def main() -> None:
    summary = run(split_series(load_features()))
    write_summary(OUT_DIR / "summary.json", summary)
    print_tables(summary)


if __name__ == "__main__":
    main()
