#!/usr/bin/env python3
"""Model explainability using SHAP and LIME.

Usage:
    python explain_predictions.py --model-path model.pkl --data data.csv --framework sklearn --method shap
    python explain_predictions.py --model-path model.pkl --data data.csv --method lime --index 42
    python explain_predictions.py --model-path model.pkl --data data.csv --method both \\
        --demographic-col gender --output-dir ./explanations
    python explain_predictions.py --model-path xgb_model.json --data data.csv --framework xgboost
    python explain_predictions.py --model-path model.pt --data data.csv --framework pytorch
"""
import argparse
import json
import logging
import os
import time

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_model(model_path, framework):
    """Load a trained model from disk."""
    if framework == "sklearn":
        import joblib
        return joblib.load(model_path)
    if framework == "xgboost":
        import xgboost as xgb
        model = xgb.XGBClassifier(); model.load_model(model_path)
        return model
    if framework == "lgbm":
        import lightgbm as lgb
        return lgb.Booster(model_file=model_path)
    if framework == "pytorch":
        import torch
        model = torch.load(model_path, map_location="cpu"); model.eval()
        return model
    raise ValueError(f"Unsupported framework: {framework}")


def get_predict_fn(model, framework):
    """Return a prediction function suitable for SHAP/LIME."""
    if framework in ("sklearn", "xgboost"):
        return model.predict_proba if hasattr(model, "predict_proba") else model.predict
    if framework == "lgbm":
        return model.predict
    if framework == "pytorch":
        import torch
        def _predict(x):
            with torch.no_grad():
                return model(torch.tensor(x, dtype=torch.float32)).numpy()
        return _predict
    raise ValueError(f"Unsupported framework: {framework}")


def _pick_shap_explainer(model, framework, background):
    """Select the best SHAP explainer for the given model type."""
    import shap
    if framework in ("xgboost", "lgbm"):
        logger.info("Using SHAP TreeExplainer for tree-based model")
        return shap.TreeExplainer(model)
    if framework == "pytorch":
        import torch
        logger.info("Using SHAP DeepExplainer for PyTorch model")
        return shap.DeepExplainer(model, torch.tensor(background, dtype=torch.float32))
    if framework == "sklearn":
        tree_names = ("RandomForest", "GradientBoosting", "DecisionTree", "ExtraTrees")
        if any(t in type(model).__name__ for t in tree_names):
            logger.info("Using SHAP TreeExplainer for sklearn tree model")
            return shap.TreeExplainer(model)
    logger.info("Using SHAP KernelExplainer (model-agnostic)")
    return shap.KernelExplainer(get_predict_fn(model, framework), background)


def compute_shap(model, framework, X, feature_names, output_dir, index=None):
    """Compute SHAP values and save global/local results."""
    import shap
    background = X.values[:min(100, len(X))]
    explainer = _pick_shap_explainer(model, framework, background)

    logger.info("Computing SHAP values ...")
    t0 = time.time()
    shap_values = explainer.shap_values(X.values)
    logger.info(f"SHAP values computed in {time.time() - t0:.1f}s")

    sv = shap_values[1] if isinstance(shap_values, list) and len(shap_values) == 2 else shap_values
    global_imp = dict(zip(feature_names, np.abs(sv).mean(axis=0).tolist()))
    global_imp = dict(sorted(global_imp.items(), key=lambda kv: kv[1], reverse=True))
    result = {"method": "shap", "global_feature_importance": global_imp}

    # Global plots (summary + bar)
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        shap.summary_plot(sv, X, feature_names=feature_names, show=False)
        plt.savefig(os.path.join(output_dir, "shap_summary.png"), bbox_inches="tight", dpi=150)
        plt.close()
        shap.summary_plot(sv, X, feature_names=feature_names, plot_type="bar", show=False)
        plt.savefig(os.path.join(output_dir, "shap_bar.png"), bbox_inches="tight", dpi=150)
        plt.close()
        logger.info("SHAP plots saved")
    except Exception as exc:
        logger.warning(f"Could not generate SHAP plots: {exc}")

    # Local explanation for a single instance
    if index is not None and 0 <= index < len(X):
        local_exp = dict(zip(feature_names, sv[index].tolist()))
        local_exp = dict(sorted(local_exp.items(), key=lambda kv: abs(kv[1]), reverse=True))
        result["local_explanation"] = {"index": index, "shap_values": local_exp}
        logger.info(f"Local SHAP for index {index}: top feature = {list(local_exp.keys())[0]}")
    elif index is not None:
        logger.warning(f"Index {index} out of range [0, {len(X)})")
    return result


def compute_lime(model, framework, X, feature_names, output_dir, index=0):
    """Compute LIME explanation for an individual prediction."""
    from lime.lime_tabular import LimeTabularExplainer
    predict_fn = get_predict_fn(model, framework)
    explainer = LimeTabularExplainer(
        X.values, feature_names=feature_names, mode="classification", discretize_continuous=True,
    )
    if index < 0 or index >= len(X):
        logger.error(f"Index {index} out of range [0, {len(X)})")
        return {"method": "lime", "error": "index out of range"}

    logger.info(f"Computing LIME explanation for index {index} ...")
    exp = explainer.explain_instance(X.values[index], predict_fn, num_features=len(feature_names))
    result = {
        "method": "lime", "index": index,
        "intercept": float(exp.intercept[1] if len(exp.intercept) > 1 else exp.intercept[0]),
        "local_weights": {f: w for f, w in exp.as_list()},
        "prediction_local": float(exp.local_pred[0]) if hasattr(exp, "local_pred") else None,
        "r2_score": float(exp.score) if hasattr(exp, "score") else None,
    }
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = exp.as_pyplot_figure()
        fig.savefig(os.path.join(output_dir, f"lime_index_{index}.png"), bbox_inches="tight", dpi=150)
        plt.close(fig)
        logger.info(f"LIME plot saved for index {index}")
    except Exception as exc:
        logger.warning(f"Could not generate LIME plot: {exc}")
    return result


def fairness_analysis(model, framework, X, feature_names, demographic_col, output_dir):
    """Compute mean |SHAP| per demographic group to assess feature attribution fairness."""
    import shap
    if demographic_col not in X.columns:
        logger.warning(f"Demographic column '{demographic_col}' not found; skipping fairness")
        return None
    X_explain = X.drop(columns=[demographic_col])
    explain_features = [f for f in feature_names if f != demographic_col]
    background = X_explain.values[:min(100, len(X))]
    explainer = _pick_shap_explainer(model, framework, background)
    sv = explainer.shap_values(X_explain.values)
    if isinstance(sv, list) and len(sv) == 2:
        sv = sv[1]
    group_importance = {}
    for g in X[demographic_col].unique():
        mask = X[demographic_col] == g
        group_importance[str(g)] = dict(zip(explain_features, np.abs(sv[mask]).mean(axis=0).tolist()))
    result = {"demographic_column": demographic_col, "group_mean_abs_shap": group_importance}
    path = os.path.join(output_dir, "fairness_shap.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Fairness analysis saved to {path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Model explainability with SHAP and LIME")
    parser.add_argument("--model-path", required=True, help="Path to saved model")
    parser.add_argument("--data", required=True, help="Path to CSV/Parquet data for explanation")
    parser.add_argument("--framework", default="sklearn", choices=["sklearn", "xgboost", "lgbm", "pytorch"])
    parser.add_argument("--method", default="shap", choices=["shap", "lime", "both"])
    parser.add_argument("--index", type=int, default=None, help="Row index for local explanation")
    parser.add_argument("--target-col", default=None, help="Target column to exclude from features")
    parser.add_argument("--demographic-col", default=None, help="Demographic column for fairness analysis")
    parser.add_argument("--output-dir", default="./explanations", help="Directory for outputs")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_parquet(args.data) if args.data.endswith(".parquet") else pd.read_csv(args.data)
    logger.info(f"Loaded data: {df.shape[0]} rows x {df.shape[1]} columns")
    drop_cols = [c for c in ([args.target_col] if args.target_col else []) if c in df.columns]
    X = df.drop(columns=drop_cols)
    feature_names = list(X.columns)
    model = load_model(args.model_path, args.framework)
    logger.info(f"Model loaded ({args.framework}): {type(model).__name__}")

    results = {}
    demo = args.demographic_col
    if args.method in ("shap", "both"):
        feats = X.drop(columns=[demo], errors="ignore") if demo else X
        fnames = [f for f in feature_names if f != demo]
        results["shap"] = compute_shap(model, args.framework, feats, fnames, args.output_dir, args.index)
    if args.method in ("lime", "both"):
        idx = args.index if args.index is not None else 0
        feats = X.drop(columns=[demo], errors="ignore") if demo else X
        fnames = [f for f in feature_names if f != demo]
        results["lime"] = compute_lime(model, args.framework, feats, fnames, args.output_dir, idx)
    if demo:
        fairness = fairness_analysis(model, args.framework, X, feature_names, demo, args.output_dir)
        if fairness:
            results["fairness"] = fairness

    out_path = os.path.join(args.output_dir, "explanation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"All results saved to {out_path}")


if __name__ == "__main__":
    main()
