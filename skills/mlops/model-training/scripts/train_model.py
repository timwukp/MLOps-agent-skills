#!/usr/bin/env python3
"""
Multi-Framework Model Training with Hyperparameter Optimization

Trains classification or regression models using scikit-learn, XGBoost, or LightGBM
with Optuna-based hyperparameter optimization and cross-validation.

Usage:
    # Train a random forest classifier with HPO
    python train_model.py --input data.csv --target label --model-type rf \
        --task classification --n-trials 50 --cv-folds 5 --output ./output

    # Train an XGBoost regressor with 100 Optuna trials
    python train_model.py --input housing.csv --target price --model-type xgb \
        --task regression --n-trials 100 --cv-folds 10 --output ./models

    # Train a LightGBM classifier with stratified CV
    python train_model.py --input churn.csv --target churned --model-type lgbm \
        --task classification --cv-strategy stratified --output ./results

    # Train an SVM classifier (no HPO, just default params)
    python train_model.py --input iris.csv --target species --model-type svm \
        --task classification --n-trials 1 --output ./results

Dependencies:
    - Python 3.8+, pandas, scikit-learn, joblib, optuna
    - Optional: xgboost, lightgbm
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy imports -- heavy packages loaded only when needed
# ---------------------------------------------------------------------------

def _import_pandas():
    import pandas as pd
    return pd


def _import_sklearn():
    from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.metrics import (
        accuracy_score, f1_score, mean_squared_error, mean_absolute_error, r2_score,
    )
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    return {
        "cross_val_score": cross_val_score,
        "KFold": KFold,
        "StratifiedKFold": StratifiedKFold,
        "RandomForestClassifier": RandomForestClassifier,
        "RandomForestRegressor": RandomForestRegressor,
        "GradientBoostingClassifier": GradientBoostingClassifier,
        "GradientBoostingRegressor": GradientBoostingRegressor,
        "SVC": SVC,
        "SVR": SVR,
        "accuracy_score": accuracy_score,
        "f1_score": f1_score,
        "mean_squared_error": mean_squared_error,
        "mean_absolute_error": mean_absolute_error,
        "r2_score": r2_score,
        "LabelEncoder": LabelEncoder,
        "StandardScaler": StandardScaler,
    }


def _import_optuna():
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna


def _import_xgboost():
    import xgboost as xgb
    return xgb


def _import_lightgbm():
    import lightgbm as lgb
    return lgb


def _import_joblib():
    import joblib
    return joblib


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(path: str, target: str) -> Tuple[Any, Any]:
    """Load a CSV/Parquet dataset and split into features and target."""
    pd = _import_pandas()
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        df = pd.read_parquet(path)
    elif ext in (".csv", ".tsv", ".txt"):
        sep = "\t" if ext == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
    else:
        df = pd.read_csv(path)

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Available: {list(df.columns)}")

    X = df.drop(columns=[target])
    y = df[target]

    # Encode string targets for classification
    if y.dtype == "object" or y.dtype.name == "category":
        sk = _import_sklearn()
        le = sk["LabelEncoder"]()
        y = pd.Series(le.fit_transform(y), name=target)
        logger.info("Encoded %d target classes: %s", len(le.classes_), list(le.classes_))

    # Drop non-numeric columns from features
    numeric_cols = X.select_dtypes(include=["number"]).columns
    dropped = set(X.columns) - set(numeric_cols)
    if dropped:
        logger.warning("Dropping non-numeric columns: %s", dropped)
        X = X[numeric_cols]

    logger.info("Dataset loaded: %d rows, %d features, target='%s'", len(X), X.shape[1], target)
    return X.values, y.values


# ---------------------------------------------------------------------------
# Search spaces per model type
# ---------------------------------------------------------------------------

def _rf_search_space(trial, task: str) -> Dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }


def _xgb_search_space(trial, task: str) -> Dict:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "verbosity": 0,
        "early_stopping_rounds": 20,
    }
    if task == "classification":
        params["eval_metric"] = "logloss"
    else:
        params["eval_metric"] = "rmse"
    return params


def _lgbm_search_space(trial, task: str) -> Dict:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "verbose": -1,
    }
    return params


def _svm_search_space(trial, task: str) -> Dict:
    kernel = trial.suggest_categorical("kernel", ["rbf", "linear", "poly"])
    params = {
        "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
        "kernel": kernel,
    }
    if kernel in ("rbf", "poly"):
        params["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])
    if kernel == "poly":
        params["degree"] = trial.suggest_int("degree", 2, 5)
    return params


SEARCH_SPACES: Dict[str, Callable] = {
    "rf": _rf_search_space,
    "xgb": _xgb_search_space,
    "lgbm": _lgbm_search_space,
    "svm": _svm_search_space,
}


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_model(model_type: str, task: str, params: Dict) -> Any:
    """Instantiate a model from type, task, and hyperparameters."""
    # Strip keys that are not constructor args (e.g. early_stopping_rounds for XGB)
    p = dict(params)
    early_stop = p.pop("early_stopping_rounds", None)
    eval_metric = p.pop("eval_metric", None)

    if model_type == "rf":
        sk = _import_sklearn()
        cls = sk["RandomForestClassifier"] if task == "classification" else sk["RandomForestRegressor"]
        return cls(**p, n_jobs=-1, random_state=42)

    if model_type == "xgb":
        xgb = _import_xgboost()
        cls = xgb.XGBClassifier if task == "classification" else xgb.XGBRegressor
        if early_stop is not None:
            p["early_stopping_rounds"] = early_stop
        if eval_metric is not None:
            p["eval_metric"] = eval_metric
        return cls(**p, n_jobs=-1, random_state=42)

    if model_type == "lgbm":
        lgb = _import_lightgbm()
        cls = lgb.LGBMClassifier if task == "classification" else lgb.LGBMRegressor
        return cls(**p, n_jobs=-1, random_state=42)

    if model_type == "svm":
        sk = _import_sklearn()
        cls = sk["SVC"] if task == "classification" else sk["SVR"]
        return cls(**p)

    raise ValueError(f"Unknown model type: {model_type}")


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def get_scoring(task: str) -> str:
    """Return the primary sklearn scoring string for cross_val_score."""
    if task == "classification":
        return "f1_weighted"
    return "neg_root_mean_squared_error"


def compute_metrics(y_true, y_pred, task: str) -> Dict[str, float]:
    """Compute all relevant metrics after final training."""
    sk = _import_sklearn()
    metrics: Dict[str, float] = {}
    if task == "classification":
        metrics["accuracy"] = float(sk["accuracy_score"](y_true, y_pred))
        metrics["f1_weighted"] = float(sk["f1_score"](y_true, y_pred, average="weighted"))
    else:
        import numpy as np
        metrics["rmse"] = float(np.sqrt(sk["mean_squared_error"](y_true, y_pred)))
        metrics["mae"] = float(sk["mean_absolute_error"](y_true, y_pred))
        metrics["r2"] = float(sk["r2_score"](y_true, y_pred))
    return metrics


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def make_objective(
    X, y, model_type: str, task: str, cv_folds: int, cv_strategy: str,
) -> Callable:
    """Return an Optuna objective function."""
    sk = _import_sklearn()
    scoring = get_scoring(task)

    if cv_strategy == "stratified" and task == "classification":
        cv = sk["StratifiedKFold"](n_splits=cv_folds, shuffle=True, random_state=42)
    else:
        cv = sk["KFold"](n_splits=cv_folds, shuffle=True, random_state=42)

    space_fn = SEARCH_SPACES[model_type]

    def objective(trial):
        params = space_fn(trial, task)
        model = build_model(model_type, task, params)
        try:
            scores = sk["cross_val_score"](model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        except Exception as exc:
            logger.warning("Trial %d failed: %s", trial.number, exc)
            raise optuna.TrialPruned()
        return scores.mean()

    # Need optuna in scope for TrialPruned; import once here
    optuna = _import_optuna()
    return objective


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def run_training(args: argparse.Namespace) -> Dict:
    """Execute the full training pipeline: HPO, retrain best, evaluate, save."""
    start_time = time.time()

    # Load data
    X, y = load_dataset(args.input, args.target)

    # HPO with Optuna
    optuna = _import_optuna()
    # sklearn scorers: f1_weighted is positive (higher=better), neg_rmse is negative
    # (higher=better), so "maximize" is correct for both task types.
    direction = "maximize"
    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    objective = make_objective(X, y, args.model_type, args.task, args.cv_folds, args.cv_strategy)

    logger.info(
        "Starting Optuna HPO: model=%s, task=%s, trials=%d, cv=%d folds (%s)",
        args.model_type, args.task, args.n_trials, args.cv_folds, args.cv_strategy,
    )
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)
    logger.info("Best trial #%d  score=%.6f", study.best_trial.number, study.best_value)
    logger.info("Best params: %s", study.best_params)

    # Retrain on full data with best params
    best_params = SEARCH_SPACES[args.model_type](study.best_trial, args.task)
    best_model = build_model(args.model_type, args.task, best_params)
    best_model.fit(X, y)
    y_pred = best_model.predict(X)

    metrics = compute_metrics(y, y_pred, args.task)
    logger.info("Final (full-data) metrics: %s", metrics)

    # Save model
    os.makedirs(args.output, exist_ok=True)
    joblib = _import_joblib()
    model_path = os.path.join(args.output, "model.joblib")
    joblib.dump(best_model, model_path)
    logger.info("Model saved to %s", model_path)

    elapsed = time.time() - start_time

    # Build training report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_type": args.model_type,
        "task": args.task,
        "input_file": args.input,
        "target_column": args.target,
        "dataset_rows": int(X.shape[0]),
        "dataset_features": int(X.shape[1]),
        "cv_folds": args.cv_folds,
        "cv_strategy": args.cv_strategy,
        "n_trials": args.n_trials,
        "best_trial": study.best_trial.number,
        "best_cv_score": float(study.best_value),
        "best_params": {k: v for k, v in study.best_params.items()},
        "final_metrics": metrics,
        "model_path": model_path,
        "training_time_seconds": round(elapsed, 2),
    }

    report_path = os.path.join(args.output, "training_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Training report saved to %s", report_path)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-framework model training with Optuna HPO.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Path to CSV or Parquet dataset")
    parser.add_argument("--target", required=True, help="Name of the target column")
    parser.add_argument(
        "--model-type", required=True, choices=["rf", "xgb", "lgbm", "svm"],
        help="Model family: rf (RandomForest), xgb (XGBoost), lgbm (LightGBM), svm (SVM)",
    )
    parser.add_argument(
        "--task", required=True, choices=["classification", "regression"],
        help="Learning task type",
    )
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna HPO trials (default: 50)")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of cross-validation folds (default: 5)")
    parser.add_argument(
        "--cv-strategy", choices=["kfold", "stratified"], default="stratified",
        help="CV strategy: kfold or stratified (default: stratified)",
    )
    parser.add_argument("--output", default="./output", help="Output directory for model and report")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        logger.error("Input file not found: %s", args.input)
        sys.exit(1)

    try:
        report = run_training(args)
        logger.info("Training complete. Best CV score: %.6f", report["best_cv_score"])
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(130)
    except Exception:
        logger.exception("Training failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
