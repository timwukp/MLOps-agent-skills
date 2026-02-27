#!/usr/bin/env python3
"""
Prefect ML Training Pipeline
==============================

A complete Prefect flow that orchestrates an end-to-end ML training pipeline.
Stages include data ingestion, data validation, feature engineering, train/test
splitting, model training, model evaluation, and conditional model registration.

Features:
- Parameterized flow for flexible execution.
- Task-level caching based on input hashing (skips redundant work).
- Per-task retry configuration with exponential backoff.
- Rich Prefect artifacts (tables, markdown) for observability.
- Structured logging via Prefect's run logger.
- Deployment configuration for scheduled and ad-hoc execution.
- Subflow composition for modular design.

Requirements:
    pip install prefect pandas scikit-learn

Usage (local):
    python prefect_pipeline.py

Usage (deployed):
    # Register the deployment
    prefect deploy --name ml-training-nightly

    # Or programmatically (see bottom of this file)
    python prefect_pipeline.py --deploy

License: Apache-2.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from prefect import flow, get_run_logger, task
from prefect.artifacts import create_markdown_artifact, create_table_artifact
from prefect.tasks import task_input_hash

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "artifact_root": "/tmp/ml_pipeline/artifacts",
    "random_seed": 42,
}


def _artifact_dir(run_name: str, stage: str) -> Path:
    """Return (and create) the artifact directory for a run and stage."""
    root = Path(DEFAULT_CONFIG["artifact_root"])
    path = root / run_name / stage
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


@task(
    name="ingest-data",
    description="Ingest raw data from source or generate synthetic demo data",
    retries=3,
    retry_delay_seconds=[10, 60, 300],
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=24),
    tags=["data", "ingestion"],
)
def ingest_data(
    data_source: str = "",
    n_samples: int = 5000,
    n_features: int = 20,
    random_seed: int = 42,
) -> dict:
    """
    Ingest raw data.

    If ``data_source`` points to an existing CSV file it is loaded.
    Otherwise, synthetic classification data is generated for demonstration.

    Returns:
        dict with ``data_path``, ``num_rows``, ``num_cols``.
    """
    logger = get_run_logger()
    run_name = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    artifact_dir = _artifact_dir(run_name, "ingestion")
    output_path = str(artifact_dir / "raw_data.csv")

    if data_source and os.path.exists(data_source):
        logger.info("Reading data from %s", data_source)
        df = pd.read_csv(data_source)
    else:
        logger.info(
            "Generating synthetic data: %d samples, %d features", n_samples, n_features
        )
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=max(2, n_features // 2),
            n_redundant=max(1, n_features // 5),
            random_state=random_seed,
        )
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        df["target"] = y

    df.to_csv(output_path, index=False)
    meta = {
        "data_path": output_path,
        "num_rows": len(df),
        "num_cols": len(df.columns),
        "run_name": run_name,
    }
    logger.info("Ingested %d rows, %d columns -> %s", meta["num_rows"], meta["num_cols"], output_path)
    return meta


@task(
    name="validate-data",
    description="Validate data quality: nulls, schema, row counts, target distribution",
    retries=1,
    retry_delay_seconds=10,
    tags=["data", "validation"],
)
def validate_data(ingest_meta: dict) -> dict:
    """
    Validate the ingested data.

    Checks:
    - Minimum row count (>=100).
    - No fully-null columns.
    - ``target`` column exists and is binary.
    - Overall null ratio < 5%.
    - Target class imbalance not extreme (minority class > 5%).

    Returns:
        dict with ``passed``, ``errors``, ``warnings``, ``data_path``, ``run_name``.
    """
    logger = get_run_logger()
    data_path = ingest_meta["data_path"]
    df = pd.read_csv(data_path)

    errors: list[str] = []
    warnings: list[str] = []

    # Row count
    if len(df) < 100:
        errors.append(f"Insufficient rows: {len(df)} (need >= 100)")

    # Fully-null columns
    fully_null = df.columns[df.isnull().all()].tolist()
    if fully_null:
        errors.append(f"Fully null columns: {fully_null}")

    # Target column
    if "target" not in df.columns:
        errors.append("Missing 'target' column")
    else:
        unique_vals = set(df["target"].dropna().unique())
        if unique_vals - {0, 1}:
            errors.append(f"Target must be binary (0/1), found: {unique_vals}")
        minority_ratio = df["target"].value_counts(normalize=True).min()
        if minority_ratio < 0.05:
            warnings.append(
                f"Severe class imbalance: minority class = {minority_ratio:.1%}"
            )

    # Null ratio
    null_ratio = df.isnull().sum().sum() / df.size
    if null_ratio > 0.05:
        errors.append(f"Null ratio {null_ratio:.2%} exceeds 5% threshold")
    elif null_ratio > 0.01:
        warnings.append(f"Null ratio {null_ratio:.2%} is notable (>1%)")

    passed = len(errors) == 0
    result = {
        "passed": passed,
        "errors": errors,
        "warnings": warnings,
        "data_path": data_path,
        "run_name": ingest_meta["run_name"],
    }

    if passed:
        logger.info("Data validation PASSED%s", f" (warnings: {warnings})" if warnings else "")
    else:
        logger.warning("Data validation FAILED: %s", errors)

    return result


@task(
    name="engineer-features",
    description="Generate derived features: interactions, polynomials, ratios, scaling",
    retries=2,
    retry_delay_seconds=30,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=12),
    tags=["features", "engineering"],
)
def engineer_features(validation_result: dict) -> dict:
    """
    Create derived features from raw data.

    Transformations:
    - Interaction features (pair-wise products).
    - Polynomial features (squares).
    - Ratio features.
    - Standard scaling with saved parameters.

    Returns:
        dict with ``features_path``, ``scaler_path``, ``num_features``, ``run_name``.
    """
    logger = get_run_logger()
    data_path = validation_result["data_path"]
    run_name = validation_result["run_name"]

    df = pd.read_csv(data_path)
    feature_cols = [c for c in df.columns if c != "target"]

    # Interaction features
    for i in range(min(3, len(feature_cols))):
        for j in range(i + 1, min(i + 2, len(feature_cols))):
            name = f"{feature_cols[i]}_x_{feature_cols[j]}"
            df[name] = df[feature_cols[i]] * df[feature_cols[j]]

    # Polynomial features
    for col in feature_cols[:5]:
        df[f"{col}_sq"] = df[col] ** 2

    # Ratio features
    for col in feature_cols[:3]:
        denom = df[feature_cols[-1]].replace(0, 1e-9)
        df[f"{col}_ratio"] = df[col] / denom

    # Standard scaling
    numeric_cols = [c for c in df.columns if c != "target"]
    means = df[numeric_cols].mean()
    stds = df[numeric_cols].std().replace(0, 1)
    df[numeric_cols] = (df[numeric_cols] - means) / stds

    artifact_dir = _artifact_dir(run_name, "features")
    features_path = str(artifact_dir / "features.csv")
    scaler_path = str(artifact_dir / "scaler_params.json")

    df.to_csv(features_path, index=False)
    with open(scaler_path, "w") as f:
        json.dump({"means": means.to_dict(), "stds": stds.to_dict()}, f)

    meta = {
        "features_path": features_path,
        "scaler_path": scaler_path,
        "num_features": len(df.columns) - 1,
        "run_name": run_name,
    }
    logger.info("Engineered %d features -> %s", meta["num_features"], features_path)
    return meta


@task(
    name="split-data",
    description="Stratified train/test split",
    retries=1,
    retry_delay_seconds=10,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=12),
    tags=["data", "splitting"],
)
def split_data(
    feature_meta: dict,
    test_ratio: float = 0.2,
    random_seed: int = 42,
) -> dict:
    """
    Split feature-engineered data into train and test sets.

    Returns:
        dict with ``train_path``, ``test_path``, ``train_rows``, ``test_rows``, ``run_name``.
    """
    logger = get_run_logger()
    features_path = feature_meta["features_path"]
    run_name = feature_meta["run_name"]

    df = pd.read_csv(features_path)
    train_df, test_df = train_test_split(
        df, test_size=test_ratio, random_state=random_seed, stratify=df["target"]
    )

    artifact_dir = _artifact_dir(run_name, "splits")
    train_path = str(artifact_dir / "train.csv")
    test_path = str(artifact_dir / "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    meta = {
        "train_path": train_path,
        "test_path": test_path,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "run_name": run_name,
    }
    logger.info("Split: train=%d rows, test=%d rows", meta["train_rows"], meta["test_rows"])
    return meta


@task(
    name="train-model",
    description="Train a RandomForestClassifier with configurable hyperparameters",
    retries=2,
    retry_delay_seconds=[30, 120],
    tags=["training", "model"],
)
def train_model(
    split_meta: dict,
    n_estimators: int = 100,
    max_depth: int = 10,
    min_samples_split: int = 5,
    random_seed: int = 42,
) -> dict:
    """
    Train a RandomForestClassifier.

    Returns:
        dict with ``model_path``, ``train_accuracy``, ``hyperparams``, ``run_name``.
    """
    logger = get_run_logger()
    train_path = split_meta["train_path"]
    run_name = split_meta["run_name"]

    train_df = pd.read_csv(train_path)
    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]

    hyperparams = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "random_state": random_seed,
    }

    logger.info("Training RandomForest with: %s", hyperparams)
    model = RandomForestClassifier(**hyperparams)
    model.fit(X_train, y_train)

    train_accuracy = accuracy_score(y_train, model.predict(X_train))

    artifact_dir = _artifact_dir(run_name, "model")
    model_path = str(artifact_dir / "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    meta = {
        "model_path": model_path,
        "train_accuracy": round(train_accuracy, 4),
        "hyperparams": hyperparams,
        "run_name": run_name,
    }
    logger.info("Training complete.  Train accuracy: %.4f", train_accuracy)
    return meta


@task(
    name="evaluate-model",
    description="Evaluate model on test set with comprehensive metrics",
    retries=1,
    retry_delay_seconds=10,
    tags=["evaluation", "metrics"],
)
def evaluate_model(model_meta: dict, split_meta: dict) -> dict:
    """
    Evaluate the trained model against the held-out test set.

    Computes accuracy, precision, recall, F1, AUC-ROC and generates
    a classification report.  Creates Prefect artifacts for UI visibility.

    Returns:
        dict with all metrics plus ``evaluation_path``, ``run_name``.
    """
    logger = get_run_logger()
    model_path = model_meta["model_path"]
    test_path = split_meta["test_path"]
    run_name = model_meta["run_name"]

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    test_df = pd.read_csv(test_path)
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "auc_roc": round(roc_auc_score(y_test, y_proba), 4),
    }

    # Persist metrics JSON
    artifact_dir = _artifact_dir(run_name, "evaluation")
    eval_path = str(artifact_dir / "metrics.json")
    with open(eval_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Persist classification report
    report_text = classification_report(y_test, y_pred)
    report_path = str(artifact_dir / "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)

    # Create Prefect table artifact
    create_table_artifact(
        key="model-evaluation-metrics",
        table=[{"Metric": k, "Value": v} for k, v in metrics.items()],
        description=f"Evaluation metrics for run {run_name}",
    )

    # Create Prefect markdown artifact
    create_markdown_artifact(
        key="classification-report",
        markdown=(
            f"## Classification Report\n\n"
            f"**Run:** {run_name}\n\n"
            f"```\n{report_text}\n```\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            + "\n".join(f"| {k} | {v} |" for k, v in metrics.items())
        ),
        description="Detailed classification report",
    )

    metrics["evaluation_path"] = eval_path
    metrics["classification_report"] = report_text
    metrics["run_name"] = run_name

    logger.info(
        "Evaluation: accuracy=%.4f, f1=%.4f, auc_roc=%.4f",
        metrics["accuracy"],
        metrics["f1"],
        metrics["auc_roc"],
    )
    return metrics


@task(
    name="register-model",
    description="Register the trained model for deployment",
    retries=2,
    retry_delay_seconds=30,
    tags=["registration", "model"],
)
def register_model(model_meta: dict, eval_metrics: dict) -> dict:
    """
    Register the model.

    In production, integrate with MLflow, Weights & Biases, SageMaker
    Model Registry, or Vertex AI.  Here we simulate registration by
    copying the model artifact and writing a manifest.

    Returns:
        dict with ``registered_model_uri``, ``version``, ``metrics``.
    """
    logger = get_run_logger()
    run_name = model_meta["run_name"]

    registry_dir = Path(DEFAULT_CONFIG["artifact_root"]) / "registry"
    registry_dir.mkdir(parents=True, exist_ok=True)

    version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    registered_path = str(registry_dir / f"model_v{version}.pkl")

    import shutil
    shutil.copy2(model_meta["model_path"], registered_path)

    registration = {
        "registered_model_uri": registered_path,
        "version": version,
        "run_name": run_name,
        "metrics": {
            "accuracy": eval_metrics["accuracy"],
            "f1": eval_metrics["f1"],
            "auc_roc": eval_metrics["auc_roc"],
        },
        "hyperparams": model_meta["hyperparams"],
        "registered_at": datetime.utcnow().isoformat(),
    }

    manifest_path = str(registry_dir / f"manifest_v{version}.json")
    with open(manifest_path, "w") as f:
        json.dump(registration, f, indent=2)

    # Create artifact for visibility
    create_markdown_artifact(
        key="model-registration",
        markdown=(
            f"## Model Registered\n\n"
            f"- **Version:** {version}\n"
            f"- **URI:** `{registered_path}`\n"
            f"- **Accuracy:** {registration['metrics']['accuracy']}\n"
            f"- **F1:** {registration['metrics']['f1']}\n"
            f"- **AUC-ROC:** {registration['metrics']['auc_roc']}\n"
            f"- **Registered At:** {registration['registered_at']}\n"
        ),
        description="Model registration details",
    )

    logger.info("Model registered: version=%s, uri=%s", version, registered_path)
    return registration


# ---------------------------------------------------------------------------
# Subflows
# ---------------------------------------------------------------------------


@flow(
    name="data-preparation",
    description="Data ingestion, validation, feature engineering, and splitting",
    retries=1,
    retry_delay_seconds=300,
    log_prints=True,
)
def data_preparation_flow(
    data_source: str = "",
    n_samples: int = 5000,
    n_features: int = 20,
    test_ratio: float = 0.2,
    random_seed: int = 42,
) -> dict:
    """
    Subflow: prepare data for training.

    Returns a dict containing split metadata and validation results.
    """
    logger = get_run_logger()
    logger.info("Starting data preparation subflow")

    ingest_meta = ingest_data(
        data_source=data_source,
        n_samples=n_samples,
        n_features=n_features,
        random_seed=random_seed,
    )

    val_result = validate_data(ingest_meta)
    if not val_result["passed"]:
        raise ValueError(f"Data validation failed: {val_result['errors']}")

    if val_result["warnings"]:
        logger.warning("Data warnings: %s", val_result["warnings"])

    feature_meta = engineer_features(val_result)
    split_meta = split_data(feature_meta, test_ratio=test_ratio, random_seed=random_seed)

    return {
        "split_meta": split_meta,
        "validation": val_result,
        "feature_meta": feature_meta,
    }


@flow(
    name="model-training-and-evaluation",
    description="Train model, evaluate, and optionally register",
    retries=1,
    retry_delay_seconds=300,
    log_prints=True,
)
def training_evaluation_flow(
    split_meta: dict,
    n_estimators: int = 100,
    max_depth: int = 10,
    min_samples_split: int = 5,
    quality_threshold: float = 0.85,
    random_seed: int = 42,
) -> dict:
    """
    Subflow: train, evaluate, and conditionally register a model.

    Returns a dict with training result, evaluation metrics, and optional
    registration metadata.
    """
    logger = get_run_logger()
    logger.info("Starting training and evaluation subflow")

    model_meta = train_model(
        split_meta,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_seed=random_seed,
    )

    eval_metrics = evaluate_model(model_meta, split_meta)

    result = {
        "model_meta": model_meta,
        "eval_metrics": eval_metrics,
        "registration": None,
        "registered": False,
    }

    if eval_metrics["accuracy"] >= quality_threshold:
        logger.info(
            "Quality gate PASSED (%.4f >= %.4f).  Registering model.",
            eval_metrics["accuracy"],
            quality_threshold,
        )
        registration = register_model(model_meta, eval_metrics)
        result["registration"] = registration
        result["registered"] = True
    else:
        logger.warning(
            "Quality gate FAILED (%.4f < %.4f).  Model NOT registered.",
            eval_metrics["accuracy"],
            quality_threshold,
        )

    return result


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------


@flow(
    name="ml-training-pipeline",
    description=(
        "End-to-end ML training pipeline: data ingestion, validation, "
        "feature engineering, splitting, training, evaluation, and "
        "conditional model registration."
    ),
    retries=0,
    log_prints=True,
    timeout_seconds=7200,
)
def ml_training_pipeline(
    # Data parameters
    data_source: str = "",
    n_samples: int = 5000,
    n_features: int = 20,
    test_ratio: float = 0.2,
    # Model parameters
    n_estimators: int = 100,
    max_depth: int = 10,
    min_samples_split: int = 5,
    # Quality gate
    quality_threshold: float = 0.85,
    # Reproducibility
    random_seed: int = 42,
) -> dict:
    """
    Orchestrate the full ML training pipeline.

    Parameters:
        data_source: Path to a CSV file.  If empty, synthetic data is generated.
        n_samples: Number of synthetic samples (if generating).
        n_features: Number of synthetic features (if generating).
        test_ratio: Fraction of data reserved for testing.
        n_estimators: Number of trees in the random forest.
        max_depth: Maximum tree depth.
        min_samples_split: Minimum samples to split an internal node.
        quality_threshold: Minimum accuracy to register the model.
        random_seed: Random seed for reproducibility.

    Returns:
        dict summarizing the pipeline run outcome.
    """
    logger = get_run_logger()
    logger.info("=" * 60)
    logger.info("ML TRAINING PIPELINE - Starting")
    logger.info("=" * 60)

    # Phase 1: Data preparation
    data_result = data_preparation_flow(
        data_source=data_source,
        n_samples=n_samples,
        n_features=n_features,
        test_ratio=test_ratio,
        random_seed=random_seed,
    )

    # Phase 2: Training and evaluation
    training_result = training_evaluation_flow(
        split_meta=data_result["split_meta"],
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        quality_threshold=quality_threshold,
        random_seed=random_seed,
    )

    # Summary
    summary = {
        "status": "success",
        "model_registered": training_result["registered"],
        "accuracy": training_result["eval_metrics"]["accuracy"],
        "f1": training_result["eval_metrics"]["f1"],
        "auc_roc": training_result["eval_metrics"]["auc_roc"],
        "quality_threshold": quality_threshold,
    }

    if training_result["registered"]:
        summary["model_version"] = training_result["registration"]["version"]
        summary["model_uri"] = training_result["registration"]["registered_model_uri"]

    logger.info("=" * 60)
    logger.info("ML TRAINING PIPELINE - Complete")
    logger.info("Summary: %s", json.dumps(summary, indent=2))
    logger.info("=" * 60)

    # Create summary artifact
    status_emoji = "PASS" if training_result["registered"] else "FAIL (below threshold)"
    create_markdown_artifact(
        key="pipeline-summary",
        markdown=(
            f"## Pipeline Run Summary\n\n"
            f"| Item | Value |\n|------|-------|\n"
            f"| Status | {summary['status']} |\n"
            f"| Quality Gate | {status_emoji} |\n"
            f"| Accuracy | {summary['accuracy']} |\n"
            f"| F1 Score | {summary['f1']} |\n"
            f"| AUC-ROC | {summary['auc_roc']} |\n"
            f"| Threshold | {summary['quality_threshold']} |\n"
            f"| Model Registered | {summary['model_registered']} |\n"
            + (
                f"| Model Version | {summary.get('model_version', 'N/A')} |\n"
                f"| Model URI | `{summary.get('model_uri', 'N/A')}` |\n"
                if training_result["registered"]
                else ""
            )
        ),
        description="End-to-end pipeline run summary",
    )

    return summary


# ---------------------------------------------------------------------------
# Deployment helpers
# ---------------------------------------------------------------------------


def create_deployment():
    """
    Create a Prefect deployment for the training pipeline.

    Run with:
        python prefect_pipeline.py --deploy
    """
    from prefect.deployments import Deployment
    from prefect.server.schemas.schedules import CronSchedule

    deployment = Deployment.build_from_flow(
        flow=ml_training_pipeline,
        name="ml-training-nightly",
        version="1.0",
        description="Nightly ML model training pipeline with quality gating",
        schedule=CronSchedule(cron="0 2 * * *", timezone="UTC"),
        parameters={
            "data_source": "",
            "n_samples": 10000,
            "n_features": 20,
            "test_ratio": 0.2,
            "n_estimators": 200,
            "max_depth": 12,
            "quality_threshold": 0.88,
            "random_seed": 42,
        },
        tags=["ml", "training", "production", "nightly"],
        work_pool_name="default-agent-pool",
    )
    deployment_id = deployment.apply()
    print(f"Deployment created: {deployment_id}")
    print(f"  Name   : {deployment.name}")
    print(f"  Schedule: {deployment.schedule}")
    return deployment_id


# ---------------------------------------------------------------------------
# Hyperparameter search flow (bonus: demonstrates fan-out / fan-in)
# ---------------------------------------------------------------------------


@flow(
    name="hyperparameter-search",
    description="Parallel hyperparameter grid search over training configurations",
    log_prints=True,
    timeout_seconds=14400,
)
def hyperparameter_search_flow(
    data_source: str = "",
    n_samples: int = 5000,
    n_features: int = 20,
    test_ratio: float = 0.2,
    random_seed: int = 42,
    quality_threshold: float = 0.85,
) -> dict:
    """
    Run a grid search over hyperparameter combinations.

    Uses Prefect's `.map()` for parallel fan-out across configurations,
    then selects the best model.
    """
    logger = get_run_logger()

    # Prepare data once
    data_result = data_preparation_flow(
        data_source=data_source,
        n_samples=n_samples,
        n_features=n_features,
        test_ratio=test_ratio,
        random_seed=random_seed,
    )
    split_meta = data_result["split_meta"]

    # Hyperparameter grid
    configs = [
        {"n_estimators": 50, "max_depth": 5, "min_samples_split": 10},
        {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5},
        {"n_estimators": 200, "max_depth": 15, "min_samples_split": 2},
        {"n_estimators": 300, "max_depth": 20, "min_samples_split": 2},
        {"n_estimators": 100, "max_depth": None, "min_samples_split": 5},
    ]

    # Fan-out: train candidates in parallel
    results = []
    for i, config in enumerate(configs):
        logger.info("Training candidate %d/%d: %s", i + 1, len(configs), config)
        model_meta = train_model(
            split_meta,
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"] if config["max_depth"] is not None else 100,
            min_samples_split=config["min_samples_split"],
            random_seed=random_seed,
        )
        eval_metrics = evaluate_model(model_meta, split_meta)
        results.append({"config": config, "model_meta": model_meta, "metrics": eval_metrics})

    # Fan-in: select best model
    best = max(results, key=lambda r: r["metrics"]["accuracy"])
    logger.info(
        "Best config: %s (accuracy=%.4f)",
        best["config"],
        best["metrics"]["accuracy"],
    )

    # Register best model if it meets the quality threshold
    if best["metrics"]["accuracy"] >= quality_threshold:
        registration = register_model(best["model_meta"], best["metrics"])
        logger.info("Best model registered: %s", registration["version"])
        return {"status": "registered", "best_config": best["config"], "registration": registration}

    logger.warning("No model met the quality threshold of %.4f", quality_threshold)
    return {"status": "not_registered", "best_config": best["config"], "best_accuracy": best["metrics"]["accuracy"]}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "--deploy" in sys.argv:
        create_deployment()
    elif "--search" in sys.argv:
        result = hyperparameter_search_flow()
        print(f"\nHyperparameter search result:\n{json.dumps(result, indent=2, default=str)}")
    else:
        result = ml_training_pipeline()
        print(f"\nPipeline result:\n{json.dumps(result, indent=2)}")
