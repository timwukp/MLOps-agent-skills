#!/usr/bin/env python3
"""
Airflow ML Training Pipeline
=============================

A complete Apache Airflow DAG that orchestrates an end-to-end ML training
pipeline. Stages include data ingestion, data validation, feature engineering,
train/test splitting, model training, model evaluation, and conditional model
registration.

Features:
- Configurable via Airflow Variables (hyperparams, thresholds, paths).
- XCom for lightweight metadata passing between tasks; large artifacts
  stored in an artifact store (local filesystem or S3).
- Per-task retry logic with exponential backoff.
- Failure and success callbacks for alerting.
- TaskGroup for logical organization.
- Idempotent tasks that can be safely retried.

Requirements:
    pip install apache-airflow pandas scikit-learn great-expectations

Usage:
    Place this file in your Airflow DAGs folder.  Configure the following
    Airflow Variables (Admin -> Variables) or let the defaults apply:

    - ml_data_source_path   (default: /tmp/ml_pipeline/raw_data.csv)
    - ml_artifact_root      (default: /tmp/ml_pipeline/artifacts)
    - ml_hyperparams        (JSON, default: {"n_estimators":100,"max_depth":10})
    - ml_quality_threshold  (float, default: 0.85)
    - ml_test_split_ratio   (float, default: 0.2)
    - ml_random_seed        (int, default: 42)

License: Apache-2.0
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

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

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

logger = logging.getLogger("ml_training_pipeline")


def _get_var(key: str, default: Any = None, deserialize_json: bool = False) -> Any:
    """Retrieve an Airflow Variable with a fallback default."""
    try:
        return Variable.get(key, default_var=default, deserialize_json=deserialize_json)
    except Exception:
        return default


def _get_artifact_dir(run_id: str, stage: str) -> Path:
    """Return (and create) the artifact directory for a given run and stage."""
    root = Path(_get_var("ml_artifact_root", "/tmp/ml_pipeline/artifacts"))
    path = root / run_id / stage
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Callback helpers
# ---------------------------------------------------------------------------


def _on_failure(context: dict) -> None:
    """Called when any task fails. Replace the print with your alerting logic."""
    ti = context["task_instance"]
    dag_id = context["dag"].dag_id
    exec_date = context["execution_date"]
    exception = context.get("exception", "N/A")
    msg = (
        f"[ALERT] Task FAILED -- DAG: {dag_id}, Task: {ti.task_id}, "
        f"Execution Date: {exec_date}, Exception: {exception}"
    )
    logger.error(msg)
    # TODO: Replace with Slack / PagerDuty / email integration
    # send_slack_alert(channel="#ml-alerts", message=msg)


def _on_success(context: dict) -> None:
    """Called when the entire DAG succeeds."""
    dag_id = context["dag"].dag_id
    exec_date = context["execution_date"]
    msg = f"[INFO] DAG succeeded -- DAG: {dag_id}, Execution Date: {exec_date}"
    logger.info(msg)


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------


def ingest_data(**context) -> dict:
    """
    Ingest raw data from the configured source.

    For demonstration purposes this function generates synthetic data when
    the source file does not exist.  In production, replace this with a
    read from S3, a database query, or an API call.

    Returns (via XCom):
        dict with keys ``data_path``, ``num_rows``, ``num_cols``.
    """
    run_id: str = context["run_id"]
    source_path = _get_var("ml_data_source_path", "/tmp/ml_pipeline/raw_data.csv")
    artifact_dir = _get_artifact_dir(run_id, "ingestion")
    output_path = str(artifact_dir / "raw_data.csv")

    if os.path.exists(source_path):
        logger.info("Reading data from %s", source_path)
        df = pd.read_csv(source_path)
    else:
        logger.info("Source not found; generating synthetic data for demo")
        X, y = make_classification(
            n_samples=5000,
            n_features=20,
            n_informative=12,
            n_redundant=4,
            random_state=int(_get_var("ml_random_seed", 42)),
        )
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        df["target"] = y

    df.to_csv(output_path, index=False)
    meta = {"data_path": output_path, "num_rows": len(df), "num_cols": len(df.columns)}
    logger.info("Ingested %d rows, %d columns -> %s", meta["num_rows"], meta["num_cols"], output_path)
    return meta


def validate_data(**context) -> dict:
    """
    Validate the ingested data.

    Checks:
    - Minimum row count (>100).
    - No fully-null columns.
    - Target column exists and is binary.
    - Null ratio below 5%.

    Returns (via XCom):
        dict with keys ``passed`` (bool), ``errors`` (list[str]), ``data_path``.
    """
    ti = context["ti"]
    ingest_meta: dict = ti.xcom_pull(task_ids="data_pipeline.ingest_data")
    data_path = ingest_meta["data_path"]

    df = pd.read_csv(data_path)
    errors: list[str] = []

    if len(df) < 100:
        errors.append(f"Insufficient rows: {len(df)} (need >= 100)")

    fully_null = df.columns[df.isnull().all()].tolist()
    if fully_null:
        errors.append(f"Fully null columns: {fully_null}")

    if "target" not in df.columns:
        errors.append("Missing 'target' column")
    elif set(df["target"].dropna().unique()) - {0, 1}:
        errors.append("Target column must be binary (0/1)")

    null_ratio = df.isnull().sum().sum() / df.size
    if null_ratio > 0.05:
        errors.append(f"Null ratio {null_ratio:.2%} exceeds 5% threshold")

    passed = len(errors) == 0
    result = {"passed": passed, "errors": errors, "data_path": data_path}
    if passed:
        logger.info("Data validation PASSED")
    else:
        logger.warning("Data validation FAILED: %s", errors)
    return result


def check_validation(**context) -> str:
    """
    Branch: proceed to feature engineering if validation passed,
    otherwise jump to the failure notification task.
    """
    ti = context["ti"]
    validation: dict = ti.xcom_pull(task_ids="data_pipeline.validate_data")
    if validation["passed"]:
        return "data_pipeline.engineer_features"
    return "notify_failure"


def engineer_features(**context) -> dict:
    """
    Create derived features from raw data.

    Transformations:
    - Interaction features (products of top informative pairs).
    - Polynomial features (squares of top features).
    - Ratio features.
    - Standard scaling.

    Returns (via XCom):
        dict with ``features_path``, ``num_features``.
    """
    ti = context["ti"]
    run_id: str = context["run_id"]
    validation: dict = ti.xcom_pull(task_ids="data_pipeline.validate_data")
    data_path = validation["data_path"]

    df = pd.read_csv(data_path)
    feature_cols = [c for c in df.columns if c != "target"]

    # Interaction features (first 3 pairs)
    for i in range(min(3, len(feature_cols))):
        for j in range(i + 1, min(i + 2, len(feature_cols))):
            col_name = f"{feature_cols[i]}_x_{feature_cols[j]}"
            df[col_name] = df[feature_cols[i]] * df[feature_cols[j]]

    # Polynomial features (squares of first 5)
    for col in feature_cols[:5]:
        df[f"{col}_sq"] = df[col] ** 2

    # Ratio features
    for col in feature_cols[:3]:
        denominator = df[feature_cols[-1]].replace(0, 1e-9)
        df[f"{col}_ratio"] = df[col] / denominator

    # Standard scaling (save scaler params for inference)
    numeric_cols = [c for c in df.columns if c != "target"]
    means = df[numeric_cols].mean()
    stds = df[numeric_cols].std().replace(0, 1)
    df[numeric_cols] = (df[numeric_cols] - means) / stds

    artifact_dir = _get_artifact_dir(run_id, "features")
    features_path = str(artifact_dir / "features.csv")
    scaler_path = str(artifact_dir / "scaler_params.json")

    df.to_csv(features_path, index=False)
    scaler_params = {"means": means.to_dict(), "stds": stds.to_dict()}
    with open(scaler_path, "w") as f:
        json.dump(scaler_params, f)

    meta = {
        "features_path": features_path,
        "scaler_path": scaler_path,
        "num_features": len(df.columns) - 1,
    }
    logger.info("Engineered %d features -> %s", meta["num_features"], features_path)
    return meta


def split_data(**context) -> dict:
    """
    Split the feature-engineered data into train and test sets.

    Returns (via XCom):
        dict with ``train_path``, ``test_path``, ``train_rows``, ``test_rows``.
    """
    ti = context["ti"]
    run_id: str = context["run_id"]
    feat_meta: dict = ti.xcom_pull(task_ids="data_pipeline.engineer_features")
    features_path = feat_meta["features_path"]

    df = pd.read_csv(features_path)
    test_ratio = float(_get_var("ml_test_split_ratio", 0.2))
    seed = int(_get_var("ml_random_seed", 42))

    train_df, test_df = train_test_split(
        df, test_size=test_ratio, random_state=seed, stratify=df["target"]
    )

    artifact_dir = _get_artifact_dir(run_id, "splits")
    train_path = str(artifact_dir / "train.csv")
    test_path = str(artifact_dir / "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    meta = {
        "train_path": train_path,
        "test_path": test_path,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
    }
    logger.info("Split data: train=%d, test=%d", meta["train_rows"], meta["test_rows"])
    return meta


def train_model(**context) -> dict:
    """
    Train a RandomForestClassifier on the training set.

    Hyperparameters are read from the Airflow Variable ``ml_hyperparams``
    (JSON), with sensible defaults.

    Returns (via XCom):
        dict with ``model_path``, ``train_accuracy``, ``hyperparams``.
    """
    ti = context["ti"]
    run_id: str = context["run_id"]
    split_meta: dict = ti.xcom_pull(task_ids="data_pipeline.split_data")
    train_path = split_meta["train_path"]

    hyperparams = _get_var(
        "ml_hyperparams",
        {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5, "random_state": 42},
        deserialize_json=True,
    )
    if hyperparams is None:
        hyperparams = {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5, "random_state": 42}

    train_df = pd.read_csv(train_path)
    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]

    logger.info("Training RandomForest with params: %s", hyperparams)
    model = RandomForestClassifier(**hyperparams)
    model.fit(X_train, y_train)

    train_accuracy = accuracy_score(y_train, model.predict(X_train))

    artifact_dir = _get_artifact_dir(run_id, "model")
    model_path = str(artifact_dir / "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    meta = {
        "model_path": model_path,
        "train_accuracy": round(train_accuracy, 4),
        "hyperparams": hyperparams,
    }
    logger.info("Model trained.  Train accuracy: %.4f -> %s", train_accuracy, model_path)
    return meta


def evaluate_model(**context) -> dict:
    """
    Evaluate the trained model on the held-out test set.

    Returns (via XCom):
        dict with accuracy, precision, recall, f1, auc_roc,
        classification_report (text), and evaluation_path.
    """
    ti = context["ti"]
    run_id: str = context["run_id"]
    model_meta: dict = ti.xcom_pull(task_ids="training.train_model")
    split_meta: dict = ti.xcom_pull(task_ids="data_pipeline.split_data")

    model_path = model_meta["model_path"]
    test_path = split_meta["test_path"]

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
        "classification_report": classification_report(y_test, y_pred),
    }

    artifact_dir = _get_artifact_dir(run_id, "evaluation")
    eval_path = str(artifact_dir / "metrics.json")
    with open(eval_path, "w") as f:
        json.dump({k: v for k, v in metrics.items() if k != "classification_report"}, f, indent=2)

    report_path = str(artifact_dir / "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(metrics["classification_report"])

    metrics["evaluation_path"] = eval_path
    logger.info(
        "Evaluation: accuracy=%.4f, f1=%.4f, auc_roc=%.4f",
        metrics["accuracy"],
        metrics["f1"],
        metrics["auc_roc"],
    )
    return metrics


def check_quality_gate(**context) -> str:
    """
    Branch: register the model if it meets the quality threshold,
    otherwise notify of failure.
    """
    ti = context["ti"]
    eval_metrics: dict = ti.xcom_pull(task_ids="training.evaluate_model")
    threshold = float(_get_var("ml_quality_threshold", 0.85))

    accuracy = eval_metrics["accuracy"]
    logger.info("Quality gate: accuracy=%.4f vs threshold=%.4f", accuracy, threshold)

    if accuracy >= threshold:
        return "registration.register_model"
    return "notify_failure"


def register_model(**context) -> dict:
    """
    Register the trained model for deployment.

    In production, this would push to MLflow, SageMaker Model Registry,
    Vertex AI Model Registry, or a similar system.  Here we simulate
    registration by copying the model and writing metadata.

    Returns (via XCom):
        dict with ``registered_model_uri``, ``version``, ``metrics``.
    """
    ti = context["ti"]
    run_id: str = context["run_id"]
    model_meta: dict = ti.xcom_pull(task_ids="training.train_model")
    eval_metrics: dict = ti.xcom_pull(task_ids="training.evaluate_model")

    model_path = model_meta["model_path"]

    # Simulate registration
    registry_dir = Path(_get_var("ml_artifact_root", "/tmp/ml_pipeline/artifacts")) / "registry"
    registry_dir.mkdir(parents=True, exist_ok=True)
    version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    registered_path = str(registry_dir / f"model_v{version}.pkl")

    import shutil
    shutil.copy2(model_path, registered_path)

    registration_meta = {
        "registered_model_uri": registered_path,
        "version": version,
        "run_id": run_id,
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
        json.dump(registration_meta, f, indent=2)

    logger.info("Model registered: version=%s, uri=%s", version, registered_path)
    return registration_meta


def notify_success(**context) -> None:
    """Send a success notification with model details."""
    ti = context["ti"]
    reg_meta: dict | None = ti.xcom_pull(task_ids="registration.register_model")
    eval_metrics: dict | None = ti.xcom_pull(task_ids="training.evaluate_model")

    if reg_meta:
        msg = (
            f"ML Pipeline SUCCESS\n"
            f"  Model Version : {reg_meta.get('version', 'N/A')}\n"
            f"  Accuracy      : {reg_meta['metrics']['accuracy']}\n"
            f"  F1 Score      : {reg_meta['metrics']['f1']}\n"
            f"  AUC-ROC       : {reg_meta['metrics']['auc_roc']}\n"
            f"  Registered At : {reg_meta.get('registered_at', 'N/A')}\n"
            f"  URI           : {reg_meta.get('registered_model_uri', 'N/A')}"
        )
    elif eval_metrics:
        msg = (
            f"ML Pipeline completed but model NOT registered "
            f"(accuracy={eval_metrics['accuracy']} below threshold)"
        )
    else:
        msg = "ML Pipeline completed (no evaluation data available)"

    logger.info(msg)
    # TODO: send_slack_message(channel="#ml-notifications", text=msg)
    # TODO: send_email(to="ml-team@company.com", subject="Pipeline Success", body=msg)


def notify_failure(**context) -> None:
    """Send a failure notification with error details."""
    ti = context["ti"]
    validation: dict | None = ti.xcom_pull(task_ids="data_pipeline.validate_data")
    eval_metrics: dict | None = ti.xcom_pull(task_ids="training.evaluate_model")

    reasons: list[str] = []
    if validation and not validation.get("passed", True):
        reasons.append(f"Data validation errors: {validation['errors']}")
    if eval_metrics:
        threshold = float(_get_var("ml_quality_threshold", 0.85))
        if eval_metrics["accuracy"] < threshold:
            reasons.append(
                f"Model accuracy {eval_metrics['accuracy']} below threshold {threshold}"
            )

    if not reasons:
        reasons.append("Unknown failure (check upstream task logs)")

    msg = f"ML Pipeline FAILURE\n  Reasons: {'; '.join(reasons)}"
    logger.warning(msg)
    # TODO: send_slack_alert(channel="#ml-alerts", text=msg)
    # TODO: send_pagerduty_event(severity="warning", summary=msg)


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "email": ["ml-alerts@company.com"],
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=30),
    "execution_timeout": timedelta(hours=2),
    "on_failure_callback": _on_failure,
}

with DAG(
    dag_id="ml_training_pipeline",
    default_args=default_args,
    description=(
        "End-to-end ML training pipeline: ingest, validate, feature engineer, "
        "train, evaluate, and register models with quality gating."
    ),
    schedule_interval="0 2 * * *",  # Daily at 2:00 AM UTC
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "training", "production"],
    on_success_callback=_on_success,
    doc_md=__doc__,
) as dag:

    # ---- Data Pipeline Task Group -------------------------------------------
    with TaskGroup("data_pipeline", tooltip="Data ingestion through splitting") as data_grp:

        t_ingest = PythonOperator(
            task_id="ingest_data",
            python_callable=ingest_data,
            doc_md="Ingest raw data from source or generate synthetic data.",
        )

        t_validate = PythonOperator(
            task_id="validate_data",
            python_callable=validate_data,
            doc_md="Validate data quality (null rates, schema, row counts).",
        )

        t_branch_validation = BranchPythonOperator(
            task_id="check_validation",
            python_callable=check_validation,
            doc_md="Branch based on data validation result.",
        )

        t_features = PythonOperator(
            task_id="engineer_features",
            python_callable=engineer_features,
            doc_md="Generate derived features: interactions, polynomials, ratios, scaling.",
        )

        t_split = PythonOperator(
            task_id="split_data",
            python_callable=split_data,
            doc_md="Stratified train/test split.",
        )

        t_ingest >> t_validate >> t_branch_validation >> t_features >> t_split

    # ---- Training Task Group ------------------------------------------------
    with TaskGroup("training", tooltip="Model training and evaluation") as training_grp:

        t_train = PythonOperator(
            task_id="train_model",
            python_callable=train_model,
            doc_md="Train RandomForestClassifier with configurable hyperparameters.",
        )

        t_evaluate = PythonOperator(
            task_id="evaluate_model",
            python_callable=evaluate_model,
            doc_md="Evaluate model on test set: accuracy, precision, recall, F1, AUC-ROC.",
        )

        t_train >> t_evaluate

    # ---- Quality Gate -------------------------------------------------------
    t_quality_gate = BranchPythonOperator(
        task_id="quality_gate",
        python_callable=check_quality_gate,
        doc_md="Branch: register if quality threshold met, else notify failure.",
    )

    # ---- Registration Task Group --------------------------------------------
    with TaskGroup("registration", tooltip="Model registration") as reg_grp:

        t_register = PythonOperator(
            task_id="register_model",
            python_callable=register_model,
            doc_md="Register the model in the model registry.",
        )

    # ---- Notifications ------------------------------------------------------
    t_notify_success = PythonOperator(
        task_id="notify_success",
        python_callable=notify_success,
        trigger_rule=TriggerRule.ALL_SUCCESS,
        doc_md="Send success notification with model details.",
    )

    t_notify_failure = PythonOperator(
        task_id="notify_failure",
        python_callable=notify_failure,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
        doc_md="Send failure notification with reasons.",
    )

    # ---- DAG wiring ---------------------------------------------------------
    # data_pipeline group -> training group
    data_grp >> training_grp

    # evaluation -> quality gate -> (register or notify_failure)
    training_grp >> t_quality_gate
    t_quality_gate >> t_register >> t_notify_success
    t_quality_gate >> t_notify_failure

    # Validation branch can also route to notify_failure
    # (already handled by BranchPythonOperator returning "notify_failure")
