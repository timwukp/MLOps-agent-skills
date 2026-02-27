---
name: ml-experiment-tracking
description: >
  Track and manage ML experiments with MLflow, Weights & Biases, Neptune, and CometML. Covers experiment logging
  (hyperparameters, metrics, artifacts), auto-logging for PyTorch, TensorFlow, scikit-learn, and XGBoost, experiment
  comparison and visualization, artifact versioning, model lineage, reproducibility (environment, code, data versioning),
  remote tracking server setup, team collaboration, experiment search and querying, and integration with training
  pipelines. Use when running experiments, comparing model versions, or setting up experiment infrastructure.
license: Apache-2.0
metadata:
  author: mlops-skills
  version: "1.0"
  category: mlops
---

# ML Experiment Tracking

## Overview

Experiment tracking is the systematic recording of ML experiments - hyperparameters,
metrics, artifacts, code, and environment - to enable reproducibility, comparison,
and collaboration.

## When to Use This Skill

- Starting a new ML project that needs experiment management
- Comparing multiple model architectures or hyperparameter configs
- Setting up team-wide experiment tracking infrastructure
- Ensuring experiment reproducibility
- Migrating between tracking tools

## Step-by-Step Instructions

### 1. MLflow Tracking

#### Basic Setup

```python
import mlflow

# Set tracking URI (local or remote)
mlflow.set_tracking_uri("http://localhost:5000")  # Remote server
# mlflow.set_tracking_uri("sqlite:///mlflow.db")  # Local SQLite

# Create/set experiment
mlflow.set_experiment("my-recommendation-model")

# Log a run
with mlflow.start_run(run_name="baseline-v1") as run:
    # Log parameters
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("model_type", "random_forest")
    mlflow.log_params({"n_estimators": 100, "max_depth": 10})

    # Log metrics
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_metric("f1_score", 0.93)

    # Log metrics over steps
    for epoch in range(10):
        mlflow.log_metric("loss", loss_val, step=epoch)

    # Log artifacts
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact("feature_importance.csv")

    # Log model
    mlflow.sklearn.log_model(model, "model")

    # Set tags
    mlflow.set_tag("team", "recommendation")
    mlflow.set_tag("dataset_version", "v2.1")
```

#### Auto-logging

```python
# scikit-learn
mlflow.sklearn.autolog()
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)  # Automatically logged

# PyTorch
mlflow.pytorch.autolog()

# TensorFlow/Keras
mlflow.tensorflow.autolog()

# XGBoost
mlflow.xgboost.autolog()

# LightGBM
mlflow.lightgbm.autolog()

# HuggingFace Transformers
import os
os.environ["HF_MLFLOW_LOG_ARTIFACTS"] = "1"
mlflow.transformers.autolog()
```

### 2. Weights & Biases

```python
import wandb

# Initialize
wandb.init(
    project="recommendation-model",
    name="baseline-v1",
    config={
        "learning_rate": 0.001,
        "batch_size": 32,
        "model_type": "transformer",
    },
    tags=["baseline", "v1"],
)

# Log metrics
wandb.log({"loss": 0.5, "accuracy": 0.85, "epoch": 1})

# Log plots
wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
    y_true=y_true, preds=y_pred, class_names=class_names
)})

# Log tables
table = wandb.Table(columns=["input", "prediction", "ground_truth"])
for i in range(100):
    table.add_data(inputs[i], preds[i], labels[i])
wandb.log({"predictions": table})

# Log model
artifact = wandb.Artifact("model", type="model")
artifact.add_file("model.pkl")
wandb.log_artifact(artifact)

wandb.finish()
```

### 3. Experiment Comparison

```python
# MLflow: Query and compare experiments
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Search runs
runs = client.search_runs(
    experiment_ids=["1"],
    filter_string="metrics.accuracy > 0.9 AND params.model_type = 'random_forest'",
    order_by=["metrics.f1_score DESC"],
    max_results=10,
)

# Compare top runs
comparison = []
for run in runs:
    comparison.append({
        "run_id": run.info.run_id,
        "run_name": run.info.run_name,
        **run.data.params,
        **run.data.metrics,
    })
comparison_df = pd.DataFrame(comparison)
print(comparison_df.to_string())
```

### 4. Reproducibility Setup

```python
import mlflow
import hashlib
import subprocess

def log_reproducibility_info():
    """Log everything needed to reproduce an experiment."""

    # Code version
    git_hash = subprocess.check_output(
        ["git", "rev-parse", "HEAD"]
    ).decode().strip()
    mlflow.set_tag("git_commit", git_hash)

    # Environment
    mlflow.log_artifact("requirements.txt")
    mlflow.log_artifact("conda.yaml")

    # Data fingerprint
    data_hash = hashlib.md5(
        open("data/train.parquet", "rb").read()
    ).hexdigest()
    mlflow.set_tag("data_hash", data_hash)

    # Random seeds
    mlflow.log_param("random_seed", 42)

    # System info
    import platform
    mlflow.set_tag("python_version", platform.python_version())
    mlflow.set_tag("os", platform.system())
```

### 5. Remote Tracking Server

```bash
# Start MLflow tracking server
mlflow server \
    --backend-store-uri postgresql://user:pass@host/mlflow \
    --default-artifact-root s3://mlflow-artifacts/ \
    --host 0.0.0.0 \
    --port 5000

# Docker deployment
docker run -p 5000:5000 \
    -e BACKEND_STORE_URI=postgresql://user:pass@host/mlflow \
    -e DEFAULT_ARTIFACT_ROOT=s3://mlflow-artifacts/ \
    ghcr.io/mlflow/mlflow:latest \
    mlflow server --host 0.0.0.0
```

### 6. Nested Runs for Hyperparameter Tuning

```python
with mlflow.start_run(run_name="hpo-sweep") as parent_run:
    mlflow.log_param("search_method", "optuna")

    for trial in study.trials:
        with mlflow.start_run(
            run_name=f"trial-{trial.number}",
            nested=True
        ):
            mlflow.log_params(trial.params)
            mlflow.log_metric("objective", trial.value)

    # Log best result in parent
    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_objective", study.best_value)
```

## Best Practices

1. **Log everything** - Parameters, metrics, artifacts, environment, code version
2. **Use auto-logging** as a baseline, add custom logging for specifics
3. **Name runs descriptively** - Include key config in the name
4. **Use tags** for filtering (team, dataset version, experiment phase)
5. **Log data fingerprints** for reproducibility
6. **Use nested runs** for hyperparameter sweeps
7. **Set up a remote server** for team collaboration
8. **Version control experiment configs** alongside code
9. **Clean up** failed/abandoned runs periodically
10. **Standardize metric names** across the team

## Scripts

- `scripts/mlflow_tracker.py` - MLflow tracking wrapper with auto-logging
- `scripts/experiment_compare.py` - Experiment comparison and reporting

## References

See [references/REFERENCE.md](references/REFERENCE.md) for tool comparisons.
