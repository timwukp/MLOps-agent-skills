---
name: model-registry
description: >
  Manage ML model lifecycle with model registries. Covers MLflow Model Registry, model versioning (semantic, hash),
  model artifact storage, model metadata and tagging, model lifecycle stages (staging, production, archived),
  model promotion workflows, model lineage tracking (data to predictions), model packaging formats (ONNX, TorchScript,
  SavedModel, joblib), model signatures and schemas, model governance and approval, model comparison, rollback
  strategies, CI/CD integration for deployment, and model cards. Use when registering, versioning, promoting,
  or managing model artifacts.
license: Apache-2.0
metadata:
  author: mlops-skills
  version: "1.0"
  category: mlops
---

# Model Registry

## Overview

A model registry is the central hub for managing ML model artifacts, versions, metadata,
and lifecycle transitions from development to production.

## When to Use This Skill

- Registering new model versions after training
- Promoting models through staging to production
- Comparing model versions before deployment
- Setting up model governance workflows
- Packaging models for deployment

## Step-by-Step Instructions

### 1. MLflow Model Registry

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register a model from a run
model_uri = f"runs:/{run_id}/model"
result = mlflow.register_model(model_uri, "recommendation-model")
print(f"Version: {result.version}")

# Add metadata
client.update_model_version(
    name="recommendation-model",
    version=result.version,
    description="XGBoost model trained on user behavior data v2.1"
)
client.set_model_version_tag(
    name="recommendation-model", version=result.version,
    key="dataset_version", value="v2.1"
)
client.set_model_version_tag(
    name="recommendation-model", version=result.version,
    key="trained_by", value="ml-team"
)

# Transition stages
client.transition_model_version_stage(
    name="recommendation-model",
    version=result.version,
    stage="Staging"
)

# After validation passes
client.transition_model_version_stage(
    name="recommendation-model",
    version=result.version,
    stage="Production"
)

# Archive old production version
client.transition_model_version_stage(
    name="recommendation-model",
    version=old_version,
    stage="Archived"
)
```

### 2. Model Versioning Strategy

```python
import hashlib
import json
from datetime import datetime

def generate_model_version(model_path, config, metrics):
    """Generate a unique model version identifier."""
    # Content-based hash
    model_hash = hashlib.md5(open(model_path, "rb").read()).hexdigest()[:8]

    # Semantic version
    version_info = {
        "major": 2,     # Breaking changes (new architecture)
        "minor": 1,     # New features (added features)
        "patch": 3,     # Bug fixes (retrained with same config)
        "hash": model_hash,
        "timestamp": datetime.utcnow().isoformat(),
    }

    return f"v{version_info['major']}.{version_info['minor']}.{version_info['patch']}-{model_hash}"
```

### 3. Model Packaging

```python
# MLflow model with signature
from mlflow.models.signature import infer_signature

signature = infer_signature(X_train, model.predict(X_train))
mlflow.sklearn.log_model(
    model, "model",
    signature=signature,
    input_example=X_train[:5],
    registered_model_name="recommendation-model",
)

# ONNX export (cross-platform)
import onnx
import torch

dummy_input = torch.randn(1, *input_shape)
torch.onnx.export(
    pytorch_model, dummy_input, "model.onnx",
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)

# TorchScript (PyTorch deployment)
scripted = torch.jit.script(model)
scripted.save("model.pt")

# TensorFlow SavedModel
tf_model.save("saved_model/")

# scikit-learn / XGBoost
import joblib
joblib.dump(model, "model.joblib")
```

### 4. Model Promotion Workflow

```python
def promote_model(model_name, version, target_stage):
    """Promote model with validation gates."""
    client = MlflowClient()

    # Gate 1: Performance check
    model_version = client.get_model_version(model_name, version)
    run = client.get_run(model_version.run_id)
    metrics = run.data.metrics

    if metrics.get("test_f1", 0) < 0.85:
        raise ValueError(f"F1 score {metrics['test_f1']} below threshold 0.85")

    # Gate 2: Compare with current production
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    if prod_versions:
        prod_run = client.get_run(prod_versions[0].run_id)
        prod_f1 = prod_run.data.metrics.get("test_f1", 0)
        if metrics["test_f1"] <= prod_f1:
            raise ValueError(f"New model F1 {metrics['test_f1']} not better than production {prod_f1}")

    # Gate 3: Data validation tag
    tags = {t.key: t.value for t in model_version.tags}
    if tags.get("data_validated") != "true":
        raise ValueError("Model data not validated")

    # Promote
    client.transition_model_version_stage(model_name, version, target_stage)
    print(f"Model {model_name} v{version} promoted to {target_stage}")
```

### 5. Model Lineage

```python
def log_model_lineage(run_id, data_source, feature_pipeline, model_config):
    """Track complete lineage from data to model."""
    with mlflow.start_run(run_id=run_id):
        # Data lineage
        mlflow.set_tag("data.source", data_source)
        mlflow.set_tag("data.version", compute_data_hash(data_source))
        mlflow.set_tag("data.row_count", str(get_row_count(data_source)))

        # Feature lineage
        mlflow.set_tag("features.pipeline_version", feature_pipeline.version)
        mlflow.set_tag("features.feature_count", str(len(feature_pipeline.features)))

        # Code lineage
        mlflow.set_tag("code.git_commit", get_git_commit())
        mlflow.set_tag("code.git_branch", get_git_branch())

        # Environment
        mlflow.log_artifact("requirements.txt")
```

### 6. Model Card Generation

```python
def generate_model_card(model_name, version):
    """Generate a model card for documentation and governance."""
    client = MlflowClient()
    mv = client.get_model_version(model_name, version)
    run = client.get_run(mv.run_id)

    card = f"""# Model Card: {model_name} v{version}

## Model Details
- **Name**: {model_name}
- **Version**: {version}
- **Created**: {mv.creation_timestamp}
- **Framework**: {run.data.params.get('model_type', 'unknown')}
- **Owner**: {run.data.tags.get('trained_by', 'unknown')}

## Training Data
- **Source**: {run.data.tags.get('data.source', 'N/A')}
- **Version**: {run.data.tags.get('data.version', 'N/A')}
- **Rows**: {run.data.tags.get('data.row_count', 'N/A')}

## Performance Metrics
"""
    for metric, value in run.data.metrics.items():
        card += f"- **{metric}**: {value:.4f}\n"

    card += """
## Intended Use
[Describe intended use cases]

## Limitations
[Describe known limitations]

## Ethical Considerations
[Describe ethical considerations]
"""
    return card
```

### 7. Rollback Strategy

```python
def rollback_model(model_name, target_version=None):
    """Rollback to a previous model version."""
    client = MlflowClient()

    if target_version is None:
        # Find last archived production version
        all_versions = client.search_model_versions(f"name='{model_name}'")
        archived = [v for v in all_versions if v.current_stage == "Archived"]
        if not archived:
            raise ValueError("No archived versions available for rollback")
        target_version = max(archived, key=lambda v: int(v.version)).version

    # Demote current production
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    for pv in prod_versions:
        client.transition_model_version_stage(model_name, pv.version, "Archived")

    # Promote rollback target
    client.transition_model_version_stage(model_name, target_version, "Production")
    print(f"Rolled back {model_name} to v{target_version}")
```

## Best Practices

1. **Always log model signatures** - Input/output schemas prevent serving errors
2. **Use content-based versioning** - Hash the model for deduplication
3. **Implement promotion gates** - Never promote without automated checks
4. **Generate model cards** - Document every production model
5. **Track full lineage** - Data, features, code, environment
6. **Test before promoting** - Run integration tests at each stage
7. **Keep rollback ready** - Always have a previous version available
8. **Clean up old versions** - Archive/delete unused model versions
9. **Use consistent naming** - `{team}-{task}-{algorithm}` convention

## Scripts

- `scripts/registry_manager.py` - Model registry operations CLI
- `scripts/model_packager.py` - Model format conversion and packaging

## References

See [references/REFERENCE.md](references/REFERENCE.md) for tool and format comparisons.
