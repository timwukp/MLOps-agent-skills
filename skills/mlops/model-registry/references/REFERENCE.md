# Model Registry Reference Guide

## Model Registry Comparison

| Feature                | MLflow Registry       | DVC                   | W&B Registry          | Vertex AI Registry    |
|------------------------|-----------------------|-----------------------|-----------------------|-----------------------|
| Model Versioning       | Auto-incrementing     | Git-based             | Auto-incrementing     | Auto-incrementing     |
| Stage Management       | Staging/Prod/Archived | Git branches/tags     | Aliases (latest, best)| Deployment states     |
| Metadata Tracking      | Tags, descriptions    | YAML metadata         | Rich metadata, tables | Labels, annotations   |
| Experiment Tracking    | Built-in              | Separate (DVCLive)    | Built-in (best)       | Vertex Experiments    |
| Access Control         | Basic (OSS), RBAC (Databricks) | Git-based    | Team-based RBAC       | IAM-based             |
| Artifact Storage       | S3, GCS, Azure, local | S3, GCS, Azure, local | W&B servers / S3      | GCS                   |
| Open Source            | Yes                   | Yes                   | Partially             | No                    |
| Cost                   | Free (OSS)            | Free                  | Free tier + paid      | Pay-per-use           |

### When to Choose What

- **MLflow**: Default choice for most teams. Open source, framework-agnostic, broad ecosystem.
- **DVC**: Best for Git-native versioning of data and models with existing Git workflows.
- **W&B**: Best experiment tracking UI, strong for research teams valuing visualization.
- **Vertex AI**: Best for GCP-native teams wanting end-to-end managed ML pipelines.

## Model Lifecycle Stages and Promotion Workflows

```
Development --> Staging --> Production --> Archived
    |              |            |
  Training     Validation   Serving + Monitoring
```

| Stage        | Purpose                                    | Automated Gates                       |
|--------------|--------------------------------------------|---------------------------------------|
| Development  | Experimental models under active iteration | None                                  |
| Staging      | Candidate models undergoing validation     | Unit tests pass, metrics met          |
| Production   | Models actively serving predictions        | Integration tests, bias audit, load test |
| Archived     | Retired models kept for reproducibility    | N/A                                   |

```python
# MLflow promotion example
from mlflow import MlflowClient
client = MlflowClient()

mv = client.create_model_version(name="fraud-detector", source="runs:/abc123/model", run_id="abc123")
client.set_registered_model_alias("fraud-detector", "staging", mv.version)
# After validation:
client.set_registered_model_alias("fraud-detector", "production", mv.version)
```

## Model Versioning Strategies

### Semantic Versioning for Models

| Component | When to Increment                            | Example Change                          |
|-----------|----------------------------------------------|-----------------------------------------|
| MAJOR     | Breaking API change, new input/output schema | Changed from 3-class to 5-class output  |
| MINOR     | Improved performance, same API               | Retrained with more data, new feature   |
| PATCH     | Bug fix, config change, no perf difference   | Fixed preprocessing bug, updated deps   |

### Naming Convention

```
{team}/{task}/{version}
  fraud-detection/transaction-classifier/v2.1.0
```

### Version Metadata to Track

Always store with each model version: training data version/hash, code commit SHA, hyperparameters, evaluation metrics on held-out set, full environment (pip freeze), and hardware used.

## Model Cards

### Structure

```markdown
# Model Card: [Model Name]
## Model Details       - Developer, version, architecture, license
## Intended Use        - Primary use cases, out-of-scope uses
## Training Data       - Dataset description, preprocessing, limitations
## Evaluation Results  - Test metrics, subgroup performance, baseline comparison
## Ethical Considerations - Bias analysis, potential harms, mitigations
## Limitations         - Known failure modes, deployment recommendations
```

### Regulatory Requirements

- **EU AI Act**: High-risk systems require detailed documentation, bias testing, human oversight.
- **NYC Local Law 144**: Automated employment decision tools need annual bias audits.
- **NIST AI RMF**: Recommends model cards as part of AI risk management.
- **FDA (medical)**: Requires clinical validation, data provenance, continuous monitoring.

## Model Packaging Formats

| Format      | Cross-Framework | Optimization   | Best Deployment Target     |
|-------------|-----------------|----------------|----------------------------|
| ONNX        | Yes             | ONNX Runtime   | Any (CPU, GPU, Edge)       |
| TorchScript | No (PyTorch)    | torch.jit      | TorchServe, C++ runtime    |
| SavedModel  | No (TensorFlow) | TF-TRT, TFLite | TF Serving, mobile/edge    |
| PMML        | Yes             | None           | Java-based enterprise      |
| joblib      | No (sklearn)    | None           | Python services            |

### Conversion Examples

```python
# PyTorch to ONNX
torch.onnx.export(model, dummy_input, "model.onnx",
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})

# sklearn to ONNX
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
initial_type = [("input", FloatTensorType([None, n_features]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

# sklearn to joblib (avoid pickle for security)
import joblib
joblib.dump(model, "model.joblib")
```

### Format Selection Guide

- **Cross-framework portability?** ONNX.
- **PyTorch model in C++?** TorchScript.
- **Mobile/edge deployment?** TFLite or ONNX Runtime Mobile.
- **Simple sklearn in Python?** joblib.
- **Legacy Java system?** PMML.

## CI/CD Integration for Model Promotion

```yaml
# GitHub Actions model promotion pipeline
name: Model Promotion
on:
  workflow_dispatch:
    inputs:
      model_name: { required: true }
      model_version: { required: true }
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - name: Performance Gate
        run: python scripts/evaluate_model.py --model ${{ inputs.model_name }}
             --version ${{ inputs.model_version }} --min-accuracy 0.95
      - name: Bias Audit
        run: python scripts/bias_audit.py --model ${{ inputs.model_name }}
             --version ${{ inputs.model_version }} --max-disparity 0.1
      - name: Integration Test
        run: python scripts/integration_test.py --model ${{ inputs.model_name }}
             --version ${{ inputs.model_version }}
      - name: Promote to Production
        if: success()
        run: python scripts/promote_model.py --model ${{ inputs.model_name }}
             --version ${{ inputs.model_version }} --stage production
```

### Recommended Gates by Stage

| Gate                     | Dev -> Staging | Staging -> Prod |
|--------------------------|:--------------:|:---------------:|
| Unit tests pass          | Yes            | Yes             |
| Metrics above threshold  | Yes            | Yes             |
| No data leakage detected | Yes            | Yes             |
| Bias audit passes        | No             | Yes             |
| Integration tests pass   | No             | Yes             |
| Load test passes         | No             | Yes             |
| Manual approval          | No             | Yes             |

## Model Lineage and Provenance Tracking

```
Raw Data --> Processed Data --> Features --> Training Run --> Model --> Deployment
  Hash       Transform SHA     Store ref    Hyperparams     Metrics   Endpoint
  Source      Config            Version     Code SHA        Bias rpt  Traffic %
```

```python
import mlflow

with mlflow.start_run():
    mlflow.set_tag("data.version", "v2.3.0")
    mlflow.set_tag("data.hash", "sha256:abc123...")
    mlflow.set_tag("mlflow.source.git.commit", "def456...")
    mlflow.log_artifact("requirements.txt")

    from mlflow.models import infer_signature
    signature = infer_signature(X_test, predictions)
    mlflow.sklearn.log_model(model, "model", signature=signature)
```

## Common Pitfalls

1. **Not pinning dependency versions**: Always freeze the full environment with each model version.
2. **Storing models without signatures**: Always log input/output schemas to catch contract violations.
3. **Manual promotion workflows**: Automate gates to avoid human error and ensure consistency.
4. **Ignoring model size**: Large models increase deployment cost and latency. Track artifact size.
5. **No rollback plan**: Always keep the previous production model version readily accessible.

## Further Reading

- [MLflow Model Registry Docs](https://mlflow.org/docs/latest/model-registry.html)
- [DVC Model Registry](https://dvc.org/doc/use-cases/model-registry)
- [W&B Model Registry](https://docs.wandb.ai/guides/model-registry)
- [ONNX Specification](https://onnx.ai/onnx/intro/)
- [Model Cards for Model Reporting (Mitchell et al.)](https://arxiv.org/abs/1810.03993)
- [Google Model Cards Toolkit](https://modelcards.withgoogle.com/about)
- [EU AI Act Overview](https://artificialintelligenceact.eu/)
- [ML Metadata (MLMD) by TensorFlow](https://www.tensorflow.org/tfx/guide/mlmd)
