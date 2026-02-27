# ML Experiment Tracking Reference Guide

## Tool Comparison

| Feature                  | MLflow                    | W&B (Weights & Biases)   | Neptune            | CometML              | Aim                      |
|--------------------------|---------------------------|--------------------------|--------------------|----------------------|--------------------------|
| **License**              | Apache 2.0 (open source)  | Freemium (SaaS)          | Freemium (SaaS)    | Freemium (SaaS)      | Apache 2.0 (open source) |
| **Self-Hosted**          | Yes                       | Enterprise only           | Enterprise only     | Enterprise only       | Yes                      |
| **Artifact Storage**     | Local, S3, GCS, Azure     | W&B servers / S3         | Neptune servers     | Comet servers / S3   | Local filesystem         |
| **Model Registry**       | Built-in                  | Registry + Lineage       | Built-in           | Built-in             | No                       |
| **Hyperparameter Viz**   | Basic (parallel coords)   | Advanced (sweeps UI)     | Advanced           | Advanced             | Advanced                 |
| **Collaboration**        | Basic (shared server)     | Teams, reports, comments | Workspaces, sharing| Teams, panels        | Basic (shared instance)  |
| **Best For**             | Open-source full lifecycle| Collaborative research   | Structured experiments | Enterprise ML    | Lightweight open-source  |

### When to Use Which

- **MLflow**: Default choice for open-source, self-hosted tracking with model registry and broad ecosystem.
- **W&B**: Research teams valuing collaborative analysis, rich visualizations, and built-in sweeps.
- **Neptune**: Teams needing structured experiment organization with flexible metadata.
- **CometML**: Enterprise teams wanting managed solution with production monitoring.
- **Aim**: Lightweight, self-hosted UI for comparing runs without vendor lock-in.

---

## MLflow Architecture

```
+---------------------+     +---------------------+     +---------------------+
|  MLflow Tracking     |     |  MLflow Projects    |     |  MLflow Models      |
|  - Experiments/Runs  |     |  - MLproject file   |     |  - Model flavors    |
|  - Params, Metrics   |     |  - Conda/Docker env |     |  - Model signatures |
|  - Artifacts         |     |  - Entry points     |     |  - Input examples   |
+---------------------+     +---------------------+     +---------------------+
         |                                                        |
         v                                                        v
+---------------------+                                +---------------------+
|  Tracking Server     |                                |  Model Registry     |
|  - Backend (SQL DB)  |                                |  - Versions/Aliases |
|  - Artifacts (S3)    |                                |  - Stage transitions|
+---------------------+                                +---------------------+
```

### Basic Tracking Example

```python
import mlflow

mlflow.set_tracking_uri("http://mlflow-server:5000")
mlflow.set_experiment("churn_prediction")

with mlflow.start_run(run_name="xgboost_v1") as run:
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("max_depth", 6)
    model = train_model(params)
    mlflow.log_metric("auc", 0.87)
    for epoch, loss in enumerate(training_losses):
        mlflow.log_metric("train_loss", loss, step=epoch)
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.xgboost.log_model(model, artifact_path="model",
                              registered_model_name="churn_model")
```

### Model Registry Workflow

```python
from mlflow import MlflowClient
client = MlflowClient()

mv = client.create_model_version(
    name="churn_model", source=f"runs:/{run_id}/model", run_id=run_id,
)
client.set_registered_model_alias("churn_model", "champion", mv.version)
champion_model = mlflow.pyfunc.load_model("models:/churn_model@champion")
```

---

## Experiment Organization Best Practices

### Naming Conventions

```
Experiment:  <project>/<task>           e.g., fraud_detection/transaction_classifier
Run name:    <model>_<date>_<desc>      e.g., xgboost_20240315_tuned_lr
```

### Tagging Strategy

```python
mlflow.set_tags({
    "team": "ml-platform", "project": "fraud_detection",
    "data_version": "v2.3", "feature_set": "feature_store_v5",
})
```

| Level          | Purpose                                    | Example                                          |
|----------------|--------------------------------------------|--------------------------------------------------|
| **Experiment** | Groups runs for a specific task            | `churn/binary_classifier`                        |
| **Run**        | Single training execution                  | `xgboost_lr01_depth6`                            |
| **Tags**       | Cross-cutting metadata for filtering       | `data_version=v3`, `team=risk`                   |
| **Nested Runs**| Group related runs (CV folds, sweeps)      | Parent: `hp_search`, Children: individual trials |

---

## Hyperparameter Search Strategies

| Strategy              | How It Works                          | Pros                            | Cons                             | Best For                     |
|-----------------------|---------------------------------------|---------------------------------|----------------------------------|------------------------------|
| **Grid Search**       | Exhaustive over all combinations      | Complete coverage               | Exponential cost                 | <4 params, small ranges      |
| **Random Search**     | Sample random combinations            | Better coverage per trial       | No learning from past trials     | Quick baseline; many params  |
| **Bayesian (Optuna)** | Probabilistic model guides next trial | Efficient; learns over time     | Sequential dependency            | Medium budgets               |
| **Hyperband / ASHA**  | Early-stopping bad configurations     | Very efficient at scale         | Requires iterative metric        | Deep learning                |
| **Population-Based**  | Evolves hyperparams during training   | Adapts schedule mid-training    | Complex to implement             | Long training runs; RL       |

### Optuna with MLflow Integration

```python
import optuna, mlflow

def objective(trial):
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    depth = trial.suggest_int("max_depth", 3, 10)
    with mlflow.start_run(nested=True):
        mlflow.log_params({"learning_rate": lr, "max_depth": depth})
        model = train_model(lr=lr, max_depth=depth)
        auc = evaluate_model(model)
        mlflow.log_metric("auc", auc)
        return auc

with mlflow.start_run(run_name="optuna_search"):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_auc", study.best_value)
```

---

## Reproducibility Checklist

| Component           | What to Track                          | How to Track                                  |
|---------------------|----------------------------------------|-----------------------------------------------|
| **Code version**    | Git commit hash, branch, dirty status  | `mlflow.log_param("git_sha", sha)`           |
| **Data version**    | Dataset hash, path, version tag        | DVC, MLflow Datasets, or manual hash logging  |
| **Environment**     | Python version, package versions       | `conda.yaml`, `requirements.txt`, Docker image|
| **Random seeds**    | numpy, torch, python seeds             | Log as parameters                             |
| **Hyperparameters** | All model and training config          | `mlflow.log_params(config)`                   |
| **Hardware**        | GPU type, CPU count, memory            | Log as tags                                   |

---

## MLflow Deployment Options

| Option                      | Complexity | Scalability | Best For                      |
|-----------------------------|------------|-------------|-------------------------------|
| **Local filesystem**        | Trivial    | Single user | Individual experimentation    |
| **Local tracking server**   | Low        | Small team  | Team sharing on one machine   |
| **Remote server + DB + S3** | Medium     | High        | Production-grade tracking     |
| **Databricks Managed**      | Low        | High        | Databricks users              |
| **Kubernetes (Helm)**       | High       | Very high   | Large orgs, multi-team        |

### Production Setup: Remote Server with PostgreSQL and S3

```bash
mlflow server \
  --backend-store-uri postgresql://user:pass@db-host:5432/mlflow \
  --default-artifact-root s3://mlflow-artifacts/ \
  --host 0.0.0.0 --port 5000
```

---

## Common Pitfalls

1. **Not logging everything**: Forgetting preprocessing steps or data splits leads to irreproducible results.
2. **Inconsistent naming**: Without conventions, experiments become impossible to navigate quickly.
3. **Wrong metric granularity**: Logging only final metrics loses the training curve. Use `step` for epoch-level metrics.
4. **Ignoring artifact size**: Logging large checkpoints for every run exhausts storage. Log selectively.
5. **No random seeds**: Stochastic training without seeds makes run comparison meaningless.
6. **Vendor lock-in**: Wrap tracking calls in an abstraction layer to ease migration.

---

## Best Practices

- Log every run, even failed ones -- they provide signal about what does not work.
- Use nested runs to group hyperparameter search trials under a parent run.
- Tag runs with metadata (data version, feature set, team) for easy filtering.
- Automate tracking in CI/CD so every model change is recorded.
- Set up model registry stage gates: candidate, champion, archived.
- Keep tracking server separate from training infrastructure for reliability.

---

## Further Reading

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Weights & Biases Documentation](https://docs.wandb.ai/)
- [Neptune Documentation](https://docs.neptune.ai/)
- [CometML Documentation](https://www.comet.com/docs/)
- [Aim Documentation](https://aimstack.readthedocs.io/)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Hidden Technical Debt in ML Systems (Google)](https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html)
