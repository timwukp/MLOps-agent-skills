# ML Pipeline Orchestration -- Reference Guide

Detailed reference material for ML pipeline orchestration, including framework comparisons, design pattern catalogs, best practices, testing strategies, and migration guides.

---

## Table of Contents

1. [Orchestrator Comparison Matrix](#1-orchestrator-comparison-matrix)
2. [Pipeline Design Patterns Catalog](#2-pipeline-design-patterns-catalog)
3. [Apache Airflow Best Practices and Anti-Patterns](#3-apache-airflow-best-practices-and-anti-patterns)
4. [Pipeline Testing Strategies](#4-pipeline-testing-strategies)
5. [Migration Guide Between Orchestrators](#5-migration-guide-between-orchestrators)
6. [Production Checklist](#6-production-checklist)
7. [Troubleshooting Common Issues](#7-troubleshooting-common-issues)
8. [Kubeflow Pipelines Deep Dive](#8-kubeflow-pipelines-deep-dive)
9. [Prefect Flows Deep Dive](#9-prefect-flows-deep-dive)
10. [Dagster Deep Dive](#10-dagster-deep-dive)
11. [ZenML Pipeline Abstraction](#11-zenml-pipeline-abstraction)
12. [Parameter Passing Between Steps](#12-parameter-passing-between-steps)
13. [Pipeline Versioning and Reproducibility](#13-pipeline-versioning-and-reproducibility)
14. [Dynamic Pipelines](#14-dynamic-pipelines)
15. [Pipeline Caching and Artifact Reuse](#15-pipeline-caching-and-artifact-reuse)
16. [Environment Management](#16-environment-management)
17. [Pipeline Templates and Reusable Components](#17-pipeline-templates-and-reusable-components)
18. [CI/CD for Pipeline Code](#18-cicd-for-pipeline-code)
19. [Pipeline Migration Code Patterns](#19-pipeline-migration-code-patterns)

---

## 1. Orchestrator Comparison Matrix

### 1.1 Feature Comparison

| Feature | Apache Airflow | Kubeflow Pipelines | Prefect | Dagster | ZenML | Argo Workflows | Metaflow |
|---------|---------------|-------------------|---------|---------|-------|----------------|----------|
| **Architecture** | Centralized scheduler + workers | Kubernetes CRDs + Argo backend | Hybrid (server + agents) | Dagster daemon + webserver | Client SDK + server | Kubernetes CRDs | Client + metadata service |
| **Pipeline definition** | Python DAGs | Python DSL -> YAML | Python (decorators) | Python (decorators) | Python (decorators) | YAML / Python SDK | Python (decorators) |
| **Execution model** | Task-based | Container-based | Task-based | Op/Asset-based | Step-based | Container-based | Step-based |
| **Scheduling** | Cron, dataset-aware, timetables | Cron, recurring runs | Cron, event-driven, automations | Cron, sensors | Cron (via orchestrator) | Cron | Cron (via external) |
| **Dynamic pipelines** | Dynamic task mapping (2.3+) | Limited (loops at compile time) | Native (Python control flow) | Dynamic graphs | Limited | Conditionals, loops | Limited |
| **Data passing** | XCom (metadata) + external stores | Typed artifacts | Return values + blocks | IO Managers | Materializers | Artifacts | Data stores |
| **Kubernetes native** | No (optional via providers) | Yes (required) | No (optional infra) | No (optional via K8s run launcher) | No (optional) | Yes (required) | No (optional via K8s) |
| **GPU support** | Via K8sPodOperator | Native | Via infrastructure blocks | Via resources | Via stack config | Native | Via decorators |
| **Caching** | Manual | Automatic (input-hash based) | Input-hash based | Memoization | Automatic | Template memoization | Namespace-based |
| **Data lineage** | Limited (dataset-aware, Lineage backend) | Pipeline graph | Moderate (UI tracking) | Excellent (asset graph) | Good (pipeline graph) | Pipeline graph | Run artifacts |
| **UI** | Rich web UI | KFP Dashboard | Prefect Cloud/Server UI | Dagit (excellent) | ZenML Dashboard | Argo UI | Metaflow UI / Cards |
| **Managed services** | Astronomer, MWAA, Cloud Composer | GCP Vertex AI, custom | Prefect Cloud | Dagster Cloud | ZenML Cloud | Akuity (Argo CD) | AWS Step Functions adapter |
| **Community size** | Very large (~35k GitHub stars) | Large (~14k stars) | Growing (~17k stars) | Growing (~11k stars) | Growing (~4k stars) | Large (~15k stars) | Growing (~8k stars) |
| **Learning curve** | Moderate | Steep | Low | Moderate | Low | Moderate-Steep | Low |
| **Best suited for** | General workflow orchestration | K8s-native ML at scale | Pythonic ML pipelines | Data asset management | Multi-infra abstraction | K8s-native CI/CD + ML | Research-to-production |

### 1.2 Infrastructure Requirements

| Orchestrator | Minimum Infrastructure | Production Infrastructure |
|--------------|----------------------|--------------------------|
| **Airflow** | Single machine (SQLite + SequentialExecutor) | PostgreSQL/MySQL + Celery/K8s Executor + Redis + multiple workers |
| **Kubeflow** | Kubernetes cluster | Multi-node K8s with GPU nodes, Istio, MinIO/GCS/S3 |
| **Prefect** | Single machine (SQLite) | Prefect Cloud or Prefect Server + PostgreSQL + work pools |
| **Dagster** | Single machine (SQLite) | Dagster Cloud or daemon + PostgreSQL + run launchers |
| **ZenML** | Single machine (local stack) | ZenML Server + configured stack (K8s, Airflow, or Vertex as orchestrator) |
| **Argo** | Kubernetes cluster | Multi-node K8s with artifact storage (S3/GCS/MinIO) |
| **Metaflow** | Single machine | AWS (S3 + Batch/Step Functions) or K8s |

### 1.3 Cost Comparison (Approximate Monthly, Medium Scale)

| Orchestrator | Self-Managed | Managed Service |
|--------------|-------------|-----------------|
| **Airflow** | $200-800 (2-4 workers on EC2/GCE) | $300-1500 (MWAA/Composer/Astronomer) |
| **Kubeflow** | $400-2000 (K8s cluster) | $500-3000 (GKE + Vertex) |
| **Prefect** | $100-500 (server + agents) | $0-750 (Prefect Cloud tiers) |
| **Dagster** | $100-500 (daemon + webserver) | $0-1000 (Dagster Cloud tiers) |
| **ZenML** | $100-400 (server) | $0-500 (ZenML Cloud tiers) |
| **Argo** | $300-1500 (K8s cluster) | $200-1000 (Akuity + K8s) |
| **Metaflow** | $100-800 (metadata service) | $200-2000 (AWS Batch/Step Functions) |

*Costs exclude compute for actual ML workloads (GPU instances, etc.).*

### 1.4 Decision Matrix

Use this matrix to choose an orchestrator based on your situation:

| Situation | Recommended | Runner-up |
|-----------|-------------|-----------|
| Already have Kubernetes | Kubeflow or Argo | Airflow on K8s Executor |
| Python-first team, minimal infra | Prefect | Dagster |
| Data engineering + ML | Airflow | Dagster |
| Data asset / lineage focus | Dagster | ZenML |
| Need to swap orchestrators later | ZenML | Custom abstraction layer |
| Research team going to production | Metaflow | Prefect |
| Enterprise, existing Airflow | Airflow | ZenML (wrapping Airflow) |
| Real-time + batch pipelines | Prefect | Dagster |
| Multi-cloud, multi-orchestrator | ZenML | Custom abstraction layer |

---

## 2. Pipeline Design Patterns Catalog

### 2.1 Standard Training Pipeline

```
[Ingest] -> [Validate] -> [Feature Eng] -> [Split] -> [Train] -> [Evaluate] -> [Gate] -> [Register]
```

**When to use:** Any supervised learning model training.
**Key decisions:** Gate threshold, evaluation metrics, registration criteria.

### 2.2 Champion-Challenger Pipeline

```
[Prepare Data] -> [Train Champion (current algo)]  -> [Evaluate Champion]  -+
                -> [Train Challenger (new algo)]    -> [Evaluate Challenger] -+-> [Compare] -> [Promote Best]
```

**When to use:** Comparing a new model against the current production model.
**Implementation:**

```python
# Airflow: fan-out to champion and challenger
prepare >> [train_champion, train_challenger]
[eval_champion, eval_challenger] >> compare >> promote

# Prefect: parallel execution
@flow
def champion_challenger():
    data = prepare_data()
    champion = train_champion.submit(data)
    challenger = train_challenger.submit(data)
    champion_metrics = evaluate.submit(champion.result())
    challenger_metrics = evaluate.submit(challenger.result())
    best = compare(champion_metrics.result(), challenger_metrics.result())
    promote(best)
```

### 2.3 Ensemble Training Pipeline

```
                            +-> [Train Model A] -+
[Prepare Data] -> [Split] -+-> [Train Model B] -+-> [Ensemble] -> [Evaluate] -> [Register]
                            +-> [Train Model C] -+
```

**When to use:** Building ensemble models (bagging, stacking, blending).
**Key pattern:** Fan-out to parallel training tasks, fan-in to ensemble step.

### 2.4 Incremental / Online Learning Pipeline

```
[Fetch New Data] -> [Validate] -> [Load Current Model] -> [Update Model] -> [Evaluate] -> [Register if Better]
```

**When to use:** Models that can learn incrementally (SGD, online trees, neural nets with warm start).
**Key decisions:** When to do full retraining vs. incremental update, staleness threshold.

### 2.5 Feature Pipeline + Training Pipeline (Decoupled)

```
Feature Pipeline (hourly):
[Ingest Sources] -> [Compute Features] -> [Validate] -> [Write to Feature Store]

Training Pipeline (daily):
[Read from Feature Store] -> [Split] -> [Train] -> [Evaluate] -> [Register]
```

**When to use:** Multiple models share the same features, feature computation is expensive.
**Key pattern:** Feature store acts as the contract between the two pipelines.

### 2.6 A/B Testing Pipeline

```
[Deploy Candidates] -> [Route Traffic] -> [Collect Metrics] -> [Statistical Test] -> [Promote Winner]
```

**When to use:** Validating model changes in production with real user traffic.
**Key decisions:** Traffic split ratio, minimum sample size, statistical significance level.

### 2.7 Continuous Training (CT) Pipeline

```
[Monitor Drift] -> [Trigger Threshold] -> [Full Training Pipeline] -> [Canary Deploy] -> [Monitor]
         ^                                                                                    |
         +------------------------------------------------------------------------------------+
```

**When to use:** Automated model refresh in response to data/concept drift.
**Key pattern:** Monitoring pipeline triggers training pipeline in a closed loop.

### 2.8 Multi-Environment Promotion Pipeline

```
[Train (Dev)] -> [Test (Staging)] -> [Approve] -> [Deploy (Production)]
```

**When to use:** Enterprise environments with strict promotion policies.
**Key pattern:** Same model artifact promoted through environments; pipeline handles environment-specific validation.

### 2.9 Data Backfill Pipeline

```
[Generate Date Range] -> [Parallel: Process Each Date] -> [Validate All] -> [Merge Results]
```

**When to use:** Reprocessing historical data for features or labels.
**Implementation:**

```python
# Airflow: dynamic task mapping for backfill
@task
def generate_dates(start: str, end: str) -> list[str]:
    return pd.date_range(start, end, freq="D").strftime("%Y-%m-%d").tolist()

@task
def process_date(date: str) -> dict:
    # Process data for a single date
    return {"date": date, "rows": 1000}

with DAG("backfill", ...):
    dates = generate_dates("2024-01-01", "2024-12-31")
    results = process_date.expand(date=dates)  # Parallel processing
```

### 2.10 Shadow Deployment Pipeline

```
[Production Model (serves)] -> [Log Predictions]
[Shadow Model (silent)]     -> [Log Predictions] -> [Compare] -> [Report]
```

**When to use:** Validating a new model against production without serving its predictions to users.

---

## 3. Apache Airflow Best Practices and Anti-Patterns

### 3.1 Best Practices

#### DAG Design

| Practice | Details |
|----------|---------|
| **Keep DAG files lightweight** | DAG files are parsed every `dag_dir_list_interval` (default 5 min). No heavy imports, API calls, or computations at the top level. |
| **Use TaskGroups** | Organize related tasks into groups for cleaner UI and logical structure. |
| **Set `max_active_runs=1` for ML DAGs** | Prevents concurrent training runs that could compete for GPU resources. |
| **Disable catchup** | `catchup=False` for ML pipelines to avoid accidentally running hundreds of historical runs. |
| **Use meaningful `dag_id` and `task_id` names** | They appear in logs, metrics, and alerts. Avoid generic names. |
| **Tag DAGs** | Use tags like `["ml", "training", "team-x"]` for filtering in the UI. |

#### Task Design

| Practice | Details |
|----------|---------|
| **Make tasks idempotent** | Running a task twice with the same inputs should produce the same result. Use overwrite semantics for artifacts. |
| **Keep tasks small and focused** | Each task does one thing (ingest, validate, train, etc.). Easier to test, debug, and retry. |
| **Use the TaskFlow API** | `@task` decorator simplifies XCom handling and makes code more Pythonic. |
| **Configure retries thoughtfully** | Training tasks: 1-2 retries (long-running). API calls: 3-5 retries with exponential backoff. |
| **Set `execution_timeout`** | Prevent runaway tasks from consuming resources indefinitely. |

#### XCom and Data Passing

| Practice | Details |
|----------|---------|
| **Pass URIs, not data** | Store artifacts (DataFrames, models) externally; pass S3/GCS paths via XCom. |
| **Use a custom XCom backend** | For larger metadata, configure S3/GCS XCom backend. |
| **Serialize with standard formats** | Use JSON, Parquet, or Pickle for consistency. Avoid custom serialization. |
| **Document XCom contracts** | Each task's XCom output schema should be documented and treated as a contract. |

#### Connections and Secrets

| Practice | Details |
|----------|---------|
| **Never hard-code credentials** | Use Airflow Connections and a secrets backend (Vault, AWS Secrets Manager, etc.). |
| **Use connection test** | Verify connections work before relying on them in DAGs. |
| **Rotate secrets regularly** | Use a secrets backend that supports rotation; Airflow re-reads on each task execution. |

#### Monitoring and Observability

| Practice | Details |
|----------|---------|
| **Enable StatsD metrics** | Monitor scheduler lag, task duration, queue times, and failure rates. |
| **Set SLA** | Use `sla=timedelta(hours=4)` on tasks to get alerts when they take too long. |
| **Use `on_failure_callback`** | Send alerts to Slack, PagerDuty, or email on task failure. |
| **Log structured data** | Use JSON-formatted logs for easier parsing by log aggregation tools. |

### 3.2 Anti-Patterns

#### The Monolith DAG

**Problem:** A single DAG file with 200+ tasks, 5000+ lines, mixing data engineering and ML logic.

**Why it's bad:** Hard to test, hard to understand, single point of failure, long DAG parse times.

**Fix:** Break into smaller DAGs connected by `TriggerDagRunOperator` or dataset-aware scheduling.

#### Top-Level Heavy Imports

**Problem:**
```python
# BAD: These run every time the DAG file is parsed
import tensorflow as tf
model = tf.keras.models.load_model("s3://...")
df = pd.read_csv("s3://large-dataset.csv")
```

**Why it's bad:** DAG files are parsed frequently. Heavy imports and I/O at the top level slow down the scheduler and waste resources.

**Fix:**
```python
# GOOD: Defer imports to task callables
def train_model(**context):
    import tensorflow as tf
    model = tf.keras.models.load_model(context["params"]["model_uri"])
    ...
```

#### XCom as a Data Lake

**Problem:** Pushing multi-MB DataFrames or model binaries through XCom.

**Why it's bad:** XCom is backed by the metadata database. Large objects bloat the DB, slow down queries, and can cause OOM errors.

**Fix:** Store artifacts in S3/GCS/Azure Blob, push only the URI via XCom.

#### depends_on_past Cascade

**Problem:** Setting `depends_on_past=True` on multiple tasks.

**Why it's bad:** One failed task blocks all future runs of that task. If the failure is not fixed quickly, a backlog accumulates.

**Fix:** Use `depends_on_past` sparingly. Prefer explicit data checks (sensors or ShortCircuitOperator) over implicit temporal dependencies.

#### Star Schema Dependencies

**Problem:**
```python
# BAD: Every task depends on every other task
for t in [task1, task2, task3, task4, task5]:
    for u in [task6, task7, task8, task9, task10]:
        t >> u
```

**Why it's bad:** Creates unnecessary bottlenecks. All of group 1 must finish before any of group 2 starts, even if there is no real data dependency.

**Fix:** Define dependencies based on actual data flow, not convenience.

#### Ignoring Idempotency

**Problem:** Tasks that append to a table or file without checking if they have already run.

**Why it's bad:** Retries or manual re-runs produce duplicate data.

**Fix:**
```python
# GOOD: Overwrite semantics
def ingest_data(**context):
    output_path = f"s3://data/{context['ds']}/data.parquet"
    df = fetch_data(context["ds"])
    df.to_parquet(output_path, mode="overwrite")  # Idempotent
```

#### Hard-Coded Everything

**Problem:** File paths, model names, thresholds, and credentials embedded directly in the DAG file.

**Fix:** Use Airflow Variables, Connections, and parameterized DAGs (`params` or Jinja templates).

---

## 4. Pipeline Testing Strategies

### 4.1 Testing Pyramid for ML Pipelines

```
                    /\
                   /  \
                  / E2E \          <- Few, slow, expensive
                 / Tests  \
                /----------\
               / Integration \     <- Moderate count
              /    Tests      \
             /----------------\
            /    Unit Tests    \   <- Many, fast, cheap
           /____________________\
```

### 4.2 Unit Testing

**What to test:** Individual pipeline step functions, data transformations, validation logic, utility functions.

**Frameworks:** pytest, unittest.

```python
# tests/unit/test_feature_engineering.py
import pytest
import pandas as pd
import numpy as np
from pipeline.steps.feature_engineering import (
    compute_interaction_features,
    compute_polynomial_features,
    apply_standard_scaling,
    engineer_features_fn,
)


class TestInteractionFeatures:
    def test_creates_expected_columns(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        result = compute_interaction_features(df, pairs=[("a", "b"), ("a", "c")])
        assert "a_x_b" in result.columns
        assert "a_x_c" in result.columns

    def test_correct_values(self):
        df = pd.DataFrame({"a": [2, 3], "b": [4, 5]})
        result = compute_interaction_features(df, pairs=[("a", "b")])
        assert result["a_x_b"].tolist() == [8, 15]

    def test_handles_nulls(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, 6]})
        result = compute_interaction_features(df, pairs=[("a", "b")])
        assert pd.isna(result["a_x_b"].iloc[1])


class TestStandardScaling:
    def test_zero_mean_unit_variance(self):
        df = pd.DataFrame({"x": np.random.randn(1000)})
        scaled, params = apply_standard_scaling(df, ["x"])
        assert abs(scaled["x"].mean()) < 0.1
        assert abs(scaled["x"].std() - 1.0) < 0.1

    def test_returns_scaler_params(self):
        df = pd.DataFrame({"x": [10, 20, 30]})
        _, params = apply_standard_scaling(df, ["x"])
        assert "means" in params
        assert "stds" in params
        assert params["means"]["x"] == pytest.approx(20.0)


class TestDataValidation:
    def test_passes_valid_data(self):
        df = pd.DataFrame({
            "feature_0": np.random.randn(500),
            "target": np.random.randint(0, 2, 500),
        })
        result = validate_data_fn(df)
        assert result["passed"] is True
        assert result["errors"] == []

    def test_fails_on_missing_target(self):
        df = pd.DataFrame({"feature_0": np.random.randn(500)})
        result = validate_data_fn(df)
        assert result["passed"] is False
        assert any("target" in e for e in result["errors"])

    def test_fails_on_insufficient_rows(self):
        df = pd.DataFrame({
            "feature_0": [1, 2, 3],
            "target": [0, 1, 0],
        })
        result = validate_data_fn(df)
        assert result["passed"] is False

    def test_fails_on_high_null_ratio(self):
        df = pd.DataFrame({
            "feature_0": [None] * 50 + [1.0] * 50,
            "feature_1": [None] * 50 + [1.0] * 50,
            "target": [0] * 50 + [1] * 50,
        })
        result = validate_data_fn(df)
        # 100 nulls out of 300 values = 33% -> should fail
        assert result["passed"] is False
```

### 4.3 Integration Testing

**What to test:** Full pipeline execution with synthetic data, inter-step data contracts, artifact storage.

```python
# tests/integration/test_training_pipeline.py
import pytest
from pathlib import Path

from sklearn.datasets import make_classification
import pandas as pd


@pytest.fixture
def synthetic_data(tmp_path):
    """Create synthetic training data for integration tests."""
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
    df["target"] = y
    data_path = tmp_path / "test_data.csv"
    df.to_csv(data_path, index=False)
    return str(data_path)


class TestTrainingPipelineIntegration:
    def test_full_pipeline_with_good_data(self, synthetic_data, tmp_path):
        """Pipeline should succeed end-to-end with valid data."""
        result = run_pipeline(
            data_source=synthetic_data,
            artifact_root=str(tmp_path / "artifacts"),
            quality_threshold=0.5,  # Low for test
            n_estimators=10,        # Small for speed
            max_depth=3,
        )
        assert result["status"] == "success"
        assert result["model_registered"] is True
        assert result["accuracy"] > 0.5

    def test_pipeline_rejects_bad_model(self, synthetic_data, tmp_path):
        """Pipeline should not register a model below quality threshold."""
        result = run_pipeline(
            data_source=synthetic_data,
            artifact_root=str(tmp_path / "artifacts"),
            quality_threshold=0.999,  # Unreachable
            n_estimators=2,
            max_depth=1,
        )
        assert result["model_registered"] is False

    def test_artifacts_are_persisted(self, synthetic_data, tmp_path):
        """All intermediate artifacts should be saved."""
        artifact_root = tmp_path / "artifacts"
        run_pipeline(
            data_source=synthetic_data,
            artifact_root=str(artifact_root),
            quality_threshold=0.5,
            n_estimators=10,
        )
        # Check that artifacts exist
        assert any(artifact_root.rglob("raw_data.csv"))
        assert any(artifact_root.rglob("features.csv"))
        assert any(artifact_root.rglob("train.csv"))
        assert any(artifact_root.rglob("test.csv"))
        assert any(artifact_root.rglob("model.pkl"))
        assert any(artifact_root.rglob("metrics.json"))

    def test_model_can_predict(self, synthetic_data, tmp_path):
        """Registered model should be loadable and functional."""
        import pickle
        result = run_pipeline(
            data_source=synthetic_data,
            artifact_root=str(tmp_path / "artifacts"),
            quality_threshold=0.5,
            n_estimators=10,
        )
        model_path = result.get("model_uri") or next(
            (tmp_path / "artifacts").rglob("model.pkl")
        )
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        test_input = pd.DataFrame(
            [[0.1] * 10], columns=[f"feature_{i}" for i in range(10)]
        )
        prediction = model.predict(test_input)
        assert prediction[0] in [0, 1]
```

### 4.4 DAG Validation Testing (Airflow-Specific)

```python
# tests/dag_validation/test_dags.py
import pytest
from airflow.models import DagBag


@pytest.fixture(scope="session")
def dag_bag():
    return DagBag(dag_folder="dags/", include_examples=False)


class TestDAGIntegrity:
    def test_no_import_errors(self, dag_bag):
        assert len(dag_bag.import_errors) == 0, (
            f"DAG import errors: {dag_bag.import_errors}"
        )

    def test_expected_dags_exist(self, dag_bag):
        expected = ["ml_training_pipeline", "feature_pipeline", "monitoring_pipeline"]
        for dag_id in expected:
            assert dag_id in dag_bag.dags, f"Missing DAG: {dag_id}"

    def test_training_dag_has_correct_tasks(self, dag_bag):
        dag = dag_bag.get_dag("ml_training_pipeline")
        expected_tasks = {
            "data_pipeline.ingest_data",
            "data_pipeline.validate_data",
            "data_pipeline.engineer_features",
            "data_pipeline.split_data",
            "training.train_model",
            "training.evaluate_model",
            "quality_gate",
        }
        actual_tasks = set(dag.task_ids)
        assert expected_tasks.issubset(actual_tasks), (
            f"Missing tasks: {expected_tasks - actual_tasks}"
        )

    def test_no_cycles(self, dag_bag):
        for dag_id, dag in dag_bag.dags.items():
            # DAGs are acyclic by definition; this validates the parser agrees
            assert dag is not None, f"DAG {dag_id} is None"

    def test_default_args(self, dag_bag):
        dag = dag_bag.get_dag("ml_training_pipeline")
        assert dag.default_args.get("retries", 0) >= 1
        assert dag.default_args.get("owner") is not None

    def test_no_top_level_queries(self, dag_bag):
        """Ensure DAGs don't make expensive calls at parse time."""
        # If DAG parsing takes >5 seconds, there's likely a top-level issue
        import time
        start = time.time()
        _ = DagBag(dag_folder="dags/", include_examples=False)
        duration = time.time() - start
        assert duration < 10, f"DAG parsing took {duration:.1f}s (too slow)"
```

### 4.5 Contract Testing Between Steps

```python
# tests/contracts/test_step_contracts.py
"""Verify that the output of each step matches the input contract of the next step."""
import pytest
import json

# Define contracts as JSON schemas
INGEST_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["data_path", "num_rows", "num_cols"],
    "properties": {
        "data_path": {"type": "string"},
        "num_rows": {"type": "integer", "minimum": 1},
        "num_cols": {"type": "integer", "minimum": 2},
    },
}

VALIDATION_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["passed", "errors", "data_path"],
    "properties": {
        "passed": {"type": "boolean"},
        "errors": {"type": "array", "items": {"type": "string"}},
        "data_path": {"type": "string"},
    },
}


def validate_schema(data: dict, schema: dict) -> list[str]:
    """Simple schema validation (use jsonschema for production)."""
    errors = []
    for field in schema.get("required", []):
        if field not in data:
            errors.append(f"Missing required field: {field}")
    return errors


class TestStepContracts:
    def test_ingest_output_matches_contract(self):
        result = ingest_data_fn(source="test_data.csv")
        errors = validate_schema(result, INGEST_OUTPUT_SCHEMA)
        assert not errors, f"Contract violations: {errors}"

    def test_validation_output_matches_contract(self):
        result = validate_data_fn(sample_dataframe)
        errors = validate_schema(result, VALIDATION_OUTPUT_SCHEMA)
        assert not errors, f"Contract violations: {errors}"

    def test_ingest_output_is_valid_validation_input(self):
        """The output of ingest can be consumed by validate."""
        ingest_result = ingest_data_fn(source="test_data.csv")
        # validate_data expects a dict with "data_path"
        assert "data_path" in ingest_result
        assert isinstance(ingest_result["data_path"], str)
```

### 4.6 Performance Testing

```python
# tests/performance/test_pipeline_performance.py
import pytest
import time

class TestPipelinePerformance:
    @pytest.mark.slow
    def test_training_completes_within_sla(self, large_synthetic_data):
        """Training should complete within 2 hours for 1M rows."""
        start = time.time()
        result = run_pipeline(
            data_source=large_synthetic_data,
            n_estimators=100,
        )
        duration = time.time() - start
        assert duration < 7200, f"Pipeline took {duration/60:.1f} min (SLA: 120 min)"
        assert result["status"] == "success"

    @pytest.mark.slow
    def test_feature_engineering_scales_linearly(self):
        """Feature engineering should scale roughly linearly with data size."""
        durations = {}
        for size in [10_000, 50_000, 100_000]:
            data = generate_synthetic_data(n_rows=size)
            start = time.time()
            engineer_features_fn(data)
            durations[size] = time.time() - start

        # Check that 10x data doesn't take more than 15x time
        ratio = durations[100_000] / durations[10_000]
        assert ratio < 15, f"Feature engineering scales poorly: {ratio:.1f}x for 10x data"
```

---

## 5. Migration Guide Between Orchestrators

### 5.1 General Migration Strategy

**Phase 1: Assessment (1-2 weeks)**
1. Inventory all existing pipelines (count, complexity, dependencies).
2. Document external integrations (databases, APIs, cloud services).
3. Identify scheduling requirements (cron, event-driven, data-driven).
4. Assess team skills and learning budget.
5. Evaluate managed vs. self-hosted options.

**Phase 2: Abstraction (2-4 weeks)**
1. Extract business logic from orchestrator-specific code.
2. Create a thin adapter layer (see Section 19 in SKILL.md).
3. Write orchestrator-agnostic unit tests for each pipeline step.
4. Standardize configuration management (YAML, environment variables).

**Phase 3: Parallel Run (2-4 weeks)**
1. Implement the most critical pipeline in the new orchestrator.
2. Run both old and new pipelines in parallel.
3. Compare outputs, performance, and reliability.
4. Build monitoring for the new orchestrator.

**Phase 4: Migration (4-8 weeks)**
1. Migrate pipelines in priority order (least critical first).
2. Update CI/CD to deploy to the new orchestrator.
3. Train the team on the new framework.
4. Decommission old pipelines one by one.

**Phase 5: Optimization (2-4 weeks)**
1. Leverage new orchestrator's native features.
2. Optimize scheduling, caching, and resource allocation.
3. Update documentation and runbooks.

### 5.2 Airflow to Prefect

| Airflow Concept | Prefect Equivalent | Migration Notes |
|----------------|-------------------|-----------------|
| DAG | `@flow` | Flows can call sub-flows (richer composition) |
| Operator / `@task` | `@task` | Prefect tasks are plain Python functions |
| XCom | Return values / task results | No size limits; objects stay in Python memory for local runs |
| Connection | Block | Create blocks via UI or code; use for S3, DBs, etc. |
| Variable | Parameters / env vars | Flow parameters are type-checked |
| Sensor | Polling task / Automation trigger | Use `while` loops or Prefect Automations |
| `schedule_interval` | `Schedule` (cron, interval, RRule) | Defined in deployment, not in flow code |
| `trigger_rule` | Python `if`/`else` in flow | Natural Python control flow |
| `BranchPythonOperator` | Python `if`/`else` | No special operator needed |
| Dynamic task mapping | `.map()` | Simpler syntax, built-in |
| TaskGroup | Sub-flow (`@flow`) | Sub-flows get their own run tracking |
| Pool | Work pool / concurrency limits | Configured via deployments |
| `catchup` | Backfill via API | Run historical flow runs via API or CLI |

**Code example -- Airflow to Prefect:**

```python
# ---- AIRFLOW ----
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def extract(**context):
    data = fetch_data()
    context["ti"].xcom_push(key="data_path", value="/tmp/data.csv")

def transform(**context):
    path = context["ti"].xcom_pull(task_ids="extract", key="data_path")
    transformed = process(path)
    context["ti"].xcom_push(key="output", value="/tmp/transformed.csv")

with DAG("etl", start_date=datetime(2024, 1, 1), schedule_interval="@daily"):
    t1 = PythonOperator(task_id="extract", python_callable=extract)
    t2 = PythonOperator(task_id="transform", python_callable=transform)
    t1 >> t2

# ---- PREFECT ----
from prefect import flow, task

@task(retries=2)
def extract() -> str:
    data = fetch_data()
    return "/tmp/data.csv"

@task(retries=2)
def transform(data_path: str) -> str:
    transformed = process(data_path)
    return "/tmp/transformed.csv"

@flow(name="etl")
def etl_flow():
    path = extract()
    output = transform(path)
```

### 5.3 Airflow to Dagster

| Airflow Concept | Dagster Equivalent | Migration Notes |
|----------------|-------------------|-----------------|
| DAG | Job (or asset graph) | Jobs are composed of ops; assets model data lineage |
| Operator | Op or Asset | `@op` for tasks, `@asset` for data artifacts |
| XCom | Op outputs / IO Manager | IO managers handle serialization transparently |
| Connection | Resource | Resources are injected via dependency injection |
| Variable | Config / RunConfig | Type-checked configuration |
| Sensor | Sensor | Similar concept; Dagster sensors yield `RunRequest` |
| `schedule_interval` | Schedule | `ScheduleDefinition` or `@schedule` |
| `execution_date` | Partition key | Dagster partitions model time-based data naturally |
| Pool | Concurrency limits (tag-based) | `max_concurrent_per_tag` on executors |
| `catchup` | Backfill (via UI or API) | Partition-based backfill is first-class |

### 5.4 Airflow to Kubeflow Pipelines

| Airflow Concept | KFP Equivalent | Migration Notes |
|----------------|---------------|-----------------|
| DAG | `@dsl.pipeline` | Compiled to YAML; each step is a container |
| PythonOperator | `@dsl.component` | Function becomes a container image |
| KubernetesPodOperator | `@dsl.component` with custom image | More natural in KFP |
| XCom | Component outputs (typed artifacts) | Strongly typed: `Input[Dataset]`, `Output[Model]` |
| Connection | K8s secrets / env vars | Mount secrets as env vars or volumes |
| Sensor | Not native; use external trigger | Trigger pipeline runs via KFP SDK or API |
| Schedule | Recurring run | Configure in KFP UI or via SDK |
| BranchPythonOperator | `dsl.If`, `dsl.Condition` | Compile-time branching |

### 5.5 Any Orchestrator to ZenML

ZenML's value proposition is that pipeline code stays the same; only the stack changes.

```python
# Step 1: Wrap existing step logic in ZenML steps
from zenml import step, pipeline

@step
def ingest_data(source: str) -> pd.DataFrame:
    # Same logic as before -- no orchestrator dependency
    return pd.read_csv(source)

@step
def train_model(data: pd.DataFrame) -> Model:
    clf = RandomForestClassifier()
    clf.fit(data.drop("target", axis=1), data["target"])
    return clf

@pipeline
def training_pipeline(source: str):
    data = ingest_data(source=source)
    model = train_model(data=data)

# Step 2: Configure stack for target orchestrator
# zenml orchestrator register airflow_orch --flavor=airflow
# zenml orchestrator register kubeflow_orch --flavor=kubeflow
# zenml stack set my_stack --orchestrator=kubeflow_orch

# Step 3: Run -- same code, different infrastructure
training_pipeline(source="s3://data/train.csv")
```

### 5.6 Common Migration Pitfalls

| Pitfall | Description | Mitigation |
|---------|-------------|------------|
| **Scheduling semantics** | Airflow's `execution_date` represents the *start* of the schedule interval; the DAG actually runs at the *end*. Prefect and Dagster run at the scheduled time. | Audit all date-dependent logic during migration. |
| **State model mismatch** | Airflow has granular task instance states (queued, running, up_for_retry, skipped, etc.). Other frameworks model state differently. | Map states explicitly; don't assume equivalence. |
| **XCom vs. return values** | Airflow XCom is pull-based (explicit pull from a task). Prefect uses return values (push-based). | Refactor pull patterns to pass-by-return. |
| **Secret management** | Each framework has its own secrets abstraction. Migrating credentials requires re-registering in the new system. | Use an external secrets manager (Vault, AWS SM) as the source of truth. |
| **Monitoring integration** | StatsD metrics, log formats, and alert hooks differ per framework. | Set up monitoring for the new framework before migrating critical pipelines. |
| **Team mental model** | Each framework has a different philosophy (Airflow: task graphs, Dagster: data assets, Prefect: Pythonic). | Invest in team training; run workshops on the new framework. |
| **Big bang migration** | Trying to migrate everything at once. | Migrate incrementally; run old and new in parallel during transition. |

---

## 6. Production Checklist

Use this checklist before promoting an ML pipeline to production.

### 6.1 Pipeline Code

- [ ] All pipeline steps are idempotent (safe to retry).
- [ ] No hard-coded paths, credentials, or magic numbers.
- [ ] Configuration is externalized (Variables, parameters, config files).
- [ ] Pipeline code is version-controlled (Git).
- [ ] Code passes linting (ruff, flake8, pylint) and type checking (mypy).
- [ ] All imports are deferred to task callables (no heavy top-level code).

### 6.2 Testing

- [ ] Unit tests for each pipeline step (>80% coverage).
- [ ] Integration test with synthetic data passes.
- [ ] DAG validation test passes (no import errors, correct structure).
- [ ] Contract tests between steps pass.
- [ ] Performance test within SLA.

### 6.3 Error Handling

- [ ] Retries configured for each task (with appropriate backoff).
- [ ] `execution_timeout` set for all tasks.
- [ ] Failure callbacks send alerts to on-call channels.
- [ ] Graceful degradation for non-critical failures.
- [ ] Catchup is disabled (or intentionally enabled with backfill strategy).

### 6.4 Monitoring

- [ ] Pipeline success/failure metrics exported (StatsD, Prometheus, CloudWatch).
- [ ] Task duration is tracked and alerted on (SLA miss detection).
- [ ] Resource utilization (CPU, memory, GPU) is monitored.
- [ ] Data quality metrics are logged per run.
- [ ] Model quality metrics are logged per run.
- [ ] Dashboard exists for pipeline health overview.

### 6.5 Security

- [ ] Credentials stored in a secrets manager (not in code or Airflow Variables).
- [ ] Network access restricted (VPC, security groups, IAM roles).
- [ ] Principle of least privilege applied to service accounts.
- [ ] Audit logging enabled for pipeline runs and data access.
- [ ] Container images scanned for vulnerabilities.

### 6.6 Operations

- [ ] Runbook exists for common failure scenarios.
- [ ] On-call rotation covers pipeline alerting channels.
- [ ] Backfill procedure is documented and tested.
- [ ] Rollback procedure is documented (revert to previous DAG version).
- [ ] Capacity planning accounts for peak loads (e.g., month-end batch).

### 6.7 Documentation

- [ ] Pipeline purpose and business context documented.
- [ ] Data flow diagram (input/output per step) up to date.
- [ ] Configuration options documented (parameters, thresholds, schedules).
- [ ] SLA and SLO defined and communicated to stakeholders.

---

## 7. Troubleshooting Common Issues

### 7.1 Airflow Issues

| Issue | Symptoms | Resolution |
|-------|----------|------------|
| **Scheduler not picking up DAGs** | DAGs don't appear in UI | Check `dag_dir_list_interval`, file permissions, import errors in `DagBag` |
| **Tasks stuck in "queued"** | Tasks never start running | Check worker availability, pool slots, executor capacity, Celery broker connectivity |
| **XCom too large** | DB errors, slow queries | Switch to custom XCom backend (S3, GCS); reduce XCom payload size |
| **DAG parse time too long** | Scheduler lag, stale DAGs | Profile with `airflow dags report`; defer heavy imports to task callables |
| **Cascading failures** | Multiple DAG runs fail in sequence | Check `depends_on_past`, clear failed task instances, fix root cause |
| **Connection timeouts** | Tasks fail with timeout errors | Increase `execution_timeout`, check network connectivity, verify connection config |
| **Worker OOM** | Workers crash mid-task | Increase worker memory, reduce task parallelism, use KubernetesPodOperator for heavy tasks |

### 7.2 Prefect Issues

| Issue | Symptoms | Resolution |
|-------|----------|------------|
| **Flow run stuck in "pending"** | Run never starts | Check work pool/agent is running, work queue matches, infrastructure has capacity |
| **Cache not working** | Tasks re-execute despite same inputs | Verify `cache_key_fn` returns consistent keys; check `cache_expiration` hasn't passed |
| **Deployment not triggering** | Scheduled runs don't appear | Check deployment schedule, work pool assignment, Prefect server time zone |
| **Task retries exhausted** | Task fails permanently | Check retry configuration, investigate root cause in task logs |
| **Subflow failures not propagated** | Parent flow succeeds despite child failure | Ensure subflow raises exceptions on failure (don't catch and swallow) |

### 7.3 Dagster Issues

| Issue | Symptoms | Resolution |
|-------|----------|------------|
| **Assets not materializing** | Auto-materialize doesn't trigger | Check `AutoMaterializePolicy`, daemon is running, freshness policies are correct |
| **IO Manager errors** | Can't load/save assets | Verify IO manager configuration, check permissions on storage backend |
| **Sensor not triggering** | Sensor runs but never yields RunRequest | Add logging to sensor function; check cursor logic; verify external condition |
| **Partition backfill slow** | Backfill takes much longer than expected | Check concurrency limits, partition count, executor configuration |

### 7.4 General Pipeline Issues

| Issue | Symptoms | Resolution |
|-------|----------|------------|
| **Data drift causes failures** | Pipeline fails on new data patterns | Add data validation with drift detection; implement schema evolution strategy |
| **GPU contention** | Training tasks queue for GPU | Use resource pools/limits; schedule GPU tasks at off-peak times; consider spot instances |
| **Secret rotation breaks pipeline** | Authentication failures after rotation | Use dynamic secret retrieval (not cached); implement graceful retry on auth failure |
| **Pipeline takes too long** | SLA misses | Profile each step; parallelize independent steps; cache expensive computations; optimize data I/O |
| **Non-deterministic results** | Different outputs on re-run | Pin random seeds everywhere; pin package versions; use deterministic algorithms; log all config |

---

## 8. Kubeflow Pipelines Deep Dive

### 8.1 Core Concepts

Kubeflow Pipelines (KFP) is Kubernetes-native and represents each pipeline step as a container.

| Concept | Description |
|---------|-------------|
| **Component** | A self-contained, containerized step with typed inputs/outputs |
| **Pipeline** | A graph of components connected by data dependencies |
| **Pipeline Run** | A single execution of a pipeline |
| **Experiment** | Groups of runs for organizational purposes |
| **Artifact** | Typed output (Dataset, Model, Metrics, ClassificationMetrics) |

### 8.2 KFP v2 Component Definition

```python
from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model, Metrics

@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas==2.1.0", "scikit-learn==1.3.0"],
)
def train_model(
    training_data: Input[Dataset],
    hyperparams: dict,
    model: Output[Model],
    metrics: Output[Metrics],
):
    """Train an ML model on the provided dataset."""
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score
    import pickle

    df = pd.read_csv(training_data.path)
    X = df.drop(columns=["target"])
    y = df["target"]

    clf = RandomForestClassifier(**hyperparams)
    clf.fit(X, y)

    preds = clf.predict(X)
    metrics.log_metric("accuracy", accuracy_score(y, preds))
    metrics.log_metric("f1_score", f1_score(y, preds, average="weighted"))

    with open(model.path, "wb") as f:
        pickle.dump(clf, f)

    model.metadata["framework"] = "sklearn"
    model.metadata["algorithm"] = "RandomForest"
```

### 8.3 Pipeline Definition (KFP v2)

```python
from kfp import dsl, compiler

@dsl.pipeline(
    name="ml-training-pipeline",
    description="End-to-end training pipeline on Kubernetes",
)
def training_pipeline(
    data_source: str = "gs://ml-data/training/latest",
    learning_rate: float = 0.01,
    n_estimators: int = 100,
    quality_threshold: float = 0.90,
):
    # Step 1: Ingest data
    ingest_task = ingest_data(source=data_source)

    # Step 2: Validate data
    validate_task = validate_data(dataset=ingest_task.outputs["dataset"])

    # Step 3: Feature engineering
    feature_task = engineer_features(
        dataset=validate_task.outputs["validated_dataset"]
    )

    # Step 4: Train model
    train_task = train_model(
        training_data=feature_task.outputs["features"],
        hyperparams={"learning_rate": learning_rate, "n_estimators": n_estimators},
    )
    # Request GPU resources
    train_task.set_gpu_limit(1)
    train_task.set_memory_limit("16Gi")
    train_task.set_cpu_limit("4")

    # Step 5: Evaluate
    eval_task = evaluate_model(
        model=train_task.outputs["model"],
        test_data=feature_task.outputs["test_set"],
    )

    # Step 6: Conditional registration
    with dsl.If(
        eval_task.outputs["accuracy"] >= quality_threshold,
        name="quality-gate",
    ):
        register_model(
            model=train_task.outputs["model"],
            metrics=eval_task.outputs["metrics"],
        )

# Compile to YAML
compiler.Compiler().compile(
    pipeline_func=training_pipeline,
    package_path="training_pipeline.yaml",
)
```

### 8.4 Kubeflow-Specific Features

- **Caching:** Automatically caches step outputs; re-runs skip steps with identical inputs.
- **Volumes:** Mount PVCs for shared storage across steps.
- **Exit handlers:** Run cleanup or notification tasks regardless of pipeline success/failure.
- **Recurring runs:** Schedule pipelines via cron expressions in the KFP UI or SDK.
- **Pipeline versioning:** Each compiled YAML is a versioned pipeline artifact.

```python
# Exit handler example
@dsl.pipeline(name="pipeline-with-cleanup")
def pipeline_with_exit():
    exit_task = notify_completion()

    with dsl.ExitHandler(exit_task):
        step1 = ingest_data()
        step2 = train_model(data=step1.output)
```

---

## 9. Prefect Flows Deep Dive

### 9.1 Core Concepts

| Concept | Description |
|---------|-------------|
| **Flow** | The top-level pipeline function decorated with `@flow` |
| **Task** | A discrete unit of work decorated with `@task` |
| **Flow Run** | A single execution of a flow |
| **Deployment** | Configuration for scheduling and remote execution |
| **Block** | Reusable configuration for external systems (S3, GCS, databases) |
| **Work Pool** | Infrastructure for running flow runs (Docker, K8s, ECS, etc.) |
| **Artifact** | Rich output attached to flow runs (tables, markdown, links) |

### 9.2 Flow and Task Definition

```python
from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from datetime import timedelta

@task(
    retries=3,
    retry_delay_seconds=[10, 60, 300],  # Exponential backoff
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=24),
    tags=["data", "ingestion"],
)
def ingest_data(source_path: str) -> dict:
    """Ingest data from source and return metadata."""
    logger = get_run_logger()
    logger.info(f"Ingesting data from {source_path}")
    # ... ingestion logic ...
    return {"path": "/tmp/data.parquet", "rows": 100000, "columns": 50}

@task(retries=2, retry_delay_seconds=30, tags=["training"])
def train_model(data_path: str, hyperparams: dict) -> dict:
    """Train model and return metrics + model path."""
    logger = get_run_logger()
    logger.info(f"Training with params: {hyperparams}")
    # ... training logic ...
    return {"model_path": "/tmp/model.pkl", "accuracy": 0.95}

@flow(
    name="ml-training-pipeline",
    description="End-to-end ML training pipeline",
    retries=1,
    retry_delay_seconds=600,
    log_prints=True,
    timeout_seconds=7200,
)
def training_pipeline(
    data_source: str = "s3://ml-data/training",
    learning_rate: float = 0.01,
    n_estimators: int = 100,
    quality_threshold: float = 0.90,
):
    data_meta = ingest_data(data_source)
    validation = validate_data(data_meta["path"])

    if not validation["passed"]:
        raise ValueError(f"Data validation failed: {validation['errors']}")

    features_path = engineer_features(data_meta["path"])
    metrics = train_model(features_path, {"lr": learning_rate, "n": n_estimators})

    if metrics["accuracy"] >= quality_threshold:
        register_model(metrics["model_path"], metrics)
    else:
        notify_failure(f"Accuracy {metrics['accuracy']} below {quality_threshold}")
```

### 9.3 Deployments and Scheduling

```python
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule
from prefect_aws.s3 import S3Bucket

# Create a deployment
deployment = Deployment.build_from_flow(
    flow=training_pipeline,
    name="nightly-training",
    version="1.0",
    schedule=CronSchedule(cron="0 2 * * *", timezone="UTC"),
    parameters={
        "data_source": "s3://ml-data/training/latest",
        "learning_rate": 0.001,
    },
    work_pool_name="ml-gpu-pool",
    tags=["production", "nightly"],
)
deployment.apply()
```

```yaml
# prefect.yaml - declarative deployment
deployments:
  - name: nightly-training
    entrypoint: flows/training.py:training_pipeline
    schedule:
      cron: "0 2 * * *"
      timezone: UTC
    parameters:
      data_source: "s3://ml-data/training/latest"
      learning_rate: 0.001
    work_pool:
      name: ml-gpu-pool
    tags:
      - production
      - nightly
```

### 9.4 Prefect Event-Driven Scheduling

```python
# Prefect - event-driven via automations
# Configure in Prefect Cloud/Server UI or via API
{
    "trigger": {
        "type": "event",
        "match": {"resource.name": "s3://ml-data/incoming/*"},
        "expect": ["data.uploaded"],
    },
    "actions": [
        {
            "type": "run-deployment",
            "deployment_id": "training-pipeline/nightly",
        }
    ],
}
```

### 9.5 Prefect Fan-Out / Fan-In

```python
from prefect import flow, task

@task
def train_single(config: dict) -> dict:
    return {"config": config, "accuracy": 0.95}

@flow
def ensemble_pipeline():
    configs = [{"algo": "rf"}, {"algo": "xgb"}, {"algo": "lgbm"}]
    futures = train_single.map(configs)  # Parallel execution
    results = [f.result() for f in futures]
    best = max(results, key=lambda x: x["accuracy"])
```

### 9.6 Prefect Monitoring via Artifacts

```python
from prefect.artifacts import create_table_artifact, create_markdown_artifact

@task
def evaluate_and_report(model, test_data):
    metrics = evaluate(model, test_data)

    create_table_artifact(
        key="model-metrics",
        table=[
            {"metric": "Accuracy", "value": metrics["accuracy"]},
            {"metric": "F1 Score", "value": metrics["f1"]},
            {"metric": "AUC-ROC", "value": metrics["auc"]},
        ],
        description="Model evaluation metrics",
    )

    create_markdown_artifact(
        key="evaluation-report",
        markdown=generate_evaluation_report(metrics),
    )
```

---

## 10. Dagster Deep Dive

### 10.1 Core Concepts

| Concept | Description |
|---------|-------------|
| **Op** | A unit of computation (like a function) |
| **Job** | A graph of ops wired together |
| **Asset** | A data artifact managed by Dagster (materialized by ops) |
| **Resource** | External dependency injected into ops (DB connections, APIs) |
| **IO Manager** | Controls how inputs/outputs are serialized and stored |
| **Schedule** | Time-based trigger for jobs |
| **Sensor** | Event-based trigger (new file, API event, etc.) |
| **Partition** | Logical subdivision of data (by date, region, etc.) |

### 10.2 Software-Defined Assets

Dagster's modern paradigm: define what data should exist (assets) rather than what to do (tasks).

```python
from dagster import asset, AssetIn, AssetKey, MetadataValue, Output
from dagster import FreshnessPolicy, AutoMaterializePolicy
import pandas as pd

@asset(
    description="Raw training data ingested from the data lake",
    group_name="ml_training",
    metadata={"source": "data_lake", "format": "parquet"},
    freshness_policy=FreshnessPolicy(maximum_lag_minutes=60 * 24),
    auto_materialize_policy=AutoMaterializePolicy.eager(),
)
def raw_training_data() -> Output[pd.DataFrame]:
    df = pd.read_parquet("s3://data-lake/training/latest.parquet")
    return Output(
        df,
        metadata={
            "num_rows": MetadataValue.int(len(df)),
            "columns": MetadataValue.json(list(df.columns)),
            "preview": MetadataValue.md(df.head().to_markdown()),
        },
    )

@asset(
    ins={"raw_data": AssetIn(key=AssetKey("raw_training_data"))},
    description="Validated and cleaned training data",
    group_name="ml_training",
)
def validated_training_data(raw_data: pd.DataFrame) -> Output[pd.DataFrame]:
    # Validation logic
    assert len(raw_data) > 1000, "Insufficient training data"
    assert raw_data.isnull().sum().sum() / raw_data.size < 0.05, "Too many nulls"
    cleaned = raw_data.dropna()
    return Output(cleaned, metadata={"num_rows": MetadataValue.int(len(cleaned))})

@asset(
    ins={"data": AssetIn(key=AssetKey("validated_training_data"))},
    description="Trained ML model",
    group_name="ml_training",
)
def trained_model(data: pd.DataFrame) -> Output:
    from sklearn.ensemble import RandomForestClassifier
    import pickle

    X = data.drop(columns=["target"])
    y = data["target"]
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    model_bytes = pickle.dumps(model)
    return Output(
        model_bytes,
        metadata={
            "algorithm": MetadataValue.text("RandomForest"),
            "n_features": MetadataValue.int(X.shape[1]),
        },
    )
```

### 10.3 Resources and IO Managers

```python
from dagster import resource, IOManager, io_manager
import boto3
import pickle

class S3ModelIOManager(IOManager):
    def __init__(self, bucket: str, prefix: str):
        self.bucket = bucket
        self.prefix = prefix
        self.s3 = boto3.client("s3")

    def handle_output(self, context, obj):
        key = f"{self.prefix}/{context.asset_key.path[-1]}/{context.run_id}.pkl"
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=pickle.dumps(obj))
        context.log.info(f"Saved to s3://{self.bucket}/{key}")

    def load_input(self, context):
        key = f"{self.prefix}/{context.asset_key.path[-1]}/latest.pkl"
        response = self.s3.get_object(Bucket=self.bucket, Key=key)
        return pickle.loads(response["Body"].read())

@io_manager(config_schema={"bucket": str, "prefix": str})
def s3_model_io_manager(init_context):
    return S3ModelIOManager(
        bucket=init_context.resource_config["bucket"],
        prefix=init_context.resource_config["prefix"],
    )
```

### 10.4 Jobs, Schedules, and Sensors

```python
from dagster import define_asset_job, ScheduleDefinition, sensor, RunRequest

# Job from assets
training_job = define_asset_job(
    name="training_job",
    selection=["raw_training_data", "validated_training_data", "trained_model"],
    tags={"team": "ml", "priority": "high"},
)

# Schedule
nightly_training_schedule = ScheduleDefinition(
    job=training_job,
    cron_schedule="0 2 * * *",
    execution_timezone="UTC",
)

# Sensor - trigger on new data
@sensor(job=training_job, minimum_interval_seconds=300)
def new_data_sensor(context):
    import boto3
    s3 = boto3.client("s3")
    response = s3.list_objects_v2(
        Bucket="ml-data", Prefix="incoming/", MaxKeys=1
    )
    if response.get("Contents"):
        latest = response["Contents"][0]["Key"]
        last_seen = context.cursor or ""
        if latest > last_seen:
            context.update_cursor(latest)
            yield RunRequest(run_key=latest, tags={"trigger": "new_data"})
```

### 10.5 Dagster Data-Driven Scheduling

```python
# Dagster sensor that checks data quality before triggering
@sensor(job=training_job, minimum_interval_seconds=600)
def data_quality_sensor(context):
    df = pd.read_parquet("s3://ml-data/latest/")
    if len(df) >= 10000 and df.isnull().mean().max() < 0.05:
        yield RunRequest(
            run_key=f"quality-check-{datetime.now().isoformat()}",
            run_config={"ops": {"ingest": {"config": {"path": "s3://ml-data/latest/"}}}},
        )
```

### 10.6 Dagster Retry Policy

```python
from dagster import RetryPolicy, Backoff, Jitter

@op(retry_policy=RetryPolicy(max_retries=3, delay=30, backoff=Backoff.EXPONENTIAL, jitter=Jitter.PLUS_MINUS))
def fetch_data():
    ...
```

---

## 11. ZenML Pipeline Abstraction

### 11.1 Core Concepts

ZenML provides a framework-agnostic abstraction that can target multiple orchestrators.

```python
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.integrations.sklearn.materializers import SklearnMaterializer

docker_settings = DockerSettings(
    required_integrations=["sklearn", "mlflow"],
    requirements=["pandas==2.1.0"],
)

@step
def ingest_data(source: str) -> pd.DataFrame:
    return pd.read_parquet(source)

@step
def train_model(data: pd.DataFrame, lr: float = 0.01) -> Output(model=..., metrics=dict):
    from sklearn.ensemble import GradientBoostingClassifier
    X, y = data.drop("target", axis=1), data["target"]
    model = GradientBoostingClassifier(learning_rate=lr)
    model.fit(X, y)
    return model, {"accuracy": model.score(X, y)}

@pipeline(settings={"docker": docker_settings})
def training_pipeline(source: str = "data/train.parquet", lr: float = 0.01):
    data = ingest_data(source=source)
    model, metrics = train_model(data=data, lr=lr)

# Run locally
training_pipeline()

# Run on Kubeflow
from zenml.integrations.kubeflow.flavors import KubeflowOrchestratorConfig
# (configured via ZenML stack)
```

### 11.2 ZenML Stack Concept

ZenML separates pipeline logic from infrastructure through stacks:

```bash
# Register components
zenml artifact-store register s3_store --flavor=s3 --path=s3://my-bucket
zenml orchestrator register kf_orch --flavor=kubeflow --kubernetes_context=my-cluster
zenml experiment-tracker register mlflow_tracker --flavor=mlflow --tracking_uri=http://mlflow:5000

# Assemble stack
zenml stack register production_stack \
    --orchestrator=kf_orch \
    --artifact-store=s3_store \
    --experiment-tracker=mlflow_tracker

# Activate stack
zenml stack set production_stack

# Same pipeline code now runs on Kubeflow
training_pipeline()
```

---

## 12. Parameter Passing Between Steps

### 12.1 Small Metadata (Metrics, Paths, Config)

| Framework | Mechanism | Limits |
|-----------|-----------|--------|
| Airflow | XCom | ~48KB default (DB-backed); unlimited with custom backend |
| Kubeflow | Component outputs | Serialized to artifact store |
| Prefect | Return values | Python objects passed in-memory or serialized |
| Dagster | Op outputs / IO Managers | Configurable serialization |
| ZenML | Step outputs / Materializers | Configurable per type |

### 12.2 Large Artifacts (Datasets, Models)

Always store in an external artifact store and pass references:

```python
# Pattern: Store artifact, pass URI
def train_model(**context):
    model = train(...)
    model_path = f"s3://models/{context['run_id']}/model.pkl"
    save_model(model, model_path)
    return {"model_uri": model_path, "accuracy": 0.95}

def deploy_model(**context):
    result = context["ti"].xcom_pull(task_ids="train_model")
    model = load_model(result["model_uri"])
    deploy(model)
```

### 12.3 Typed Parameter Passing (KFP, ZenML, Dagster)

```python
# KFP v2 - strongly typed
@dsl.component
def train(data: Input[Dataset], lr: float) -> Output[Model]:
    ...

# Dagster - typed IO
from dagster import In, Out

@op(
    ins={"data": In(dagster_type=pd.DataFrame)},
    out={"model": Out(dagster_type=bytes), "metrics": Out(dagster_type=dict)},
)
def train_op(data):
    ...
    yield Output(model_bytes, "model")
    yield Output(metrics_dict, "metrics")
```

---

## 13. Pipeline Versioning and Reproducibility

### 13.1 What to Version

| Artifact | How to Version |
|----------|----------------|
| Pipeline code | Git (branch, commit SHA, tag) |
| Pipeline config | Git or config management (Hydra, OmegaConf) |
| Data | DVC, Delta Lake, LakeFS, dataset version IDs |
| Model | Model registry (MLflow, Weights & Biases, SageMaker) |
| Environment | Docker images, conda lock files, pip freeze |
| Infrastructure | Terraform, Pulumi, CDK |

### 13.2 Reproducibility Checklist

```python
# Every pipeline run should log:
run_metadata = {
    "pipeline_version": get_git_sha(),
    "pipeline_branch": get_git_branch(),
    "data_version": data_hash_or_version,
    "config": pipeline_config,
    "docker_image": os.environ.get("DOCKER_IMAGE", "local"),
    "python_version": sys.version,
    "package_versions": {pkg.key: pkg.version for pkg in pkg_resources.working_set},
    "hardware": {
        "cpu_count": os.cpu_count(),
        "gpu": get_gpu_info(),
        "memory_gb": get_memory_gb(),
    },
    "random_seed": config.seed,
    "start_time": datetime.utcnow().isoformat(),
}
```

### 13.3 Configuration Management

```python
# Using Hydra for pipeline configuration
# config/training.yaml
"""
data:
  source: s3://ml-data/training
  version: "2024-01-15"
  test_split: 0.2

model:
  algorithm: random_forest
  hyperparams:
    n_estimators: 100
    max_depth: 10
    min_samples_split: 5

quality_gate:
  min_accuracy: 0.90
  min_f1: 0.85
  max_inference_latency_ms: 100

notifications:
  on_success: ["ml-team@company.com"]
  on_failure: ["ml-team@company.com", "on-call@company.com"]
"""
```

---

## 14. Dynamic Pipelines

### 14.1 Conditional Execution

```python
# Airflow - BranchPythonOperator
from airflow.operators.python import BranchPythonOperator

def choose_training_branch(**context):
    data_size = context["ti"].xcom_pull(task_ids="check_data_size")
    if data_size > 1_000_000:
        return "train_distributed"
    elif data_size > 10_000:
        return "train_single_gpu"
    else:
        return "train_cpu"

branch = BranchPythonOperator(
    task_id="choose_training_method",
    python_callable=choose_training_branch,
)

branch >> [train_distributed, train_single_gpu, train_cpu]
[train_distributed, train_single_gpu, train_cpu] >> evaluate

# Kubeflow - dsl.If / dsl.Condition
with dsl.If(eval_task.outputs["accuracy"] >= 0.95):
    deploy_to_production(model=train_task.outputs["model"])
with dsl.Elif(eval_task.outputs["accuracy"] >= 0.90):
    deploy_to_staging(model=train_task.outputs["model"])
with dsl.Else():
    notify_failure(metrics=eval_task.outputs["metrics"])
```

### 14.2 Dynamic Task Generation

```python
# Airflow 2.3+ - Dynamic Task Mapping
@task
def get_hyperparameter_grid() -> list[dict]:
    return [
        {"lr": 0.001, "layers": [64, 32]},
        {"lr": 0.01, "layers": [128, 64]},
        {"lr": 0.1, "layers": [256, 128, 64]},
    ]

@task
def train_candidate(params: dict) -> dict:
    model = train_with_params(params)
    return {"params": params, "score": model.score}

@task
def select_champion(candidates: list[dict]) -> dict:
    return max(candidates, key=lambda c: c["score"])

with DAG("hyperparameter_search", ...):
    grid = get_hyperparameter_grid()
    candidates = train_candidate.expand(params=grid)
    champion = select_champion(candidates)
```

```python
# Prefect - dynamic task mapping
@flow
def hyperparameter_search():
    grid = generate_grid()
    results = train_candidate.map(grid)  # Parallel execution
    best = select_best([r.result() for r in results])
```

---

## 15. Pipeline Caching and Artifact Reuse

### 15.1 Task-Level Caching

```python
# Prefect - input-based caching
from prefect.tasks import task_input_hash

@task(
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=24),
)
def expensive_feature_computation(data_path: str, config: dict) -> str:
    # Only re-runs if data_path or config change
    ...

# Custom cache key
def data_version_cache_key(context, parameters):
    """Cache based on data version, not content."""
    data_path = parameters["data_path"]
    version = get_data_version(data_path)
    return f"features-{version}-{hash(frozenset(parameters['config'].items()))}"

@task(cache_key_fn=data_version_cache_key, cache_expiration=timedelta(days=7))
def compute_features(data_path: str, config: dict) -> str:
    ...
```

```python
# Kubeflow - automatic caching (enabled by default)
# Disable for specific steps:
train_task.set_caching_options(False)

# Airflow - manual caching via artifact store
def train_if_needed(**context):
    config_hash = hash_config(context["params"])
    cached_model = check_cache(f"models/{config_hash}")
    if cached_model:
        return cached_model
    model = train(...)
    save_to_cache(model, f"models/{config_hash}")
    return model
```

### 15.2 Cross-Pipeline Artifact Reuse

```python
# Pattern: Feature store as shared cache between pipelines
# Feature pipeline writes to feature store
@task
def materialize_features(data: pd.DataFrame) -> None:
    feature_store.materialize(
        features=compute_features(data),
        entity="customer",
        version=get_data_version(),
    )

# Training pipeline reads from feature store
@task
def get_training_features(entity_ids: list, feature_list: list) -> pd.DataFrame:
    return feature_store.get_features(
        entities=entity_ids,
        features=feature_list,
        as_of=datetime.utcnow(),
    )
```

---

## 16. Environment Management

### 16.1 Container-Based Isolation

```python
# Airflow - KubernetesPodOperator per-task environments
preprocess_task = KubernetesPodOperator(
    task_id="preprocess",
    image="ml-preprocess:v1.2",  # Pandas, NumPy
    ...
)

train_task = KubernetesPodOperator(
    task_id="train",
    image="ml-training:v3.0",  # PyTorch, CUDA
    resources={"limits": {"nvidia.com/gpu": "1"}},
    ...
)

eval_task = KubernetesPodOperator(
    task_id="evaluate",
    image="ml-evaluation:v2.1",  # Scikit-learn, SHAP
    ...
)

# Kubeflow - per-component base images
@dsl.component(base_image="pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime")
def train_pytorch(data: Input[Dataset]) -> Output[Model]:
    ...
```

### 16.2 Virtual Environment Isolation

```python
# Airflow - PythonVirtualenvOperator
from airflow.operators.python import PythonVirtualenvOperator

train = PythonVirtualenvOperator(
    task_id="train_model",
    python_callable=train_fn,
    requirements=["scikit-learn==1.3.0", "pandas==2.1.0", "mlflow==2.8.0"],
    python_version="3.11",
    system_site_packages=False,
)

# Prefect - infrastructure blocks
from prefect.infrastructure import DockerContainer

docker_block = DockerContainer(
    image="ml-training:latest",
    image_pull_policy="ALWAYS",
    auto_remove=True,
    env={"CUDA_VISIBLE_DEVICES": "0"},
)
docker_block.save("ml-gpu-container")
```

---

## 17. Pipeline Templates and Reusable Components

### 17.1 Template Pattern

```python
# Parameterized pipeline factory
from typing import Callable, Optional

def create_training_pipeline(
    name: str,
    data_loader: Callable,
    feature_engineer: Callable,
    model_trainer: Callable,
    evaluator: Callable,
    quality_gate: Optional[Callable] = None,
    notifier: Optional[Callable] = None,
) -> DAG:
    """Factory function to create standardized training pipelines."""
    with DAG(
        dag_id=f"training_{name}",
        default_args=STANDARD_DEFAULT_ARGS,
        schedule_interval="0 2 * * *",
        catchup=False,
        tags=["ml", "training", name],
    ) as dag:
        load_task = PythonOperator(
            task_id="load_data", python_callable=data_loader
        )
        feature_task = PythonOperator(
            task_id="engineer_features", python_callable=feature_engineer
        )
        train_task = PythonOperator(
            task_id="train_model", python_callable=model_trainer
        )
        eval_task = PythonOperator(
            task_id="evaluate_model", python_callable=evaluator
        )

        load_task >> feature_task >> train_task >> eval_task

        if quality_gate:
            gate_task = PythonOperator(
                task_id="quality_gate", python_callable=quality_gate
            )
            eval_task >> gate_task

        if notifier:
            notify_task = PythonOperator(
                task_id="notify", python_callable=notifier,
                trigger_rule=TriggerRule.ALL_DONE,
            )
            (gate_task if quality_gate else eval_task) >> notify_task

    return dag

# Instantiate for different models
churn_pipeline = create_training_pipeline(
    name="churn_model",
    data_loader=load_churn_data,
    feature_engineer=churn_features,
    model_trainer=train_xgboost,
    evaluator=evaluate_classification,
    quality_gate=check_churn_thresholds,
    notifier=slack_notification,
)
```

### 17.2 Reusable Component Library (Kubeflow)

```python
# components/data_validation/component.yaml
name: Data Validation
description: Validates a dataset against a schema using Great Expectations
inputs:
  - name: dataset
    type: Dataset
  - name: schema_path
    type: String
outputs:
  - name: validation_report
    type: Artifact
  - name: is_valid
    type: Boolean
implementation:
  container:
    image: ml-components/data-validation:v1.0
    command: [python, validate.py]
    args:
      - --dataset
      - {inputPath: dataset}
      - --schema
      - {inputValue: schema_path}
      - --report-path
      - {outputPath: validation_report}
```

---

## 18. CI/CD for Pipeline Code

### 18.1 Pipeline CI/CD Workflow

```yaml
# .github/workflows/ml-pipeline-ci.yml
name: ML Pipeline CI/CD

on:
  push:
    paths: ["pipelines/**", "components/**"]
  pull_request:
    paths: ["pipelines/**", "components/**"]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements-dev.txt

      - name: Lint pipeline code
        run: |
          ruff check pipelines/ components/
          mypy pipelines/ components/

      - name: Run unit tests
        run: pytest tests/unit/ -v --cov=pipelines --cov-report=xml

      - name: Validate DAGs
        run: python -c "
          from airflow.models import DagBag;
          bag = DagBag('pipelines/dags/', include_examples=False);
          assert not bag.import_errors, bag.import_errors
          "

      - name: Run integration tests
        run: pytest tests/integration/ -v -m "not slow"

  deploy-staging:
    needs: lint-and-test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Deploy DAGs to staging
        run: |
          aws s3 sync pipelines/dags/ s3://airflow-staging-dags/dags/
          # Or for Prefect:
          # prefect deploy --all --work-pool staging-pool

  deploy-production:
    needs: deploy-staging
    if: github.ref == 'refs/heads/main'
    environment: production
    runs-on: ubuntu-latest
    steps:
      - name: Run smoke tests on staging
        run: pytest tests/smoke/ -v

      - name: Deploy DAGs to production
        run: |
          aws s3 sync pipelines/dags/ s3://airflow-production-dags/dags/
```

### 18.2 Pipeline Promotion Strategy

```
Feature Branch -> PR (lint + unit tests + DAG validation)
    -> main (integration tests -> deploy to staging)
    -> Staging (smoke tests -> manual approval)
    -> Production (gradual rollout -> monitoring)
```

---

## 19. Pipeline Migration Code Patterns

### 19.1 Abstraction Layer Strategy

Build a thin abstraction over pipeline logic so the orchestrator becomes a deployment detail:

```python
# pipeline_core.py - Orchestrator-agnostic business logic
class TrainingStep:
    def ingest_data(self, source: str) -> str:
        """Returns path to ingested data."""
        ...

    def validate_data(self, data_path: str) -> dict:
        """Returns validation report."""
        ...

    def train_model(self, data_path: str, config: dict) -> dict:
        """Returns model path and metrics."""
        ...

# adapters/airflow_adapter.py
from pipeline_core import TrainingStep

steps = TrainingStep()

with DAG("training", ...) as dag:
    t1 = PythonOperator(task_id="ingest", python_callable=steps.ingest_data, ...)
    t2 = PythonOperator(task_id="validate", python_callable=steps.validate_data, ...)
    t3 = PythonOperator(task_id="train", python_callable=steps.train_model, ...)
    t1 >> t2 >> t3

# adapters/prefect_adapter.py
from pipeline_core import TrainingStep

steps = TrainingStep()

@flow
def training():
    data = steps.ingest_data("s3://...")
    report = steps.validate_data(data)
    result = steps.train_model(data, config)
```

### 19.2 Migration Checklist

| From -> To | Key Changes |
|------------|-------------|
| Airflow -> Prefect | Replace DAGs with flows, Operators with tasks, XCom with return values, Connections with Blocks |
| Airflow -> Dagster | Replace DAGs with jobs/asset graphs, Operators with ops/assets, XCom with IO managers |
| Airflow -> Kubeflow | Replace operators with KFP components, DAGs with pipeline DSL, connections with K8s secrets |
| Prefect -> Dagster | Replace flows with jobs, tasks with ops/assets, blocks with resources |
| Any -> ZenML | Wrap step logic in `@step`, wire in `@pipeline`, configure stack for target infra |
