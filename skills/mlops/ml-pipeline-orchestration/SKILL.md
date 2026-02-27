---
name: ml-pipeline-orchestration
description: "ML pipeline orchestration skill covering end-to-end workflow automation with Apache Airflow DAGs, Kubeflow Pipelines, Prefect flows, Dagster ops and assets, ZenML abstractions, and Argo Workflows. Includes pipeline scheduling (cron, event-driven, data-driven triggers), task dependency management, fan-out/fan-in patterns, parameter passing via XComs and artifacts, pipeline retry and error handling, pipeline monitoring and alerting, dynamic conditional branching, pipeline caching, artifact reuse, pipeline versioning, reproducibility, CI/CD integration, reusable pipeline templates, and orchestrator migration strategies for production ML workflow automation."
license: Apache-2.0
metadata:
  author: mlops-skills
  version: "1.0"
---

# ML Pipeline Orchestration

Guide to designing, building, scheduling, and operating ML pipelines across major orchestration frameworks. Covers platform-agnostic patterns and framework-specific implementations for Apache Airflow, Kubeflow Pipelines, Prefect, Dagster, and ZenML.

For framework-specific deep dives (Kubeflow, Prefect, Dagster, ZenML), parameter passing, caching, environment management, templates, CI/CD, and migration guides, see `references/REFERENCE.md`.

---

## 1. ML Pipeline Design Patterns

### 1.1 Training Pipeline

The most common ML pipeline. Orchestrates the path from raw data to a registered, validated model.

```
Data Ingestion -> Data Validation -> Feature Engineering -> Data Splitting
    -> Model Training -> Model Evaluation -> Model Registration -> Notification
```

**Design principles:**
- Each stage is independently testable and idempotent.
- Artifacts (datasets, models, metrics) are persisted in an artifact store (S3, GCS, ADLS).
- The pipeline is parameterized: dataset version, hyperparameters, model type, target environment.
- A gate between evaluation and registration enforces quality thresholds.

### 1.2 Other Pipeline Types

| Pipeline Type | Flow | Key Consideration |
|---------------|------|-------------------|
| **Inference (Batch)** | Load Model -> Fetch Data -> Predict -> Store Results -> Monitor Drift | Feature engineering MUST match training; log predictions for monitoring |
| **Feature Pipeline** | Ingest Sources -> Transform -> Validate -> Write to Feature Store | Schedule aligned with freshness requirements; point-in-time correctness |
| **Monitoring Pipeline** | Collect Predictions + Actuals -> Compute Drift -> Check Thresholds -> Alert | Runs independently; can trigger retraining (closed-loop MLOps) |
| **Composite / Meta-Pipeline** | Orchestrates other pipelines in sequence | Example: nightly feature -> training -> deploy -> monitor |

---

## 2. Apache Airflow for ML

### 2.1 Core Concepts

| Concept | ML Usage |
|---------|----------|
| **DAG** | ML pipeline as a directed acyclic graph of tasks |
| **Operator** | Single step (PythonOperator, KubernetesPodOperator) |
| **Sensor** | Waits for external conditions (new data in S3, model registry update) |
| **XCom** | Passes small metadata between tasks (metrics, file paths, model URIs) |
| **Connection** | Stores credentials for external systems |
| **Dataset** | (Airflow 2.4+) Enables data-aware scheduling |

### 2.2 DAG Definition Best Practices

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta

default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "email_on_failure": True,
    "email": ["ml-alerts@company.com"],
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=30),
    "execution_timeout": timedelta(hours=2),
}

with DAG(
    dag_id="ml_training_pipeline",
    default_args=default_args,
    description="End-to-end ML training pipeline",
    schedule_interval="0 2 * * *",  # Daily at 2 AM
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "training", "production"],
) as dag:
    pass
```

**Anti-patterns to avoid:**
- Storing large objects (DataFrames, models) in XCom -- use artifact stores and pass URIs.
- Top-level heavy code that runs at DAG parse time -- defer to task callables.
- Excessive `depends_on_past=True` -- creates cascading failures.
- Hard-coded paths and credentials -- use Connections and Variables.

### 2.3 Key Operators for ML

```python
# PythonOperator - most common for ML tasks
train_task = PythonOperator(
    task_id="train_model",
    python_callable=train_model_fn,
    op_kwargs={"hyperparams": "{{ var.json.hyperparams }}"},
)

# KubernetesPodOperator - for GPU training or isolated environments
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator

gpu_train = KubernetesPodOperator(
    task_id="gpu_training",
    name="gpu-training-pod",
    namespace="ml-workloads",
    image="ml-training:latest",
    arguments=["--epochs", "100", "--lr", "0.001"],
    resources={
        "requests": {"cpu": "4", "memory": "16Gi", "nvidia.com/gpu": "1"},
        "limits": {"cpu": "8", "memory": "32Gi", "nvidia.com/gpu": "1"},
    },
    is_delete_operator_pod=True,
    get_logs=True,
)

# S3Sensor - wait for new data
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor

wait_for_data = S3KeySensor(
    task_id="wait_for_new_data",
    bucket_name="ml-data",
    bucket_key="incoming/{{ ds }}/data.parquet",
    poke_interval=300,
    timeout=3600,
    mode="reschedule",
)
```

### 2.4 XCom for Metadata Passing

```python
def train_model_fn(**context):
    metrics = {"accuracy": 0.95, "f1": 0.93, "model_uri": "s3://models/v42"}
    context["ti"].xcom_push(key="training_metrics", value=metrics)
    return metrics  # Also pushed automatically with key "return_value"

def evaluate_model_fn(**context):
    metrics = context["ti"].xcom_pull(task_ids="train_model", key="training_metrics")
    model_uri = metrics["model_uri"]
```

**XCom size limits:** Default backend stores in the metadata DB (~48KB). For larger payloads, use a custom XCom backend backed by S3/GCS.

See `scripts/airflow_pipeline.py` for a complete, executable Airflow DAG.

---

## 3. Orchestrator Comparison

| Criterion | Airflow | Kubeflow | Prefect | Dagster | ZenML |
|-----------|---------|----------|---------|---------|-------|
| **Best for** | General workflow orchestration | Kubernetes-native ML | Python-native ML pipelines | Data asset management | Multi-orchestrator abstraction |
| **Learning curve** | Moderate | Steep | Low | Moderate | Low |
| **K8s required** | No | Yes | No | No | No |
| **Dynamic pipelines** | Good (2.3+) | Limited | Excellent | Good | Good |
| **Data lineage** | Limited | Limited | Moderate | Excellent | Good |
| **Managed offering** | Astronomer, MWAA, Composer | GCP Vertex, AWS native | Prefect Cloud | Dagster Cloud | ZenML Cloud |
| **GPU support** | Via K8s/Docker operators | Native | Via infra blocks | Via resources | Via stack config |

For detailed framework-specific guides (Kubeflow, Prefect, Dagster, ZenML), see `references/REFERENCE.md` sections 8-11.

---

## 4. Pipeline Scheduling Strategies

### 4.1 Cron-Based Scheduling

```
0 2 * * *          # Daily at 2 AM (nightly retraining)
0 */6 * * *        # Every 6 hours (frequent retraining)
0 2 * * 1          # Weekly on Monday at 2 AM
*/30 * * * *       # Every 30 minutes (feature pipeline)
```

### 4.2 Event-Driven Scheduling

```python
# Airflow - Dataset-aware scheduling (Airflow 2.4+)
from airflow.datasets import Dataset

data_landing = Dataset("s3://ml-data/incoming/")

# Producer DAG
with DAG("data_producer", ...):
    task = PythonOperator(
        task_id="produce_data",
        python_callable=produce_data_fn,
        outlets=[data_landing],
    )

# Consumer DAG - triggered when dataset is updated
with DAG("training_pipeline", schedule=[data_landing], ...):
    pass
```

### 4.3 Hybrid Scheduling

Combine strategies: run on a cron schedule but skip if no new data.

```python
with DAG("smart_training", schedule_interval="0 */4 * * *", ...):
    check = ShortCircuitOperator(
        task_id="check_data",
        python_callable=check_new_data,
    )
    train = PythonOperator(task_id="train", ...)
    check >> train  # train only runs if check returns True
```

---

## 5. Task Dependency Management

### 5.1 Linear Dependencies

```python
# Airflow
ingest >> validate >> feature_eng >> train >> evaluate >> register

# Prefect - implicit via function calls
data = ingest()
validated = validate(data)
model = train(engineer(validated))

# Dagster - implicit via asset dependencies
@asset
def features(raw_data): ...  # Depends on raw_data automatically
```

### 5.2 Fan-Out / Fan-In (Parallel Branches)

```python
# Airflow - fan-out to multiple evaluation tasks, fan-in to registration
train_task >> [eval_accuracy, eval_fairness, eval_latency] >> register_task

# Airflow - dynamic fan-out with mapped tasks (Airflow 2.3+)
@task
def get_model_configs() -> list[dict]:
    return [{"algorithm": "rf"}, {"algorithm": "xgb"}, {"algorithm": "lgbm"}]

@task
def train_single_model(config: dict) -> dict:
    return {"config": config, "accuracy": 0.95}

with DAG("ensemble_training", ...):
    configs = get_model_configs()
    results = train_single_model.expand(config=configs)  # Dynamic fan-out
    best = select_best_model(results)  # Fan-in
```

### 5.3 Cross-DAG Dependencies

```python
# Airflow - TriggerDagRunOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

trigger_deployment = TriggerDagRunOperator(
    task_id="trigger_deployment",
    trigger_dag_id="model_deployment_pipeline",
    conf={"model_uri": "{{ ti.xcom_pull(task_ids='register') }}"},
    wait_for_completion=True,
)
```

---

## 6. Error Handling, Retries, and Failure Notifications

### 6.1 Retry Strategies

```python
# Airflow - per-task retry configuration
PythonOperator(
    task_id="train_model",
    python_callable=train_fn,
    retries=3,
    retry_delay=timedelta(minutes=5),
    retry_exponential_backoff=True,
    max_retry_delay=timedelta(hours=1),
    on_retry_callback=lambda context: log_retry(context),
)
```

### 6.2 Custom Failure Callbacks

```python
def on_failure_callback(context):
    task_id = context["task_instance"].task_id
    dag_id = context["dag"].dag_id
    exception = context.get("exception", "Unknown")
    log_url = context["task_instance"].log_url
    message = f"Task FAILED: {dag_id}.{task_id}\nException: {exception}\nLog: {log_url}"
    send_slack_alert(channel="#ml-alerts", message=message)

with DAG("training_pipeline", on_failure_callback=on_failure_callback, ...):
    ...
```

### 6.3 Graceful Degradation

```python
# Pattern: Fallback to previous model if training fails
@task
def training_with_fallback(data_path: str, config: dict) -> str:
    try:
        model = train_new_model(data_path, config)
        if validate_model(model):
            return save_model(model)
        else:
            return get_current_production_model_uri()
    except Exception as e:
        logger.error(f"Training failed: {e}, falling back to current model")
        return get_current_production_model_uri()
```

---

## 7. Pipeline Testing

### 7.1 Unit Testing Pipeline Steps

```python
def test_feature_engineering():
    input_df = pd.DataFrame({"age": [25, 30], "income": [50000, 75000], "target": [0, 1]})
    result = engineer_features_fn(input_df)
    assert "age_income_ratio" in result.columns
    assert not result.isnull().any().any()

def test_data_validation():
    bad_df = pd.DataFrame({"age": [25, None], "income": [50000, -1]})
    report = validate_data_fn(bad_df)
    assert not report["passed"]
```

### 7.2 DAG Validation Testing (Airflow)

```python
def test_dag_loading():
    from airflow.models import DagBag
    dag_bag = DagBag(dag_folder="dags/", include_examples=False)
    assert len(dag_bag.import_errors) == 0

def test_dag_structure():
    dag_bag = DagBag(dag_folder="dags/", include_examples=False)
    dag = dag_bag.get_dag("ml_training_pipeline")
    assert dag is not None
    assert "train_model" in dag.task_ids
```

For comprehensive testing strategies (integration, contract, performance tests), see `references/REFERENCE.md` Section 4.

---

## 8. Pipeline Monitoring and Alerting

### 8.1 Metrics to Monitor

| Category | Metrics |
|----------|---------|
| **Pipeline Health** | Success rate, failure rate, SLA compliance |
| **Task Performance** | Duration per task, duration trends, queue time |
| **Resource Usage** | CPU, memory, GPU utilization per task |
| **Data Quality** | Schema changes, null rates, distribution drift |
| **Model Quality** | Training metrics over time, evaluation score trends |

### 8.2 Alerting Rules

```yaml
# Prometheus alerting rules
groups:
  - name: ml_pipeline_alerts
    rules:
      - alert: PipelineFailureRate
        expr: |
          rate(airflow_dag_run_duration_failed_total{dag_id="training_pipeline"}[1h])
          / rate(airflow_dag_run_duration_total{dag_id="training_pipeline"}[1h]) > 0.3
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "Training pipeline failure rate > 30%"

      - alert: PipelineSLAMiss
        expr: |
          airflow_dag_run_duration_seconds{dag_id="training_pipeline", state="running"} > 7200
        for: 5m
        labels:
          severity: warning
```

---

## Quick Reference

| Topic | Key Guidance |
|-------|-------------|
| **Scheduling** | Use cron for stable patterns, event-driven for irregular data, hybrid for cost efficiency |
| **Dependencies** | Linear for simple flows, fan-out/fan-in for parallel work, cross-DAG for pipeline chaining |
| **Data passing** | Small metadata via XCom/returns; large artifacts via S3/GCS URIs |
| **Error handling** | Exponential backoff retries, failure callbacks to Slack/PagerDuty, graceful degradation |
| **Testing** | Unit test steps, validate DAG structure, integration test with synthetic data |
| **Monitoring** | Track success rates, task durations, SLA compliance; alert on anomalies |
| **Caching** | Input-hash caching (Prefect, KFP native); manual for Airflow |
| **Environment** | Container-per-step (KubernetesPodOperator, KFP components) for isolation |
| **CI/CD** | Lint + unit test + DAG validation in PR; deploy to staging then production |
| **Migration** | Extract business logic first; run old/new in parallel; migrate incrementally |

---

## Scripts

- **`scripts/airflow_pipeline.py`** - Complete Airflow DAG for an end-to-end ML training pipeline with data validation, feature engineering, model training, evaluation, and conditional registration.
- **`scripts/prefect_pipeline.py`** - Complete Prefect flow with cached tasks, retries, deployment configuration, and parameterized execution.

## References

- **`references/REFERENCE.md`** - Detailed orchestrator comparisons, framework-specific deep dives (Kubeflow, Prefect, Dagster, ZenML), design pattern catalog, parameter passing, versioning, dynamic pipelines, caching, environment management, templates, CI/CD, testing strategies, migration guides, production checklist, and troubleshooting.
