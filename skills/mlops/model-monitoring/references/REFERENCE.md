# Model Monitoring Reference Guide

## Monitoring Tools Comparison

| Feature              | Evidently           | Whylogs             | NannyML             | Arize               | Fiddler             |
|----------------------|---------------------|---------------------|---------------------|---------------------|---------------------|
| Open Source          | Yes                 | Yes                 | Yes (core)          | No                  | No                  |
| Data Drift Detection| PSI, KS, Wasserstein| Statistical profiling| Multivariate drift  | PSI, KL, JS, KS    | JSD, PSI, KS       |
| No-label Monitoring | Limited             | No                  | Yes (best-in-class) | Yes                 | Yes                 |
| Real-time Support   | Batch-oriented      | Streaming profiles  | Batch-oriented      | Real-time + batch   | Real-time + batch   |
| Dashboards          | HTML reports, UI    | WhyLabs platform    | Built-in UI         | Full platform       | Full platform       |
| Explainability      | No                  | No                  | No                  | SHAP integration    | Built-in XAI        |
| Cost                | Free                | Free / paid SaaS    | Free / paid         | Paid                | Paid                |

### When to Choose What

- **Evidently**: Best starting point; rich open-source reports, excellent for batch monitoring.
- **Whylogs**: Best for high-volume streaming data profiling with minimal overhead.
- **NannyML**: Unique capability to estimate performance without ground truth labels (CBPE).
- **Arize**: Enterprise-grade real-time monitoring with root cause analysis.
- **Fiddler**: Strong explainability, good for regulated industries needing transparency.

## What to Monitor

### Data Quality

| Metric                   | Description                                 | Alert Threshold Example       |
|--------------------------|---------------------------------------------|-------------------------------|
| Missing value rate       | % of null/NaN values per feature            | > 5% or 2x increase          |
| Schema violations        | Unexpected types, new categories            | Any violation                 |
| Volume anomalies         | Prediction request count changes            | +/- 30% from rolling average  |
| Feature range violations | Values outside training distribution        | > 1% out-of-range             |
| Cardinality changes      | Unique values for categorical features      | Change > 20% from baseline    |

### Model Performance

Track classification metrics (accuracy, precision, recall, F1, AUC-ROC, calibration), regression metrics (MAE, RMSE, MAPE, R-squared), and ranking metrics (NDCG@k, MRR, CTR) over sliding time windows.

### Feature Drift Detection

| Method                         | Type             | Best For                       | Sensitivity  |
|-------------------------------|------------------|--------------------------------|-------------|
| Population Stability Index    | Binned           | Production monitoring standard | Medium      |
| Kolmogorov-Smirnov (KS)      | Non-parametric   | Continuous features            | High        |
| Jensen-Shannon Divergence     | Information      | Probability distributions      | Medium      |
| Wasserstein Distance          | Optimal transport| Continuous, sensitive to shifts | High        |
| Chi-squared Test              | Statistical      | Categorical features           | Medium      |
| Page-Hinkley Test             | Sequential       | Streaming / online detection   | Configurable|

### Prediction Drift

Monitor model output distributions over time, independent of ground truth. Prediction drift often precedes performance degradation.

```python
# Evidently 0.7+ (TargetDriftPreset is removed; scope DataDriftPreset
# to the prediction column instead)
from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset

definition = DataDefinition(numerical_columns=["prediction"])
reference = Dataset.from_pandas(reference_df, data_definition=definition)
current = Dataset.from_pandas(current_df, data_definition=definition)

report = Report([DataDriftPreset(columns=["prediction"])])
snapshot = report.run(current, reference)
snapshot.save_html("prediction_drift_report.html")
```

## Alerting Strategies and Thresholds

### Tiered Alerting

| Severity | Condition                                    | Response Time | Channel          |
|----------|----------------------------------------------|---------------|------------------|
| Critical | Serving errors > 1%, monitoring failure      | 15 min        | PagerDuty, SMS   |
| High     | Performance drop > 5%, PSI > 0.2             | 1 hour        | Slack, email     |
| Medium   | Feature drift (PSI 0.1-0.2), volume anomaly  | 4 hours       | Slack channel    |
| Low      | Minor shift (PSI < 0.1), slight latency rise | Next day      | Dashboard, digest|

### PSI Interpretation

| PSI Value  | Interpretation               | Action                              |
|------------|------------------------------|-------------------------------------|
| < 0.1      | No significant change        | No action needed                    |
| 0.1 - 0.2 | Moderate shift               | Investigate, consider retraining    |
| 0.2 - 0.5 | Significant shift            | Retrain, root cause analysis        |
| > 0.5     | Severe distribution change   | Immediate investigation, fallback   |

Note: 0.1/0.2 is the credit-risk industry convention (used consistently with the model-drift-detection skill); some references use 0.25 as the "significant" cutoff. Pick one convention and apply it uniformly.

```yaml
# Alert configuration example
alerts:
  drift:
    feature_drift_psi: { warning: 0.1, critical: 0.2 }
    prediction_drift_ks: { warning: 0.05, critical: 0.1 }
  performance:
    accuracy_drop: { warning: 0.02, critical: 0.05 }
    latency_p99_ms: { warning: 200, critical: 500 }
  volume:
    request_count: { low_warning_pct: -30, high_warning_pct: 50 }
```

## Monitoring Dashboard Design

### Key Panels

**Overview**: Model health status (green/yellow/red), prediction volume, overall metric trend.

**Drift**: PSI heatmap per feature over time, top drifted features, prediction distribution overlay.

**Performance**: Primary metric trend with confidence intervals, per-segment breakdown.

**Data Quality**: Missing value rates, schema validation pass/fail, feature distribution histograms.

**Operational**: Request latency (p50/p95/p99), error rates, model version, resource utilization.

```
+---------------------------+---------------------------+
|     MODEL HEALTH          |    PREDICTION VOLUME      |
+---------------------------+---------------------------+
|     FEATURE DRIFT         |    MODEL PERFORMANCE      |
|   [PSI Heatmap]           |   [Metric Trend + CI]     |
+---------------------------+---------------------------+
|     DATA QUALITY          |    OPERATIONAL METRICS    |
|   [Missing Rate Trends]   |   [Latency Distribution]  |
+---------------------------+---------------------------+
```

## Performance Degradation Root Cause Analysis

### Systematic Workflow

1. **Check Data Quality** -- Missing values increased? Schema changes? Volume anomalies?
2. **Check Feature Drift** -- Which features drifted most? Gradual or sudden? External events?
3. **Check Prediction Drift** -- Output distribution shifted? Certain classes affected more?
4. **Check Upstream Systems** -- Pipeline changes? Feature engineering code changes?
5. **Segment Analysis** -- Which user segments affected? Geographic/temporal patterns?
6. **Remediate** -- Short-term: rollback or rule override. Medium: retrain. Long: fix root cause.

### Common Root Causes

| Symptom                    | Likely Cause                          | Quick Fix                      |
|---------------------------|---------------------------------------|--------------------------------|
| All features drifted      | Upstream data pipeline change         | Contact data engineering       |
| Single feature drifted    | Feature engineering bug               | Fix feature, retrain           |
| Sudden performance drop   | Data pipeline failure, schema change  | Rollback, fix pipeline         |
| Gradual performance decay | Concept drift, changing behavior      | Scheduled retraining           |
| Performance drop on segment| New user segment                     | Add segment to training data   |

## Ground Truth Delay Handling

| Domain              | Typical Delay        | Strategy                          |
|---------------------|---------------------|-----------------------------------|
| Fraud detection     | 30-90 days          | Proxy labels, NannyML CBPE        |
| Credit risk         | 6-24 months         | Early indicators, cohort analysis  |
| Recommendations     | Minutes to hours    | Implicit feedback (clicks, views)  |
| Churn prediction    | 30-90 days          | Early engagement signals           |

### Strategies

**CBPE (Confidence-Based Performance Estimation):**
```python
import nannyml as nml
estimator = nml.CBPE(y_pred_proba="pred_proba", y_pred="prediction",
    y_true="target", metrics=["roc_auc"], problem_type="classification_binary")
estimator.fit(reference_df)
results = estimator.estimate(analysis_df)  # no labels needed
```

**Proxy Labels**: Use approximate labels available sooner (e.g., "reported fraud" in 7 days vs "confirmed fraud" in 90 days). Track proxy-to-true-label correlation.

**Cohort Backtesting**: When labels arrive, compare actual performance to drift metrics from that period to calibrate thresholds.

## Monitoring Infrastructure Patterns

### Batch Monitoring

```
Prediction Logs --> Data Warehouse --> Scheduled Job (Airflow) --> Reports/Alerts
```

Lower cost, simpler. Acceptable when hours of detection delay is tolerable. Tools: Evidently + Airflow.

### Streaming Monitoring

```
Prediction Events (Kafka) --> Stream Processor (Flink) --> Real-time Metrics --> Alerts
```

Sub-minute detection. Higher cost and complexity. Required for high-stakes systems. Tools: Whylogs, Arize.

### Hybrid (Recommended)

Stream critical operational metrics (latency, errors, volume). Batch deeper statistical analysis (drift, performance, fairness). Balance cost and speed based on business requirements.

## Common Pitfalls

1. **Monitoring only accuracy**: Track data quality, drift, and operational metrics alongside performance.
2. **Static thresholds**: Adapt alert thresholds to seasonal patterns and business cycles.
3. **Alert fatigue**: Start with fewer, high-confidence alerts. Tune thresholds based on false alert rates.
4. **Ignoring reference data staleness**: Update reference dataset periodically as the world changes.
5. **Not monitoring the monitor**: Ensure monitoring pipelines themselves have health checks.
6. **One-size-fits-all drift detection**: Different features need different statistical tests.

## Production Runbooks

Concrete incident playbooks wired to this repo's scripts. Severity levels follow the
Tiered Alerting table above; rollback always means the model-registry skill's alias flip
(`registry_manager.py --action promote --alias champion --version <previous>`), which
re-points serving without a redeploy.

### Runbook 1: Drift Alert Fired

**Severity**: PSI 0.1-0.2 on features = Medium; PSI > 0.2 on features or any prediction
drift = High; PSI > 0.5 = Critical.

**Triage**
1. Confirm and localize with the model-drift-detection skill:
   `python detect_drift.py --reference ref.parquet --current current.parquet --tests psi ks`
   — which features, how far over threshold?
2. Rule out data-quality causes first (missing-value spike, schema change, volume anomaly) —
   an upstream pipeline bug looks like "all features drifted at once".
3. Check prediction drift and performance:
   `python scripts/monitor_model.py --reference ref.parquet --current current.parquet
   --target label --thresholds thresholds.json --report drift_incident.json`.
   If labels are delayed, use proxy labels or NannyML CBPE (see Ground Truth Delay Handling).
4. Correlate with external events (campaign launch, seasonality, new user segment).

**Escalate when**: prediction distribution shifted significantly (mean shift > 2 std),
performance metric violates thresholds, or PSI > 0.5 on a top-importance feature.

**Remediate**: single-feature drift → fix feature pipeline, retrain. Broad genuine drift →
trigger retraining on recent data, register and gate the new version, promote via alias.
Performance already degraded and no fixed candidate → **rollback** (alias flip above).

### Runbook 2: Endpoint Error-Rate / 5xx Spike

**Severity**: error rate > 1% or endpoint down = Critical (15 min response);
0.1-1% or latency p99 > 500 ms = High.

**Triage**
1. Check the serving metrics endpoint (`GET /metrics` on `serve_model.py`):
   `model_serving_error_rate`, `model_serving_errors_total`, latency p50/p95/p99.
2. `GET /health` — is the process up and the model loaded? Check pod restarts / OOM kills
   and recent deploys (`kubectl get events`, deployment history).
3. Classify errors from serving logs: 400s (malformed client input / schema change
   upstream) vs 500s (prediction exceptions, resource exhaustion).
4. If the spike started at a model rollout: compare A/B legs (`model_used` in responses)
   to isolate whether only the new model errors.

**Escalate when**: errors persist > 15 min, affect > 1% of traffic, or coincide with a
model/infra change you cannot revert yourself.

**Remediate**: bad rollout → **rollback via alias flip** and restart serving (or set
`--ab-ratio 0` to drain the B leg). Resource exhaustion → scale replicas / raise limits.
Client-side 400s → notify the upstream team; do not "fix" by loosening input validation.

### Runbook 3: Data-Quality Alert

**Severity**: schema violation or missing-rate > 5x baseline = High; missing-rate 2-5x or
volume anomaly +/- 30% = Medium.

**Triage**
1. Quantify: run `monitor_model.py` (or the data-validation skill's `validate_data.py`)
   on the current window — which columns, what violation type, since when?
2. Check the ingestion layer: upstream schema migration, late-arriving partition, source
   outage, or a new client version sending different payloads.
3. Assess model exposure: is the affected feature high-importance? Are predictions already
   shifting (Runbook 1 step 3)?

**Escalate when**: violations feed a high-importance feature, predictions are visibly
degraded, or the upstream owner cannot give an ETA.

**Remediate**: quarantine bad batches before they hit the feature store and reference data;
if predictions are compromised, prefer holding scoring or serving the previous champion
(**alias-flip rollback**) over serving on corrupt inputs. Never silently patch the
reference dataset — fix the source, then rebuild the reference.

### Ops Handover Checklist

- [ ] **Dashboards**: overview/drift/performance/quality/operational panels (layout above)
      linked and access granted; `/metrics` scraped by Prometheus.
- [ ] **Alert routing**: `setup_alerts.py` config reviewed — severities map to channels
      (critical → PagerDuty/SMS, high → Slack/email) per the Tiered Alerting table;
      test-fired via `--action test-alert`.
- [ ] **Thresholds**: `thresholds.json` values agreed with the model owner and recorded
      next to the model version in the registry.
- [ ] **Retrain criteria**: written trigger (e.g. PSI > 0.2 sustained 3 days, or primary
      metric down > 5%) plus who approves promotion.
- [ ] **Rollback drill**: on-call has run the alias flip once end-to-end and knows the
      previous-champion version number.
- [ ] **Ownership**: named owner for model, data pipeline, and serving infra; escalation
      contacts in the alert config, ground-truth delay documented.

## Managed Monitoring: SageMaker Model Monitor & Clarify

For models served on SageMaker endpoints (or batch transform), Amazon SageMaker Model Monitor provides fully managed monitoring jobs. Four monitor types map to `MonitoringType` in the `CreateMonitoringSchedule` API: `DataQuality`, `ModelQuality`, `ModelBias`, and `ModelExplainability`. Prerequisite: enable data capture on the endpoint (`DataCaptureConfig`) so requests/responses land in S3.

### Data Quality Monitor (baseline + schedule)

```python
# SageMaker Python SDK v3: monitor classes live in sagemaker.core.model_monitor
from sagemaker.core.model_monitor import (
    CronExpressionGenerator, DataCaptureConfig, DatasetFormat,
    DefaultModelMonitor, EndpointInput,
)

monitor = DefaultModelMonitor(role=role, instance_type="ml.m5.xlarge", instance_count=1)

# 1. Baseline job: profiles training data -> statistics.json + constraints.json
monitor.suggest_baseline(
    baseline_dataset="s3://bucket/train/train.csv",
    dataset_format=DatasetFormat.csv(header=True),
    output_s3_uri="s3://bucket/monitor/baseline",
)

# 2. Hourly schedule: each run compares captured traffic against the baseline
monitor.create_monitoring_schedule(
    monitor_schedule_name="churn-data-quality",
    endpoint_input=EndpointInput(endpoint_name="churn-endpoint", destination="/opt/ml/processing/input"),
    statistics=monitor.baseline_statistics(),
    constraints=monitor.suggested_constraints(),
    schedule_cron_expression=CronExpressionGenerator.hourly(),
    output_s3_uri="s3://bucket/monitor/reports",
    enable_cloudwatch_metrics=True,
)
```

Violations (missing columns, type mismatches, distribution drift per feature) are written to `constraint_violations.json`; `ModelQualityMonitor` works the same way but joins captured predictions with ground-truth labels you upload to S3 and computes accuracy/AUC/etc. against baseline thresholds.

### Clarify Bias & Explainability Monitors

`ModelBiasMonitor` and `ModelExplainabilityMonitor` (same module) run SageMaker Clarify jobs on schedule: bias monitors track post-training metrics (DPPL, DI, etc.) using a `BiasConfig` (facet column, e.g. gender/age) and ground-truth labels; explainability monitors track SHAP feature-attribution drift against a baseline `SHAPConfig`. Both use `suggest_baseline(...)` then `create_monitoring_schedule(...)` like the data-quality monitor. Raw APIs if working in boto3: `create_data_quality_job_definition`, `create_model_quality_job_definition`, `create_model_bias_job_definition`, `create_model_explainability_job_definition` + `create_monitoring_schedule(MonitoringScheduleConfig={"MonitoringJobDefinitionName": ..., "MonitoringType": ..., "ScheduleConfig": {...}})`.

### CloudWatch Alarm Wiring

With `enable_cloudwatch_metrics=True`, each run emits per-feature metrics to the `aws/sagemaker/Endpoints/data-metrics` namespace (model-quality metrics to `.../model-metrics`). Alarm on them like any CloudWatch metric:

```python
cloudwatch.put_metric_alarm(
    AlarmName="churn-feature-drift",
    Namespace="aws/sagemaker/Endpoints/data-metrics",
    MetricName="feature_baseline_drift_age",
    Dimensions=[{"Name": "Endpoint", "Value": "churn-endpoint"},
                {"Name": "MonitoringSchedule", "Value": "churn-data-quality"}],
    ComparisonOperator="GreaterThanThreshold", Threshold=0.1,
    EvaluationPeriods=1, Period=3600, Statistic="Average",
    AlarmActions=[sns_topic_arn],  # feeds the tiered alerting channels above
)
```

### vs Evidently (and other OSS tools)

| Aspect | SageMaker Model Monitor | Evidently |
|--------|-------------------------|-----------|
| Scope | SageMaker endpoints / batch transform only | Any serving stack |
| Ops burden | Fully managed jobs; pay per processing run | You host and schedule it |
| Drift methods | Managed baseline comparison (per-feature drift check) | PSI, KS, Wasserstein, JS -- your choice per feature |
| Bias / explainability | Built-in via Clarify | Not built-in |
| Custom metrics | Bring-your-own container/analysis possible but heavier | Trivial (Python) |
| Alerting | CloudWatch-native | Build your own (or Evidently Cloud) |

Rule of thumb: if the model runs on a SageMaker endpoint, start with Model Monitor for managed drift/quality checks and CloudWatch alerting; add Evidently when you need custom statistical tests, richer reports, or monitoring outside SageMaker.

## Further Reading

- [Evidently AI Documentation](https://docs.evidentlyai.com/)
- [Whylogs Documentation](https://whylogs.readthedocs.io/)
- [NannyML Documentation](https://nannyml.readthedocs.io/)
- [Arize AI Documentation](https://docs.arize.com/)
- [Failing Loudly: Methods for Detecting Dataset Shift](https://arxiv.org/abs/1810.11953)
- [Monitoring ML Models in Production (Made With ML)](https://madewithml.com/courses/mlops/monitoring/)
- [Hidden Technical Debt in ML Systems (Sculley et al.)](https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html)
