# Model Drift Detection Reference Guide

Comprehensive reference for detecting and responding to drift in production ML systems.
Covers drift types, statistical tests, detection strategies, tooling, and retraining workflows.

---

## Table of Contents

1. [Drift Types](#1-drift-types)
2. [Statistical Tests Comparison](#2-statistical-tests-comparison)
3. [Drift Detection Window Strategies](#3-drift-detection-window-strategies)
4. [Threshold Selection Guide](#4-threshold-selection-guide)
5. [Drift Detection Tools Comparison](#5-drift-detection-tools-comparison)
6. [Retraining Strategies](#6-retraining-strategies)
7. [False Positive Management](#7-false-positive-management)
8. [Further Reading](#8-further-reading)

---

## 1. Drift Types

### Definitions and Examples

| Drift Type | Definition | Example | Detection Approach |
|------------|-----------|---------|-------------------|
| **Data Drift** (covariate shift) | Change in the input feature distribution P(X) while P(Y\|X) remains unchanged | A loan model receives applicants from a new demographic region with different income distributions | Compare feature distributions between reference and production windows |
| **Concept Drift** | Change in the relationship P(Y\|X) between inputs and target | Customer purchasing behavior shifts during a recession; the same features now predict different outcomes | Monitor model performance metrics; compare predicted vs. actual labels over time |
| **Prediction Drift** | Change in the model's output distribution P(Y_hat) | A fraud model begins predicting "fraud" at a higher rate than historical baselines even though inputs look similar | Compare prediction distributions over time |
| **Label Drift** (prior probability shift) | Change in the target variable distribution P(Y) | Fraud prevalence doubles from 1% to 2% due to a new attack vector | Compare ground-truth label distributions when available |

### Concept Drift Subtypes

| Subtype | Pattern | Example | Detection Difficulty |
|---------|---------|---------|---------------------|
| **Sudden** | Abrupt change at a point in time | Regulatory change alters customer behavior overnight | Easy -- sharp metric drop |
| **Gradual** | Slow transition from old concept to new | Consumer preferences evolve over months | Moderate -- requires trend analysis |
| **Incremental** | Very slow, continuous shift | Language usage evolving over years | Hard -- indistinguishable from noise short-term |
| **Recurring** | Periodic concept changes | Seasonal purchasing patterns | Moderate -- detectable with calendar-aware baselines |

---

## 2. Statistical Tests Comparison

### When to Use Each Test

| Test | Full Name | Data Type | Sensitivity | Computational Cost | Best For |
|------|-----------|-----------|-------------|-------------------|----------|
| **PSI** | Population Stability Index | Categorical / binned numerical | Low-moderate | Very low | Production monitoring dashboards; quick health checks |
| **KS Test** | Kolmogorov-Smirnov | Continuous univariate | Moderate | Low | Single numerical feature distributions |
| **Chi-Squared** | Chi-Squared Test | Categorical | Moderate | Low | Categorical feature distributions |
| **Wasserstein** | Wasserstein Distance (Earth Mover's Distance) | Continuous univariate | High | Moderate | Detecting magnitude of distributional shift; cost-sensitive applications |
| **Jensen-Shannon** | Jensen-Shannon Divergence | Any (via binning or density estimation) | Moderate-high | Moderate | Symmetric divergence measure; comparing probability distributions |
| **MMD** | Maximum Mean Discrepancy | Multivariate / high-dimensional | High | High | Multivariate drift detection; embedding spaces; image/text features |

### Test Details

**PSI (Population Stability Index)**:
```
PSI = sum((actual_pct - expected_pct) * ln(actual_pct / expected_pct))

Interpretation:
  PSI < 0.1   -> No significant drift
  PSI 0.1-0.2 -> Moderate drift, investigate
  PSI > 0.2   -> Significant drift, action required
```
- Requires binning continuous features (typically 10-20 bins).
- Not a formal hypothesis test; provides a scalar score.
- Widely adopted in financial services due to regulatory familiarity.

**KS Test (Kolmogorov-Smirnov)**:
```python
from scipy.stats import ks_2samp
statistic, p_value = ks_2samp(reference_data, production_data)
# Reject H0 (no drift) if p_value < alpha (e.g., 0.05)
```
- Non-parametric; no distributional assumptions.
- Sensitive to changes in location, scale, and shape.
- Less powerful for large samples (nearly everything becomes significant).
- Only works on univariate data.

**Chi-Squared Test**:
```python
from scipy.stats import chi2_contingency
chi2, p_value, dof, expected = chi2_contingency(contingency_table)
```
- Designed for categorical data.
- Requires sufficient counts per category (>5 expected per cell).
- Can be applied to binned continuous data.

**Wasserstein Distance (Earth Mover's Distance)**:
```python
from scipy.stats import wasserstein_distance
distance = wasserstein_distance(reference_data, production_data)
```
- Measures the minimum "work" to transform one distribution into another.
- Sensitive to the magnitude of shift, not just the presence of shift.
- No p-value by default; requires bootstrapping or thresholds.
- Meaningful when the metric space has physical interpretation (e.g., prices, distances).

**Jensen-Shannon Divergence**:
```python
from scipy.spatial.distance import jensenshannon
# Both must be probability distributions (sum to 1)
js_distance = jensenshannon(p_distribution, q_distribution)
js_divergence = js_distance ** 2  # JSD is the square of JS distance
```
- Symmetric (unlike KL divergence).
- Bounded between 0 and 1 (using log base 2) or 0 and ln(2) (natural log).
- Requires converting data to probability distributions (histogram or KDE).

**MMD (Maximum Mean Discrepancy)**:
```python
# Kernel-based test for multivariate distributions
# Available in alibi-detect
from alibi_detect.cd import MMDDrift
detector = MMDDrift(reference_data, backend='pytorch', p_val=0.05)
result = detector.predict(production_data)
```
- Works on multivariate data natively.
- Kernel choice matters: RBF kernel is standard; bandwidth selection is critical.
- Computationally expensive: O(n^2) for naive implementation; use linear-time approximations for production.
- Ideal for embedding-based drift detection (NLP, CV feature spaces).

---

## 3. Drift Detection Window Strategies

### Strategy Comparison

| Strategy | Description | Pros | Cons | Best For |
|----------|------------|------|------|----------|
| **Fixed Window** | Compare production data in fixed time windows (e.g., daily, weekly) against a reference set | Simple to implement; consistent reporting cadence | May miss drift within a window; window size selection is critical | Regular reporting; batch inference systems |
| **Sliding Window** | Rolling window of the most recent N samples compared against reference | Smoother detection; reduces noise | Higher compute cost; N must be tuned | Real-time inference; streaming data |
| **Adaptive (ADWIN)** | Automatically adjusts window size based on detected rate of change | Optimal window size without tuning; detects both sudden and gradual drift | More complex implementation; harder to explain to stakeholders | Research; systems with mixed drift patterns |
| **Page-Hinkley** | Sequential test that monitors cumulative deviation from the mean | Detects gradual drift well; low memory | Requires threshold tuning; sensitive to parameter choices | Streaming data; gradual concept drift |
| **CUSUM** | Cumulative sum of deviations from a target value | Well-established; detects small persistent shifts | Assumes known pre-change distribution | Manufacturing QC-style monitoring |

### Window Size Guidelines

| Data Volume | Recommended Window | Rationale |
|-------------|-------------------|-----------|
| < 100 samples/day | Weekly or biweekly | Need sufficient samples for statistical power |
| 100-1,000 samples/day | Daily | Enough for stable daily statistics |
| 1,000-100,000 samples/day | Daily with hourly checks | Can detect intra-day drift |
| > 100,000 samples/day | Hourly or sliding (1,000-10,000 samples) | Sufficient volume for fine-grained detection |

### Reference Window Best Practices

- Use a **stable period** of known-good model performance as the reference.
- Reference should be large enough for reliable statistics (minimum 500-1,000 samples).
- **Update the reference** after confirmed non-drift periods or successful retraining.
- Store reference statistics (not raw data) to reduce storage and compute costs.

---

## 4. Threshold Selection Guide

### Recommended Thresholds by Test

| Test | Warning Threshold | Alert Threshold | Notes |
|------|------------------|-----------------|-------|
| **PSI** | 0.1 | 0.2 | Industry standard from credit risk modeling |
| **KS Test (p-value)** | p < 0.05 | p < 0.01 | Apply Bonferroni correction for multiple features |
| **Chi-Squared (p-value)** | p < 0.05 | p < 0.01 | Check cell counts > 5 |
| **Wasserstein** | > 0.1 * feature_std | > 0.3 * feature_std | Scale by feature standard deviation; no universal threshold |
| **Jensen-Shannon** | > 0.05 | > 0.1 | On 0-1 scale (log base 2) |
| **MMD (p-value)** | p < 0.05 | p < 0.01 | Permutation test p-value |

### Multiple Testing Correction

When monitoring many features simultaneously, raw p-values produce excessive false positives.

| Method | Description | Strictness |
|--------|------------|------------|
| **Bonferroni** | Divide alpha by number of tests | Very strict; reduces false positives but misses real drift |
| **Holm-Bonferroni** | Step-down version of Bonferroni | Slightly less strict; better power |
| **Benjamini-Hochberg** | Controls false discovery rate (FDR) | Moderate; recommended for exploratory drift detection |
| **No correction + voting** | Alert if > K of N features drift | Practical; easy to explain; tune K empirically |

**Recommendation**: Use Benjamini-Hochberg FDR control at 0.05 for automated alerting, or a voting scheme (alert if more than 20% of features show drift).

---

## 5. Drift Detection Tools Comparison

| Feature | Evidently | Alibi Detect | NannyML | River |
|---------|-----------|-------------|---------|-------|
| **Focus** | ML monitoring and reporting | Statistical drift detection | Post-deployment performance estimation | Online / streaming ML |
| **Data Drift** | Yes (PSI, KS, Wasserstein, Jensen-Shannon) | Yes (KS, MMD, Chi-Squared, learned detectors) | Yes (univariate methods) | Yes (ADWIN, Page-Hinkley, DDM) |
| **Concept Drift** | Via performance monitoring | Spot checks, context-aware | CBPE and DLE performance estimation without labels | Yes (native streaming detectors) |
| **Prediction Drift** | Yes | Yes | Yes | Yes |
| **Multivariate** | Limited (per-feature) | Yes (MMD, LSDD, classifier-based) | Limited | No |
| **Labels Required** | Optional | Optional | Not required for estimation | Depends on method |
| **Visual Reports** | Excellent (HTML dashboards) | Basic (matplotlib) | Good (interactive dashboards) | Minimal |
| **Streaming Support** | No (batch) | No (batch) | No (batch) | Yes (native streaming) |
| **Integration** | Airflow, MLflow, Grafana, Prometheus | Standalone | Standalone, APIs | Standalone |
| **License** | Apache 2.0 | Business Source (BSL 1.1) | Apache 2.0 | BSD 3-Clause |
| **Best For** | Team dashboards, stakeholder reports | Research, advanced multivariate detection | Performance monitoring without ground truth | Real-time streaming applications |

### Quick Start Examples

**Evidently**:
```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=ref_df, current_data=prod_df)
report.save_html("drift_report.html")
```

**Alibi Detect**:
```python
from alibi_detect.cd import KSDrift
detector = KSDrift(reference_data.values, p_val=0.05)
result = detector.predict(production_data.values)
print(f"Drift detected: {result['data']['is_drift']}")
```

**NannyML**:
```python
import nannyml as nml
estimator = nml.CBPE(
    y_pred_proba='predicted_probability',
    y_pred='prediction',
    y_true='target',
    problem_type='classification_binary',
    metrics=['roc_auc']
)
estimator.fit(reference_df)
results = estimator.estimate(production_df)
results.plot().show()
```

**River (streaming)**:
```python
from river import drift

adwin = drift.ADWIN()
for i, val in enumerate(data_stream):
    adwin.update(val)
    if adwin.drift_detected:
        print(f"Drift detected at index {i}")
```

---

## 6. Retraining Strategies

### Strategy Comparison

| Strategy | Trigger | Pros | Cons | Best For |
|----------|---------|------|------|----------|
| **Scheduled** | Fixed interval (daily, weekly, monthly) | Simple; predictable costs | May retrain unnecessarily or too late | Stable environments; compliance-driven |
| **Performance-triggered** | Metrics drop below threshold | Efficient; retrains only when needed | Requires ground-truth labels (often delayed) | Supervised tasks with fast label feedback |
| **Drift-triggered** | Drift test exceeds threshold | Proactive; acts before performance degrades | Risk of false positive triggers; requires good thresholds | High-stakes applications; unsupervised monitoring |
| **Hybrid** | Scheduled baseline + drift/performance triggers | Balances responsiveness and stability | More complex orchestration | Production enterprise systems |

### Retraining Data Selection

| Approach | Description | When to Use |
|----------|------------|-------------|
| **Full retrain** | Train on all available historical data | Model benefits from long history; compute is not constrained |
| **Sliding window retrain** | Train on most recent N months/samples | Non-stationary environments where old data hurts |
| **Weighted retrain** | Assign higher weight to recent data | Gradual drift; want to preserve older patterns while adapting |
| **Incremental update** | Update model weights on new data only | Streaming; compute-constrained; compatible architectures (e.g., SGD-based) |

### Retraining Workflow Checklist

```
1. Drift signal confirmed (passed false positive checks)
2. New labeled data collected or most recent window identified
3. Data quality validation passed (schema, completeness, distributions)
4. Retrained model evaluated on holdout set
5. Retrained model compared to current production model (challenger/champion)
6. A/B test or shadow deployment (if applicable)
7. Promote new model only if it outperforms current model
8. Update reference window for future drift detection
9. Log retraining event with metadata (trigger, data version, metrics)
```

---

## 7. False Positive Management

Drift detection in production generates false alarms. Managing them is critical for team trust.

### Common Causes of False Positives

| Cause | Description | Mitigation |
|-------|------------|------------|
| **Small sample sizes** | Noisy estimates with limited data | Increase minimum window size; require N > 500 |
| **Multiple testing** | Testing many features inflates Type I error | Apply Bonferroni or Benjamini-Hochberg correction |
| **Seasonal patterns** | Regular calendar-driven shifts flagged as drift | Use calendar-aware baselines; seasonal reference windows |
| **Deployment artifacts** | Logging changes, schema updates cause spurious distribution shifts | Validate data pipeline integrity before drift analysis |
| **One-off events** | Holidays, outages, marketing campaigns | Annotate known events; suppress alerts during known anomalies |

### False Positive Reduction Strategies

1. **Confirmation window**: Require drift to persist for N consecutive checks before alerting.
2. **Severity scoring**: Combine drift magnitude, number of affected features, and performance impact into a composite score.
3. **Two-stage detection**: Use a fast, cheap test for screening (PSI) and a rigorous test for confirmation (MMD or classifier-based).
4. **Human-in-the-loop**: Route borderline alerts to a review queue rather than triggering automated retraining.
5. **Alert fatigue tracking**: Monitor alert-to-action ratio; if < 20% of alerts lead to action, tighten thresholds.

### Recommended Alerting Tiers

| Tier | Condition | Action |
|------|-----------|--------|
| **Info** | Mild drift in < 3 features, no performance impact | Log for weekly review |
| **Warning** | Moderate drift in multiple features OR mild performance drop | Notify ML engineer; investigate within 24 hours |
| **Critical** | Significant drift across many features OR performance below SLA | Page on-call; begin retraining evaluation immediately |

---

## 8. Further Reading

### Official Documentation

- **Evidently AI**: https://docs.evidentlyai.com/
- **Alibi Detect**: https://docs.seldon.io/projects/alibi-detect/
- **NannyML**: https://nannyml.readthedocs.io/
- **River**: https://riverml.xyz/

### Key Papers

- Gama et al. (2014) "A Survey on Concept Drift Adaptation" -- comprehensive taxonomy of drift types
- Lu et al. (2019) "Learning under Concept Drift: A Review" -- modern drift detection methods
- Rabanser et al. (2019) "Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift" -- benchmark of statistical tests
- Bifet & Gavalda (2007) "Learning from Time-Changing Data with Adaptive Windowing" -- ADWIN algorithm
- Page (1954) "Continuous Inspection Schemes" -- CUSUM foundations

### Industry References

- Google ML Best Practices: https://developers.google.com/machine-learning/guides/rules-of-ml
- AWS SageMaker Model Monitor: https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor.html
- Azure ML Data Drift: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-monitor-datasets
