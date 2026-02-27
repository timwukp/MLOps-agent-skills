# ML Testing Reference Guide

Comprehensive reference for testing machine learning systems. Covers the ML testing pyramid,
framework comparisons, behavioral testing, data validation, model quality gates, regression
strategies, and deployment testing patterns.

---

## Table of Contents

1. [ML Testing Pyramid](#1-ml-testing-pyramid)
2. [Testing Frameworks Comparison](#2-testing-frameworks-comparison)
3. [Behavioral Testing Patterns](#3-behavioral-testing-patterns)
4. [Data Validation Testing Patterns](#4-data-validation-testing-patterns)
5. [Model Quality Gates for CI/CD](#5-model-quality-gates-for-cicd)
6. [Regression Testing Strategies](#6-regression-testing-strategies)
7. [Testing Checklist for ML Pipeline Deployment](#7-testing-checklist-for-ml-pipeline-deployment)
8. [Shadow Deployment Testing](#8-shadow-deployment-testing)
9. [Further Reading](#9-further-reading)

---

## 1. ML Testing Pyramid

The ML testing pyramid extends the traditional software testing pyramid with ML-specific layers.

```
                    /\
                   /  \
                  / E2E \          <- End-to-end / System Tests
                 /--------\           Full pipeline: data -> training -> serving -> prediction
                / Behavioral\      <- Behavioral Tests
               /--------------\       Invariance, directional, minimum functionality
              /  Integration    \  <- Integration Tests
             /-------------------\    Feature pipelines, model-serving contracts, data flows
            /       Unit          \<- Unit Tests
           /------------------------\  Data transforms, feature functions, pre/post processing
```

### Layer Details

| Layer | Scope | Speed | Frequency | Examples |
|-------|-------|-------|-----------|---------|
| **Unit Tests** | Individual functions, transforms, utilities | Fast (ms) | Every commit | Feature engineering functions return correct types/values; preprocessing handles edge cases; custom loss functions compute correctly |
| **Integration Tests** | Component interactions, data flow between stages | Moderate (seconds) | Every PR / merge | Feature pipeline produces expected schema; model loads and serves predictions; training script runs end-to-end on sample data |
| **Behavioral Tests** | Model behavior properties | Moderate (seconds-minutes) | Every model training run | Invariance to irrelevant perturbations; directional expectations; minimum accuracy on known examples |
| **System / E2E Tests** | Full pipeline from data to prediction | Slow (minutes-hours) | Before deployment; nightly | Full training pipeline on subset; serving latency under load; prediction consistency across model versions |

### What to Test at Each Layer

**Unit Tests**:
- Data transformation functions produce expected output for known inputs.
- Feature engineering handles nulls, NaN, Inf, empty strings, extreme values.
- Custom metrics, loss functions, and evaluation code are correct.
- Preprocessing is deterministic (same input yields same output).
- Serialization/deserialization preserves model and data integrity.

**Integration Tests**:
- Feature pipeline output matches expected schema (column names, types, shapes).
- Model training script runs without errors on a small data sample.
- Trained model can be serialized, loaded, and produces predictions.
- Serving endpoint returns valid responses for well-formed requests.
- Data pipeline stages connect correctly (output of stage N is valid input for stage N+1).

**Behavioral Tests**:
- See [Section 3: Behavioral Testing Patterns](#3-behavioral-testing-patterns).

**System / E2E Tests**:
- Full training pipeline completes on a representative subset of data.
- Model performance meets minimum thresholds on held-out test set.
- Serving infrastructure handles expected load with acceptable latency.
- Predictions are consistent between model versions (no silent regressions).

---

## 2. Testing Frameworks Comparison

| Feature | pytest | Great Expectations | Deepchecks | CheckList |
|---------|--------|-------------------|------------|-----------|
| **Type** | General testing framework | Data validation framework | ML validation suite | Behavioral testing for NLP |
| **Focus** | Code correctness | Data quality and contracts | Model + data validation | NLP model behavioral testing |
| **Data validation** | Manual (with assertions) | Native (expectations, profiling) | Native (data checks) | No |
| **Model validation** | Manual (with assertions) | No | Native (performance, drift, integrity) | Yes (NLP-focused) |
| **Schema validation** | Manual | Yes (column types, nullability, uniqueness) | Yes (data integrity checks) | No |
| **Statistical tests** | Via scipy/statsmodels | Yes (distributional expectations) | Yes (drift, distribution checks) | No |
| **CI/CD integration** | Native | Yes (checkpoint with actions) | Yes (suite results as pass/fail) | Manual |
| **Report generation** | Via plugins (pytest-html) | Data Docs (HTML) | Rich HTML reports | Visual report |
| **Learning curve** | Low | Moderate | Low-moderate | Low |
| **Best for** | All code-level ML tests | Data pipeline validation | Pre-deployment model checks | NLP behavioral testing |
| **License** | MIT | Apache 2.0 | AGPL 3.0 (open) / Commercial | MIT |

### When to Use Each

| Scenario | Recommended Tool |
|----------|-----------------|
| Unit tests for feature engineering, preprocessing, utilities | pytest |
| Data pipeline contract tests (schema, distributions, completeness) | Great Expectations |
| Pre-deployment model validation (performance, fairness, drift) | Deepchecks |
| NLP model behavioral testing (invariance, directional, MFT) | CheckList |
| All of the above in a single CI/CD pipeline | pytest as the runner; Great Expectations, Deepchecks, and CheckList as libraries called within pytest |

### Quick Start Examples

**pytest (ML unit test)**:
```python
import pytest
import numpy as np

def test_feature_engineering_handles_nulls():
    """Feature function should replace nulls with median."""
    input_data = [1.0, 2.0, None, 4.0, None]
    result = compute_feature(input_data)
    assert not any(v is None for v in result)
    assert result[2] == 2.0  # median of [1, 2, 4]

def test_model_prediction_shape():
    """Model output should match expected shape."""
    model = load_model("model_v1.pkl")
    X = np.random.randn(10, 5)
    preds = model.predict(X)
    assert preds.shape == (10,)

def test_model_prediction_range():
    """Probability predictions should be in [0, 1]."""
    model = load_model("model_v1.pkl")
    X = np.random.randn(100, 5)
    proba = model.predict_proba(X)
    assert np.all(proba >= 0) and np.all(proba <= 1)
```

**Great Expectations (data validation)**:
```python
import great_expectations as gx

context = gx.get_context()
validator = context.sources.pandas_default.read_csv("features.csv")

validator.expect_column_to_exist("user_age")
validator.expect_column_values_to_be_between("user_age", min_value=0, max_value=150)
validator.expect_column_values_to_not_be_null("user_id")
validator.expect_column_mean_to_be_between("purchase_amount", min_value=10, max_value=500)
validator.expect_table_row_count_to_be_between(min_value=1000, max_value=10_000_000)

results = validator.validate()
assert results.success, f"Data validation failed: {results}"
```

**Deepchecks (model validation)**:
```python
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import model_evaluation

train_ds = Dataset(train_df, label='target', cat_features=cat_cols)
test_ds = Dataset(test_df, label='target', cat_features=cat_cols)

suite = model_evaluation()
result = suite.run(train_ds, test_ds, model)
result.save_as_html("model_validation_report.html")

# Custom check with pass condition
from deepchecks.tabular.checks import TrainTestPerformance
check = TrainTestPerformance().add_condition_train_test_relative_degradation_less_than(0.1)
check_result = check.run(train_ds, test_ds, model)
assert check_result.passed_conditions()
```

**CheckList (NLP behavioral testing)**:
```python
from checklist.editor import Editor
from checklist.test_types import MFT, INV, DIR

editor = Editor()

# Minimum Functionality Test
test_data = editor.template("I am {adj} about this.", adj=["happy", "thrilled", "excited"])
mft = MFT(test_data.data, labels=1, name="Positive sentiment with positive words")

# Invariance Test
inv = INV(data, lambda x, *args: x.replace("John", "Mary"),
          name="Name change should not affect sentiment")

# Directional Test
dir_test = DIR(data, lambda x, *args: x + " This is terrible.",
               expect=lambda orig, new, *args: new <= orig,
               name="Adding negative sentence should decrease sentiment")
```

---

## 3. Behavioral Testing Patterns

Behavioral tests verify what a model should do rather than specific numerical outputs.

### Invariance Tests (INV)

The model's prediction should not change when irrelevant perturbations are applied.

| Domain | Perturbation | Example |
|--------|-------------|---------|
| **NLP** | Name substitution | Sentiment should not change when "John" is replaced with "Priya" |
| **NLP** | Typo injection | Intent classification should tolerate minor typos |
| **Tabular** | Unit conversion | Prediction unchanged when feature is converted between equivalent units |
| **CV** | Minor rotation / crop | Object detection should tolerate small affine transformations |
| **General** | Irrelevant feature change | Prediction unchanged when a feature known to be irrelevant is modified |

### Directional Tests (DIR)

The model's prediction should change in a known direction when a meaningful perturbation is applied.

| Domain | Perturbation | Expected Direction |
|--------|-------------|-------------------|
| **NLP** | Add negative clause | Sentiment score should decrease |
| **Tabular (credit)** | Increase income | Default risk should decrease |
| **Tabular (medical)** | Increase risk factor value | Disease probability should increase |
| **Pricing** | Increase demand signal | Price recommendation should increase |

### Minimum Functionality Tests (MFT)

The model should produce the correct output for simple, unambiguous examples.

| Domain | Test | Expected Behavior |
|--------|------|-------------------|
| **NLP** | "I love this product!" | Positive sentiment |
| **NLP** | "This is terrible and broken" | Negative sentiment |
| **Tabular (fraud)** | Known fraudulent pattern | Flag as fraud |
| **CV** | Clear, centered, well-lit canonical image | Correct classification |
| **Recommendation** | User who bought item A always buys item B | Recommend item B |

### Implementation Pattern

```python
import pytest

class TestModelBehavior:
    """Behavioral test suite for sentiment model."""

    @pytest.fixture
    def model(self):
        return load_model("sentiment_v2.pkl")

    @pytest.mark.parametrize("text,expected", [
        ("I absolutely love this!", "positive"),
        ("This is the worst thing ever.", "negative"),
        ("The product arrived on Tuesday.", "neutral"),
    ])
    def test_minimum_functionality(self, model, text, expected):
        """MFT: Model should correctly classify unambiguous examples."""
        assert model.predict(text) == expected

    @pytest.mark.parametrize("name_a,name_b", [
        ("John", "Maria"), ("David", "Wei"), ("Sarah", "Aisha"),
    ])
    def test_name_invariance(self, model, name_a, name_b):
        """INV: Sentiment should not depend on person's name."""
        template = "{name} said the food was delicious."
        pred_a = model.predict_proba(template.format(name=name_a))
        pred_b = model.predict_proba(template.format(name=name_b))
        assert abs(pred_a - pred_b) < 0.05

    def test_negative_addition_directional(self, model):
        """DIR: Adding negative text should decrease sentiment score."""
        base = "The hotel room was clean."
        modified = "The hotel room was clean. But the service was awful."
        assert model.predict_proba(modified) < model.predict_proba(base)
```

---

## 4. Data Validation Testing Patterns

### Schema Tests

| Check | Description | Example Assertion |
|-------|------------|-------------------|
| Column existence | Required columns are present | `assert set(required_cols).issubset(df.columns)` |
| Column types | Columns have expected data types | `assert df['age'].dtype == np.float64` |
| No unexpected columns | No extra columns from upstream changes | `assert set(df.columns) == set(expected_cols)` |
| Row count bounds | Dataset has reasonable number of rows | `assert 1000 <= len(df) <= 10_000_000` |

### Completeness Tests

| Check | Description | Threshold |
|-------|------------|-----------|
| Null rate per column | Missing values below threshold | < 5% for most features; 0% for IDs and labels |
| Row completeness | Minimum non-null columns per row | > 80% of columns non-null per row |
| Required fields | Critical fields never null | 0% null for label, ID, timestamp |

### Distribution Tests

| Check | Description | Implementation |
|-------|------------|----------------|
| Feature ranges | Values within expected bounds | `assert df['age'].between(0, 150).all()` |
| Mean / stddev stability | Summary statistics within expected range | `assert abs(df['amount'].mean() - historical_mean) < 3 * historical_std` |
| Cardinality bounds | Categorical features have expected number of unique values | `assert 2 <= df['country'].nunique() <= 250` |
| Class balance | Label distribution within expected range | `assert 0.001 <= df['fraud'].mean() <= 0.05` |
| No constant columns | Features are not degenerate | `assert df['feature_x'].nunique() > 1` |

### Referential Integrity Tests

| Check | Description | Example |
|-------|------------|---------|
| Foreign key validity | Referenced IDs exist in lookup table | `assert df['user_id'].isin(users_df['id']).all()` |
| Temporal ordering | Events are chronologically ordered | `assert df['timestamp'].is_monotonic_increasing` |
| No future data leakage | Training data does not contain future information | `assert (df['event_date'] < training_cutoff_date).all()` |

---

## 5. Model Quality Gates for CI/CD

### Gate Structure

```
Code Commit -> Unit Tests -> Data Validation -> Model Training -> Model Validation -> Deployment
                  |               |                    |                |
                GATE 1         GATE 2              GATE 3           GATE 4
              (code)          (data)             (training)       (model quality)
```

### Gate Definitions

| Gate | Checks | Fail Action |
|------|--------|-------------|
| **Gate 1: Code Quality** | Unit tests pass; linting pass; type checks pass; no secrets in code | Block merge |
| **Gate 2: Data Validation** | Schema valid; null rates OK; distributions stable; no data leakage | Block training; alert data team |
| **Gate 3: Training Completion** | Training converges; no NaN/Inf in loss; training time within budget; resource usage acceptable | Alert ML engineer; retry or investigate |
| **Gate 4: Model Quality** | Performance above threshold; no regression vs. champion; fairness metrics pass; latency within SLA | Block deployment; require human review |

### Gate 4: Model Quality Checks (Detailed)

| Check | Metric | Threshold Example | Rationale |
|-------|--------|-------------------|-----------|
| **Absolute performance** | Accuracy, F1, AUC, RMSE | AUC > 0.85 | Minimum acceptable model quality |
| **Relative performance** | Delta vs. current production model | AUC regression < 1% | Prevent silent degradation |
| **Slice performance** | Metrics per critical subgroup | No subgroup AUC < 0.75 | Catch subgroup-level regressions |
| **Fairness** | Demographic parity ratio, equalized odds difference | DP ratio > 0.8 | Legal and ethical compliance |
| **Latency** | P50, P95, P99 inference latency | P99 < 100ms | SLA compliance |
| **Model size** | Serialized model file size | < 500MB | Deployment feasibility |
| **Prediction distribution** | KL divergence or PSI vs. expected | PSI < 0.2 | Detect degenerate models |

### Example CI/CD Configuration

```yaml
# .github/workflows/model-ci.yml (conceptual)
model_quality_gate:
  performance:
    auc_roc: {min: 0.85}
    f1_score: {min: 0.75}
    regression_vs_champion: {max_degradation_pct: 1.0}
  fairness:
    demographic_parity_ratio: {min: 0.8}
    equalized_odds_difference: {max: 0.1}
  latency:
    p99_ms: {max: 100}
  slices:
    - name: "high_value_customers"
      auc_roc: {min: 0.80}
    - name: "new_customers"
      auc_roc: {min: 0.78}
```

---

## 6. Regression Testing Strategies

### Types of ML Regressions

| Regression Type | Description | Detection Method |
|----------------|------------|-----------------|
| **Performance regression** | Overall metric degrades | Compare metrics against champion model on same test set |
| **Slice regression** | Specific subgroup degrades while aggregate holds | Compute metrics per slice; flag if any slice drops |
| **Behavioral regression** | Model violates known behavioral properties | Run behavioral test suite (INV, DIR, MFT) on every candidate |
| **Latency regression** | Inference speed degrades | Benchmark inference latency on standardized hardware |
| **Prediction consistency** | Predictions change unexpectedly on golden set | Compare predictions on fixed golden dataset |

### Golden Dataset Testing

Maintain a curated, versioned "golden dataset" of examples with known expected outputs.

```python
def test_golden_dataset_consistency(model, golden_df, tolerance=0.02):
    """Predictions on golden dataset should not change significantly."""
    preds = model.predict(golden_df.drop('expected', axis=1))
    agreement_rate = (preds == golden_df['expected']).mean()
    assert agreement_rate >= (1 - tolerance), \
        f"Golden dataset agreement {agreement_rate:.2%} below {1-tolerance:.2%} threshold"
```

**Golden dataset guidelines**:
- Include 100-500 carefully curated examples spanning all important categories.
- Include edge cases and known difficult examples.
- Version the golden dataset alongside the model code.
- Update only with deliberate, reviewed changes.

### Back-to-Back Testing

Compare the new model against the current production model on the same data.

```python
def test_back_to_back(new_model, champion_model, test_data, max_degradation=0.01):
    """New model should not significantly underperform champion."""
    new_metrics = evaluate(new_model, test_data)
    champ_metrics = evaluate(champion_model, test_data)
    for metric_name in ['auc', 'f1', 'precision', 'recall']:
        degradation = champ_metrics[metric_name] - new_metrics[metric_name]
        assert degradation < max_degradation, \
            f"{metric_name} regressed by {degradation:.4f} (max allowed: {max_degradation})"
```

---

## 7. Testing Checklist for ML Pipeline Deployment

### Pre-Deployment

- [ ] All unit tests pass (feature engineering, preprocessing, postprocessing)
- [ ] Data validation suite passes on latest data (schema, completeness, distributions)
- [ ] Model trains successfully on sample data in CI environment
- [ ] Model quality gates pass (performance, fairness, latency, model size)
- [ ] Behavioral test suite passes (invariance, directional, minimum functionality)
- [ ] No regression on golden dataset
- [ ] Back-to-back comparison against champion model passes
- [ ] Model artifact is serializable, loadable, and produces deterministic predictions
- [ ] Feature pipeline integration test passes (model receives correct features at serving time)
- [ ] Serving endpoint returns valid responses for well-formed requests
- [ ] Serving endpoint returns proper errors for malformed requests
- [ ] Load test passes: throughput and latency within SLA under expected traffic
- [ ] Model card or documentation updated with new version details

### During Deployment

- [ ] Shadow deployment running and comparison metrics are healthy (see Section 8)
- [ ] Canary deployment (if applicable) showing stable metrics on live traffic
- [ ] Rollback procedure tested and ready
- [ ] Monitoring dashboards active (predictions, latency, errors, drift)
- [ ] Alerting configured for critical metric thresholds

### Post-Deployment

- [ ] Production metrics match pre-deployment expectations (within 5%)
- [ ] No increase in error rates or latency
- [ ] Prediction distribution matches expected distribution
- [ ] Ground-truth labels (when available) confirm expected performance
- [ ] Drift monitoring baseline updated with new model's reference data

---

## 8. Shadow Deployment Testing

### What is Shadow Deployment?

Shadow (dark launch) deployment routes live traffic to both the production model and the candidate model. The production model serves responses; the candidate's predictions are logged but not returned.

```
Live Traffic
     |
     v
  [Router]
   /    \
  v      v
[Production Model]    [Shadow Model]
  |                      |
  v                      v
Response to User    Log Predictions (not returned)
                         |
                         v
                   Compare & Analyze
```

### Shadow Deployment Test Plan

| Phase | Duration | Success Criteria |
|-------|----------|-----------------|
| **Warm-up** | 1-4 hours | Shadow model serves without errors; latency stable |
| **Comparison** | 1-7 days | Prediction agreement rate > 95%; no systematic divergence |
| **Analysis** | 1-2 days | Review disagreements; confirm shadow model is better on labeled subset |
| **Promotion** | -- | If all criteria met, promote shadow to production |

### Metrics to Compare

| Metric | How to Compare | Alert If |
|--------|---------------|----------|
| Prediction agreement | % of predictions that match between models | < 90% agreement |
| Prediction distribution | KL divergence or PSI between output distributions | PSI > 0.2 |
| Disagreement analysis | Characterize where models disagree by segment | Systematic pattern in disagreements |
| Latency | P50, P95, P99 comparison | Shadow > 2x production latency |
| Error rate | % of requests that fail or timeout | Shadow error rate > 1% |
| Resource usage | CPU, memory, GPU utilization | Shadow > 1.5x production resources |

### Shadow Deployment Pitfalls

| Pitfall | Description | Mitigation |
|---------|------------|------------|
| **Stale features** | Shadow model may use different feature versions | Ensure both models receive identical feature vectors |
| **Side effects** | Shadow predictions accidentally affect downstream systems | Strictly isolate shadow output from production data flows |
| **Insufficient traffic** | Not enough traffic to draw statistical conclusions | Run longer or route 100% of traffic to both models |
| **Label delay** | Cannot validate accuracy until labels arrive | Use proxy metrics (agreement, distribution) during shadow period |
| **Cost** | Running two models doubles inference cost | Budget for shadow period; limit to 1-2 weeks |

---

## 9. Further Reading

### Official Documentation

- **pytest**: https://docs.pytest.org/
- **Great Expectations**: https://docs.greatexpectations.io/
- **Deepchecks**: https://docs.deepchecks.com/
- **CheckList**: https://github.com/marcotcr/checklist

### Key Papers

- Ribeiro et al. (2020) "Beyond Accuracy: Behavioral Testing of NLP Models with CheckList" -- behavioral testing framework
- Breck et al. (2017) "The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction" -- Google's ML testing rubric
- Sculley et al. (2015) "Hidden Technical Debt in Machine Learning Systems" -- seminal paper on ML systems technical debt
- Amershi et al. (2019) "Software Engineering for Machine Learning: A Case Study" -- Microsoft's ML engineering practices
- Polyzotis et al. (2019) "Data Lifecycle Challenges in Production Machine Learning" -- data validation at scale

### Industry Guides

- Google ML Testing Guide: https://developers.google.com/machine-learning/testing-debugging
- Microsoft Responsible AI Toolbox: https://responsibleaitoolbox.ai/
- MLOps Community Testing Patterns: https://ml-ops.org/content/mlops-principles
