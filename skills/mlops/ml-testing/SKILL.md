---
name: ml-testing
description: >
  Test ML models, data pipelines, and features comprehensively. Covers ML testing pyramid (data, feature, model,
  integration, system tests), model unit tests (prediction shape, output range, determinism), quality gate tests
  (accuracy thresholds, performance benchmarks), behavioral testing (invariance, directional, minimum functionality),
  metamorphic testing, regression testing, smoke tests, shadow testing, A/B testing, load testing, data pipeline
  testing, property-based testing with Hypothesis, pytest fixtures, synthetic data generation, CI/CD integration,
  and test coverage for ML code. Use when writing tests for ML systems or setting up quality gates.
license: Apache-2.0
metadata:
  author: mlops-skills
  version: "1.0"
  category: mlops
---

# ML Testing

## Overview

ML testing goes beyond traditional software testing to validate data quality, model behavior,
and pipeline correctness. A comprehensive test suite catches bugs before they reach production.

## ML Testing Pyramid

```
          ┌─────────────┐
          │   System    │  End-to-end pipeline tests
          │   Tests     │  (slow, expensive)
         ─┤            ├─
        ┌─┴─────────────┴─┐
        │  Integration    │  Component interaction tests
        │  Tests          │
       ─┤                ├─
      ┌─┴─────────────────┴─┐
      │  Model Tests        │  Quality, behavior, regression
      │                     │
     ─┤                    ├─
    ┌─┴─────────────────────┴─┐
    │  Feature / Transform    │  Pipeline correctness
    │  Tests                  │
   ─┤                        ├─
  ┌─┴─────────────────────────┴─┐
  │  Data Tests                 │  Schema, quality, freshness
  │                             │  (fast, cheap)
  └─────────────────────────────┘
```

## Step-by-Step Instructions

### 1. Data Tests

```python
import pytest
import pandas as pd
import pandera as pa

class TestDataQuality:
    @pytest.fixture
    def training_data(self):
        return pd.read_parquet("data/train.parquet")

    def test_no_nulls_in_required_columns(self, training_data):
        required = ["user_id", "target", "timestamp"]
        for col in required:
            assert training_data[col].isnull().sum() == 0, f"Nulls in {col}"

    def test_no_duplicate_ids(self, training_data):
        assert training_data["user_id"].is_unique

    def test_target_distribution(self, training_data):
        """Target should not be severely imbalanced."""
        ratio = training_data["target"].mean()
        assert 0.05 < ratio < 0.95, f"Target ratio: {ratio}"

    def test_feature_ranges(self, training_data):
        assert training_data["age"].between(0, 150).all()
        assert training_data["score"].between(0, 1).all()

    def test_data_freshness(self, training_data):
        max_ts = pd.to_datetime(training_data["timestamp"]).max()
        staleness = (pd.Timestamp.now() - max_ts).total_seconds() / 3600
        assert staleness < 48, f"Data is {staleness:.1f} hours old"

    def test_schema(self, training_data):
        schema = pa.DataFrameSchema({
            "user_id": pa.Column(int, nullable=False, unique=True),
            "age": pa.Column(int, pa.Check.in_range(0, 150)),
            "target": pa.Column(int, pa.Check.isin([0, 1])),
        })
        schema.validate(training_data)
```

### 2. Model Unit Tests

```python
import numpy as np
import joblib

class TestModel:
    @pytest.fixture
    def model(self):
        return joblib.load("models/model.joblib")

    @pytest.fixture
    def sample_input(self):
        return np.random.randn(1, 10)  # 10 features

    def test_prediction_shape(self, model, sample_input):
        pred = model.predict(sample_input)
        assert pred.shape == (1,)

    def test_prediction_type(self, model, sample_input):
        pred = model.predict(sample_input)
        assert pred.dtype in [np.int64, np.float64]

    def test_prediction_range(self, model, sample_input):
        """Predictions should be valid class labels."""
        pred = model.predict(sample_input)
        assert all(p in [0, 1] for p in pred)

    def test_probability_range(self, model, sample_input):
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(sample_input)
            assert np.all(proba >= 0) and np.all(proba <= 1)
            assert np.allclose(proba.sum(axis=1), 1.0)

    def test_determinism(self, model, sample_input):
        """Same input should give same output."""
        pred1 = model.predict(sample_input)
        pred2 = model.predict(sample_input)
        np.testing.assert_array_equal(pred1, pred2)

    def test_batch_prediction(self, model):
        """Model should handle batch inputs."""
        batch = np.random.randn(100, 10)
        pred = model.predict(batch)
        assert pred.shape == (100,)
```

### 3. Behavioral Tests

```python
class TestModelBehavior:
    """Behavioral tests check expected model behavior without exact values."""

    @pytest.fixture
    def model(self):
        return joblib.load("models/model.joblib")

    def test_invariance(self, model):
        """Prediction should not change for irrelevant feature perturbations."""
        base = np.array([[25, 50000, 1, 0, 0, 0, 0, 0, 0, 0]])  # base input
        perturbed = base.copy()
        perturbed[0, -1] = 1  # Change irrelevant feature

        assert model.predict(base) == model.predict(perturbed)

    def test_directional(self, model):
        """Higher income should increase credit approval probability."""
        low_income = np.array([[30, 30000, 1, 0, 0, 0, 0, 0, 0, 0]])
        high_income = np.array([[30, 100000, 1, 0, 0, 0, 0, 0, 0, 0]])

        prob_low = model.predict_proba(low_income)[0, 1]
        prob_high = model.predict_proba(high_income)[0, 1]
        assert prob_high >= prob_low

    def test_minimum_functionality(self, model):
        """Model should correctly predict obvious cases."""
        obvious_positive = create_obvious_positive_case()
        obvious_negative = create_obvious_negative_case()

        assert model.predict(obvious_positive)[0] == 1
        assert model.predict(obvious_negative)[0] == 0
```

### 4. Quality Gate Tests

```python
class TestModelQuality:
    """Quality gates that must pass before deployment."""

    @pytest.fixture
    def model_and_data(self):
        model = joblib.load("models/model.joblib")
        test_df = pd.read_parquet("data/test.parquet")
        X = test_df.drop("target", axis=1)
        y = test_df["target"]
        return model, X, y

    def test_accuracy_threshold(self, model_and_data):
        model, X, y = model_and_data
        accuracy = accuracy_score(y, model.predict(X))
        assert accuracy >= 0.85, f"Accuracy {accuracy} below 0.85"

    def test_f1_threshold(self, model_and_data):
        model, X, y = model_and_data
        f1 = f1_score(y, model.predict(X), average="weighted")
        assert f1 >= 0.80, f"F1 {f1} below 0.80"

    def test_no_regression(self, model_and_data):
        """New model should not be worse than baseline."""
        model, X, y = model_and_data
        baseline = joblib.load("models/baseline.joblib")

        new_f1 = f1_score(y, model.predict(X), average="weighted")
        baseline_f1 = f1_score(y, baseline.predict(X), average="weighted")

        assert new_f1 >= baseline_f1 - 0.01, \
            f"Regression: new F1 {new_f1} < baseline {baseline_f1}"

    def test_latency(self, model_and_data):
        """Prediction latency should be within SLA."""
        model, X, _ = model_and_data
        import time
        start = time.time()
        model.predict(X[:1])
        latency_ms = (time.time() - start) * 1000
        assert latency_ms < 100, f"Latency {latency_ms}ms exceeds 100ms SLA"
```

### 5. Pipeline Integration Tests

```python
class TestTrainingPipeline:
    def test_pipeline_end_to_end(self, tmp_path):
        """Full pipeline should produce a valid model."""
        config = {
            "data": {"path": "data/sample.parquet"},
            "model": {"type": "random_forest", "params": {"n_estimators": 10}},
            "output": {"path": str(tmp_path / "model.joblib")},
        }
        result = run_training_pipeline(config)
        assert result["status"] == "success"
        assert (tmp_path / "model.joblib").exists()

    def test_pipeline_idempotent(self, tmp_path):
        """Running pipeline twice should produce equivalent results."""
        config = {"seed": 42, "output": str(tmp_path)}
        result1 = run_training_pipeline(config)
        result2 = run_training_pipeline(config)
        assert abs(result1["metrics"]["f1"] - result2["metrics"]["f1"]) < 0.001
```

## Best Practices

1. **Write data tests first** - Catch issues at the source
2. **Use behavioral tests** - More robust than exact value assertions
3. **Automate quality gates** in CI/CD before model promotion
4. **Test on representative slices** not just overall metrics
5. **Keep test data separate** from training data
6. **Use synthetic data** for edge case testing
7. **Test model serving** endpoints, not just model objects
8. **Run tests on every commit** and before every deployment
9. **Track test metrics over time** to catch gradual degradation
10. **Test the tests** - Verify tests catch known failures

## Scripts

- `scripts/test_model.py` - Comprehensive pytest-based model test suite
- `scripts/test_data_pipeline.py` - Data pipeline test suite

## References

See [references/REFERENCE.md](references/REFERENCE.md) for testing strategies by model type.
