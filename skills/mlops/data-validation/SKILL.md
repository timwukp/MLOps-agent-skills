---
name: data-validation
description: >
  Validate data quality for ML pipelines using Great Expectations, Pandera, Pydantic, and Deequ. Covers schema validation,
  data profiling, data quality checks (completeness, uniqueness, consistency, accuracy, timeliness), data contracts,
  anomaly detection in datasets, automated data testing in CI/CD, handling schema evolution and breaking changes,
  data quality dashboards, alerting on validation failures, and integration with orchestrators like Airflow and Prefect.
  Use when building data quality gates, profiling datasets, enforcing data contracts, or debugging data issues.
license: Apache-2.0
metadata:
  author: mlops-skills
  version: "1.0"
  category: mlops
---

# Data Validation for ML Pipelines

## Overview

Data validation ensures that data feeding ML models meets quality, schema, and business
requirements. Poor data quality is the #1 cause of ML model failures in production.

## When to Use This Skill

- Adding quality gates to data pipelines
- Profiling new datasets before model training
- Setting up data contracts between teams
- Debugging unexpected model performance degradation
- Automating data quality checks in CI/CD

## Data Quality Dimensions

| Dimension | Description | Example Check |
|-----------|-------------|---------------|
| Completeness | No missing required values | `null_count(col) == 0` |
| Uniqueness | No unexpected duplicates | `unique_count(id) == row_count` |
| Consistency | Values conform to rules | `min(age) >= 0` |
| Accuracy | Values are correct | `mean(price) within expected range` |
| Timeliness | Data is fresh enough | `max(timestamp) > now() - 1h` |
| Validity | Values in expected domain | `col IN ('A', 'B', 'C')` |

## Step-by-Step Instructions

### 1. Great Expectations Validation

```python
import great_expectations as gx

# Initialize context
context = gx.get_context()

# Connect to data
datasource = context.data_sources.add_pandas("my_datasource")
asset = datasource.add_dataframe_asset("my_asset")
batch = asset.add_batch_definition_whole_dataframe("my_batch").get_batch(
    batch_parameters={"dataframe": df}
)

# Create expectations
suite = context.suites.add(gx.ExpectationSuite(name="feature_quality"))
suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="user_id"))
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(
    column="age", min_value=0, max_value=150
))
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeUnique(column="user_id"))
suite.add_expectation(gx.expectations.ExpectTableRowCountToBeBetween(
    min_value=1000, max_value=10000000
))

# Validate
results = batch.validate(suite)
if not results.success:
    for result in results.results:
        if not result.success:
            print(f"FAILED: {result.expectation_config}")
```

### 2. Pandera Schema Validation

```python
import pandera as pa
from pandera import Column, Check, DataFrameSchema

# Define schema
schema = DataFrameSchema({
    "user_id": Column(int, Check.gt(0), unique=True, nullable=False),
    "age": Column(int, Check.in_range(0, 150), nullable=False),
    "income": Column(float, Check.gt(0), nullable=True),
    "category": Column(str, Check.isin(["A", "B", "C"]), nullable=False),
    "score": Column(float, Check.in_range(0.0, 1.0)),
    "created_at": Column(pa.DateTime, nullable=False),
}, coerce=True)

# Validate
validated_df = schema.validate(df, lazy=True)  # lazy=True collects all errors

# Decorator-based validation
@pa.check_input(schema)
def train_model(df):
    """Input data is automatically validated."""
    pass
```

### 3. Data Profiling

```python
# Using ydata-profiling (formerly pandas-profiling)
from ydata_profiling import ProfileReport

profile = ProfileReport(df, title="Training Data Profile", minimal=True)
profile.to_file("data_profile.html")

# Using whylogs
import whylogs as why

results = why.log(df)
profile = results.profile()
view = profile.view()

# Get summary statistics
summary = view.to_pandas()
print(summary[["distribution/mean", "distribution/stddev", "types/fractional"]])
```

### 4. Data Contracts

```yaml
# data_contract.yaml
contract:
  name: user_features_v2
  owner: ml-team
  description: User feature table for recommendation model
  sla:
    freshness: 1h
    completeness: 99.5%
    row_count_min: 100000

  schema:
    columns:
      - name: user_id
        type: integer
        nullable: false
        unique: true
      - name: age
        type: integer
        nullable: false
        checks:
          - min: 0
          - max: 150
      - name: lifetime_value
        type: float
        nullable: true
        checks:
          - min: 0.0

  quality_rules:
    - rule: "duplicate_ratio < 0.001"
    - rule: "null_ratio(income) < 0.05"
    - rule: "mean(age) BETWEEN 25 AND 55"
```

```python
import yaml

def enforce_contract(df, contract_path):
    """Enforce a data contract against a DataFrame."""
    with open(contract_path) as f:
        contract = yaml.safe_load(f)["contract"]

    violations = []

    # Check SLA
    if "sla" in contract:
        sla = contract["sla"]
        if "row_count_min" in sla and len(df) < sla["row_count_min"]:
            violations.append(f"Row count {len(df)} < min {sla['row_count_min']}")

    # Check schema
    for col_spec in contract["schema"]["columns"]:
        col = col_spec["name"]
        if col not in df.columns:
            violations.append(f"Missing column: {col}")
            continue
        if not col_spec.get("nullable", True) and df[col].isnull().any():
            violations.append(f"Null values in non-nullable column: {col}")

    return violations
```

### 5. Automated Data Testing in CI/CD

```python
# tests/test_data_quality.py
import pytest
import pandera as pa

@pytest.fixture
def training_data():
    return pd.read_parquet("data/training.parquet")

def test_no_nulls_in_target(training_data):
    assert training_data["target"].isnull().sum() == 0

def test_feature_ranges(training_data):
    assert training_data["age"].between(0, 150).all()
    assert training_data["score"].between(0, 1).all()

def test_no_duplicates(training_data):
    assert training_data["user_id"].is_unique

def test_data_freshness(training_data):
    max_ts = training_data["updated_at"].max()
    assert (pd.Timestamp.now() - max_ts).total_seconds() < 3600

def test_distribution_stability(training_data, reference_data):
    """Check that feature distributions haven't shifted dramatically."""
    for col in ["age", "income", "score"]:
        ref_mean = reference_data[col].mean()
        cur_mean = training_data[col].mean()
        assert abs(cur_mean - ref_mean) / ref_mean < 0.2  # 20% tolerance
```

### 6. Schema Evolution Handling

```python
def validate_schema_evolution(current_schema, new_schema):
    """Check if schema change is backward compatible."""
    breaking_changes = []

    current_cols = {c.name: c for c in current_schema}
    new_cols = {c.name: c for c in new_schema}

    # Removed columns are breaking
    for col in current_cols:
        if col not in new_cols:
            breaking_changes.append(f"Column removed: {col}")

    # Type changes are breaking
    for col in current_cols:
        if col in new_cols and current_cols[col].type != new_cols[col].type:
            breaking_changes.append(
                f"Type changed for {col}: {current_cols[col].type} -> {new_cols[col].type}"
            )

    # New non-nullable columns are breaking
    for col in new_cols:
        if col not in current_cols and not new_cols[col].nullable:
            breaking_changes.append(f"New non-nullable column: {col}")

    return breaking_changes
```

## Best Practices

1. **Validate early** - Check data at ingestion, not just before training
2. **Use data contracts** - Formal agreements between data producers and consumers
3. **Profile regularly** - Track statistical properties over time to detect drift
4. **Automate in CI/CD** - Run data tests alongside code tests
5. **Alert, don't just log** - Failed validations should trigger notifications
6. **Version your expectations** - Track validation rules in version control
7. **Start simple** - Begin with null checks and type validation, add complexity later
8. **Test edge cases** - Empty datasets, single-row datasets, extreme values
9. **Document exceptions** - When you skip validation, document why
10. **Monitor validation pass rates** - Track the trend of validation failures

## Scripts

- `scripts/validate_data.py` - Comprehensive data validation with GX and Pandera
- `scripts/data_contract.py` - Data contract enforcement engine

## References

See [references/REFERENCE.md](references/REFERENCE.md) for tool comparisons and patterns.
