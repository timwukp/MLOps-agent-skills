# Data Validation Reference Guide

## Tool Comparison

| Feature                  | Great Expectations       | Pandera              | Cerberus             | Pydantic              |
|--------------------------|--------------------------|----------------------|----------------------|-----------------------|
| **Primary Use Case**     | Data pipeline validation | DataFrame validation | Dict/JSON validation | Data model validation |
| **Data Backends**        | Pandas, Spark, SQL       | Pandas, Polars, Pyspark | Python dicts      | Python objects        |
| **Schema Definition**    | JSON/YAML + Python API   | Python classes/decorators | Python dicts/YAML | Python type hints     |
| **Built-in Profiling**   | Yes (automatic suite)    | No                   | No                   | No                    |
| **Data Docs / Reports**  | Rich HTML reports        | Error summaries      | Error dicts          | Error dicts           |
| **Checkpoint System**    | Yes                      | No                   | No                   | No                    |
| **Pipeline Integration** | Airflow, Prefect, Dagster| Lightweight / any    | Any                  | FastAPI, any          |
| **Learning Curve**       | Steep                    | Moderate             | Low                  | Low                   |
| **Best For**             | Enterprise data pipelines| Pandas-centric tests | API input validation | API models & configs  |

### When to Use Which

- **Great Expectations**: Large-scale pipelines needing HTML reports, profiling, and orchestrator integration.
- **Pandera**: Lightweight DataFrame validation with Pythonic schema definitions.
- **Cerberus**: Validating configuration dictionaries, JSON payloads, or nested documents.
- **Pydantic**: Structured data models in application code, especially with FastAPI.

---

## Great Expectations Suite Setup and Checkpoint Configuration (GX 1.x)

### Define an Expectation Suite

```python
import great_expectations as gx

context = gx.get_context()
suite = context.suites.add(gx.ExpectationSuite(name="orders_suite"))

suite.add_expectation(
    gx.expectations.ExpectColumnValuesToNotBeNull(column="order_id")
)
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeBetween(
        column="quantity", min_value=1, max_value=10000
    )
)
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeUnique(column="order_id")
)
```

### Configure and Run a Checkpoint

```python
# Data source -> asset -> batch definition
datasource = context.data_sources.add_pandas("my_datasource")
asset = datasource.add_dataframe_asset("orders")
batch_definition = asset.add_batch_definition_whole_dataframe("orders_batch")

# Bind suite + batch definition into a validation definition
validation_definition = context.validation_definitions.add(
    gx.ValidationDefinition(
        name="orders_validation",
        data=batch_definition,
        suite=suite,
    )
)

# Checkpoint wraps validation definitions plus post-run actions
checkpoint = context.checkpoints.add(
    gx.Checkpoint(
        name="orders_checkpoint",
        validation_definitions=[validation_definition],
        actions=[gx.checkpoint.UpdateDataDocsAction(name="update_data_docs")],
    )
)

result = checkpoint.run(batch_parameters={"dataframe": df})
assert result.success, "Data validation failed!"
```

---

## Pandera Schema Definition Examples

### Class-Based Schema (Recommended)

```python
import pandera.pandas as pa  # pandas-specific namespace (pandera 0.20+)
from pandera.typing import Series, DataFrame

class OrderSchema(pa.DataFrameModel):
    order_id: Series[int] = pa.Field(gt=0, unique=True, nullable=False)
    customer_id: Series[int] = pa.Field(gt=0, nullable=False)
    amount: Series[float] = pa.Field(ge=0.01, le=1_000_000)
    status: Series[str] = pa.Field(isin=["pending", "shipped", "delivered"])
    created_at: Series[pa.DateTime] = pa.Field(nullable=False)

    class Config:
        strict = True
        coerce = True

@pa.check_types
def process_orders(df: DataFrame[OrderSchema]) -> DataFrame[OrderSchema]:
    return df  # input and output validated automatically
```

---

## Data Contract Patterns

A **data contract** is a formal agreement between a data producer and consumer specifying schema, quality rules, SLAs, and ownership.

**Key components:**

1. **Schema definition**: Column names, types, nullability, constraints.
2. **Quality expectations**: Freshness, completeness thresholds, uniqueness rules.
3. **SLA**: Maximum delivery latency, update frequency.
4. **Ownership**: Team or individual responsible for the data.
5. **Change management**: How breaking changes are communicated and versioned.

**Enforcement strategy**: Producers validate before publishing (shift-left). Consumers validate after ingestion (defense-in-depth). A central schema registry stores versioned contracts. CI/CD pipelines run contract tests on schema-modifying pull requests.

---

## Data Quality Dimensions

| Dimension       | Definition                                    | Example Check                               |
|-----------------|-----------------------------------------------|---------------------------------------------|
| **Completeness**| All required data is present                  | Null rate for column < 1%                   |
| **Accuracy**    | Data correctly represents real-world values   | Zip codes match a reference table           |
| **Consistency** | No contradictions across sources              | `start_date` <= `end_date` in every row     |
| **Timeliness**  | Data available within the expected window     | Partition arrives within 2 hours of event   |
| **Uniqueness**  | No unintended duplicate records               | Primary key has zero duplicates             |
| **Validity**    | Data conforms to defined formats and ranges   | Email column matches regex pattern          |

---

## Integration with Airflow and Prefect

### Airflow: Great Expectations Operators (provider 1.0+)

```python
# GreatExpectationsOperator is legacy (provider <1.0). Provider 1.0+ ships
# GX 1.x-native operators:
from great_expectations_provider.operators.validate_checkpoint import (
    GXValidateCheckpointOperator,
)
from great_expectations_provider.operators.validate_dataframe import (
    GXValidateDataFrameOperator,
)

# Run a pre-configured checkpoint
validate_task = GXValidateCheckpointOperator(
    task_id="validate_orders",
    configure_checkpoint=configure_checkpoint,  # callable returning a gx.Checkpoint
)

# Or validate an in-memory DataFrame directly
validate_df_task = GXValidateDataFrameOperator(
    task_id="validate_orders_df",
    configure_dataframe=get_orders_df,  # callable returning a DataFrame
    expect=orders_suite,                # gx.ExpectationSuite or single expectation
)
# DAG: extract >> validate_task >> transform >> load
```

### Prefect: Pandera Validation in a Flow

```python
from prefect import flow, task

@task
def validate_data(df, schema):
    return schema.validate(df)

@flow(name="orders_pipeline")
def orders_pipeline():
    raw_df = extract()
    validated = validate_data(raw_df, OrderSchema)
    transformed = transform(validated)
    load(transformed)
```

---

## Common Pitfalls

1. **Validating only in production**: Run validation in CI with sample data to catch schema regressions before deployment.
2. **Ignoring distribution drift**: Static range checks miss gradual shifts. Add statistical tests (KS test, PSI) for continuous columns.
3. **Overly strict schemas**: Making every column non-nullable leads to brittle pipelines. Distinguish hard failures from warnings.
4. **No alerting on partial failures**: A suite passing at 99% can hide critical column-level failures. Set per-expectation thresholds.
5. **Skipping validation on backfills**: Historical data often has different quality. Validate backfill batches separately.
6. **Not versioning expectation suites**: Store suites in version control alongside pipeline code.

---

## Best Practices

- Validate at every stage boundary (ingestion, transformation, serving).
- Use profiling to bootstrap initial expectations, then refine manually.
- Tag expectations by severity (critical vs. warning) to allow graceful degradation.
- Store validation results in a queryable backend for historical trend analysis.
- Automate data contract review as part of the pull request process.

---

## Further Reading

- [Great Expectations Documentation](https://docs.greatexpectations.io/)
- [Pandera Documentation](https://pandera.readthedocs.io/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Data Contracts - PayPal Engineering Blog](https://medium.com/paypal-tech/the-next-big-data-challenge-data-contracts-d1e4ee5e0f38)
- [Data Quality Fundamentals (O'Reilly)](https://www.oreilly.com/library/view/data-quality-fundamentals/9781098112035/)
- [DAMA DMBOK - Data Quality Chapter](https://www.dama.org/cpages/body-of-knowledge)
