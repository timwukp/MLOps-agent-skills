# Feature Store Reference Guide

## Tool Comparison

| Feature                    | Feast                     | Tecton                   | Hopsworks                | Databricks Feature Store  |
|----------------------------|---------------------------|--------------------------|--------------------------|---------------------------|
| **License**                | Apache 2.0 (open source)  | Commercial (SaaS)        | Open source + Enterprise | Commercial (Databricks)   |
| **Offline Store**          | BigQuery, Redshift, Snowflake, file | Spark, Rift (native) | Hive, HopsFS       | Delta Lake                |
| **Online Store**           | Redis, DynamoDB, Datastore| DynamoDB, Redis          | RonDB (MySQL NDB)        | Cosmos DB, DynamoDB       |
| **Streaming Ingestion**    | Push-based (limited)      | Native (Spark Streaming) | Native (Kafka, Spark)    | Delta Live Tables         |
| **Feature Transformations**| External (pre-compute)    | Native (Python, SQL)     | Native (PySpark, SQL)    | Native (SQL, PySpark)     |
| **Point-in-Time Joins**    | Yes                       | Yes                      | Yes                      | Yes                       |
| **Feature Monitoring**     | Limited                   | Built-in                 | Built-in                 | Built-in (via Lakehouse)  |
| **Access Control**         | Basic                     | Enterprise RBAC          | Project-based RBAC       | Unity Catalog             |
| **Best For**               | Open-source lightweight   | Real-time ML at scale    | Full ML platform         | Databricks ecosystem      |

---

## Feast Architecture

```
+-------------------+       +-------------------+       +-------------------+
|   Feature         |       |   Offline Store    |       |   Online Store    |
|   Definitions     |------>|   (BigQuery /      |------>|   (Redis /        |
|   (Python)        |       |    Redshift)       |       |    DynamoDB)      |
+-------------------+       +-------------------+       +-------------------+
        |                           |                           |
        v                           v                           v
+-------------------+       +-------------------+       +-------------------+
|   Registry        |       | get_historical_   |       | get_online_       |
|   (file/SQL/S3)   |       | features (train)  |       | features (serve)  |
+-------------------+       +-------------------+       +-------------------+
```

- **Registry**: Stores feature view definitions, data sources, and entity metadata.
- **Offline Store**: Warehouse holding historical feature values for training.
- **Online Store**: Low-latency key-value store for real-time serving.

### Feast Project Setup

```yaml
# feature_store.yaml
project: my_ml_project
registry: s3://my-bucket/feast/registry.db
provider: aws
online_store:
  type: redis
  connection_string: "redis-host:6379"
offline_store:
  type: redshift
  region: us-east-1
  cluster_id: my-redshift-cluster
  database: features
```

### Define Entities and Feature Views

```python
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String
from datetime import timedelta

customer = Entity(name="customer_id", join_keys=["customer_id"])

customer_features = FeatureView(
    name="customer_features",
    entities=[customer],
    ttl=timedelta(days=365),
    schema=[
        Field(name="total_purchases", dtype=Int64),
        Field(name="avg_order_value", dtype=Float32),
        Field(name="loyalty_tier", dtype=String),
    ],
    source=FileSource(
        path="s3://my-bucket/data/customer_features.parquet",
        timestamp_field="event_timestamp",
    ),
    online=True,
)
```

---

## Point-in-Time Joins

PIT joins prevent **data leakage** by ensuring feature values correspond to what was known at the time of each training example.

### ASCII Diagram

```
Entity Events (Labels)                  Feature Table (Versioned)
+-----------+------------+-------+      +-----------+------------+-------+
| entity_id | event_time | label |      | entity_id | feature_ts | value |
+-----------+------------+-------+      +-----------+------------+-------+
| A         | 2024-03-15 | 1     |      | A         | 2024-01-01 | 10    |
| A         | 2024-06-20 | 0     |      | A         | 2024-04-01 | 20    |
| B         | 2024-05-10 | 1     |      | A         | 2024-07-01 | 30    |
+-----------+------------+-------+      | B         | 2024-02-01 | 50    |
                                        | B         | 2024-06-01 | 60    |
         |                              +-----------+------------+-------+
         v
  PIT Join Result (uses latest value BEFORE event_time)
  +-----------+------------+-------+---------+
  | entity_id | event_time | label | value   |
  +-----------+------------+-------+---------+
  | A         | 2024-03-15 | 1     | 10      |  <-- latest before 03-15
  | A         | 2024-06-20 | 0     | 20      |  <-- latest before 06-20
  | B         | 2024-05-10 | 1     | 50      |  <-- latest before 05-10
  +-----------+------------+-------+---------+
```

### Feast Historical Retrieval

```python
from feast import FeatureStore
import pandas as pd

store = FeatureStore(repo_path=".")
entity_df = pd.DataFrame({
    "customer_id": [1001, 1002, 1001],
    "event_timestamp": pd.to_datetime([
        "2024-03-15 10:00:00", "2024-05-10 14:00:00", "2024-06-20 09:00:00",
    ]),
})
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=["customer_features:total_purchases", "customer_features:avg_order_value"],
).to_df()
```

---

## Online vs Offline Feature Serving

| Aspect              | Offline Serving                          | Online Serving                       |
|---------------------|------------------------------------------|--------------------------------------|
| **Use Case**        | Training data generation, batch scoring  | Real-time inference                  |
| **Latency**         | Seconds to minutes                       | Single-digit milliseconds            |
| **Data Volume**     | Full historical range                    | Latest feature values only           |
| **Storage**         | Data warehouse / data lake               | Key-value store (Redis, DynamoDB)    |
| **Join Type**       | Point-in-time join on timestamps         | Simple key lookup                    |

---

## Feature Freshness and Consistency Patterns

| Tier               | Latency        | Pattern                                          | Example                 |
|--------------------|----------------|--------------------------------------------------|-------------------------|
| **Batch**          | Hours to days  | Scheduled ETL writes to offline/online store     | Daily aggregates        |
| **Near-real-time** | Minutes        | Micro-batch streaming (every 5 min)              | 15-min rolling averages |
| **Real-time**      | Seconds        | Stream processing writes directly to online store| Session click count     |
| **On-demand**      | At request time| Computed at inference, not stored                | Request metadata        |

### Consistency: Dual-Write with Reconciliation

```
  Event Stream (Kafka)
        |
        +---> Stream Processor ---> Online Store (Redis)
        |
        +---> Batch Pipeline -----> Offline Store (Warehouse)
                   |
                   +---> Reconciliation Job (compare online vs offline)
```

---

## Feature Store Integration with Pipelines

### Training Pipeline

```python
store = FeatureStore(repo_path="feature_repo/")
entity_df = build_entity_dataframe(label_source="s3://labels/training.parquet")
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=["customer_features:total_purchases", "customer_features:loyalty_tier"],
).to_df()
model = train_model(training_df)
```

### Serving Pipeline

```python
from fastapi import FastAPI
from feast import FeatureStore

app = FastAPI()
store = FeatureStore(repo_path="feature_repo/")

@app.post("/predict")
def predict(customer_id: int):
    features = store.get_online_features(
        features=["customer_features:total_purchases"],
        entity_rows=[{"customer_id": customer_id}],
    ).to_dict()
    return {"prediction": model.predict(features)}
```

---

## Common Pitfalls

1. **Skipping point-in-time joins**: Using latest feature values for all training rows causes future data leakage.
2. **Training/serving skew**: Computing features differently in batch vs. online. Use the feature store as the single source of truth.
3. **Ignoring feature TTL**: Without time-to-live, stale values persist in the online store indefinitely.
4. **Over-materializing**: Pushing every feature to the online store wastes resources. Only materialize what real-time serving needs.
5. **Missing monitoring**: Not tracking feature staleness, null rates, or distribution shifts.

---

## Best Practices

- Treat feature definitions as code: version control, code review, CI/CD.
- Use consistent entity keys across all feature views for easy joins.
- Document each feature with a description, owner, and data lineage.
- Set appropriate TTLs based on feature freshness requirements.
- Start with batch features; add real-time features only when latency demands it.

---

## Managed: SageMaker Feature Store

Amazon SageMaker Feature Store is a fully managed feature store: an online store (low-latency key-value lookups; `Standard` or `InMemory` storage) and an offline store (S3, registered as a Glue table -- `Default`/`Glue` or `Iceberg` table format) with automatic replication from online to offline. Every feature group requires a **record identifier** (entity key) and an **event time** feature -- the same two columns that drive point-in-time correctness in the diagrams above.

### Create a Feature Group (boto3)

```python
sm = boto3.client("sagemaker")
sm.create_feature_group(
    FeatureGroupName="customer-features",
    RecordIdentifierFeatureName="customer_id",
    EventTimeFeatureName="event_time",
    FeatureDefinitions=[
        {"FeatureName": "customer_id", "FeatureType": "String"},
        {"FeatureName": "event_time", "FeatureType": "String"},   # ISO-8601 or epoch
        {"FeatureName": "total_purchases", "FeatureType": "Integral"},
        {"FeatureName": "avg_order_value", "FeatureType": "Fractional"},
    ],
    OnlineStoreConfig={"EnableOnlineStore": True},
    OfflineStoreConfig={"S3StorageConfig": {"S3Uri": "s3://my-bucket/feature-store/"},
                        "TableFormat": "Iceberg"},
    RoleArn=role_arn,
)
```

### Ingestion and Online Reads

Writes go through the `sagemaker-featurestore-runtime` client (`put_record`, `batch_write_record`); online reads use `get_record` / `batch_get_record` for serving:

```python
fs_runtime = boto3.client("sagemaker-featurestore-runtime")
fs_runtime.put_record(
    FeatureGroupName="customer-features",
    Record=[{"FeatureName": "customer_id", "ValueAsString": "1001"},
            {"FeatureName": "event_time", "ValueAsString": "2026-07-27T00:00:00Z"},
            {"FeatureName": "total_purchases", "ValueAsString": "42"}],
)
rec = fs_runtime.get_record(FeatureGroupName="customer-features",
                            RecordIdentifierValueAsString="1001")
```

The SageMaker Python SDK v3 wraps this as `FeatureGroupManager` (`sagemaker.mlops.feature_store`) with `create`, `put_record`/`batch_write_record`, `get_record`, and pandas ingestion helpers (`ingestion_manager_pandas`).

### Offline Queries via Athena

The offline store is queryable with standard SQL through Athena (Glue catalog table is auto-created). The SDK's `AthenaQuery` / `DatasetBuilder` (`sagemaker.mlops.feature_store`) build training sets, including point-in-time joins against an entity dataframe:

```sql
-- latest value per entity as of a cutoff (dedup on event_time)
SELECT * FROM (
  SELECT *, ROW_NUMBER() OVER (
    PARTITION BY customer_id ORDER BY event_time DESC) AS rn
  FROM "sagemaker_featurestore"."customer_features_1690000000"
  WHERE event_time <= '2026-07-01T00:00:00Z') WHERE rn = 1;
```

### vs Feast

| Aspect | SageMaker Feature Store | Feast |
|--------|-------------------------|-------|
| Hosting | Fully managed (AWS) | Self-hosted (open source) |
| Online store | Managed KV (`Standard`) or `InMemory` tier | Redis, DynamoDB, Datastore -- you operate it |
| Offline store | S3 + Glue/Iceberg, Athena SQL | BigQuery, Redshift, Snowflake, files |
| Point-in-time joins | Via `DatasetBuilder` / Athena SQL | Native `get_historical_features` |
| Feature definitions | API/console (imperative) | Python files in git (declarative repo) |
| TTL / freshness | `TtlDuration` on online store | Per-feature-view `ttl` |
| Ecosystem fit | Tight SageMaker integration (pipelines, lineage, Clarify) | Cloud-agnostic, portable |
| Cost model | Pay per read/write/storage | Infra cost of stores you run |

Choose SageMaker Feature Store when the ML platform is already SageMaker and you want zero-ops online serving with Athena-queryable history; choose Feast when you need portability across clouds, git-reviewed feature definitions, or an existing Redis/BigQuery footprint.

---

## Further Reading

- [Feast Documentation](https://docs.feast.dev/)
- [Tecton Documentation](https://docs.tecton.ai/)
- [Hopsworks Feature Store](https://www.hopsworks.ai/feature-store)
- [Feature Stores for ML (Chip Huyen)](https://www.featurestore.org/)
- [Google Vertex AI Feature Store](https://cloud.google.com/vertex-ai/docs/featurestore/overview)
- [Uber Michelangelo Feature Store](https://www.uber.com/blog/michelangelo-machine-learning-platform/)
