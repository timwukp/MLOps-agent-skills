---
name: data-ingestion
description: >
  Design and implement data ingestion pipelines for ML workflows. Covers batch ingestion from CSV, Parquet, JSON, Avro,
  databases (PostgreSQL, MySQL, BigQuery), and cloud storage (S3, GCS, Azure Blob). Streaming ingestion with Kafka,
  Kinesis, Pulsar, and Flink. ETL/ELT pipeline design, data lake ingestion (Delta Lake, Iceberg, Hudi), schema evolution,
  data versioning with DVC and LakeFS, data catalog integration, incremental loading, partitioning, compression,
  error handling, retry logic, dead letter queues, idempotency, and exactly-once semantics for ML data pipelines.
license: Apache-2.0
metadata:
  author: mlops-skills
  version: "1.0"
  category: mlops
---

# Data Ingestion for ML Pipelines

## Overview

Data ingestion is the first step in any ML pipeline. This skill covers patterns for reliably
collecting, transforming, and loading data from diverse sources into ML-ready formats.

## When to Use This Skill

- Setting up new data pipelines for ML projects
- Migrating from batch to streaming ingestion
- Integrating new data sources into existing pipelines
- Designing data lake/warehouse ingestion architectures
- Troubleshooting data pipeline failures

## Architecture Patterns

### Batch Ingestion Pipeline

```
Source -> Extract -> Validate -> Transform -> Load -> Catalog
  |                    |                        |
  v                    v                        v
Schedule          Dead Letter              Partition &
(cron/trigger)      Queue                  Compress
```

### Streaming Ingestion Pipeline

```
Source -> Message Queue -> Consumer -> Buffer -> Write -> Store
  |         (Kafka)          |          |                  |
  v                          v          v                  v
Producer              Deserialize   Micro-batch      Checkpoint
Config                & Validate    Accumulate       & Commit
```

## Step-by-Step Instructions

### 1. Batch Data Ingestion

#### From Files (CSV, Parquet, JSON, Avro)

```python
import pandas as pd
import pyarrow.parquet as pq

# CSV with schema enforcement
df = pd.read_csv("data.csv", dtype={"id": int, "value": float})

# Parquet (preferred for ML - columnar, compressed, typed)
df = pq.read_table("data.parquet").to_pandas()

# JSON Lines (common for API responses)
df = pd.read_json("data.jsonl", lines=True)

# Avro (schema-embedded, good for streaming)
import fastavro
with open("data.avro", "rb") as f:
    records = list(fastavro.reader(f))
```

#### From Databases

```python
from sqlalchemy import create_engine

engine = create_engine("postgresql://user:pass@host:5432/db")

# Full load
df = pd.read_sql("SELECT * FROM features", engine)

# Incremental load with watermark
df = pd.read_sql(
    "SELECT * FROM features WHERE updated_at > :watermark",
    engine,
    params={"watermark": last_watermark}
)
```

#### Incremental Loading Strategy

```python
def incremental_load(source, watermark_col, last_watermark, target_path):
    """Load only new/changed records since last watermark."""
    query = f"SELECT * FROM {source} WHERE {watermark_col} > '{last_watermark}'"
    new_data = pd.read_sql(query, engine)

    if len(new_data) > 0:
        # Append to existing dataset
        new_data.to_parquet(
            target_path,
            partition_cols=["date"],
            engine="pyarrow",
            existing_data_behavior="overwrite_or_ignore"
        )
        # Update watermark
        new_watermark = new_data[watermark_col].max()
        save_watermark(new_watermark)
    return len(new_data)
```

### 2. Streaming Data Ingestion

#### Kafka Consumer Pattern

```python
from confluent_kafka import Consumer, KafkaError
import json

conf = {
    "bootstrap.servers": "localhost:9092",
    "group.id": "ml-feature-consumer",
    "auto.offset.reset": "earliest",
    "enable.auto.commit": False,  # Manual commit for exactly-once
}

consumer = Consumer(conf)
consumer.subscribe(["feature-events"])

batch = []
BATCH_SIZE = 1000

while True:
    msg = consumer.poll(timeout=1.0)
    if msg is None:
        continue
    if msg.error():
        handle_error(msg)
        continue

    record = json.loads(msg.value().decode("utf-8"))
    batch.append(record)

    if len(batch) >= BATCH_SIZE:
        write_batch(batch)
        consumer.commit()
        batch = []
```

### 3. Data Lake Ingestion

#### Delta Lake

```python
from delta import DeltaTable
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.1.0") \
    .getOrCreate()

# Write with ACID transactions
df.write.format("delta") \
    .mode("append") \
    .partitionBy("date") \
    .save("/data/lake/features")

# Upsert (merge) for incremental updates
delta_table = DeltaTable.forPath(spark, "/data/lake/features")
delta_table.alias("target").merge(
    new_data.alias("source"),
    "target.id = source.id"
).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()
```

#### Apache Iceberg

```python
# Iceberg with PySpark
df.writeTo("catalog.db.features") \
    .using("iceberg") \
    .partitionedBy(days("timestamp")) \
    .createOrReplace()
```

### 4. Data Versioning

#### DVC (Data Version Control)

```bash
# Initialize DVC
dvc init
dvc remote add -d storage s3://my-bucket/dvc-store

# Track dataset
dvc add data/training_data.parquet
git add data/training_data.parquet.dvc .gitignore
git commit -m "Add training data v1"
dvc push

# Create a version tag
git tag -a "data-v1.0" -m "Initial training dataset"
```

#### LakeFS

```python
import lakefs_client
from lakefs_client.api import branches_api, objects_api

# Create branch for new data version
branches_api.create_branch(
    repository="ml-data",
    branch_creation={"name": "new-features", "source": "main"}
)

# Upload data
objects_api.upload_object(
    repository="ml-data",
    branch="new-features",
    path="features/v2/data.parquet",
    content=open("data.parquet", "rb")
)

# Commit and merge
commits_api.commit(repository="ml-data", branch="new-features",
                   commit_creation={"message": "Add new features"})
```

### 5. Schema Evolution

```python
import pyarrow as pa

# Define evolving schema
schema_v1 = pa.schema([
    ("user_id", pa.int64()),
    ("feature_a", pa.float64()),
])

schema_v2 = pa.schema([
    ("user_id", pa.int64()),
    ("feature_a", pa.float64()),
    ("feature_b", pa.string()),  # New column
])

# Merge schemas for backward compatibility
merged = pa.unify_schemas([schema_v1, schema_v2])
```

### 6. Data Partitioning Strategies

| Strategy | Best For | Example |
|----------|----------|---------|
| Date-based | Time-series data | `partition_cols=["year", "month"]` |
| Hash-based | Uniform distribution | `partition_cols=[hash(id) % N]` |
| Range-based | Ordered data | `partition_cols=[price_bucket]` |
| Composite | Complex queries | `partition_cols=["region", "date"]` |

### 7. Compression Comparison

| Format | Compression | Splittable | Schema | Best For |
|--------|-------------|------------|--------|----------|
| Parquet | Snappy/ZSTD | Yes | Embedded | Analytics, ML features |
| Avro | Deflate/Snappy | Yes | Embedded | Streaming, schema evolution |
| ORC | ZLIB/Snappy | Yes | Embedded | Hive workloads |
| CSV+GZIP | GZIP | No | External | Simple interchange |

### 8. Error Handling & Retry

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
)
def ingest_with_retry(source_config):
    """Ingest data with exponential backoff retry."""
    try:
        data = extract(source_config)
        validated = validate(data)
        load(validated, target_config)
    except ValidationError as e:
        send_to_dead_letter_queue(data, error=str(e))
        raise
    except ConnectionError:
        raise  # Will be retried
```

## Best Practices

1. **Use Parquet** as the default storage format for ML data (columnar, compressed, typed)
2. **Always validate schemas** before loading to catch upstream changes early
3. **Implement idempotent writes** (upsert/merge instead of append when possible)
4. **Partition by query patterns** (usually date for time-series ML data)
5. **Version your datasets** with DVC or LakeFS for reproducibility
6. **Use dead letter queues** for records that fail validation
7. **Monitor ingestion lag** for streaming pipelines
8. **Compress with Snappy** for speed or ZSTD for size
9. **Log data lineage** (source, timestamp, record count, schema version)
10. **Test pipelines** with both happy path and edge cases

## Common Edge Cases

- Schema changes in upstream sources (add nullable columns, handle gracefully)
- Timezone handling (always store in UTC, convert at query time)
- Null/missing value conventions (empty string vs NULL vs NaN)
- Character encoding issues (enforce UTF-8)
- Duplicate records in streaming (use deduplication windows)
- Late-arriving data (use watermarks with tolerance windows)
- Large file handling (chunk processing, memory-mapped files)

## Scripts

- `scripts/ingest_batch.py` - Batch ingestion from files and databases
- `scripts/ingest_streaming.py` - Kafka streaming ingestion with checkpointing

## References

See [references/REFERENCE.md](references/REFERENCE.md) for:
- Tool comparison (Airbyte, Meltano, dlt, Singer, NiFi)
- Format benchmarks
- Connection patterns for 10+ data sources
