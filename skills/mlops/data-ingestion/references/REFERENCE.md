# Data Ingestion Reference

## Tool Comparison

| Tool | Type | Best For | Connectors | OSS | Language |
|------|------|----------|------------|-----|----------|
| Airbyte | ELT | 300+ connectors, easy setup | 300+ | Yes | Java/Python |
| Meltano | ELT | Singer ecosystem, CLI-first | 600+ (Singer) | Yes | Python |
| dlt (data load tool) | ELT | Python-native, lightweight | Many | Yes | Python |
| Apache NiFi | ETL | Visual flows, enterprise | Many | Yes | Java |
| Singer | ELT | Tap/target ecosystem | 200+ | Yes | Python |
| Fivetran | ELT | Managed, low maintenance | 300+ | No | Managed |
| Apache Flink | Stream | Low-latency streaming | Kafka, Kinesis | Yes | Java/Python |
| Apache Spark | Batch/Stream | Large-scale processing | Many | Yes | Scala/Python |
| Apache Kafka Connect | Stream/Batch | Kafka ecosystem, CDC | 100+ | Yes | Java |
| Debezium | CDC | Database change capture | 10+ DBs | Yes | Java |

### When to Choose What

- **Small team, Python-first**: dlt or Meltano
- **Enterprise, many sources**: Airbyte or Fivetran
- **Real-time ML features**: Flink or Spark Structured Streaming
- **CDC for incremental ML data**: Debezium + Kafka Connect
- **Visual pipeline builder**: Apache NiFi

## Ingestion Architecture Patterns

### Pattern 1: Simple Batch (ETL)

```
Source DB ──> Extract ──> Transform ──> Load ──> Data Warehouse
                                                       │
                                                  ML Training
```

**When to use**: Nightly/hourly model retraining, historical feature computation.
**Pros**: Simple, debuggable, well-understood.
**Cons**: High latency, full reprocessing on failure.

### Pattern 2: ELT with Staging

```
Sources ──> Raw Zone (land as-is) ──> Staging (validate/clean) ──> Curated (ML-ready)
```

**When to use**: Multiple data sources, schema evolution expected, data lake architecture.
**Pros**: Raw data preserved, reprocessable, schema-on-read flexibility.
**Cons**: Storage cost for raw zone, more complex pipeline.

### Pattern 3: Streaming with Micro-batch Sink

```
Event Stream ──> Kafka ──> Consumer ──> Buffer ──> Micro-batch Write ──> Feature Store
                              │                          │
                         Deserialize              Checkpoint/Commit
                         & Validate                (exactly-once)
```

**When to use**: Real-time features, event-driven ML, fraud detection.
**Pros**: Low latency, continuous processing, backpressure handling.
**Cons**: Complex exactly-once semantics, harder to debug.

### Pattern 4: Change Data Capture (CDC)

```
Source DB ──> WAL/Binlog ──> Debezium ──> Kafka ──> Consumer ──> Feature Store
                                            │
                                     Schema Registry
                                  (Avro/Protobuf schemas)
```

**When to use**: Syncing operational DB changes to ML pipelines without impacting source.
**Pros**: No impact on source DB, captures all changes (insert/update/delete), low latency.
**Cons**: Requires DB log access, schema evolution complexity.

### Pattern 5: Lambda Architecture (Batch + Stream)

```
                    ┌──> Batch Layer (historical) ──┐
Event Stream ──>    │                                ├──> Serving Layer ──> ML Model
                    └──> Speed Layer (real-time)  ──┘
```

**When to use**: Need both historical accuracy and real-time features.
**Cons**: Two code paths to maintain. Consider Kappa architecture (stream-only) instead.

## Format Comparison

| Format | Columnar | Schema | Compression | Splittable | Read Speed | Write Speed | ML Use Case |
|--------|----------|--------|-------------|------------|------------|-------------|-------------|
| Parquet | Yes | Embedded | Snappy/ZSTD | Yes | Fast | Medium | Feature storage, training data |
| Avro | No | Embedded | Deflate | Yes | Medium | Fast | Streaming events, schema evolution |
| ORC | Yes | Embedded | ZLIB/Snappy | Yes | Fast | Medium | Hive/Spark workloads |
| CSV | No | External | GZIP | No* | Slow | Fast | Data exchange, small datasets |
| JSON | No | External | GZIP | No* | Slow | Fast | API responses, configs |
| Delta Lake | Yes | Embedded | Snappy/ZSTD | Yes | Fast | Medium | Lakehouse, ACID on data lake |
| Iceberg | Yes | Embedded | Snappy/ZSTD | Yes | Fast | Medium | Analytics, time travel |
| Arrow IPC | Yes | Embedded | LZ4/ZSTD | No | Very fast | Very fast | In-memory exchange, zero-copy |

\* Splittable with newline-delimited format and no compression (or with bzip2).

**Recommendation for ML**: Use Parquet as default. Use Delta Lake or Iceberg when you need ACID transactions, time travel, or schema evolution on a data lake.

## Compression Comparison

| Algorithm | Ratio | Speed | CPU | Best For |
|-----------|-------|-------|-----|----------|
| Snappy | Low | Very fast | Low | Real-time, default for Parquet |
| LZ4 | Low | Very fast | Low | Real-time, in-memory |
| ZSTD | High | Fast | Medium | Storage optimization, archival |
| GZIP | High | Slow | High | Maximum compression, legacy |
| Brotli | Very high | Slow | High | Web/archive |

**Rule of thumb**: Use Snappy for hot data (frequent reads), ZSTD for warm/cold data (storage cost matters).

## Database Connection Patterns

### PostgreSQL
```python
from sqlalchemy import create_engine
engine = create_engine(
    "postgresql://user:password@host:5432/database",
    pool_size=5, max_overflow=10, pool_timeout=30
)
```

### MySQL
```python
engine = create_engine(
    "mysql+pymysql://user:password@host:3306/database",
    pool_recycle=3600  # Reconnect after 1hr (MySQL timeout)
)
```

### BigQuery
```python
# Via pandas-gbq
df = pd.read_gbq("SELECT * FROM dataset.table", project_id="my-project")

# Via SQLAlchemy (recommended for large queries)
engine = create_engine("bigquery://my-project/my-dataset")
df = pd.read_sql("SELECT * FROM table WHERE date > '2024-01-01'", engine)
```

### Snowflake
```python
engine = create_engine(
    "snowflake://user:pass@account/db/schema?warehouse=wh&role=role"
)
```

### MongoDB
```python
from pymongo import MongoClient
client = MongoClient("mongodb://user:pass@host:27017/")
df = pd.DataFrame(list(client.db.collection.find({}, {"_id": 0})))
```

### S3 / Cloud Object Storage
```python
import pyarrow.parquet as pq
import s3fs

# Direct read from S3
fs = s3fs.S3FileSystem()
df = pq.read_table("s3://bucket/path/data.parquet", filesystem=fs).to_pandas()

# With predicate pushdown (read only matching row groups)
df = pq.read_table(
    "s3://bucket/path/",
    filesystem=fs,
    filters=[("date", ">=", "2024-01-01"), ("region", "=", "us-east")]
).to_pandas()
```

### DynamoDB
```python
import boto3
dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table("features")
response = table.scan(FilterExpression=Attr("updated_at").gt(watermark))
df = pd.DataFrame(response["Items"])
```

## Data Versioning Tools

| Tool | Approach | Storage | Git Integration | Best For |
|------|----------|---------|-----------------|----------|
| DVC | Pointer files (.dvc) | Any remote (S3/GCS/Azure) | Strong (git-based) | ML data + models |
| LakeFS | Git-like branches | S3-compatible API | Separate | Data lake versioning |
| Delta Lake | Transaction log | Object storage | None | Lakehouse, ACID |
| Iceberg | Snapshot-based | Object storage | None | Analytics, time travel |
| Pachyderm | Data-aware pipelines | Object storage | None | Data lineage pipelines |

### DVC Workflow for ML Data

```bash
# Setup
dvc init && dvc remote add -d store s3://bucket/dvc

# Version a dataset
dvc add data/train.parquet      # Creates .dvc pointer
git add data/train.parquet.dvc
git commit -m "Training data v1.0"
git tag data-v1.0
dvc push

# Switch to a previous version
git checkout data-v1.0
dvc checkout
```

## Exactly-Once Semantics

Achieving exactly-once processing in streaming ingestion:

| Strategy | How It Works | Trade-off |
|----------|-------------|-----------|
| **Idempotent writes** | Use upsert/merge with natural keys | Simplest, works for most ML use cases |
| **Transactional commit** | Commit offset + write in same transaction | Requires transactional sink (DB, Delta Lake) |
| **Deduplication window** | Track seen IDs in a time window | Memory/storage for dedup state |
| **Kafka transactions** | Producer + consumer in same transaction | Kafka-to-Kafka only |

### Idempotent Write Pattern (Recommended for ML)

```python
def idempotent_upsert(new_records, target_table, key_columns):
    """Upsert pattern - safe to replay without duplicates."""
    from sqlalchemy import text

    keys = ", ".join(key_columns)
    cols = ", ".join(new_records.columns)
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in new_records.columns if c not in key_columns)

    sql = text(f"""
        INSERT INTO {target_table} ({cols})
        VALUES ({', '.join([f':{c}' for c in new_records.columns])})
        ON CONFLICT ({keys}) DO UPDATE SET {updates}
    """)
    with engine.begin() as conn:
        conn.execute(sql, new_records.to_dict("records"))
```

## Backfill Strategies

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| **Full reload** | Drop and recreate entire dataset | Schema changes, data corruption |
| **Incremental backfill** | Replay from watermark | Missed windows, partial failures |
| **Parallel partition backfill** | Backfill partitions independently | Large datasets, fast recovery |
| **Shadow pipeline** | Run backfill alongside live pipeline | Zero-downtime backfill |

### Incremental Backfill Example

```python
def backfill(start_date, end_date, partition_size="1d"):
    """Backfill data in date-sized chunks."""
    current = start_date
    while current < end_date:
        next_date = current + pd.Timedelta(partition_size)
        logger.info(f"Backfilling {current} to {next_date}")
        data = extract(start=current, end=next_date)
        validated = validate(data)
        load(validated, partition_key=current.strftime("%Y-%m-%d"))
        current = next_date
```

## Monitoring Ingestion Pipelines

### Key Metrics to Track

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| **Records ingested** | Count per batch/window | Drop >50% from baseline |
| **Ingestion latency** | Time from source to destination | >SLA (e.g., 5min for streaming) |
| **Schema violations** | Records failing validation | >1% of batch |
| **Dead letter queue size** | Failed records accumulating | >0 (investigate immediately) |
| **Consumer lag** | Kafka offset lag | >10,000 messages |
| **Pipeline duration** | End-to-end batch time | >2x historical average |
| **Data freshness** | Time since last successful load | >expected interval + buffer |
| **Null rate per column** | Fraction of nulls | Sudden increase >5% |

### Alerting Example

```python
def check_ingestion_health(run_stats):
    alerts = []
    if run_stats["record_count"] < run_stats["expected_min"] * 0.5:
        alerts.append(f"CRITICAL: Only {run_stats['record_count']} records (expected {run_stats['expected_min']}+)")
    if run_stats["error_rate"] > 0.01:
        alerts.append(f"WARNING: Error rate {run_stats['error_rate']:.2%} exceeds 1% threshold")
    if run_stats["duration_s"] > run_stats["sla_seconds"]:
        alerts.append(f"WARNING: Pipeline took {run_stats['duration_s']}s (SLA: {run_stats['sla_seconds']}s)")
    return alerts
```

## Performance Tuning Tips

1. **Read only needed columns**: `pd.read_parquet(path, columns=["col1", "col2"])`
2. **Use predicate pushdown**: `pq.read_table(path, filters=[("date", ">=", "2024-01-01")])`
3. **Partition by query patterns**: Date for time-series, region for geographic data
4. **Optimal row group size**: 128MB for Parquet (default is usually fine)
5. **Parallel reads**: Use `pyarrow.dataset` or Dask for multi-file reads
6. **Memory mapping**: Use Arrow memory-mapped files for large datasets
7. **Chunk processing**: `pd.read_csv(path, chunksize=10000)` for CSV files that exceed RAM
8. **Connection pooling**: Use SQLAlchemy engine with `pool_size` parameter
9. **Batch inserts**: Use `executemany` or `COPY` instead of row-by-row inserts
10. **Async I/O**: Use `asyncio` + `aiohttp` for API-based ingestion

### Benchmark: Parquet vs CSV Read Performance

| Operation | CSV (1GB) | Parquet (1GB) | Speedup |
|-----------|-----------|---------------|---------|
| Full read | 45s | 3s | 15x |
| Read 3 columns | 45s | 0.8s | 56x |
| Filter + read | 45s | 1.2s | 37x |
| Count rows | 45s | 0.01s | 4500x |

## Best Practices & Anti-Patterns

### Best Practices

1. **Validate early**: Check schema and data quality immediately after extraction, before any transformation
2. **Idempotent pipelines**: Design so re-running produces the same result (use upsert, not append)
3. **Log lineage metadata**: Record source, timestamp, record count, schema hash per batch
4. **Use dead letter queues**: Route invalid records to DLQ instead of dropping or failing the batch
5. **Separate extraction from transformation**: Makes debugging and reprocessing easier
6. **Schema registry**: Use Confluent Schema Registry or AWS Glue Schema Registry for streaming
7. **Test with production-like data**: Include edge cases (nulls, Unicode, large values, empty strings)
8. **Monitor data freshness**: Alert when data is stale, not just when pipelines fail

### Anti-Patterns to Avoid

| Anti-Pattern | Problem | Better Approach |
|-------------|---------|-----------------|
| No schema enforcement | Silent data corruption | Define and validate schemas at ingestion |
| Append-only without dedup | Duplicate records in training data | Use upsert/merge or dedup step |
| Storing everything as CSV | Slow reads, no types, no compression | Use Parquet or Delta Lake |
| Hardcoded connection strings | Security risk, inflexible | Use env vars or secret managers |
| No retry logic | Transient failures crash pipeline | Exponential backoff with tenacity |
| Processing in single thread | Slow ingestion for large sources | Parallel partition processing |
| No data quality checks | Garbage in, garbage out for ML | Validate with Great Expectations or Pandera |
| Ignoring late-arriving data | Incomplete training datasets | Watermarks with tolerance windows |

## Further Reading

- [dlt Documentation](https://dlthub.com/docs/intro) - Python-first data loading
- [Delta Lake Guide](https://docs.delta.io/latest/index.html) - ACID on data lakes
- [Debezium CDC](https://debezium.io/documentation/) - Change data capture
- [The Data Engineering Cookbook](https://github.com/andkret/Cookbook) - Comprehensive patterns
- [Fundamentals of Data Engineering](https://www.oreilly.com/library/view/fundamentals-of-data/9781098108298/) - O'Reilly book by Joe Reis & Matt Housley
