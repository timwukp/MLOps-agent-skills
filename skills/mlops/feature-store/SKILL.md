---
name: feature-store
description: >
  Set up and manage feature stores for ML systems. Covers Feast, Hopsworks, and Tecton configuration, online and
  offline store architecture, feature materialization (batch and streaming), point-in-time correct feature retrieval
  for training, real-time feature serving for inference, feature freshness and staleness management, feature sharing
  across teams, feature registry and catalog, feature lineage and provenance, feature store monitoring, and migration
  strategies. Use when building feature infrastructure, sharing features across models, or serving features in production.
license: Apache-2.0
metadata:
  author: mlops-skills
  version: "1.0"
  category: mlops
---

# Feature Store for ML

## Overview

A feature store is centralized infrastructure for managing ML features - from feature
engineering to serving. It ensures consistency between training and inference, enables
feature reuse across teams, and provides point-in-time correct data retrieval.

## When to Use This Skill

- Building shared feature infrastructure for multiple ML models
- Need consistent features between training and serving
- Setting up real-time feature serving for online inference
- Managing feature freshness and materialization pipelines
- Creating a feature catalog for team collaboration

## Architecture

```
                    Feature Store Architecture
┌─────────────────────────────────────────────────────────┐
│                    Feature Registry                      │
│            (metadata, schemas, lineage)                  │
├──────────────────┬──────────────────────────────────────┤
│   Offline Store  │           Online Store                │
│   (historical)   │           (low-latency)               │
│                  │                                       │
│  ┌────────────┐  │  ┌─────────────┐  ┌───────────────┐  │
│  │  Parquet/  │  │  │   Redis/    │  │  DynamoDB/    │  │
│  │  BigQuery/ │  │  │   Postgres  │  │  Cassandra    │  │
│  │  Redshift  │  │  │             │  │               │  │
│  └─────┬──────┘  │  └──────┬──────┘  └───────┬───────┘  │
│        │         │         │                  │          │
│  Training Data   │    Real-time Serving    Batch Serving │
│  (point-in-time) │    (< 10ms latency)                  │
└──────────────────┴──────────────────────────────────────┘
```

## Step-by-Step Instructions

### 1. Feast Setup

#### Installation and Project Structure

```bash
pip install feast[redis]

# Initialize project
feast init my_feature_store
cd my_feature_store
```

```
my_feature_store/
├── feature_repo/
│   ├── feature_store.yaml    # Store configuration
│   ├── entities.py           # Entity definitions
│   ├── features.py           # Feature view definitions
│   └── data_sources.py       # Data source definitions
```

#### Configuration

```yaml
# feature_store.yaml
project: my_project
provider: local
registry: data/registry.db
online_store:
  type: redis
  connection_string: localhost:6379
offline_store:
  type: file  # or bigquery, redshift, snowflake
entity_key_serialization_version: 2
```

#### Define Entities and Feature Views

```python
# entities.py
from feast import Entity, ValueType

user = Entity(
    name="user_id",
    value_type=ValueType.INT64,
    description="Unique user identifier",
)

# features.py
from feast import FeatureView, Field
from feast.types import Float64, Int64, String
from datetime import timedelta

user_features = FeatureView(
    name="user_features",
    entities=[user],
    ttl=timedelta(days=1),
    schema=[
        Field(name="age", dtype=Int64),
        Field(name="lifetime_value", dtype=Float64),
        Field(name="account_type", dtype=String),
        Field(name="login_count_7d", dtype=Int64),
    ],
    source=user_source,  # FileSource, BigQuerySource, etc.
    online=True,
    tags={"team": "ml-platform", "version": "v2"},
)
```

#### Materialize and Serve

```python
from feast import FeatureStore
from datetime import datetime, timedelta

store = FeatureStore(repo_path="feature_repo")

# Apply definitions
# CLI: feast apply

# Materialize to online store (batch)
store.materialize(
    start_date=datetime.now() - timedelta(days=7),
    end_date=datetime.now(),
)

# Incremental materialization
store.materialize_incremental(end_date=datetime.now())

# Get training data (point-in-time join)
entity_df = pd.DataFrame({
    "user_id": [1, 2, 3],
    "event_timestamp": [datetime(2024, 1, 15)] * 3,
})
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=["user_features:age", "user_features:lifetime_value"],
).to_df()

# Get online features (real-time serving)
online_features = store.get_online_features(
    features=["user_features:age", "user_features:lifetime_value"],
    entity_rows=[{"user_id": 1}, {"user_id": 2}],
).to_dict()
```

### 2. Feature Freshness Management

```python
def check_feature_freshness(store, feature_view_name, max_staleness_hours=24):
    """Check if materialized features are fresh enough."""
    from feast.infra.registry.registry import Registry

    fv = store.get_feature_view(feature_view_name)
    last_materialization = fv.materialization_intervals[-1].end_date

    staleness = datetime.utcnow() - last_materialization
    is_fresh = staleness.total_seconds() < max_staleness_hours * 3600

    return {
        "feature_view": feature_view_name,
        "last_materialized": last_materialization.isoformat(),
        "staleness_hours": staleness.total_seconds() / 3600,
        "is_fresh": is_fresh,
    }
```

### 3. Feature Registry / Catalog

```python
def list_feature_catalog(store):
    """List all available features with metadata."""
    catalog = []
    for fv in store.list_feature_views():
        for field in fv.schema:
            catalog.append({
                "feature_view": fv.name,
                "feature_name": field.name,
                "dtype": str(field.dtype),
                "entities": [e.name for e in fv.entities],
                "ttl": str(fv.ttl),
                "tags": fv.tags,
                "online": fv.online,
            })
    return pd.DataFrame(catalog)
```

### 4. Point-in-Time Join (Explained)

Point-in-time joins prevent data leakage by only using feature values that were
available at the time of each training example:

```
Timeline:
─────────────────────────────────────────────►
  Feature   Feature   Training   Feature
  Update 1  Update 2  Event      Update 3
  (t=1)     (t=5)     (t=7)      (t=10)

For training event at t=7:
  ✓ Use Feature Update 2 (t=5, most recent before event)
  ✗ NOT Feature Update 3 (t=10, future data leakage!)
```

### 5. Streaming Feature Pipeline

```python
from feast import FeatureStore
from confluent_kafka import Consumer

def streaming_feature_pipeline(store, topic, feature_view_name):
    """Ingest streaming features into online store."""
    consumer = Consumer({
        "bootstrap.servers": "localhost:9092",
        "group.id": "feature-ingestion",
    })
    consumer.subscribe([topic])

    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            continue

        record = json.loads(msg.value())
        entity_key = record["entity_key"]
        features = record["features"]
        timestamp = datetime.fromisoformat(record["timestamp"])

        # Write to online store
        store.write_to_online_store(
            feature_view_name=feature_view_name,
            df=pd.DataFrame([{**entity_key, **features, "event_timestamp": timestamp}]),
        )
```

## Best Practices

1. **Use point-in-time joins** for training data to prevent leakage
2. **Separate online/offline** stores for latency vs cost optimization
3. **Set appropriate TTLs** to avoid serving stale features
4. **Tag features** with owner, version, and team for discoverability
5. **Monitor materialization lag** and alert on staleness
6. **Version feature views** when making breaking changes
7. **Use streaming materialization** for features needing sub-minute freshness
8. **Document feature semantics** in the registry
9. **Centralize features** to avoid duplication across teams

## Scripts

- `scripts/feast_setup.py` - Feast project initialization and operations
- `scripts/feature_registry.py` - Feature catalog and registry management

## References

See [references/REFERENCE.md](references/REFERENCE.md) for feature store tool comparisons.
