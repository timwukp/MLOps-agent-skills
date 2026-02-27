#!/usr/bin/env python3
"""Feast feature store setup and management script.

Initializes a Feast project, defines feature views / entities / data sources,
applies definitions to the registry, materializes features, and retrieves
online or historical feature vectors.

Usage:
    python feast_setup.py --action init --repo-path ./my_feature_repo
    python feast_setup.py --action apply --repo-path ./my_feature_repo
    python feast_setup.py --action materialize --repo-path ./my_feature_repo
    python feast_setup.py --action get-online --repo-path ./my_feature_repo \
        --entity-key customer_id=1001 --features customer_fv:daily_spend,customer_fv:lifetime_value
    python feast_setup.py --action get-historical --repo-path ./my_feature_repo \
        --entity-key customer_id --features customer_fv:daily_spend
"""
import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project initialisation
# ---------------------------------------------------------------------------

FEATURE_STORE_YAML_TEMPLATE = """\
project: {project_name}
provider: local
registry: {registry_path}
online_store:
  type: {online_store_type}
{online_store_extra}offline_store:
  type: file
entity_key_serialization_version: 2
"""


def init_project(repo_path, online_store_type="sqlite", redis_url=None):
    """Generate a feature_store.yaml and scaffold directories."""
    repo = Path(repo_path)
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "data").mkdir(exist_ok=True)

    extra = ""
    if online_store_type == "redis":
        url = redis_url or "redis://localhost:6379"
        extra = f"  connection_string: {url}\n"

    yaml_content = FEATURE_STORE_YAML_TEMPLATE.format(
        project_name=repo.name.replace("-", "_"),
        registry_path=str(repo / "registry.pb"),
        online_store_type=online_store_type,
        online_store_extra=extra,
    )

    yaml_path = repo / "feature_store.yaml"
    yaml_path.write_text(yaml_content)
    logger.info("Wrote %s", yaml_path)
    return yaml_path


# ---------------------------------------------------------------------------
# Programmatic feature definitions
# ---------------------------------------------------------------------------

def build_demo_definitions():
    """Return sample Feast objects (entity, source, feature view).

    Imports are deferred so the module can be loaded without Feast installed
    when only the ``init`` action is used.
    """
    from feast import Entity, FeatureView, Field, FileSource, ValueType
    from feast.types import Float64, Int64, String

    customer = Entity(
        name="customer_id",
        value_type=ValueType.INT64,
        description="Unique customer identifier",
    )

    source = FileSource(
        path="data/customer_features.parquet",
        timestamp_field="event_timestamp",
        created_timestamp_column="created_at",
    )

    customer_fv = FeatureView(
        name="customer_fv",
        entities=[customer],
        schema=[
            Field(name="daily_spend", dtype=Float64),
            Field(name="lifetime_value", dtype=Float64),
            Field(name="transaction_count", dtype=Int64),
            Field(name="segment", dtype=String),
        ],
        source=source,
        ttl=timedelta(days=1),
        online=True,
        tags={"team": "growth", "version": "1"},
    )

    return [customer], [source], [customer_fv]


# ---------------------------------------------------------------------------
# Apply / materialise / retrieve
# ---------------------------------------------------------------------------

def get_store(repo_path):
    from feast import FeatureStore
    return FeatureStore(repo_path=str(repo_path))


def apply_definitions(repo_path):
    """Apply feature definitions to the Feast registry."""
    store = get_store(repo_path)
    entities, _sources, feature_views = build_demo_definitions()
    store.apply(entities + feature_views)
    logger.info("Applied %d entities and %d feature views", len(entities), len(feature_views))


def materialize_features(repo_path, start_date=None, end_date=None):
    """Materialize features from the offline store into the online store."""
    store = get_store(repo_path)
    end = end_date or datetime.utcnow()
    start = start_date or (end - timedelta(days=7))
    logger.info("Materializing features from %s to %s", start.isoformat(), end.isoformat())
    store.materialize(start_date=start, end_date=end)
    logger.info("Materialization complete")


def get_online_features(repo_path, entity_rows, feature_refs):
    """Retrieve features from the online store for given entity keys.

    Parameters
    ----------
    entity_rows : list[dict]
        e.g. [{"customer_id": 1001}]
    feature_refs : list[str]
        e.g. ["customer_fv:daily_spend", "customer_fv:lifetime_value"]
    """
    store = get_store(repo_path)
    response = store.get_online_features(
        features=feature_refs,
        entity_rows=entity_rows,
    )
    result = response.to_dict()
    logger.info("Online features retrieved for %d entities", len(entity_rows))
    return result


def get_historical_features(repo_path, entity_df_path, feature_refs):
    """Point-in-time join to retrieve historical features.

    Parameters
    ----------
    entity_df_path : str
        Path to a Parquet file containing entity keys and an
        ``event_timestamp`` column used for point-in-time correctness.
    feature_refs : list[str]
        Feature references, e.g. ["customer_fv:daily_spend"].
    """
    import pandas as pd
    from feast import FeatureStore

    store = FeatureStore(repo_path=str(repo_path))
    entity_df = pd.read_parquet(entity_df_path)

    if "event_timestamp" not in entity_df.columns:
        raise ValueError("Entity DataFrame must contain an 'event_timestamp' column")

    job = store.get_historical_features(
        entity_df=entity_df,
        features=feature_refs,
    )
    df = job.to_df()
    logger.info(
        "Historical features retrieved: %d rows x %d columns",
        len(df),
        len(df.columns),
    )
    return df


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def parse_entity_key(raw):
    """Parse 'key=value' into a dict, coercing numeric values."""
    pairs = {}
    for part in raw.split(","):
        k, v = part.split("=", 1)
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                pass
        pairs[k.strip()] = v
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Feast feature store setup and management")
    parser.add_argument(
        "--action",
        required=True,
        choices=["init", "apply", "materialize", "get-online", "get-historical"],
        help="Action to perform",
    )
    parser.add_argument("--repo-path", default="./feature_repo", help="Feast repo path")
    parser.add_argument(
        "--online-store",
        default="sqlite",
        choices=["sqlite", "redis"],
        help="Online store backend (used with init)",
    )
    parser.add_argument("--redis-url", default=None, help="Redis URL (when online-store=redis)")
    parser.add_argument("--entity-key", default=None, help="Entity key(s) as key=value pairs")
    parser.add_argument("--entity-df", default=None, help="Parquet path for historical retrieval")
    parser.add_argument("--features", default=None, help="Comma-separated feature references")
    parser.add_argument("--start-date", default=None, help="Materialization start (ISO format)")
    parser.add_argument("--end-date", default=None, help="Materialization end (ISO format)")

    args = parser.parse_args()
    repo = Path(args.repo_path)

    try:
        if args.action == "init":
            init_project(repo, args.online_store, args.redis_url)

        elif args.action == "apply":
            apply_definitions(repo)

        elif args.action == "materialize":
            start = datetime.fromisoformat(args.start_date) if args.start_date else None
            end = datetime.fromisoformat(args.end_date) if args.end_date else None
            materialize_features(repo, start, end)

        elif args.action == "get-online":
            if not args.entity_key or not args.features:
                parser.error("--entity-key and --features are required for get-online")
            entity_rows = [parse_entity_key(args.entity_key)]
            feature_refs = [f.strip() for f in args.features.split(",")]
            result = get_online_features(repo, entity_rows, feature_refs)
            print(json.dumps(result, indent=2, default=str))

        elif args.action == "get-historical":
            if not args.entity_df or not args.features:
                parser.error("--entity-df and --features are required for get-historical")
            feature_refs = [f.strip() for f in args.features.split(",")]
            df = get_historical_features(repo, args.entity_df, feature_refs)
            print(df.to_string(index=False))

    except Exception as exc:
        logger.error("Action '%s' failed: %s", args.action, exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
