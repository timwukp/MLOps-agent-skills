#!/usr/bin/env python3
"""Batch data ingestion tool for ML pipelines.

Supports CSV, Parquet, JSON, Avro files and database sources.
Includes incremental loading, schema validation, and error handling.

Usage:
    python ingest_batch.py --source data.csv --target output/ --format parquet
    python ingest_batch.py --source "postgresql://user:pass@host/db" --query "SELECT * FROM features" --target output/
    python ingest_batch.py --source data/ --target output/ --incremental --watermark-col updated_at
"""
import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_file(path, file_format=None):
    """Load data from a file."""
    import pandas as pd

    path = str(path)
    fmt = file_format or Path(path).suffix.lstrip(".")

    loaders = {
        "csv": lambda p: pd.read_csv(p),
        "parquet": lambda p: pd.read_parquet(p),
        "json": lambda p: pd.read_json(p, lines=True),
        "jsonl": lambda p: pd.read_json(p, lines=True),
    }

    if fmt in loaders:
        logger.info(f"Loading {fmt} file: {path}")
        return loaders[fmt](path)
    else:
        raise ValueError(f"Unsupported format: {fmt}. Supported: {list(loaders.keys())}")


def load_database(connection_string, query):
    """Load data from a database."""
    import pandas as pd
    from sqlalchemy import create_engine

    logger.info(f"Connecting to database...")
    engine = create_engine(connection_string)
    logger.info(f"Executing query: {query[:100]}...")
    df = pd.read_sql(query, engine)
    logger.info(f"Loaded {len(df)} rows from database")
    return df


def load_directory(dir_path, file_format=None):
    """Load all files from a directory."""
    import pandas as pd

    dir_path = Path(dir_path)
    fmt = file_format or "parquet"
    pattern = f"*.{fmt}"

    files = sorted(dir_path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No {pattern} files found in {dir_path}")

    logger.info(f"Loading {len(files)} {fmt} files from {dir_path}")
    dfs = [load_file(f, fmt) for f in files]
    return pd.concat(dfs, ignore_index=True)


def validate_schema(df, schema_path=None):
    """Validate DataFrame against expected schema."""
    issues = []

    # Basic checks
    if len(df) == 0:
        issues.append("DataFrame is empty")
        return issues

    # Check for all-null columns
    for col in df.columns:
        if df[col].isnull().all():
            issues.append(f"Column '{col}' is entirely null")

    # Null percentage check
    for col in df.columns:
        null_pct = df[col].isnull().mean()
        if null_pct > 0.5:
            issues.append(f"Column '{col}' has {null_pct:.1%} null values")

    # Duplicate check
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        issues.append(f"{dup_count} duplicate rows ({dup_count/len(df):.1%})")

    if schema_path:
        with open(schema_path) as f:
            schema = json.load(f)
        for col_spec in schema.get("columns", []):
            col = col_spec["name"]
            if col not in df.columns:
                issues.append(f"Missing expected column: {col}")

    return issues


def save_data(df, target_path, output_format="parquet", partition_cols=None, compression="snappy"):
    """Save DataFrame to target format."""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "parquet":
        if partition_cols:
            df.to_parquet(target_path, partition_cols=partition_cols,
                         compression=compression, index=False)
        else:
            df.to_parquet(target_path, compression=compression, index=False)
    elif output_format == "csv":
        df.to_csv(target_path, index=False)
    elif output_format == "json":
        df.to_json(target_path, orient="records", lines=True)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

    logger.info(f"Saved {len(df)} rows to {target_path} ({output_format})")


def compute_data_fingerprint(df):
    """Compute a hash fingerprint of the data for versioning."""
    content = df.to_csv(index=False).encode("utf-8")
    return hashlib.md5(content).hexdigest()


def incremental_load(df, watermark_col, watermark_file):
    """Filter to only new records since last watermark."""
    import pandas as pd

    last_watermark = None
    if os.path.exists(watermark_file):
        with open(watermark_file) as f:
            last_watermark = f.read().strip()
        logger.info(f"Last watermark: {last_watermark}")

    if last_watermark:
        df[watermark_col] = pd.to_datetime(df[watermark_col])
        df = df[df[watermark_col] > pd.to_datetime(last_watermark)]
        logger.info(f"Filtered to {len(df)} new records since {last_watermark}")

    if len(df) > 0:
        new_watermark = str(df[watermark_col].max())
        with open(watermark_file, "w") as f:
            f.write(new_watermark)
        logger.info(f"Updated watermark to {new_watermark}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Batch data ingestion for ML pipelines")
    parser.add_argument("--source", required=True, help="Source file/directory/connection string")
    parser.add_argument("--target", required=True, help="Target path for output")
    parser.add_argument("--format", default="parquet", help="Output format (parquet, csv, json)")
    parser.add_argument("--source-format", default=None, help="Source file format")
    parser.add_argument("--query", default=None, help="SQL query for database sources")
    parser.add_argument("--schema", default=None, help="Path to schema JSON for validation")
    parser.add_argument("--partition-cols", nargs="*", default=None, help="Columns to partition by")
    parser.add_argument("--compression", default="snappy", help="Compression (snappy, gzip, zstd)")
    parser.add_argument("--incremental", action="store_true", help="Enable incremental loading")
    parser.add_argument("--watermark-col", default=None, help="Column for incremental watermark")
    parser.add_argument("--watermark-file", default=".watermark", help="File to store watermark")
    parser.add_argument("--validate", action="store_true", help="Validate data before saving")

    args = parser.parse_args()

    try:
        # Load data
        if args.query:
            df = load_database(args.source, args.query)
        elif os.path.isdir(args.source):
            df = load_directory(args.source, args.source_format)
        else:
            df = load_file(args.source, args.source_format)

        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

        # Incremental filtering
        if args.incremental and args.watermark_col:
            df = incremental_load(df, args.watermark_col, args.watermark_file)
            if len(df) == 0:
                logger.info("No new data to ingest")
                return

        # Validate
        if args.validate:
            issues = validate_schema(df, args.schema)
            if issues:
                for issue in issues:
                    logger.warning(f"Validation issue: {issue}")

        # Compute fingerprint
        fingerprint = compute_data_fingerprint(df)
        logger.info(f"Data fingerprint: {fingerprint}")

        # Save
        save_data(df, args.target, args.format, args.partition_cols, args.compression)

        # Summary
        summary = {
            "source": args.source,
            "target": args.target,
            "rows": len(df),
            "columns": len(df.columns),
            "fingerprint": fingerprint,
            "timestamp": datetime.utcnow().isoformat(),
        }
        logger.info(f"Ingestion complete: {json.dumps(summary)}")

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
