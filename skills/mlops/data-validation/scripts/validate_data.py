#!/usr/bin/env python3
"""Data validation tool using Great Expectations and Pandera.

Usage:
    python validate_data.py --data data.parquet --checks basic
    python validate_data.py --data data.csv --checks full --report report.html
    python validate_data.py --data data.parquet --schema schema.yaml
"""
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_data(path):
    import pandas as pd
    path = str(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    elif path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".json") or path.endswith(".jsonl"):
        return pd.read_json(path, lines=path.endswith(".jsonl"))
    else:
        raise ValueError(f"Unsupported format: {path}")


def basic_checks(df):
    """Run basic data quality checks."""
    import numpy as np
    results = {"passed": [], "failed": [], "warnings": []}

    # Row count
    if len(df) > 0:
        results["passed"].append(f"Row count: {len(df)}")
    else:
        results["failed"].append("DataFrame is empty")

    # Duplicate check
    dup_count = df.duplicated().sum()
    if dup_count == 0:
        results["passed"].append("No duplicate rows")
    else:
        results["warnings"].append(f"{dup_count} duplicate rows ({dup_count/len(df):.1%})")

    # Null check per column
    for col in df.columns:
        null_pct = df[col].isnull().mean()
        if null_pct == 0:
            results["passed"].append(f"Column '{col}': no nulls")
        elif null_pct < 0.05:
            results["warnings"].append(f"Column '{col}': {null_pct:.1%} nulls")
        else:
            results["failed"].append(f"Column '{col}': {null_pct:.1%} nulls (>5%)")

    # Constant columns
    for col in df.columns:
        if df[col].nunique() <= 1:
            results["warnings"].append(f"Column '{col}' is constant (nunique={df[col].nunique()})")

    # Numeric range checks
    for col in df.select_dtypes(include=[np.number]).columns:
        if np.isinf(df[col]).any():
            results["failed"].append(f"Column '{col}' contains infinity values")

    return results


def statistical_profile(df):
    """Generate statistical profile of the dataset."""
    import numpy as np
    profile = {"columns": {}, "dataset": {}}

    profile["dataset"] = {
        "rows": len(df),
        "columns": len(df.columns),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    for col in df.columns:
        col_profile = {
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isnull().sum()),
            "null_pct": round(df[col].isnull().mean(), 4),
            "unique_count": int(df[col].nunique()),
        }

        if df[col].dtype in ["int64", "float64", "int32", "float32"]:
            col_profile.update({
                "mean": round(float(df[col].mean()), 4),
                "std": round(float(df[col].std()), 4),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "median": float(df[col].median()),
                "q25": float(df[col].quantile(0.25)),
                "q75": float(df[col].quantile(0.75)),
            })
        elif df[col].dtype == "object":
            top_values = df[col].value_counts().head(5).to_dict()
            col_profile["top_values"] = {str(k): int(v) for k, v in top_values.items()}

        profile["columns"][col] = col_profile

    return profile


def validate_with_schema(df, schema_path):
    """Validate data against a YAML schema definition."""
    import yaml

    with open(schema_path) as f:
        schema = yaml.safe_load(f)

    results = {"passed": [], "failed": []}

    for col_spec in schema.get("columns", []):
        col = col_spec["name"]

        if col not in df.columns:
            results["failed"].append(f"Missing column: {col}")
            continue

        if not col_spec.get("nullable", True) and df[col].isnull().any():
            results["failed"].append(f"Column '{col}': has nulls but nullable=false")
        else:
            results["passed"].append(f"Column '{col}': nullable check passed")

        if "min" in col_spec:
            if (df[col].dropna() < col_spec["min"]).any():
                results["failed"].append(f"Column '{col}': values below min {col_spec['min']}")

        if "max" in col_spec:
            if (df[col].dropna() > col_spec["max"]).any():
                results["failed"].append(f"Column '{col}': values above max {col_spec['max']}")

        if "allowed_values" in col_spec:
            invalid = ~df[col].dropna().isin(col_spec["allowed_values"])
            if invalid.any():
                results["failed"].append(
                    f"Column '{col}': {invalid.sum()} values not in allowed set")

    return results


def main():
    parser = argparse.ArgumentParser(description="Data validation for ML pipelines")
    parser.add_argument("--data", required=True, help="Path to data file")
    parser.add_argument("--checks", default="basic", choices=["basic", "full"],
                       help="Check level")
    parser.add_argument("--schema", default=None, help="YAML schema file for validation")
    parser.add_argument("--report", default=None, help="Output report path (JSON)")
    parser.add_argument("--profile", action="store_true", help="Generate statistical profile")
    parser.add_argument("--fail-on-warning", action="store_true",
                       help="Exit with error on warnings")

    args = parser.parse_args()

    logger.info(f"Loading data from {args.data}")
    df = load_data(args.data)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    report = {"timestamp": datetime.utcnow().isoformat(), "source": args.data}

    # Basic checks
    results = basic_checks(df)
    report["basic_checks"] = results

    # Schema validation
    if args.schema:
        schema_results = validate_with_schema(df, args.schema)
        report["schema_validation"] = schema_results
        results["failed"].extend(schema_results["failed"])
        results["passed"].extend(schema_results["passed"])

    # Profile
    if args.profile or args.checks == "full":
        profile = statistical_profile(df)
        report["profile"] = profile

    # Summary
    total_passed = len(results["passed"])
    total_failed = len(results["failed"])
    total_warnings = len(results["warnings"])

    logger.info(f"Results: {total_passed} passed, {total_failed} failed, {total_warnings} warnings")

    for item in results["failed"]:
        logger.error(f"FAILED: {item}")
    for item in results["warnings"]:
        logger.warning(f"WARNING: {item}")

    # Save report
    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Report saved to {args.report}")

    # Exit code
    if total_failed > 0:
        sys.exit(1)
    if args.fail_on_warning and total_warnings > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
