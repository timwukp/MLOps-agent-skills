#!/usr/bin/env python3
"""
Data Pipeline Testing Suite — Validate data quality, schema, and distributions.

Features:
    - Schema validation (column names, types, nullable constraints)
    - Data quality checks (null rates, duplicates, value ranges, cardinality)
    - Distribution tests (KS test: train vs test / current vs reference)
    - Freshness checks (data timestamp recency)
    - Volume checks (row count within expected range)
    - Referential integrity checks (foreign key relationships)
    - JSON report with per-check pass/fail status

Usage:
    python test_data_pipeline.py --help
    python test_data_pipeline.py --data data.csv --schema schema.yaml
    python test_data_pipeline.py --data data.csv --schema schema.yaml --reference ref.csv
    python test_data_pipeline.py --data data.parquet --schema schema.yaml \
        --output report.json --fail-on-error
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("data_pipeline_test")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data(path: str):
    """Load a dataset from CSV, Parquet, or JSON/JSONL."""
    import pandas as pd

    p = str(path)
    if p.endswith(".parquet"):
        return pd.read_parquet(p)
    if p.endswith(".csv"):
        return pd.read_csv(p)
    if p.endswith(".json") or p.endswith(".jsonl"):
        return pd.read_json(p, lines=p.endswith(".jsonl"))
    raise ValueError(f"Unsupported data format: {p}")


def load_schema(path: str) -> Dict[str, Any]:
    """Load a YAML schema definition."""
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def validate_schema(df, schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Check column presence, types, and nullable constraints."""
    checks: List[Dict[str, Any]] = []
    expected_cols = schema.get("columns", [])

    expected_names = {c["name"] for c in expected_cols}
    actual_names = set(df.columns)

    # Missing columns
    for missing in expected_names - actual_names:
        checks.append({"check": "schema.column_present", "column": missing,
                        "pass": False, "detail": "column missing from data"})

    # Extra columns (warning, not failure)
    for extra in actual_names - expected_names:
        checks.append({"check": "schema.unexpected_column", "column": extra,
                        "pass": True, "detail": "unexpected column present (warning)"})

    for col_spec in expected_cols:
        col = col_spec["name"]
        if col not in df.columns:
            continue

        # Type check
        expected_dtype = col_spec.get("dtype")
        if expected_dtype:
            actual_dtype = str(df[col].dtype)
            type_ok = expected_dtype in actual_dtype or actual_dtype.startswith(expected_dtype)
            checks.append({"check": "schema.dtype", "column": col, "pass": type_ok,
                            "detail": f"expected={expected_dtype}, actual={actual_dtype}"})

        # Nullable
        if not col_spec.get("nullable", True):
            has_nulls = bool(df[col].isnull().any())
            checks.append({"check": "schema.nullable", "column": col,
                            "pass": not has_nulls,
                            "detail": f"nulls found={has_nulls}"})
    return checks


# ---------------------------------------------------------------------------
# Data quality
# ---------------------------------------------------------------------------

def quality_checks(df, schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Null rates, duplicate rates, value ranges, cardinality."""
    import numpy as np

    checks: List[Dict[str, Any]] = []
    quality_cfg = schema.get("quality", {})

    # Global duplicate rate
    max_dup_rate = quality_cfg.get("max_duplicate_rate", 0.01)
    dup_rate = float(df.duplicated().mean())
    checks.append({"check": "quality.duplicate_rate", "pass": dup_rate <= max_dup_rate,
                    "detail": f"rate={dup_rate:.4f}, max={max_dup_rate}"})

    for col_spec in schema.get("columns", []):
        col = col_spec["name"]
        if col not in df.columns:
            continue

        # Null rate
        max_null_rate = col_spec.get("max_null_rate")
        if max_null_rate is not None:
            null_rate = float(df[col].isnull().mean())
            checks.append({"check": "quality.null_rate", "column": col,
                            "pass": null_rate <= max_null_rate,
                            "detail": f"rate={null_rate:.4f}, max={max_null_rate}"})

        # Value range
        if "min" in col_spec:
            below = (df[col].dropna() < col_spec["min"]).any()
            checks.append({"check": "quality.min_value", "column": col,
                            "pass": not bool(below),
                            "detail": f"min_allowed={col_spec['min']}"})
        if "max" in col_spec:
            above = (df[col].dropna() > col_spec["max"]).any()
            checks.append({"check": "quality.max_value", "column": col,
                            "pass": not bool(above),
                            "detail": f"max_allowed={col_spec['max']}"})

        # Allowed values
        if "allowed_values" in col_spec:
            invalid_count = int((~df[col].dropna().isin(col_spec["allowed_values"])).sum())
            checks.append({"check": "quality.allowed_values", "column": col,
                            "pass": invalid_count == 0,
                            "detail": f"invalid_count={invalid_count}"})

        # Cardinality
        min_card = col_spec.get("min_cardinality")
        max_card = col_spec.get("max_cardinality")
        if min_card is not None or max_card is not None:
            card = int(df[col].nunique())
            ok = True
            if min_card is not None and card < min_card:
                ok = False
            if max_card is not None and card > max_card:
                ok = False
            checks.append({"check": "quality.cardinality", "column": col,
                            "pass": ok,
                            "detail": f"cardinality={card}, min={min_card}, max={max_card}"})
    return checks


# ---------------------------------------------------------------------------
# Distribution tests
# ---------------------------------------------------------------------------

def distribution_tests(df, ref_df, schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    """KS test per numeric feature comparing *df* against *ref_df*."""
    import numpy as np
    from scipy.stats import ks_2samp

    checks: List[Dict[str, Any]] = []
    alpha = schema.get("distribution", {}).get("ks_alpha", 0.05)
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if col not in ref_df.columns:
            continue
        stat, pvalue = ks_2samp(df[col].dropna(), ref_df[col].dropna())
        passed = pvalue >= alpha
        checks.append({
            "check": "distribution.ks_test", "column": col,
            "pass": passed,
            "detail": f"statistic={stat:.4f}, p-value={pvalue:.4f}, alpha={alpha}",
        })

    # Categorical distribution: compare value proportions
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        if col not in ref_df.columns:
            continue
        dist_curr = df[col].value_counts(normalize=True).to_dict()
        dist_ref = ref_df[col].value_counts(normalize=True).to_dict()
        all_keys = set(dist_curr) | set(dist_ref)
        max_diff = max(abs(dist_curr.get(k, 0) - dist_ref.get(k, 0)) for k in all_keys)
        passed = max_diff < 0.10
        checks.append({
            "check": "distribution.categorical_drift", "column": col,
            "pass": passed,
            "detail": f"max_proportion_diff={max_diff:.4f}",
        })
    return checks


# ---------------------------------------------------------------------------
# Freshness checks
# ---------------------------------------------------------------------------

def freshness_checks(df, schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Verify that timestamp columns are recent enough."""
    import pandas as pd

    checks: List[Dict[str, Any]] = []
    freshness_cfg = schema.get("freshness", {})
    ts_col = freshness_cfg.get("timestamp_column")
    max_age_hours = freshness_cfg.get("max_age_hours", 24)

    if not ts_col or ts_col not in df.columns:
        return checks

    try:
        ts_series = pd.to_datetime(df[ts_col], utc=True)
        latest = ts_series.max()
        now = pd.Timestamp.now(tz="UTC")
        age_hours = (now - latest).total_seconds() / 3600
        passed = age_hours <= max_age_hours
        checks.append({
            "check": "freshness.max_age",
            "pass": passed,
            "detail": f"latest={latest.isoformat()}, age_hours={age_hours:.1f}, max={max_age_hours}",
        })
    except Exception as exc:
        checks.append({"check": "freshness.max_age", "pass": False,
                        "detail": f"error parsing timestamps: {exc}"})
    return checks


# ---------------------------------------------------------------------------
# Volume checks
# ---------------------------------------------------------------------------

def volume_checks(df, schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Row count within expected range."""
    checks: List[Dict[str, Any]] = []
    vol_cfg = schema.get("volume", {})
    min_rows = vol_cfg.get("min_rows")
    max_rows = vol_cfg.get("max_rows")
    row_count = len(df)

    if min_rows is not None:
        checks.append({"check": "volume.min_rows", "pass": row_count >= min_rows,
                        "detail": f"rows={row_count}, min={min_rows}"})
    if max_rows is not None:
        checks.append({"check": "volume.max_rows", "pass": row_count <= max_rows,
                        "detail": f"rows={row_count}, max={max_rows}"})
    return checks


# ---------------------------------------------------------------------------
# Referential integrity
# ---------------------------------------------------------------------------

def referential_integrity_checks(df, schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Check foreign-key relationships across columns or external files."""
    checks: List[Dict[str, Any]] = []
    refs = schema.get("referential_integrity", [])

    for ref in refs:
        col = ref.get("column")
        ref_source = ref.get("reference")  # column name or file path

        if col not in df.columns:
            checks.append({"check": "referential_integrity", "column": col,
                            "pass": False, "detail": "column missing"})
            continue

        # Reference is another column in the same dataframe
        if ref_source in df.columns:
            valid_keys = set(df[ref_source].dropna().unique())
        elif Path(ref_source).exists():
            ref_df = load_data(ref_source)
            ref_col = ref.get("reference_column", ref_df.columns[0])
            valid_keys = set(ref_df[ref_col].dropna().unique())
        else:
            checks.append({"check": "referential_integrity", "column": col,
                            "pass": False, "detail": f"reference '{ref_source}' not found"})
            continue

        orphans = int((~df[col].dropna().isin(valid_keys)).sum())
        checks.append({
            "check": "referential_integrity", "column": col,
            "pass": orphans == 0,
            "detail": f"orphan_rows={orphans}, reference={ref_source}",
        })
    return checks


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report(checks: List[Dict[str, Any]], data_path: str) -> Dict[str, Any]:
    """Aggregate all checks into a final report."""
    total = len(checks)
    passed = sum(1 for c in checks if c["pass"])
    failed = total - passed

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_path": data_path,
        "summary": {"total_checks": total, "passed": passed, "failed": failed},
        "overall_pass": failed == 0,
        "checks": checks,
    }


def report_to_markdown(report: Dict[str, Any]) -> str:
    """Render the report as Markdown."""
    lines = [
        "# Data Pipeline Test Report", "",
        f"**Timestamp:** {report['timestamp']}",
        f"**Data:** `{report['data_path']}`",
        f"**Overall:** {'PASS' if report['overall_pass'] else 'FAIL'}", "",
        f"**Checks:** {report['summary']['passed']}/{report['summary']['total_checks']} passed", "",
        "## Results", "",
        "| Check | Column | Pass | Detail |",
        "|-------|--------|:----:|--------|",
    ]

    for c in report["checks"]:
        col = c.get("column", "-")
        status = "Yes" if c["pass"] else "**No**"
        detail = c.get("detail", "")
        lines.append(f"| {c['check']} | {col} | {status} | {detail} |")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Data pipeline testing and validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data", required=True, help="Path to the dataset to validate")
    parser.add_argument("--schema", required=True, help="Path to YAML schema definition")
    parser.add_argument("--reference", default=None,
                        help="Path to reference dataset for distribution tests")
    parser.add_argument("--output", default=None, help="Output path for JSON report")
    parser.add_argument("--markdown", default=None, help="Output path for Markdown report")
    parser.add_argument("--fail-on-error", action="store_true",
                        help="Exit with non-zero code if any check fails")

    args = parser.parse_args()

    logger.info("Loading data from %s", args.data)
    df = load_data(args.data)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    logger.info("Loading schema from %s", args.schema)
    schema = load_schema(args.schema)

    all_checks: List[Dict[str, Any]] = []

    # 1. Schema validation
    logger.info("Running schema validation ...")
    all_checks.extend(validate_schema(df, schema))

    # 2. Data quality
    logger.info("Running data quality checks ...")
    all_checks.extend(quality_checks(df, schema))

    # 3. Distribution tests (requires reference data)
    if args.reference:
        logger.info("Loading reference data from %s", args.reference)
        ref_df = load_data(args.reference)
        logger.info("Running distribution tests ...")
        all_checks.extend(distribution_tests(df, ref_df, schema))
    else:
        logger.info("Skipping distribution tests (no --reference provided)")

    # 4. Freshness
    logger.info("Running freshness checks ...")
    all_checks.extend(freshness_checks(df, schema))

    # 5. Volume
    logger.info("Running volume checks ...")
    all_checks.extend(volume_checks(df, schema))

    # 6. Referential integrity
    logger.info("Running referential integrity checks ...")
    all_checks.extend(referential_integrity_checks(df, schema))

    # Build report
    report = build_report(all_checks, args.data)

    # Console summary
    logger.info("=" * 60)
    logger.info("OVERALL: %s  (%d/%d passed)",
                "PASS" if report["overall_pass"] else "FAIL",
                report["summary"]["passed"], report["summary"]["total_checks"])
    for c in all_checks:
        if not c["pass"]:
            logger.warning("  FAILED: %s [%s] — %s",
                           c["check"], c.get("column", "-"), c.get("detail", ""))
    logger.info("=" * 60)

    # Save JSON report
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("JSON report saved to %s", args.output)

    # Save Markdown report
    md_text = report_to_markdown(report)
    if args.markdown:
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        with open(args.markdown, "w") as f:
            f.write(md_text)
        logger.info("Markdown report saved to %s", args.markdown)

    if args.fail_on_error and not report["overall_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
