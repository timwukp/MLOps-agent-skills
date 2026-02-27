#!/usr/bin/env python3
"""Drift detection tool for ML model monitoring.

Supports PSI, KS test, chi-squared test, and Wasserstein distance.

Usage:
    python detect_drift.py --reference ref.parquet --current prod.parquet
    python detect_drift.py --reference ref.parquet --current prod.parquet --tests psi ks --threshold 0.2
"""
import argparse
import json
import logging
import sys
from datetime import datetime

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def calculate_psi(reference, current, n_bins=10):
    """Population Stability Index."""
    breakpoints = np.quantile(reference.dropna(), np.linspace(0, 1, n_bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    breakpoints = np.unique(breakpoints)

    ref_counts = np.histogram(reference.dropna(), bins=breakpoints)[0]
    cur_counts = np.histogram(current.dropna(), bins=breakpoints)[0]

    ref_pct = (ref_counts + 1) / (len(reference.dropna()) + len(breakpoints) - 1)
    cur_pct = (cur_counts + 1) / (len(current.dropna()) + len(breakpoints) - 1)

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return {"test": "psi", "statistic": psi, "drift_detected": psi > 0.2}


def ks_test(reference, current, significance=0.05):
    """Kolmogorov-Smirnov test for numerical features."""
    from scipy import stats
    stat, p_value = stats.ks_2samp(reference.dropna(), current.dropna())
    return {
        "test": "ks", "statistic": float(stat),
        "p_value": float(p_value), "drift_detected": p_value < significance,
    }


def chi2_test(reference, current, significance=0.05):
    """Chi-squared test for categorical features."""
    from scipy import stats
    categories = set(reference.dropna().unique()) | set(current.dropna().unique())
    ref_counts = reference.value_counts().reindex(categories, fill_value=0)
    cur_counts = current.value_counts().reindex(categories, fill_value=0)
    expected = (ref_counts / ref_counts.sum()) * cur_counts.sum()
    expected = expected.clip(lower=1)
    stat, p_value = stats.chisquare(cur_counts, f_exp=expected)
    return {
        "test": "chi2", "statistic": float(stat),
        "p_value": float(p_value), "drift_detected": p_value < significance,
    }


def wasserstein_test(reference, current, threshold=0.1):
    """Wasserstein (Earth Mover's) distance."""
    from scipy import stats
    dist = stats.wasserstein_distance(reference.dropna(), current.dropna())
    ref_std = reference.std()
    normalized = dist / ref_std if ref_std > 0 else dist
    return {
        "test": "wasserstein", "statistic": float(dist),
        "normalized": float(normalized), "drift_detected": normalized > threshold,
    }


def detect_drift(ref_df, cur_df, columns=None, tests=None, significance=0.05):
    """Run drift detection on all specified columns."""
    columns = columns or ref_df.columns.tolist()
    tests = tests or ["psi", "ks"]
    results = {}

    for col in columns:
        if col not in ref_df.columns or col not in cur_df.columns:
            logger.warning(f"Column '{col}' missing, skipping")
            continue

        col_results = {}
        is_numeric = pd.api.types.is_numeric_dtype(ref_df[col])

        if is_numeric:
            if "psi" in tests:
                col_results["psi"] = calculate_psi(ref_df[col], cur_df[col])
            if "ks" in tests:
                col_results["ks"] = ks_test(ref_df[col], cur_df[col], significance)
            if "wasserstein" in tests:
                col_results["wasserstein"] = wasserstein_test(ref_df[col], cur_df[col])
        else:
            if "chi2" in tests or "psi" in tests:
                col_results["chi2"] = chi2_test(ref_df[col], cur_df[col], significance)

        any_drift = any(r.get("drift_detected", False) for r in col_results.values())
        results[col] = {"tests": col_results, "drift_detected": any_drift, "type": "numerical" if is_numeric else "categorical"}

    # Summary
    drifted = [col for col, r in results.items() if r["drift_detected"]]
    drift_score = len(drifted) / len(results) if results else 0

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "columns_checked": len(results),
        "columns_drifted": len(drifted),
        "drifted_columns": drifted,
        "drift_score": round(drift_score, 3),
        "overall_drift": drift_score > 0.3,
        "details": results,
    }


def recommend_action(results):
    """Recommend action based on drift severity."""
    score = results["drift_score"]
    if score > 0.5:
        return "RETRAIN", "Significant drift in >50% of features. Immediate retraining recommended."
    elif score > 0.3:
        return "INVESTIGATE", "Moderate drift. Investigate root cause and consider retraining."
    elif score > 0.1:
        return "MONITOR", "Minor drift detected. Increase monitoring frequency."
    return "OK", "No significant drift detected."


def main():
    parser = argparse.ArgumentParser(description="Drift detection for ML models")
    parser.add_argument("--reference", required=True, help="Reference data path")
    parser.add_argument("--current", required=True, help="Current/production data path")
    parser.add_argument("--columns", nargs="*", default=None, help="Columns to check")
    parser.add_argument("--tests", nargs="*", default=["psi", "ks"],
                       choices=["psi", "ks", "chi2", "wasserstein"], help="Tests to run")
    parser.add_argument("--significance", type=float, default=0.05)
    parser.add_argument("--report", default=None, help="Output report path (JSON)")
    parser.add_argument("--fail-on-drift", action="store_true")

    args = parser.parse_args()

    ref_df = pd.read_parquet(args.reference) if args.reference.endswith(".parquet") else pd.read_csv(args.reference)
    cur_df = pd.read_parquet(args.current) if args.current.endswith(".parquet") else pd.read_csv(args.current)

    logger.info(f"Reference: {len(ref_df)} rows, Current: {len(cur_df)} rows")

    results = detect_drift(ref_df, cur_df, args.columns, args.tests, args.significance)
    action, reason = recommend_action(results)

    logger.info(f"Drift score: {results['drift_score']:.1%} ({results['columns_drifted']}/{results['columns_checked']} columns)")
    logger.info(f"Recommendation: {action} - {reason}")

    if results["drifted_columns"]:
        logger.warning(f"Drifted columns: {', '.join(results['drifted_columns'])}")

    if args.report:
        results["recommendation"] = {"action": action, "reason": reason}
        with open(args.report, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Report saved to {args.report}")

    if args.fail_on_drift and results["overall_drift"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
