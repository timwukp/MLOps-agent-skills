#!/usr/bin/env python3
"""Model monitoring tool using Evidently and custom metrics.

Usage:
    python monitor_model.py --reference ref.parquet --current prod.parquet --target target_col
    python monitor_model.py --reference ref.parquet --current prod.parquet --report report.html
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


def compute_classification_metrics(y_true, y_pred):
    """Compute classification metrics."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def compute_regression_metrics(y_true, y_pred):
    """Compute regression metrics."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100),
    }


def check_prediction_distribution(reference_preds, current_preds):
    """Compare prediction distributions."""
    ref_stats = {
        "mean": float(np.mean(reference_preds)),
        "std": float(np.std(reference_preds)),
        "median": float(np.median(reference_preds)),
    }
    cur_stats = {
        "mean": float(np.mean(current_preds)),
        "std": float(np.std(current_preds)),
        "median": float(np.median(current_preds)),
    }

    mean_shift = abs(cur_stats["mean"] - ref_stats["mean"]) / max(ref_stats["std"], 1e-10)

    return {
        "reference_stats": ref_stats,
        "current_stats": cur_stats,
        "mean_shift_std": float(mean_shift),
        "significant_shift": mean_shift > 2.0,
    }


def check_thresholds(metrics, thresholds):
    """Check metrics against thresholds."""
    violations = []
    for metric, value in metrics.items():
        if metric in thresholds:
            threshold = thresholds[metric]
            if isinstance(threshold, dict):
                if "min" in threshold and value < threshold["min"]:
                    violations.append({
                        "metric": metric, "value": value,
                        "threshold": threshold["min"], "type": "below_min",
                    })
                if "max" in threshold and value > threshold["max"]:
                    violations.append({
                        "metric": metric, "value": value,
                        "threshold": threshold["max"], "type": "above_max",
                    })
            else:
                if value < threshold:
                    violations.append({
                        "metric": metric, "value": value, "threshold": threshold, "type": "below",
                    })
    return violations


def generate_evidently_report(reference_df, current_df, output_path):
    """Generate Evidently drift and quality reports."""
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset

        report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
        report.run(reference_data=reference_df, current_data=current_df)

        if output_path.endswith(".html"):
            report.save_html(output_path)
        else:
            result = report.as_dict()
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2, default=str)

        logger.info(f"Evidently report saved to {output_path}")
        return True
    except ImportError:
        logger.warning("Evidently not installed. Skipping Evidently report.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Model monitoring tool")
    parser.add_argument("--reference", required=True, help="Reference data path")
    parser.add_argument("--current", required=True, help="Current production data path")
    parser.add_argument("--target", default=None, help="Target/label column name")
    parser.add_argument("--prediction", default="prediction", help="Prediction column name")
    parser.add_argument("--task", default="classification", choices=["classification", "regression"])
    parser.add_argument("--thresholds", default=None, help="JSON file with metric thresholds")
    parser.add_argument("--report", default=None, help="Output report path")
    parser.add_argument("--evidently-report", default=None, help="Evidently HTML report path")
    parser.add_argument("--fail-on-violation", action="store_true")

    args = parser.parse_args()

    ref_df = pd.read_parquet(args.reference) if args.reference.endswith(".parquet") else pd.read_csv(args.reference)
    cur_df = pd.read_parquet(args.current) if args.current.endswith(".parquet") else pd.read_csv(args.current)

    logger.info(f"Reference: {len(ref_df)} rows, Current: {len(cur_df)} rows")

    report = {"timestamp": datetime.utcnow().isoformat()}

    # Performance metrics
    if args.target and args.target in cur_df.columns and args.prediction in cur_df.columns:
        if args.task == "classification":
            metrics = compute_classification_metrics(cur_df[args.target], cur_df[args.prediction])
        else:
            metrics = compute_regression_metrics(cur_df[args.target], cur_df[args.prediction])
        report["performance_metrics"] = metrics
        logger.info(f"Performance: {json.dumps(metrics, indent=2)}")

    # Prediction distribution
    if args.prediction in ref_df.columns and args.prediction in cur_df.columns:
        pred_dist = check_prediction_distribution(ref_df[args.prediction], cur_df[args.prediction])
        report["prediction_distribution"] = pred_dist
        if pred_dist["significant_shift"]:
            logger.warning("Significant prediction distribution shift detected!")

    # Threshold violations
    violations = []
    if args.thresholds:
        with open(args.thresholds) as f:
            thresholds = json.load(f)
        violations = check_thresholds(report.get("performance_metrics", {}), thresholds)
        report["threshold_violations"] = violations
        for v in violations:
            logger.error(f"VIOLATION: {v['metric']} = {v['value']:.4f} (threshold: {v['threshold']})")

    # Evidently report
    if args.evidently_report:
        generate_evidently_report(ref_df, cur_df, args.evidently_report)

    # Save report
    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Report saved to {args.report}")

    if args.fail_on_violation and violations:
        sys.exit(1)


if __name__ == "__main__":
    main()
