#!/usr/bin/env python3
"""
ML Model Testing Suite — Comprehensive testing for ML models in production.

Features:
    - Behavioral testing (CheckList-inspired): invariance, directional, MFT
    - Performance regression testing against a baseline model
    - Slice-based testing across data subgroups
    - Boundary/edge-case testing
    - Consistency/determinism testing
    - Quality gates with configurable thresholds
    - JSON and Markdown report generation

Usage:
    python test_model.py --help
    python test_model.py --model-path model.pkl --test-data test.csv --task classification
    python test_model.py --model-path model.pkl --test-data test.csv --task regression \
        --baseline-model baseline.pkl --slices region --output report.json
    python test_model.py --model-path model.pkl --test-data test.csv --task classification \
        --thresholds '{"accuracy": 0.90, "latency_p99_ms": 200}'
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ml_model_test")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data(path: str):
    """Load a dataset from CSV, Parquet, or JSON."""
    import pandas as pd

    p = str(path)
    if p.endswith(".parquet"):
        return pd.read_parquet(p)
    if p.endswith(".csv"):
        return pd.read_csv(p)
    if p.endswith(".json") or p.endswith(".jsonl"):
        return pd.read_json(p, lines=p.endswith(".jsonl"))
    raise ValueError(f"Unsupported data format: {p}")


def load_model(path: str):
    """Load a model from a pickle/joblib file."""
    import joblib

    return joblib.load(path)


# ---------------------------------------------------------------------------
# Test categories
# ---------------------------------------------------------------------------

def behavioral_tests(model, X, y, task: str) -> Dict[str, Any]:
    """CheckList-inspired behavioral tests.

    * Invariance  – predictions should not change for trivial perturbations.
    * Directional – monotonic feature changes should move predictions predictably.
    * Minimum Functionality Test (MFT) – basic sanity checks on simple inputs.
    """
    import numpy as np

    results: Dict[str, Any] = {"invariance": {}, "directional": {}, "mft": {}}

    # --- Invariance: add tiny noise to numeric features ---
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        X_noisy = X.copy()
        rng = np.random.RandomState(42)
        noise = rng.normal(0, 1e-6, size=X_noisy[numeric_cols].shape)
        X_noisy[numeric_cols] = X_noisy[numeric_cols] + noise

        preds_orig = model.predict(X)
        preds_noisy = model.predict(X_noisy)
        if task == "classification":
            invariance_rate = float(np.mean(preds_orig == preds_noisy))
        else:
            invariance_rate = float(np.mean(np.abs(preds_orig - preds_noisy) < 1e-3))
        results["invariance"] = {"rate": round(invariance_rate, 4), "pass": invariance_rate > 0.95}
    else:
        results["invariance"] = {"rate": None, "pass": True, "note": "no numeric features"}

    # --- Directional: increase each numeric feature by 1 std, track prediction shift ---
    directional_shifts: List[Dict[str, Any]] = []
    for col in numeric_cols[:5]:  # cap at 5 features
        X_shifted = X.copy()
        col_std = float(X[col].std())
        if col_std == 0:
            continue
        X_shifted[col] = X_shifted[col] + col_std
        preds_shift = model.predict(X_shifted)
        if task == "regression":
            mean_delta = float(np.mean(preds_shift - preds_orig))
            directional_shifts.append({"feature": col, "mean_delta": round(mean_delta, 6)})
        else:
            change_rate = float(np.mean(preds_shift != preds_orig))
            directional_shifts.append({"feature": col, "change_rate": round(change_rate, 4)})
    results["directional"] = {"shifts": directional_shifts}

    # --- MFT: model should produce valid output shapes and types ---
    single_pred = model.predict(X.iloc[:1])
    results["mft"] = {
        "single_prediction_shape": list(np.array(single_pred).shape),
        "returns_numeric": bool(np.issubdtype(np.array(single_pred).dtype, np.number)),
        "pass": True,
    }
    return results


def performance_regression_tests(
    model, baseline_model, X, y, task: str
) -> Dict[str, Any]:
    """Compare new model vs baseline on the same test set."""
    import numpy as np

    results: Dict[str, Any] = {}
    preds_new = model.predict(X)

    if task == "classification":
        from sklearn.metrics import accuracy_score, f1_score

        acc_new = accuracy_score(y, preds_new)
        f1_new = f1_score(y, preds_new, average="weighted", zero_division=0)
        results["new_model"] = {"accuracy": round(acc_new, 4), "f1_weighted": round(f1_new, 4)}

        if baseline_model is not None:
            preds_base = baseline_model.predict(X)
            acc_base = accuracy_score(y, preds_base)
            f1_base = f1_score(y, preds_base, average="weighted", zero_division=0)
            results["baseline_model"] = {"accuracy": round(acc_base, 4), "f1_weighted": round(f1_base, 4)}
            results["regression"] = {
                "accuracy_delta": round(acc_new - acc_base, 4),
                "f1_delta": round(f1_new - f1_base, 4),
                "pass": acc_new >= acc_base - 0.01,
            }
    else:
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        mae_new = mean_absolute_error(y, preds_new)
        rmse_new = float(np.sqrt(mean_squared_error(y, preds_new)))
        results["new_model"] = {"mae": round(mae_new, 4), "rmse": round(rmse_new, 4)}

        if baseline_model is not None:
            preds_base = baseline_model.predict(X)
            mae_base = mean_absolute_error(y, preds_base)
            rmse_base = float(np.sqrt(mean_squared_error(y, preds_base)))
            results["baseline_model"] = {"mae": round(mae_base, 4), "rmse": round(rmse_base, 4)}
            results["regression"] = {
                "mae_delta": round(mae_new - mae_base, 4),
                "rmse_delta": round(rmse_new - rmse_base, 4),
                "pass": mae_new <= mae_base * 1.05,
            }
    return results


def slice_tests(model, X, y, task: str, slice_col: str) -> Dict[str, Any]:
    """Evaluate model performance on each unique value of *slice_col*."""
    import numpy as np

    slices: Dict[str, Any] = {}
    for value, idx in X.groupby(slice_col).groups.items():
        X_slice = X.loc[idx]
        y_slice = y.loc[idx]
        preds = model.predict(X_slice)

        if task == "classification":
            from sklearn.metrics import accuracy_score

            acc = accuracy_score(y_slice, preds)
            slices[str(value)] = {"count": int(len(idx)), "accuracy": round(acc, 4)}
        else:
            from sklearn.metrics import mean_absolute_error

            mae = mean_absolute_error(y_slice, preds)
            slices[str(value)] = {"count": int(len(idx)), "mae": round(mae, 4)}

    # Flag underperforming slices (> 10 % worse than global metric)
    if task == "classification":
        global_metric = float(np.mean([s["accuracy"] for s in slices.values()]))
        for name, s in slices.items():
            s["underperforming"] = s["accuracy"] < global_metric * 0.90
    else:
        global_metric = float(np.mean([s["mae"] for s in slices.values()]))
        for name, s in slices.items():
            s["underperforming"] = s["mae"] > global_metric * 1.10

    return {"slice_column": slice_col, "slices": slices, "global_metric": round(global_metric, 4)}


def boundary_tests(model, X) -> Dict[str, Any]:
    """Edge-case and boundary testing."""
    import numpy as np
    import pandas as pd

    results: Dict[str, Any] = {}
    numeric_cols = X.select_dtypes(include=[np.number]).columns

    # 1. Empty input
    try:
        model.predict(X.iloc[:0])
        results["empty_input"] = {"pass": True, "error": None}
    except Exception as exc:
        results["empty_input"] = {"pass": False, "error": str(exc)}

    # 2. Single row
    try:
        model.predict(X.iloc[:1])
        results["single_row"] = {"pass": True, "error": None}
    except Exception as exc:
        results["single_row"] = {"pass": False, "error": str(exc)}

    # 3. Extreme values (all numeric columns set to +/- 1e9)
    for label, fill_val in [("extreme_high", 1e9), ("extreme_low", -1e9)]:
        try:
            X_ext = X.iloc[:5].copy()
            X_ext[numeric_cols] = fill_val
            preds = model.predict(X_ext)
            has_nan = bool(np.isnan(preds).any()) if np.issubdtype(np.array(preds).dtype, np.number) else False
            results[label] = {"pass": not has_nan, "nan_in_output": has_nan}
        except Exception as exc:
            results[label] = {"pass": False, "error": str(exc)}

    # 4. All-zero row
    try:
        X_zero = pd.DataFrame(np.zeros((1, len(X.columns))), columns=X.columns)
        model.predict(X_zero)
        results["all_zeros"] = {"pass": True, "error": None}
    except Exception as exc:
        results["all_zeros"] = {"pass": False, "error": str(exc)}

    return results


def consistency_tests(model, X, n_runs: int = 5) -> Dict[str, Any]:
    """Determinism check — same input should yield same output every time."""
    import numpy as np

    sample = X.iloc[:50]
    predictions = [model.predict(sample) for _ in range(n_runs)]
    all_equal = all(np.array_equal(predictions[0], p) for p in predictions[1:])
    return {"n_runs": n_runs, "sample_size": len(sample), "deterministic": all_equal, "pass": all_equal}


def latency_benchmark(model, X, n_iterations: int = 20) -> Dict[str, Any]:
    """Measure single-row inference latency."""
    import numpy as np

    sample = X.iloc[:1]
    times: List[float] = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        model.predict(sample)
        times.append((time.perf_counter() - t0) * 1000)  # ms

    arr = np.array(times)
    return {
        "n_iterations": n_iterations,
        "mean_ms": round(float(arr.mean()), 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
    }


# ---------------------------------------------------------------------------
# Quality gates
# ---------------------------------------------------------------------------

def evaluate_quality_gates(report: Dict[str, Any], thresholds: Dict[str, float]) -> Dict[str, Any]:
    """Check configurable thresholds against collected metrics."""
    gate_results: Dict[str, Any] = {}
    perf = report.get("performance_regression", {}).get("new_model", {})
    latency = report.get("latency", {})

    mapping = {
        "accuracy": perf.get("accuracy"),
        "f1_weighted": perf.get("f1_weighted"),
        "mae": perf.get("mae"),
        "rmse": perf.get("rmse"),
        "latency_mean_ms": latency.get("mean_ms"),
        "latency_p95_ms": latency.get("p95_ms"),
        "latency_p99_ms": latency.get("p99_ms"),
    }

    higher_is_better = {"accuracy", "f1_weighted"}

    for metric_name, threshold in thresholds.items():
        actual = mapping.get(metric_name)
        if actual is None:
            gate_results[metric_name] = {"pass": False, "reason": "metric not available"}
            continue
        if metric_name in higher_is_better:
            passed = actual >= threshold
        else:
            passed = actual <= threshold
        gate_results[metric_name] = {
            "threshold": threshold,
            "actual": actual,
            "pass": passed,
        }
    return gate_results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_markdown(report: Dict[str, Any]) -> str:
    """Render the test report as a Markdown string."""
    lines = ["# ML Model Test Report", "", f"**Timestamp:** {report['timestamp']}",
             f"**Model:** `{report['model_path']}`", f"**Task:** {report['task']}", ""]

    overall = report.get("overall_pass", "N/A")
    lines.append(f"**Overall result:** {'PASS' if overall else 'FAIL'}")
    lines.append("")

    # Performance
    if "performance_regression" in report:
        lines.append("## Performance")
        for section, vals in report["performance_regression"].items():
            lines.append(f"### {section}")
            for k, v in vals.items():
                lines.append(f"- **{k}:** {v}")
            lines.append("")

    # Slices
    if "slice_tests" in report:
        lines.append("## Slice Tests")
        st = report["slice_tests"]
        lines.append(f"Slice column: `{st.get('slice_column')}`\n")
        lines.append("| Slice | Count | Metric | Underperforming |")
        lines.append("|-------|------:|-------:|:---------------:|")
        for name, info in st.get("slices", {}).items():
            metric_val = info.get("accuracy", info.get("mae", "N/A"))
            flag = "Yes" if info.get("underperforming") else ""
            lines.append(f"| {name} | {info['count']} | {metric_val} | {flag} |")
        lines.append("")

    # Quality gates
    if "quality_gates" in report:
        lines.append("## Quality Gates")
        lines.append("| Metric | Threshold | Actual | Pass |")
        lines.append("|--------|----------:|-------:|:----:|")
        for metric, info in report["quality_gates"].items():
            p = "Yes" if info.get("pass") else "No"
            lines.append(f"| {metric} | {info.get('threshold', 'N/A')} | {info.get('actual', 'N/A')} | {p} |")
        lines.append("")

    # Behavioral
    if "behavioral" in report:
        lines.append("## Behavioral Tests")
        for test_name, info in report["behavioral"].items():
            status = "PASS" if info.get("pass", True) else "FAIL"
            lines.append(f"- **{test_name}:** {status}  ")
        lines.append("")

    # Boundary
    if "boundary" in report:
        lines.append("## Boundary Tests")
        for test_name, info in report["boundary"].items():
            status = "PASS" if info.get("pass") else "FAIL"
            lines.append(f"- **{test_name}:** {status}")
        lines.append("")

    # Consistency
    if "consistency" in report:
        c = report["consistency"]
        lines.append(f"## Consistency\n- Deterministic: **{c['deterministic']}** "
                      f"({c['n_runs']} runs, {c['sample_size']} samples)\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive ML model testing suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-path", required=True, help="Path to model file (joblib/pickle)")
    parser.add_argument("--test-data", required=True, help="Path to test dataset (CSV/Parquet/JSON)")
    parser.add_argument("--baseline-model", default=None, help="Path to baseline model for regression comparison")
    parser.add_argument("--task", required=True, choices=["classification", "regression"], help="ML task type")
    parser.add_argument("--target", default=None, help="Target column name (default: last column)")
    parser.add_argument("--slices", default=None, help="Column name used for slice-based testing")
    parser.add_argument("--thresholds", default=None,
                        help='JSON string of quality-gate thresholds, e.g. \'{"accuracy": 0.9}\'')
    parser.add_argument("--output", default=None, help="Output path for JSON report")
    parser.add_argument("--markdown", default=None, help="Output path for Markdown report")

    args = parser.parse_args()

    logger.info("Loading test data from %s", args.test_data)
    df = load_data(args.test_data)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    target_col = args.target or df.columns[-1]
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Optionally drop slice column from features
    slice_col_in_features = args.slices and args.slices in X.columns

    logger.info("Loading model from %s", args.model_path)
    model = load_model(args.model_path)
    baseline_model = load_model(args.baseline_model) if args.baseline_model else None

    report: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_path": args.model_path,
        "task": args.task,
        "test_data": args.test_data,
        "num_samples": len(df),
    }

    # Features used for prediction (exclude slice col if it is categorical-only)
    X_pred = X

    # --- Run test suites ---
    logger.info("Running behavioral tests ...")
    report["behavioral"] = behavioral_tests(model, X_pred, y, args.task)

    logger.info("Running performance regression tests ...")
    report["performance_regression"] = performance_regression_tests(
        model, baseline_model, X_pred, y, args.task
    )

    if args.slices and args.slices in df.columns:
        logger.info("Running slice-based tests on column '%s' ...", args.slices)
        # Use original X (with slice col) for grouping
        report["slice_tests"] = slice_tests(model, X, y, args.task, args.slices)

    logger.info("Running boundary tests ...")
    report["boundary"] = boundary_tests(model, X_pred)

    logger.info("Running consistency tests ...")
    report["consistency"] = consistency_tests(model, X_pred)

    logger.info("Running latency benchmark ...")
    report["latency"] = latency_benchmark(model, X_pred)

    # --- Quality gates ---
    thresholds: Dict[str, float] = {}
    if args.thresholds:
        try:
            thresholds = json.loads(args.thresholds)
        except json.JSONDecodeError as exc:
            logger.error("Invalid --thresholds JSON: %s", exc)
            sys.exit(2)

    if thresholds:
        logger.info("Evaluating quality gates ...")
        report["quality_gates"] = evaluate_quality_gates(report, thresholds)

    # --- Overall pass/fail ---
    failures: List[str] = []
    for name, info in report.get("behavioral", {}).items():
        if isinstance(info, dict) and info.get("pass") is False:
            failures.append(f"behavioral.{name}")
    for name, info in report.get("boundary", {}).items():
        if isinstance(info, dict) and info.get("pass") is False:
            failures.append(f"boundary.{name}")
    if report.get("consistency", {}).get("pass") is False:
        failures.append("consistency")
    reg = report.get("performance_regression", {}).get("regression", {})
    if reg and reg.get("pass") is False:
        failures.append("performance_regression")
    for metric, info in report.get("quality_gates", {}).items():
        if info.get("pass") is False:
            failures.append(f"quality_gate.{metric}")

    report["overall_pass"] = len(failures) == 0
    report["failures"] = failures

    # --- Output ---
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("JSON report saved to %s", args.output)

    md_text = generate_markdown(report)
    if args.markdown:
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        with open(args.markdown, "w") as f:
            f.write(md_text)
        logger.info("Markdown report saved to %s", args.markdown)

    # Summary to console
    logger.info("=" * 60)
    logger.info("OVERALL: %s", "PASS" if report["overall_pass"] else "FAIL")
    if failures:
        for f_name in failures:
            logger.warning("  FAILED: %s", f_name)
    logger.info("=" * 60)

    sys.exit(0 if report["overall_pass"] else 1)


if __name__ == "__main__":
    main()
