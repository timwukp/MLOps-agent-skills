#!/usr/bin/env python3
"""
MLflow Experiment Tracking Wrapper

Provides a high-level ExperimentTracker class around MLflow for structured
experiment logging, nested runs, tag management, artifact logging (confusion
matrix, feature importance), model signature inference, and run comparison.

Usage:
    # Create an experiment
    python mlflow_tracker.py --action create-experiment --experiment my-project

    # List runs sorted by a metric
    python mlflow_tracker.py --action list-runs --experiment my-project --metric accuracy --top-n 10

    # Compare top runs
    python mlflow_tracker.py --action compare --experiment my-project --metric f1_score --top-n 5

    # Cleanup failed/abandoned runs
    python mlflow_tracker.py --action cleanup --experiment my-project

Dependencies:
    - Python 3.8+
    - mlflow
    - scikit-learn (optional, for integration examples)
    - matplotlib (optional, for confusion matrix artifact)

Optional:
    - torch, xgboost (for framework-specific auto-logging)
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------

def _import_mlflow():
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        return mlflow, MlflowClient
    except ImportError:
        logger.error("mlflow is required: pip install mlflow")
        sys.exit(1)


def _import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        logger.warning("matplotlib not available; artifact plots will be skipped")
        return None


def _import_numpy():
    try:
        import numpy as np
        return np
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Tag helpers
# ---------------------------------------------------------------------------

def _get_git_hash() -> Optional[str]:
    """Return the current git HEAD commit hash, or None."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def default_tags() -> Dict[str, str]:
    """Build a set of default tags for reproducibility."""
    import platform

    tags: Dict[str, str] = {
        "python_version": platform.python_version(),
        "os": platform.system(),
        "timestamp": datetime.utcnow().isoformat(),
    }
    git_hash = _get_git_hash()
    if git_hash:
        tags["git_commit"] = git_hash

    data_version = os.environ.get("DATA_VERSION")
    if data_version:
        tags["data_version"] = data_version

    env_name = os.environ.get("CONDA_DEFAULT_ENV") or os.environ.get("VIRTUAL_ENV")
    if env_name:
        tags["environment"] = env_name

    return tags


# ---------------------------------------------------------------------------
# ExperimentTracker
# ---------------------------------------------------------------------------

class ExperimentTracker:
    """High-level wrapper around MLflow for structured experiment tracking."""

    def __init__(self, experiment_name: str, tracking_uri: Optional[str] = None):
        self.mlflow, self.MlflowClient = _import_mlflow()
        if tracking_uri:
            self.mlflow.set_tracking_uri(tracking_uri)
        self.experiment_name = experiment_name
        self.mlflow.set_experiment(experiment_name)
        self.client = self.MlflowClient()
        self._active_run = None
        logger.info("Tracker initialised for experiment '%s'", experiment_name)

    # -- run lifecycle -----------------------------------------------------

    def start_run(self, run_name: str, tags: Optional[Dict[str, str]] = None,
                  nested: bool = False) -> Any:
        """Start a new MLflow run with default + custom tags."""
        all_tags = default_tags()
        if tags:
            all_tags.update(tags)
        self._active_run = self.mlflow.start_run(run_name=run_name,
                                                  nested=nested,
                                                  tags=all_tags)
        logger.info("Started run '%s' (id=%s, nested=%s)",
                     run_name, self._active_run.info.run_id, nested)
        return self._active_run

    def end_run(self, status: str = "FINISHED") -> None:
        self.mlflow.end_run(status=status)
        self._active_run = None
        logger.info("Ended run with status %s", status)

    # -- logging helpers ---------------------------------------------------

    def log_params(self, params: Dict[str, Any]) -> None:
        self.mlflow.log_params({k: str(v) for k, v in params.items()})

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        for key, value in metrics.items():
            self.mlflow.log_metric(key, value, step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        self.mlflow.log_artifact(local_path, artifact_path)

    def set_tags(self, tags: Dict[str, str]) -> None:
        for k, v in tags.items():
            self.mlflow.set_tag(k, v)

    # -- model logging with signature inference ----------------------------

    def log_model(self, model: Any, artifact_path: str = "model",
                  input_example: Any = None) -> None:
        """Log a model with automatic flavour detection and signature inference."""
        np = _import_numpy()
        try:
            from mlflow.models.signature import infer_signature
        except ImportError:
            infer_signature = None

        signature = None
        if infer_signature and input_example is not None:
            try:
                preds = model.predict(input_example[:5] if hasattr(input_example, '__getitem__') else input_example)
                signature = infer_signature(input_example[:5] if hasattr(input_example, '__getitem__') else input_example, preds)
                logger.info("Inferred model signature")
            except Exception as exc:
                logger.warning("Could not infer signature: %s", exc)

        # Detect framework and log accordingly
        model_type = type(model).__module__.split(".")[0] if hasattr(type(model), "__module__") else ""
        try:
            if "sklearn" in model_type or "sklearn" in str(type(model)):
                self.mlflow.sklearn.log_model(model, artifact_path,
                                              signature=signature,
                                              input_example=input_example)
            elif "xgboost" in model_type:
                self.mlflow.xgboost.log_model(model, artifact_path,
                                              signature=signature,
                                              input_example=input_example)
            elif "torch" in model_type:
                self.mlflow.pytorch.log_model(model, artifact_path)
            else:
                # Fallback: try sklearn flavour, then pickle
                self.mlflow.sklearn.log_model(model, artifact_path,
                                              signature=signature,
                                              input_example=input_example)
            logger.info("Logged model to '%s'", artifact_path)
        except Exception as exc:
            logger.error("Failed to log model: %s", exc)

    # -- artifact helpers --------------------------------------------------

    def log_confusion_matrix(self, y_true, y_pred, labels=None) -> None:
        """Save a confusion matrix plot as an artifact."""
        plt = _import_matplotlib()
        np = _import_numpy()
        if plt is None or np is None:
            logger.warning("Skipping confusion matrix (matplotlib/numpy unavailable)")
            return
        try:
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        except ImportError:
            logger.warning("scikit-learn required for confusion matrix")
            return

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap="Blues")
        ax.set_title("Confusion Matrix")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            fig.savefig(tmp.name, bbox_inches="tight", dpi=150)
            plt.close(fig)
            self.log_artifact(tmp.name, artifact_path="plots")
            os.unlink(tmp.name)
        logger.info("Logged confusion matrix artifact")

    def log_feature_importance(self, feature_names: List[str],
                               importances: List[float]) -> None:
        """Save feature importance as CSV and bar chart artifacts."""
        np = _import_numpy()
        plt = _import_matplotlib()

        with tempfile.TemporaryDirectory() as tmpdir:
            # CSV
            csv_path = os.path.join(tmpdir, "feature_importance.csv")
            with open(csv_path, "w") as f:
                f.write("feature,importance\n")
                for name, imp in sorted(zip(feature_names, importances),
                                        key=lambda x: x[1], reverse=True):
                    f.write(f"{name},{imp:.6f}\n")
            self.log_artifact(csv_path, artifact_path="features")

            # Plot
            if plt is not None and np is not None:
                indices = np.argsort(importances)[-20:]  # top 20
                fig, ax = plt.subplots(figsize=(10, max(6, len(indices) * 0.35)))
                ax.barh([feature_names[i] for i in indices],
                        [importances[i] for i in indices])
                ax.set_xlabel("Importance")
                ax.set_title("Feature Importance (top 20)")
                png_path = os.path.join(tmpdir, "feature_importance.png")
                fig.savefig(png_path, bbox_inches="tight", dpi=150)
                plt.close(fig)
                self.log_artifact(png_path, artifact_path="plots")

        logger.info("Logged feature importance artifacts")

    # -- auto-log convenience ----------------------------------------------

    def enable_autolog(self, framework: str = "sklearn") -> None:
        """Enable MLflow auto-logging for the given framework."""
        autolog_map = {
            "sklearn": lambda: self.mlflow.sklearn.autolog(),
            "pytorch": lambda: self.mlflow.pytorch.autolog(),
            "tensorflow": lambda: self.mlflow.tensorflow.autolog(),
            "xgboost": lambda: self.mlflow.xgboost.autolog(),
            "lightgbm": lambda: self.mlflow.lightgbm.autolog(),
        }
        fn = autolog_map.get(framework)
        if fn is None:
            logger.warning("Unknown framework '%s'; skipping autolog", framework)
            return
        try:
            fn()
            logger.info("Auto-logging enabled for %s", framework)
        except Exception as exc:
            logger.warning("Could not enable autolog for %s: %s", framework, exc)

    # -- run comparison / querying -----------------------------------------

    def get_experiment_id(self) -> Optional[str]:
        exp = self.client.get_experiment_by_name(self.experiment_name)
        return exp.experiment_id if exp else None

    def list_runs(self, metric: str = "accuracy", top_n: int = 10,
                  status: str = "") -> List[Dict]:
        """Fetch runs from the experiment, sorted by *metric* descending."""
        exp_id = self.get_experiment_id()
        if exp_id is None:
            logger.error("Experiment '%s' not found", self.experiment_name)
            return []

        filter_str = f"attributes.status = '{status}'" if status else ""
        runs = self.client.search_runs(
            experiment_ids=[exp_id],
            filter_string=filter_str,
            order_by=[f"metrics.{metric} DESC"],
            max_results=top_n,
        )
        results = []
        for run in runs:
            results.append({
                "run_id": run.info.run_id,
                "run_name": run.info.run_name or "",
                "status": run.info.status,
                "params": dict(run.data.params),
                "metrics": dict(run.data.metrics),
                "tags": {k: v for k, v in run.data.tags.items()
                         if not k.startswith("mlflow.")},
            })
        return results

    def compare_runs(self, metric: str = "accuracy", top_n: int = 5) -> str:
        """Return a formatted table comparing the top-N runs by *metric*."""
        runs = self.list_runs(metric=metric, top_n=top_n)
        if not runs:
            return "No runs found."

        # Collect all metric keys across runs
        all_metrics = sorted({k for r in runs for k in r["metrics"]})
        header = f"{'Run Name':<25} {'Status':<10} " + "  ".join(f"{m:>12}" for m in all_metrics)
        sep = "-" * len(header)
        lines = [sep, header, sep]
        for r in runs:
            vals = "  ".join(f"{r['metrics'].get(m, 0):>12.4f}" for m in all_metrics)
            lines.append(f"{r['run_name']:<25} {r['status']:<10} {vals}")
        lines.append(sep)
        return "\n".join(lines)

    # -- cleanup -----------------------------------------------------------

    def cleanup(self, delete_failed: bool = True,
                delete_unfinished: bool = True) -> int:
        """Delete failed and/or unfinished runs. Returns count deleted."""
        exp_id = self.get_experiment_id()
        if exp_id is None:
            logger.error("Experiment '%s' not found", self.experiment_name)
            return 0

        statuses = []
        if delete_failed:
            statuses.append("FAILED")
        if delete_unfinished:
            statuses.append("RUNNING")

        deleted = 0
        for status in statuses:
            runs = self.client.search_runs(
                experiment_ids=[exp_id],
                filter_string=f"attributes.status = '{status}'",
            )
            for run in runs:
                self.client.delete_run(run.info.run_id)
                deleted += 1
                logger.info("Deleted run %s (%s)", run.info.run_id, status)

        logger.info("Cleaned up %d runs from '%s'", deleted, self.experiment_name)
        return deleted


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MLflow experiment tracking wrapper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mlflow_tracker.py --action create-experiment --experiment my-project
  python mlflow_tracker.py --action list-runs --experiment my-project --metric accuracy --top-n 10
  python mlflow_tracker.py --action compare --experiment my-project --metric f1_score --top-n 5
  python mlflow_tracker.py --action cleanup --experiment my-project
        """,
    )
    parser.add_argument("--action", required=True,
                        choices=["create-experiment", "list-runs", "compare", "cleanup"],
                        help="Action to perform")
    parser.add_argument("--experiment", required=True, help="Experiment name")
    parser.add_argument("--tracking-uri", default=None,
                        help="MLflow tracking URI (default: local ./mlruns)")
    parser.add_argument("--metric", default="accuracy",
                        help="Metric to sort/compare by (default: accuracy)")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Number of top runs to display (default: 10)")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    tracker = ExperimentTracker(args.experiment, tracking_uri=args.tracking_uri)

    if args.action == "create-experiment":
        exp_id = tracker.get_experiment_id()
        if exp_id:
            logger.info("Experiment '%s' already exists (id=%s)", args.experiment, exp_id)
        else:
            mlflow, _ = _import_mlflow()
            exp_id = mlflow.create_experiment(args.experiment)
            logger.info("Created experiment '%s' (id=%s)", args.experiment, exp_id)
        if args.json:
            print(json.dumps({"experiment": args.experiment, "id": exp_id}))

    elif args.action == "list-runs":
        runs = tracker.list_runs(metric=args.metric, top_n=args.top_n)
        if args.json:
            print(json.dumps(runs, indent=2, default=str))
        else:
            for r in runs:
                metric_val = r["metrics"].get(args.metric, "N/A")
                print(f"  {r['run_name']:<25} {r['status']:<10} "
                      f"{args.metric}={metric_val}")

    elif args.action == "compare":
        if args.json:
            runs = tracker.list_runs(metric=args.metric, top_n=args.top_n)
            print(json.dumps(runs, indent=2, default=str))
        else:
            print(tracker.compare_runs(metric=args.metric, top_n=args.top_n))

    elif args.action == "cleanup":
        count = tracker.cleanup()
        if args.json:
            print(json.dumps({"deleted": count}))
        else:
            print(f"Deleted {count} runs.")


if __name__ == "__main__":
    main()
