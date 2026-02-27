#!/usr/bin/env python3
"""
Experiment Comparison and Analysis

Fetches runs from an MLflow experiment and produces comparison tables,
statistical summaries, markdown/JSON reports, and parallel-coordinates
plots for hyperparameter search visualization.

Usage:
    # Table of top runs sorted by accuracy
    python experiment_compare.py --experiment my-project --metric accuracy --top-n 10

    # Markdown report
    python experiment_compare.py --experiment my-project --metric f1_score --output-format markdown

    # JSON report saved to file
    python experiment_compare.py --experiment my-project --metric accuracy --output-format json > report.json

    # Generate parallel coordinates plot
    python experiment_compare.py --experiment my-project --metric accuracy --plot hparam_plot.png

Dependencies:
    - Python 3.8+
    - mlflow

Optional:
    - matplotlib (for --plot)
    - numpy (for statistical analysis)
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

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
        logger.warning("matplotlib not available; plot generation will be skipped")
        return None


def _import_numpy():
    try:
        import numpy as np
        return np
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_runs(experiment_name: str, tracking_uri: Optional[str] = None,
               status_filter: Optional[str] = None,
               tag_filters: Optional[Dict[str, str]] = None,
               start_date: Optional[str] = None,
               end_date: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetch all runs from an MLflow experiment with optional filters.

    Parameters
    ----------
    status_filter : e.g. "FINISHED", "FAILED", "RUNNING"
    tag_filters   : dict of tag_key -> tag_value to match
    start_date / end_date : ISO-8601 date strings (YYYY-MM-DD)
    """
    mlflow, MlflowClient = _import_mlflow()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        logger.error("Experiment '%s' not found", experiment_name)
        return []

    # Build filter string
    parts: List[str] = []
    if status_filter:
        parts.append(f"attributes.status = '{status_filter}'")
    if tag_filters:
        for k, v in tag_filters.items():
            parts.append(f"tags.{k} = '{v}'")
    if start_date:
        ts = int(datetime.fromisoformat(start_date).timestamp() * 1000)
        parts.append(f"attributes.start_time >= {ts}")
    if end_date:
        ts = int(datetime.fromisoformat(end_date).timestamp() * 1000)
        parts.append(f"attributes.start_time <= {ts}")

    filter_string = " AND ".join(parts) if parts else ""

    raw_runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=filter_string,
        max_results=5000,
    )

    runs: List[Dict[str, Any]] = []
    for r in raw_runs:
        runs.append({
            "run_id": r.info.run_id,
            "run_name": r.info.run_name or "",
            "status": r.info.status,
            "start_time": datetime.fromtimestamp(
                r.info.start_time / 1000).isoformat() if r.info.start_time else None,
            "params": dict(r.data.params),
            "metrics": dict(r.data.metrics),
            "tags": {k: v for k, v in r.data.tags.items()
                     if not k.startswith("mlflow.")},
        })

    logger.info("Fetched %d runs from experiment '%s'", len(runs), experiment_name)
    return runs


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------

def analyse_runs(runs: List[Dict], metric: str) -> Dict[str, Any]:
    """Compute statistics over the target metric and correlate with params."""
    np = _import_numpy()

    values = [r["metrics"][metric] for r in runs if metric in r["metrics"]]
    if not values:
        return {"error": f"No runs contain metric '{metric}'"}

    stats: Dict[str, Any] = {
        "metric": metric,
        "count": len(values),
        "best": max(values),
        "worst": min(values),
        "mean": sum(values) / len(values),
    }

    if np is not None and len(values) > 1:
        arr = np.array(values, dtype=float)
        stats["std"] = float(np.std(arr, ddof=1))
        stats["median"] = float(np.median(arr))
    elif len(values) > 1:
        mean = stats["mean"]
        stats["std"] = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5

    # Identify best run
    best_run = max(
        (r for r in runs if metric in r["metrics"]),
        key=lambda r: r["metrics"][metric],
    )
    stats["best_run_id"] = best_run["run_id"]
    stats["best_run_name"] = best_run["run_name"]

    # Metric trend (by start_time)
    timed = sorted(
        ((r["start_time"], r["metrics"][metric]) for r in runs
         if r.get("start_time") and metric in r["metrics"]),
        key=lambda x: x[0],
    )
    if timed:
        stats["trend"] = [{"time": t, "value": v} for t, v in timed]

    # Hyperparameter correlation with metric
    stats["param_correlation"] = _param_correlation(runs, metric)

    return stats


def _param_correlation(runs: List[Dict], metric: str) -> Dict[str, float]:
    """Pearson correlation of each numeric param with the target metric."""
    np = _import_numpy()
    if np is None or len(runs) < 3:
        return {}

    metric_vals = []
    param_vecs: Dict[str, List[float]] = {}

    for r in runs:
        if metric not in r["metrics"]:
            continue
        metric_vals.append(r["metrics"][metric])
        for k, v in r["params"].items():
            try:
                param_vecs.setdefault(k, []).append(float(v))
            except (ValueError, TypeError):
                pass

    correlations: Dict[str, float] = {}
    m_arr = np.array(metric_vals, dtype=float)
    for k, vals in param_vecs.items():
        if len(vals) != len(metric_vals):
            continue
        p_arr = np.array(vals, dtype=float)
        if np.std(p_arr) == 0 or np.std(m_arr) == 0:
            continue
        corr = float(np.corrcoef(p_arr, m_arr)[0, 1])
        if not (corr != corr):  # skip NaN
            correlations[k] = round(corr, 4)

    return dict(sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True))


# ---------------------------------------------------------------------------
# Formatting / reports
# ---------------------------------------------------------------------------

def format_table(runs: List[Dict], metric: str, top_n: int) -> str:
    """Pretty-print a comparison table to the terminal."""
    sorted_runs = sorted(
        [r for r in runs if metric in r["metrics"]],
        key=lambda r: r["metrics"][metric],
        reverse=True,
    )[:top_n]

    if not sorted_runs:
        return f"No runs with metric '{metric}' found."

    all_metrics = sorted({k for r in sorted_runs for k in r["metrics"]})
    header = f"{'#':<4} {'Run Name':<25} {'Status':<10} " + "  ".join(
        f"{m:>14}" for m in all_metrics)
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for i, r in enumerate(sorted_runs, 1):
        vals = "  ".join(f"{r['metrics'].get(m, 0):>14.5f}" for m in all_metrics)
        lines.append(f"{i:<4} {r['run_name']:<25} {r['status']:<10} {vals}")
    lines.append(sep)
    return "\n".join(lines)


def generate_markdown(runs: List[Dict], metric: str, top_n: int,
                      stats: Dict) -> str:
    """Generate a Markdown comparison report."""
    sorted_runs = sorted(
        [r for r in runs if metric in r["metrics"]],
        key=lambda r: r["metrics"][metric],
        reverse=True,
    )[:top_n]

    lines = [
        f"# Experiment Comparison Report",
        f"",
        f"_Generated: {datetime.utcnow().isoformat()}_",
        f"",
        f"## Summary",
        f"",
        f"| Statistic | Value |",
        f"|-----------|-------|",
        f"| Metric | {stats.get('metric', metric)} |",
        f"| Runs analysed | {stats.get('count', len(runs))} |",
        f"| Best | {stats.get('best', 'N/A'):.5f} |",
        f"| Worst | {stats.get('worst', 'N/A'):.5f} |",
        f"| Mean | {stats.get('mean', 'N/A'):.5f} |",
        f"| Std | {stats.get('std', 'N/A'):.5f} |",
        f"| Best run | {stats.get('best_run_name', '')} (`{stats.get('best_run_id', '')[:8]}`) |",
        f"",
        f"## Top {top_n} Runs",
        f"",
    ]

    # Table header
    all_metrics = sorted({k for r in sorted_runs for k in r["metrics"]})
    lines.append("| # | Run Name | " + " | ".join(all_metrics) + " |")
    lines.append("|---|---------" + "".join("| ---:" for _ in all_metrics) + " |")
    for i, r in enumerate(sorted_runs, 1):
        vals = " | ".join(f"{r['metrics'].get(m, 0):.5f}" for m in all_metrics)
        lines.append(f"| {i} | {r['run_name']} | {vals} |")

    # Parameter correlations
    corr = stats.get("param_correlation", {})
    if corr:
        lines.extend([
            "",
            "## Hyperparameter Correlation with Metric",
            "",
            "| Parameter | Correlation |",
            "|-----------|------------|",
        ])
        for param, val in corr.items():
            lines.append(f"| {param} | {val:+.4f} |")

    lines.append("")
    return "\n".join(lines)


def generate_json_report(runs: List[Dict], metric: str, top_n: int,
                         stats: Dict) -> str:
    sorted_runs = sorted(
        [r for r in runs if metric in r["metrics"]],
        key=lambda r: r["metrics"][metric],
        reverse=True,
    )[:top_n]

    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "statistics": stats,
        "top_runs": sorted_runs,
    }
    return json.dumps(report, indent=2, default=str)


# ---------------------------------------------------------------------------
# Parallel coordinates plot
# ---------------------------------------------------------------------------

def parallel_coordinates_plot(runs: List[Dict], metric: str,
                              output_path: str) -> bool:
    """Create a parallel coordinates plot for hyperparameter search and save as PNG."""
    plt = _import_matplotlib()
    np = _import_numpy()
    if plt is None or np is None:
        logger.error("matplotlib and numpy are required for plotting")
        return False

    scored_runs = [r for r in runs if metric in r["metrics"]]
    if len(scored_runs) < 2:
        logger.warning("Need at least 2 runs with metric '%s' for a plot", metric)
        return False

    # Collect numeric params present in all scored runs
    param_keys: List[str] = []
    for key in scored_runs[0]["params"]:
        try:
            vals = [float(r["params"][key]) for r in scored_runs if key in r["params"]]
            if len(vals) == len(scored_runs):
                param_keys.append(key)
        except (ValueError, TypeError):
            continue

    if not param_keys:
        logger.warning("No numeric params shared across all runs; cannot plot")
        return False

    axes_labels = param_keys + [metric]
    n_axes = len(axes_labels)

    # Build data matrix (runs x axes), normalised to [0, 1] per axis
    data = np.zeros((len(scored_runs), n_axes))
    for i, r in enumerate(scored_runs):
        for j, key in enumerate(param_keys):
            data[i, j] = float(r["params"][key])
        data[i, -1] = r["metrics"][metric]

    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    normed = (data - mins) / ranges

    # Colour by metric value
    metric_normed = normed[:, -1]

    fig, axes = plt.subplots(1, n_axes - 1, sharey=False,
                             figsize=(max(8, n_axes * 1.8), 5))
    if n_axes - 1 == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        for j in range(len(scored_runs)):
            ax.plot([0, 1], [normed[j, i], normed[j, i + 1]],
                    color=plt.cm.viridis(metric_normed[j]),
                    alpha=0.6, linewidth=1.2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([axes_labels[i], axes_labels[i + 1]],
                           fontsize=8, rotation=30, ha="right")
        ax.tick_params(axis="y", labelsize=7)

        # Show original scale on y-axis
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels([
            f"{mins[i]:.3g}", f"{(mins[i]+maxs[i])/2:.3g}", f"{maxs[i]:.3g}",
        ], fontsize=7)

    sm = plt.cm.ScalarMappable(cmap="viridis",
                               norm=plt.Normalize(mins[-1], maxs[-1]))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, pad=0.02, aspect=40)
    cbar.set_label(metric, fontsize=9)

    fig.suptitle("Parallel Coordinates: Hyperparameter Search", fontsize=11)
    fig.tight_layout(rect=[0, 0, 0.92, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved parallel coordinates plot to %s", output_path)
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare and analyse MLflow experiment runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiment_compare.py --experiment my-project --metric accuracy --top-n 10
  python experiment_compare.py --experiment my-project --metric f1_score --output-format markdown
  python experiment_compare.py --experiment my-project --metric accuracy --output-format json
  python experiment_compare.py --experiment my-project --metric accuracy --plot hparams.png
        """,
    )
    parser.add_argument("--experiment", required=True, help="MLflow experiment name")
    parser.add_argument("--tracking-uri", default=None,
                        help="MLflow tracking URI (default: local ./mlruns)")
    parser.add_argument("--metric", default="accuracy",
                        help="Primary metric to rank by (default: accuracy)")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Number of top runs to include (default: 10)")
    parser.add_argument("--output-format", default="table",
                        choices=["table", "json", "markdown"],
                        help="Report output format (default: table)")
    parser.add_argument("--plot", default=None, metavar="FILE.png",
                        help="Save parallel coordinates plot to this path")
    parser.add_argument("--status", default=None,
                        help="Filter runs by status (e.g. FINISHED, FAILED)")
    parser.add_argument("--start-date", default=None,
                        help="Filter runs starting on or after date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=None,
                        help="Filter runs starting on or before date (YYYY-MM-DD)")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    runs = fetch_runs(
        experiment_name=args.experiment,
        tracking_uri=args.tracking_uri,
        status_filter=args.status,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    if not runs:
        logger.error("No runs returned; exiting")
        sys.exit(1)

    stats = analyse_runs(runs, args.metric)

    if args.output_format == "table":
        print(format_table(runs, args.metric, args.top_n))
    elif args.output_format == "markdown":
        print(generate_markdown(runs, args.metric, args.top_n, stats))
    elif args.output_format == "json":
        print(generate_json_report(runs, args.metric, args.top_n, stats))

    if args.plot:
        parallel_coordinates_plot(runs, args.metric, args.plot)


if __name__ == "__main__":
    main()
