#!/usr/bin/env python3
"""
ML Cost Analyzer

Estimates training costs, analyzes experiment cost history, identifies optimization
opportunities, compares GPU instance costs, and generates cost reports.

Usage:
    python cost_analyzer.py estimate --model-params 350 --dataset-gb 50 --epochs 10 --gpu a100_40gb
    python cost_analyzer.py compare-gpus --model-params 350 --dataset-gb 50 --epochs 10
    python cost_analyzer.py analyze-history --history experiments.json
    python cost_analyzer.py optimize --history experiments.json
    python cost_analyzer.py report --history experiments.json --output report.json

Dependencies:
    - Python 3.8+
    - No external dependencies required (stdlib only)

Optional:
    - matplotlib (for chart generation in reports)
"""

import argparse
import json
import os
import sys
import math
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# GPU catalog
# ---------------------------------------------------------------------------

GPU_CATALOG = {
    "t4": {
        "name": "NVIDIA T4",
        "vram_gb": 16,
        "fp32_tflops": 8.1,
        "fp16_tflops": 65.0,
        "tensor_cores": True,
        "on_demand_price": 0.35,
        "spot_price": 0.12,
        "throughput_factor": 1.0,
        "best_for": "inference, small training, budget-friendly",
    },
    "v100_16gb": {
        "name": "NVIDIA V100 16GB",
        "vram_gb": 16,
        "fp32_tflops": 15.7,
        "fp16_tflops": 125.0,
        "tensor_cores": True,
        "on_demand_price": 1.50,
        "spot_price": 0.45,
        "throughput_factor": 2.5,
        "best_for": "general training, medium models",
    },
    "v100_32gb": {
        "name": "NVIDIA V100 32GB",
        "vram_gb": 32,
        "fp32_tflops": 15.7,
        "fp16_tflops": 125.0,
        "tensor_cores": True,
        "on_demand_price": 2.48,
        "spot_price": 0.74,
        "throughput_factor": 2.5,
        "best_for": "general training, large batch sizes",
    },
    "a10g": {
        "name": "NVIDIA A10G",
        "vram_gb": 24,
        "fp32_tflops": 31.2,
        "fp16_tflops": 62.5,
        "tensor_cores": True,
        "on_demand_price": 1.00,
        "spot_price": 0.35,
        "throughput_factor": 3.0,
        "best_for": "inference, fine-tuning, medium models",
    },
    "a100_40gb": {
        "name": "NVIDIA A100 40GB",
        "vram_gb": 40,
        "fp32_tflops": 19.5,
        "fp16_tflops": 312.0,
        "tensor_cores": True,
        "on_demand_price": 3.50,
        "spot_price": 1.10,
        "throughput_factor": 8.0,
        "best_for": "large model training, high throughput",
    },
    "a100_80gb": {
        "name": "NVIDIA A100 80GB",
        "vram_gb": 80,
        "fp32_tflops": 19.5,
        "fp16_tflops": 312.0,
        "tensor_cores": True,
        "on_demand_price": 4.00,
        "spot_price": 1.50,
        "throughput_factor": 9.0,
        "best_for": "very large models, multi-task training",
    },
    "h100": {
        "name": "NVIDIA H100",
        "vram_gb": 80,
        "fp32_tflops": 51.0,
        "fp16_tflops": 990.0,
        "tensor_cores": True,
        "on_demand_price": 6.00,
        "spot_price": 2.50,
        "throughput_factor": 15.0,
        "best_for": "LLM training, maximum throughput",
    },
    "l4": {
        "name": "NVIDIA L4",
        "vram_gb": 24,
        "fp32_tflops": 30.3,
        "fp16_tflops": 121.0,
        "tensor_cores": True,
        "on_demand_price": 0.50,
        "spot_price": 0.18,
        "throughput_factor": 2.8,
        "best_for": "cost-effective inference, fine-tuning",
    },
}

STORAGE_COSTS = {
    "s3_standard": 0.023,        # $/GB/month
    "s3_ia": 0.0125,             # $/GB/month  (Infrequent Access)
    "s3_glacier": 0.004,         # $/GB/month
    "gcs_standard": 0.020,       # $/GB/month
    "gcs_nearline": 0.010,       # $/GB/month
    "gcs_coldline": 0.004,       # $/GB/month
    "azure_hot": 0.018,          # $/GB/month
    "azure_cool": 0.010,         # $/GB/month
    "azure_archive": 0.002,      # $/GB/month
    "local_ssd": 0.08,           # $/GB/month (estimated)
    "local_hdd": 0.02,           # $/GB/month (estimated)
}

DATA_TRANSFER_COSTS = {
    "aws_egress_per_gb": 0.09,
    "gcp_egress_per_gb": 0.12,
    "azure_egress_per_gb": 0.087,
    "cross_region_per_gb": 0.02,
}


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

@dataclass
class TrainingCostEstimate:
    """Detailed training cost breakdown."""
    gpu_type: str
    gpu_name: str
    model_params_millions: float
    dataset_size_gb: float
    num_epochs: int
    num_gpus: int
    use_spot: bool
    mixed_precision: bool
    estimated_gpu_hours: float
    gpu_cost_per_hour: float
    compute_cost: float
    storage_cost: float
    data_transfer_cost: float
    total_cost: float
    cost_with_spot: float
    spot_savings: float
    mixed_precision_savings: float

    def to_dict(self):
        return asdict(self)


def estimate_gpu_hours(
    model_params_millions: float,
    dataset_size_gb: float,
    num_epochs: int,
    gpu_type: str,
    mixed_precision: bool = True,
) -> float:
    """
    Estimate GPU hours for a training run.

    The formula is a rough heuristic based on:
    - Larger models take longer (roughly linear with parameters)
    - Larger datasets take longer (roughly linear with data size)
    - More epochs take longer (linear)
    - Faster GPUs reduce time (throughput_factor)
    - Mixed precision provides ~1.5x speedup on average

    Base assumption: 100M parameters, 1 GB data, 1 epoch takes ~1 hour on a T4.
    """
    gpu = GPU_CATALOG[gpu_type]
    base_hours = (model_params_millions / 100.0) * dataset_size_gb * num_epochs
    adjusted_hours = base_hours / gpu["throughput_factor"]

    if mixed_precision and gpu["tensor_cores"]:
        adjusted_hours /= 1.5

    return max(adjusted_hours, 0.1)  # Minimum 0.1 hours


def estimate_training_cost(
    model_params_millions: float,
    dataset_size_gb: float,
    num_epochs: int,
    gpu_type: str = "a100_40gb",
    num_gpus: int = 1,
    use_spot: bool = False,
    mixed_precision: bool = True,
    storage_class: str = "s3_standard",
    data_transfer_gb: float = 0.0,
    num_hpo_trials: int = 0,
) -> TrainingCostEstimate:
    """Produce a detailed training cost estimate."""

    gpu = GPU_CATALOG[gpu_type]

    # GPU hours estimate
    hours = estimate_gpu_hours(
        model_params_millions, dataset_size_gb, num_epochs, gpu_type, mixed_precision
    )
    total_gpu_hours = hours * num_gpus

    # Compute cost
    price = gpu["spot_price"] if use_spot else gpu["on_demand_price"]
    compute_cost = total_gpu_hours * price

    # HPO additional cost (trials are typically shorter -- ~30% of full training)
    if num_hpo_trials > 0:
        hpo_hours = hours * 0.3 * num_hpo_trials * num_gpus
        compute_cost += hpo_hours * price
        total_gpu_hours += hpo_hours

    # Storage cost (estimate: 2x dataset size for artifacts, 1 month)
    artifact_gb = dataset_size_gb * 2.0
    storage_rate = STORAGE_COSTS.get(storage_class, 0.023)
    storage_cost = artifact_gb * storage_rate

    # Data transfer cost
    data_transfer_cost = data_transfer_gb * DATA_TRANSFER_COSTS.get("aws_egress_per_gb", 0.09)

    total = compute_cost + storage_cost + data_transfer_cost

    # Savings calculations
    on_demand_compute = total_gpu_hours * gpu["on_demand_price"]
    spot_compute = total_gpu_hours * gpu["spot_price"]
    spot_savings = on_demand_compute - spot_compute

    hours_without_mp = estimate_gpu_hours(
        model_params_millions, dataset_size_gb, num_epochs, gpu_type, mixed_precision=False
    ) * num_gpus
    mp_savings = (hours_without_mp - total_gpu_hours) * price

    return TrainingCostEstimate(
        gpu_type=gpu_type,
        gpu_name=gpu["name"],
        model_params_millions=model_params_millions,
        dataset_size_gb=dataset_size_gb,
        num_epochs=num_epochs,
        num_gpus=num_gpus,
        use_spot=use_spot,
        mixed_precision=mixed_precision,
        estimated_gpu_hours=round(total_gpu_hours, 2),
        gpu_cost_per_hour=price,
        compute_cost=round(compute_cost, 2),
        storage_cost=round(storage_cost, 2),
        data_transfer_cost=round(data_transfer_cost, 2),
        total_cost=round(total, 2),
        cost_with_spot=round(spot_compute + storage_cost + data_transfer_cost, 2),
        spot_savings=round(spot_savings, 2),
        mixed_precision_savings=round(mp_savings, 2),
    )


# ---------------------------------------------------------------------------
# GPU comparison
# ---------------------------------------------------------------------------

def compare_gpus(
    model_params_millions: float,
    dataset_size_gb: float,
    num_epochs: int,
    num_gpus: int = 1,
) -> List[Dict]:
    """Compare all GPUs for a given training workload, sorted by total cost."""
    results = []
    for gpu_key in GPU_CATALOG:
        est = estimate_training_cost(
            model_params_millions=model_params_millions,
            dataset_size_gb=dataset_size_gb,
            num_epochs=num_epochs,
            gpu_type=gpu_key,
            num_gpus=num_gpus,
            use_spot=False,
            mixed_precision=True,
        )
        results.append({
            "gpu": gpu_key,
            "name": est.gpu_name,
            "vram_gb": GPU_CATALOG[gpu_key]["vram_gb"],
            "estimated_hours": est.estimated_gpu_hours,
            "on_demand_cost": est.total_cost,
            "spot_cost": est.cost_with_spot,
            "best_for": GPU_CATALOG[gpu_key]["best_for"],
        })
    results.sort(key=lambda x: x["on_demand_cost"])
    return results


# ---------------------------------------------------------------------------
# Experiment history analysis
# ---------------------------------------------------------------------------

@dataclass
class ExperimentRecord:
    """Represents a single experiment's cost data."""
    experiment_id: str
    model_name: str
    gpu_type: str
    gpu_hours: float
    cost_usd: float
    dataset_size_gb: float
    num_epochs: int
    final_metric: float
    metric_name: str = "accuracy"
    timestamp: str = ""
    status: str = "completed"
    tags: Dict = field(default_factory=dict)


def load_experiment_history(path: str) -> List[ExperimentRecord]:
    """Load experiment history from a JSON file."""
    with open(path, "r") as f:
        data = json.load(f)

    records = []
    for item in data:
        records.append(ExperimentRecord(
            experiment_id=item.get("experiment_id", "unknown"),
            model_name=item.get("model_name", "unknown"),
            gpu_type=item.get("gpu_type", "unknown"),
            gpu_hours=float(item.get("gpu_hours", 0)),
            cost_usd=float(item.get("cost_usd", 0)),
            dataset_size_gb=float(item.get("dataset_size_gb", 0)),
            num_epochs=int(item.get("num_epochs", 0)),
            final_metric=float(item.get("final_metric", 0)),
            metric_name=item.get("metric_name", "accuracy"),
            timestamp=item.get("timestamp", ""),
            status=item.get("status", "completed"),
            tags=item.get("tags", {}),
        ))
    return records


def analyze_experiment_history(records: List[ExperimentRecord]) -> Dict:
    """Analyze experiment history for cost patterns and insights."""
    if not records:
        return {"error": "No experiment records provided"}

    total_cost = sum(r.cost_usd for r in records)
    total_gpu_hours = sum(r.gpu_hours for r in records)
    completed = [r for r in records if r.status == "completed"]
    failed = [r for r in records if r.status == "failed"]

    # Cost by GPU type
    cost_by_gpu = {}
    for r in records:
        cost_by_gpu.setdefault(r.gpu_type, {"cost": 0, "hours": 0, "count": 0})
        cost_by_gpu[r.gpu_type]["cost"] += r.cost_usd
        cost_by_gpu[r.gpu_type]["hours"] += r.gpu_hours
        cost_by_gpu[r.gpu_type]["count"] += 1

    # Cost by model
    cost_by_model = {}
    for r in records:
        cost_by_model.setdefault(r.model_name, {"cost": 0, "hours": 0, "count": 0})
        cost_by_model[r.model_name]["cost"] += r.cost_usd
        cost_by_model[r.model_name]["hours"] += r.gpu_hours
        cost_by_model[r.model_name]["count"] += 1

    # Cost efficiency (metric improvement per dollar)
    cost_efficiency = []
    for r in completed:
        if r.cost_usd > 0:
            cost_efficiency.append({
                "experiment_id": r.experiment_id,
                "model": r.model_name,
                "metric_per_dollar": round(r.final_metric / r.cost_usd, 4),
                "cost": round(r.cost_usd, 2),
                "metric": r.final_metric,
            })
    cost_efficiency.sort(key=lambda x: x["metric_per_dollar"], reverse=True)

    # Waste from failed experiments
    failed_cost = sum(r.cost_usd for r in failed)
    failed_hours = sum(r.gpu_hours for r in failed)

    # Cost trend (by timestamp if available)
    costs_sorted = sorted(
        [r for r in records if r.timestamp],
        key=lambda r: r.timestamp,
    )
    monthly_costs = {}
    for r in costs_sorted:
        month_key = r.timestamp[:7]  # "YYYY-MM"
        monthly_costs.setdefault(month_key, 0)
        monthly_costs[month_key] += r.cost_usd

    analysis = {
        "summary": {
            "total_experiments": len(records),
            "completed": len(completed),
            "failed": len(failed),
            "total_cost_usd": round(total_cost, 2),
            "total_gpu_hours": round(total_gpu_hours, 2),
            "avg_cost_per_experiment": round(total_cost / len(records), 2),
            "avg_gpu_hours_per_experiment": round(total_gpu_hours / len(records), 2),
            "wasted_cost_from_failures": round(failed_cost, 2),
            "wasted_hours_from_failures": round(failed_hours, 2),
        },
        "cost_by_gpu_type": {
            k: {key: round(val, 2) if isinstance(val, float) else val for key, val in v.items()}
            for k, v in cost_by_gpu.items()
        },
        "cost_by_model": {
            k: {key: round(val, 2) if isinstance(val, float) else val for key, val in v.items()}
            for k, v in cost_by_model.items()
        },
        "top_cost_efficient_experiments": cost_efficiency[:10],
        "monthly_costs": {k: round(v, 2) for k, v in monthly_costs.items()},
    }

    return analysis


# ---------------------------------------------------------------------------
# Optimization recommendations
# ---------------------------------------------------------------------------

def identify_optimizations(records: List[ExperimentRecord]) -> List[Dict]:
    """Identify cost optimization opportunities from experiment history."""
    recommendations = []

    if not records:
        return [{"type": "info", "message": "No experiment records to analyze."}]

    completed = [r for r in records if r.status == "completed"]
    failed = [r for r in records if r.status == "failed"]

    # 1. High failure rate
    if len(records) > 5:
        failure_rate = len(failed) / len(records)
        if failure_rate > 0.2:
            wasted = sum(r.cost_usd for r in failed)
            recommendations.append({
                "type": "failure_reduction",
                "priority": "HIGH",
                "message": (
                    f"Failure rate is {failure_rate:.0%} ({len(failed)}/{len(records)} experiments). "
                    f"${wasted:.2f} wasted on failed runs."
                ),
                "suggestion": (
                    "Implement pre-flight checks (data validation, config validation, small "
                    "smoke-test runs) before launching full training."
                ),
                "estimated_savings_pct": round(failure_rate * 100, 1),
            })

    # 2. GPU right-sizing opportunity
    cost_by_gpu = {}
    for r in records:
        cost_by_gpu.setdefault(r.gpu_type, [])
        cost_by_gpu[r.gpu_type].append(r)

    for gpu_type, gpu_records in cost_by_gpu.items():
        if gpu_type in GPU_CATALOG:
            gpu_info = GPU_CATALOG[gpu_type]
            # If average dataset is small but using expensive GPU
            avg_data = statistics.mean(r.dataset_size_gb for r in gpu_records)
            avg_params = 0  # We don't have params in history, so check by data size
            if avg_data < 5 and gpu_info["on_demand_price"] > 3.0:
                recommendations.append({
                    "type": "gpu_right_sizing",
                    "priority": "HIGH",
                    "message": (
                        f"Using {gpu_info['name']} (${gpu_info['on_demand_price']}/hr) for "
                        f"small datasets (avg {avg_data:.1f} GB). Consider a smaller GPU."
                    ),
                    "suggestion": (
                        f"Try {GPU_CATALOG['a10g']['name']} (${GPU_CATALOG['a10g']['on_demand_price']}/hr) "
                        f"or {GPU_CATALOG['t4']['name']} (${GPU_CATALOG['t4']['on_demand_price']}/hr) "
                        f"for small workloads."
                    ),
                    "estimated_savings_pct": 50,
                })

    # 3. Spot instance recommendation
    all_on_demand = all(
        r.tags.get("instance_type", "on_demand") == "on_demand" for r in records
    )
    if all_on_demand and len(records) > 3:
        total_cost = sum(r.cost_usd for r in records)
        recommendations.append({
            "type": "spot_instances",
            "priority": "MEDIUM",
            "message": (
                f"All {len(records)} experiments used on-demand instances. "
                f"Spot instances offer 60-70% savings."
            ),
            "suggestion": (
                "Use spot/preemptible instances for training jobs with robust checkpointing. "
                "Implement fault-tolerant training with periodic checkpoint saves."
            ),
            "estimated_savings_pct": 65,
        })

    # 4. Long-running experiments
    long_runs = [r for r in records if r.gpu_hours > 24]
    if long_runs:
        recommendations.append({
            "type": "long_running_optimization",
            "priority": "MEDIUM",
            "message": (
                f"{len(long_runs)} experiments ran for more than 24 GPU-hours. "
                f"Total cost: ${sum(r.cost_usd for r in long_runs):.2f}."
            ),
            "suggestion": (
                "For long training runs: (1) Enable mixed precision training for 1.5x speedup, "
                "(2) Use learning rate schedulers for faster convergence, "
                "(3) Implement early stopping to avoid wasted epochs."
            ),
            "estimated_savings_pct": 30,
        })

    # 5. Experiment deduplication
    model_metric_pairs = {}
    for r in completed:
        key = f"{r.model_name}_{r.gpu_type}_{r.num_epochs}"
        model_metric_pairs.setdefault(key, [])
        model_metric_pairs[key].append(r)

    duplicate_groups = {k: v for k, v in model_metric_pairs.items() if len(v) > 2}
    if duplicate_groups:
        dup_cost = sum(
            sum(r.cost_usd for r in group[2:])
            for group in duplicate_groups.values()
        )
        recommendations.append({
            "type": "experiment_deduplication",
            "priority": "LOW",
            "message": (
                f"Found {len(duplicate_groups)} groups of highly similar experiments. "
                f"Estimated redundant cost: ${dup_cost:.2f}."
            ),
            "suggestion": (
                "Use experiment tracking to check for existing results before launching new runs. "
                "Implement a configuration deduplication check."
            ),
            "estimated_savings_pct": 15,
        })

    # Sort by priority
    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    recommendations.sort(key=lambda x: priority_order.get(x.get("priority", "LOW"), 3))

    return recommendations


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_cost_report(
    records: List[ExperimentRecord],
    output_path: Optional[str] = None,
) -> Dict:
    """Generate a comprehensive cost report."""
    analysis = analyze_experiment_history(records)
    optimizations = identify_optimizations(records)

    # Calculate potential savings
    total_cost = analysis["summary"]["total_cost_usd"]
    potential_savings = 0
    for opt in optimizations:
        savings_pct = opt.get("estimated_savings_pct", 0)
        # Apply savings to the portion of cost that the recommendation addresses
        # (rough: assume each optimization addresses ~30% of total cost)
        potential_savings += total_cost * 0.3 * (savings_pct / 100)

    report = {
        "report_generated_at": datetime.now().isoformat(),
        "analysis": analysis,
        "optimizations": optimizations,
        "potential_monthly_savings_usd": round(potential_savings, 2),
        "optimization_summary": {
            "total_recommendations": len(optimizations),
            "high_priority": sum(1 for o in optimizations if o.get("priority") == "HIGH"),
            "medium_priority": sum(1 for o in optimizations if o.get("priority") == "MEDIUM"),
            "low_priority": sum(1 for o in optimizations if o.get("priority") == "LOW"),
        },
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {output_path}")

    return report


# ---------------------------------------------------------------------------
# Demo / sample data generation
# ---------------------------------------------------------------------------

def generate_sample_history(num_experiments: int = 20) -> List[Dict]:
    """Generate sample experiment history for demonstration purposes."""
    import random
    random.seed(42)

    models = ["bert-base", "resnet50", "gpt2-medium", "efficientnet-b4", "distilbert"]
    gpus = ["t4", "v100_16gb", "a100_40gb", "a100_80gb", "h100"]
    statuses = ["completed"] * 8 + ["failed"] * 2  # 20% failure rate

    experiments = []
    base_date = datetime(2025, 1, 1)

    for i in range(num_experiments):
        model = random.choice(models)
        gpu = random.choice(gpus)
        gpu_info = GPU_CATALOG[gpu]
        status = random.choice(statuses)

        dataset_gb = round(random.uniform(1, 100), 1)
        epochs = random.randint(3, 50)
        gpu_hours = round(random.uniform(0.5, 48), 2)
        cost = round(gpu_hours * gpu_info["on_demand_price"], 2)

        metric = round(random.uniform(0.6, 0.98), 4) if status == "completed" else 0

        timestamp = (base_date + timedelta(days=random.randint(0, 365))).isoformat()

        experiments.append({
            "experiment_id": f"exp-{i+1:04d}",
            "model_name": model,
            "gpu_type": gpu,
            "gpu_hours": gpu_hours,
            "cost_usd": cost,
            "dataset_size_gb": dataset_gb,
            "num_epochs": epochs,
            "final_metric": metric,
            "metric_name": "accuracy",
            "timestamp": timestamp,
            "status": status,
            "tags": {"instance_type": "on_demand"},
        })

    return experiments


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_estimate(est: TrainingCostEstimate) -> str:
    """Format a cost estimate for terminal output."""
    lines = [
        "",
        "=" * 60,
        "  ML TRAINING COST ESTIMATE",
        "=" * 60,
        "",
        f"  Model parameters:     {est.model_params_millions}M",
        f"  Dataset size:         {est.dataset_size_gb} GB",
        f"  Epochs:               {est.num_epochs}",
        f"  GPU:                  {est.gpu_name} ({est.gpu_type})",
        f"  Number of GPUs:       {est.num_gpus}",
        f"  Mixed precision:      {'Yes' if est.mixed_precision else 'No'}",
        f"  Spot instances:       {'Yes' if est.use_spot else 'No'}",
        "",
        "-" * 60,
        "  COST BREAKDOWN",
        "-" * 60,
        "",
        f"  Estimated GPU hours:  {est.estimated_gpu_hours}",
        f"  GPU cost ($/hr):      ${est.gpu_cost_per_hour:.2f}",
        f"  Compute cost:         ${est.compute_cost:.2f}",
        f"  Storage cost:         ${est.storage_cost:.2f}",
        f"  Data transfer cost:   ${est.data_transfer_cost:.2f}",
        "",
        f"  TOTAL COST:           ${est.total_cost:.2f}",
        "",
        "-" * 60,
        "  SAVINGS OPPORTUNITIES",
        "-" * 60,
        "",
        f"  With spot instances:  ${est.cost_with_spot:.2f}  (save ${est.spot_savings:.2f})",
        f"  Mixed precision gain: ${est.mixed_precision_savings:.2f} saved",
        "",
        "=" * 60,
    ]
    return "\n".join(lines)


def format_gpu_comparison(results: List[Dict]) -> str:
    """Format GPU comparison as a table."""
    lines = [
        "",
        "=" * 90,
        "  GPU COST COMPARISON FOR WORKLOAD",
        "=" * 90,
        "",
        f"  {'GPU':<20} {'VRAM':>6} {'Hours':>8} {'On-Demand':>12} {'Spot':>10} {'Best For'}",
        f"  {'-'*18:<20} {'----':>6} {'-----':>8} {'---------':>12} {'----':>10} {'--------'}",
    ]
    for r in results:
        lines.append(
            f"  {r['name']:<20} {r['vram_gb']:>4}GB {r['estimated_hours']:>7.1f}h "
            f"${r['on_demand_cost']:>10.2f} ${r['spot_cost']:>8.2f}  {r['best_for']}"
        )
    lines.extend(["", "=" * 90, ""])
    return "\n".join(lines)


def format_analysis(analysis: Dict) -> str:
    """Format experiment history analysis."""
    s = analysis["summary"]
    lines = [
        "",
        "=" * 60,
        "  EXPERIMENT COST ANALYSIS",
        "=" * 60,
        "",
        f"  Total experiments:           {s['total_experiments']}",
        f"  Completed:                   {s['completed']}",
        f"  Failed:                      {s['failed']}",
        f"  Total cost:                  ${s['total_cost_usd']:.2f}",
        f"  Total GPU hours:             {s['total_gpu_hours']:.1f}",
        f"  Avg cost per experiment:     ${s['avg_cost_per_experiment']:.2f}",
        f"  Avg GPU hours per experiment:{s['avg_gpu_hours_per_experiment']:.1f}",
        f"  Wasted on failures:          ${s['wasted_cost_from_failures']:.2f}",
        "",
        "-" * 60,
        "  COST BY GPU TYPE",
        "-" * 60,
    ]
    for gpu, info in analysis["cost_by_gpu_type"].items():
        lines.append(f"  {gpu:<20} ${info['cost']:>8.2f}  ({info['hours']:.1f} hrs, {info['count']} runs)")

    lines.extend(["", "-" * 60, "  COST BY MODEL", "-" * 60])
    for model, info in analysis["cost_by_model"].items():
        lines.append(f"  {model:<20} ${info['cost']:>8.2f}  ({info['hours']:.1f} hrs, {info['count']} runs)")

    if analysis.get("top_cost_efficient_experiments"):
        lines.extend(["", "-" * 60, "  TOP COST-EFFICIENT EXPERIMENTS", "-" * 60])
        for exp in analysis["top_cost_efficient_experiments"][:5]:
            lines.append(
                f"  {exp['experiment_id']:<15} {exp['model']:<15} "
                f"metric/$ = {exp['metric_per_dollar']:.4f}  (cost: ${exp['cost']:.2f})"
            )

    lines.extend(["", "=" * 60, ""])
    return "\n".join(lines)


def format_optimizations(optimizations: List[Dict]) -> str:
    """Format optimization recommendations."""
    if not optimizations:
        return "\n  No optimization recommendations at this time.\n"

    lines = [
        "",
        "=" * 60,
        "  COST OPTIMIZATION RECOMMENDATIONS",
        "=" * 60,
    ]
    for i, opt in enumerate(optimizations, 1):
        priority = opt.get("priority", "INFO")
        lines.extend([
            "",
            f"  [{priority}] Recommendation #{i}: {opt.get('type', 'general')}",
            f"  {'-' * 50}",
            f"  Issue:      {opt.get('message', '')}",
            f"  Suggestion: {opt.get('suggestion', '')}",
            f"  Est. savings: ~{opt.get('estimated_savings_pct', 0)}%",
        ])
    lines.extend(["", "=" * 60, ""])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ML Cost Analyzer: estimate, compare, and optimize ML training costs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Estimate training cost
  python cost_analyzer.py estimate --model-params 350 --dataset-gb 50 --epochs 10 --gpu a100_40gb

  # Compare all GPUs for a workload
  python cost_analyzer.py compare-gpus --model-params 350 --dataset-gb 50 --epochs 10

  # Analyze experiment history
  python cost_analyzer.py analyze-history --history experiments.json

  # Get optimization recommendations
  python cost_analyzer.py optimize --history experiments.json

  # Generate comprehensive cost report
  python cost_analyzer.py report --history experiments.json --output report.json

  # Generate sample experiment data for testing
  python cost_analyzer.py generate-sample --output sample_experiments.json --count 30
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Estimate command
    est = subparsers.add_parser("estimate", help="Estimate training cost")
    est.add_argument("--model-params", type=float, required=True, help="Model parameters in millions")
    est.add_argument("--dataset-gb", type=float, required=True, help="Dataset size in GB")
    est.add_argument("--epochs", type=int, required=True, help="Number of training epochs")
    est.add_argument("--gpu", type=str, default="a100_40gb", choices=list(GPU_CATALOG.keys()),
                     help="GPU type (default: a100_40gb)")
    est.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs")
    est.add_argument("--spot", action="store_true", help="Use spot instance pricing")
    est.add_argument("--no-mixed-precision", action="store_true", help="Disable mixed precision")
    est.add_argument("--hpo-trials", type=int, default=0, help="Number of HPO trials")
    est.add_argument("--json", action="store_true", help="Output as JSON")

    # Compare GPUs command
    cmp = subparsers.add_parser("compare-gpus", help="Compare GPU costs for a workload")
    cmp.add_argument("--model-params", type=float, required=True, help="Model parameters in millions")
    cmp.add_argument("--dataset-gb", type=float, required=True, help="Dataset size in GB")
    cmp.add_argument("--epochs", type=int, required=True, help="Number of training epochs")
    cmp.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs")
    cmp.add_argument("--json", action="store_true", help="Output as JSON")

    # Analyze history command
    hist = subparsers.add_parser("analyze-history", help="Analyze experiment cost history")
    hist.add_argument("--history", type=str, required=True, help="Path to experiments JSON file")
    hist.add_argument("--json", action="store_true", help="Output as JSON")

    # Optimize command
    opt = subparsers.add_parser("optimize", help="Identify cost optimization opportunities")
    opt.add_argument("--history", type=str, required=True, help="Path to experiments JSON file")
    opt.add_argument("--json", action="store_true", help="Output as JSON")

    # Report command
    rep = subparsers.add_parser("report", help="Generate comprehensive cost report")
    rep.add_argument("--history", type=str, required=True, help="Path to experiments JSON file")
    rep.add_argument("--output", type=str, help="Output file path for the report (JSON)")
    rep.add_argument("--json", action="store_true", help="Output as JSON to stdout")

    # Generate sample data
    gen = subparsers.add_parser("generate-sample", help="Generate sample experiment history")
    gen.add_argument("--output", type=str, default="sample_experiments.json",
                     help="Output file path")
    gen.add_argument("--count", type=int, default=20, help="Number of experiments to generate")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "estimate":
        est = estimate_training_cost(
            model_params_millions=args.model_params,
            dataset_size_gb=args.dataset_gb,
            num_epochs=args.epochs,
            gpu_type=args.gpu,
            num_gpus=args.num_gpus,
            use_spot=args.spot,
            mixed_precision=not args.no_mixed_precision,
            num_hpo_trials=args.hpo_trials,
        )
        if args.json:
            print(json.dumps(est.to_dict(), indent=2))
        else:
            print(format_estimate(est))

    elif args.command == "compare-gpus":
        results = compare_gpus(
            model_params_millions=args.model_params,
            dataset_size_gb=args.dataset_gb,
            num_epochs=args.epochs,
            num_gpus=args.num_gpus,
        )
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print(format_gpu_comparison(results))

    elif args.command == "analyze-history":
        records = load_experiment_history(args.history)
        analysis = analyze_experiment_history(records)
        if args.json:
            print(json.dumps(analysis, indent=2))
        else:
            print(format_analysis(analysis))

    elif args.command == "optimize":
        records = load_experiment_history(args.history)
        optimizations = identify_optimizations(records)
        if args.json:
            print(json.dumps(optimizations, indent=2))
        else:
            print(format_optimizations(optimizations))

    elif args.command == "report":
        records = load_experiment_history(args.history)
        report = generate_cost_report(records, output_path=args.output)
        if args.json or not args.output:
            print(json.dumps(report, indent=2))

    elif args.command == "generate-sample":
        data = generate_sample_history(args.count)
        with open(args.output, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Generated {args.count} sample experiments -> {args.output}")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
