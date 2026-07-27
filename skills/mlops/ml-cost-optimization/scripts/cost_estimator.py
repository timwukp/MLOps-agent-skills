#!/usr/bin/env python3
"""
Pre-Sales ML Workload Cost Estimator

Quick monthly-cost sizing for customer conversations: GPU training clusters,
GPU inference endpoints, and LLM API usage. Every estimate ships with its
assumptions stated explicitly -- these are sizing numbers for scoping and
budgeting, not quotes.

GPU pricing reuses the GPU_CATALOG from cost_analyzer.py in this directory
(AWS us-east-1 on-demand, per GPU) so the two tools never disagree.

Usage:
    # Training: 4x A100 40GB, 200 GPU-cluster-hours per month
    python cost_estimator.py --workload training --gpu-type a100_40gb \
        --instances 4 --hours-per-month 200

    # Inference sized from traffic: 2M requests/day within a 50 ms budget
    python cost_estimator.py --workload inference --gpu-type l4 \
        --requests-per-day 2000000 --latency-budget-ms 50 --model-size-gb 8

    # Inference sized from always-on hours
    python cost_estimator.py --workload inference --gpu-type a10g \
        --instances 2 --hours-per-month 730

    # LLM API: 100k requests/day, ~1500 tokens in / 400 out per request
    python cost_estimator.py --workload llm-api --llm-model claude-sonnet-5 \
        --requests-per-day 100000 --tokens-in 1500 --tokens-out 400

    # JSON output for pipelines / proposals
    python cost_estimator.py --workload training --gpu-type h100 \
        --instances 8 --hours-per-month 400 --json

Dependencies:
    - Python 3.8+, stdlib only (imports GPU_CATALOG from cost_analyzer.py)
"""

import argparse
import json
import math
import os
import sys

# Import the shared GPU pricing table from the sibling cost_analyzer.py so
# both tools quote identical prices.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cost_analyzer import GPU_CATALOG  # noqa: E402

HOURS_PER_MONTH = 730  # average hours in a month (24 * 365 / 12)
DAYS_PER_MONTH = 30.4

# ---------------------------------------------------------------------------
# LLM API pricing ($ per million tokens). Point-in-time list prices --
# re-check the provider pricing page before quoting a real budget.
# ---------------------------------------------------------------------------

LLM_PRICING = {
    "claude-sonnet-5": {"input_per_mtok": 3.00, "output_per_mtok": 15.00},
    "claude-opus-5": {"input_per_mtok": 5.00, "output_per_mtok": 25.00},
    "claude-haiku-4-5": {"input_per_mtok": 1.00, "output_per_mtok": 5.00},
    "gpt-5": {"input_per_mtok": 1.25, "output_per_mtok": 10.00},
}

# Sizing assumptions for traffic-based inference estimates.
PEAK_TO_AVERAGE_RATIO = 3.0     # peak-hour traffic vs daily average
TARGET_UTILIZATION = 0.6        # keep instances at <= 60% at peak (headroom)


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

def estimate_training(args) -> dict:
    """Monthly cost of a training cluster: instances x hours x $/GPU-hr."""
    gpu = GPU_CATALOG[args.gpu_type]
    hours = args.hours_per_month
    assumptions = [
        f"AWS us-east-1 on-demand pricing, ${gpu['on_demand_price']:.3f}/GPU-hr "
        f"({gpu['instance']}); re-check current list prices before quoting.",
        f"{args.instances} GPU(s) busy {hours:.0f} hr/month each "
        f"(= {args.instances * hours:.0f} GPU-hours).",
        "Compute only: excludes storage, egress, and managed-platform markup "
        "(SageMaker adds ~20-40% over raw EC2).",
    ]

    on_demand = args.instances * hours * gpu["on_demand_price"]
    spot = args.instances * hours * gpu["spot_price"]
    assumptions.append(
        f"Spot alternative uses a recent-average rate (${gpu['spot_price']:.2f}/GPU-hr); "
        "spot is a live market and capacity for this GPU may be unavailable."
    )

    result = {
        "monthly_cost_usd": round(on_demand, 2),
        "monthly_cost_spot_usd": round(spot, 2),
        "gpu": gpu["name"],
        "gpu_hours_per_month": round(args.instances * hours, 1),
        "price_per_gpu_hour": gpu["on_demand_price"],
    }
    _check_vram(args, gpu, assumptions)
    return {"estimate": result, "assumptions": assumptions}


def estimate_inference(args) -> dict:
    """Monthly cost of a GPU inference fleet, sized by hours or by traffic."""
    gpu = GPU_CATALOG[args.gpu_type]
    assumptions = [
        f"AWS us-east-1 on-demand pricing, ${gpu['on_demand_price']:.3f}/GPU-hr "
        f"({gpu['instance']}); re-check current list prices before quoting.",
        "Compute only: excludes load balancer, storage, egress, and managed-endpoint markup.",
    ]

    if args.hours_per_month is not None:
        instances = args.instances or 1
        hours = args.hours_per_month
        assumptions.append(
            f"Fleet of {instances} instance(s), each running {hours:.0f} hr/month."
        )
    else:
        # Size the fleet from traffic + latency budget.
        # Throughput model: one request occupies the GPU for its full latency
        # budget (sequential, no batching) => 1000/latency req/s per instance.
        per_instance_rps = 1000.0 / args.latency_budget_ms
        avg_rps = args.requests_per_day / 86400.0
        peak_rps = avg_rps * PEAK_TO_AVERAGE_RATIO
        instances = max(1, math.ceil(peak_rps / (per_instance_rps * TARGET_UTILIZATION)))
        hours = HOURS_PER_MONTH
        assumptions.extend([
            f"Throughput model: 1 request occupies the GPU for the full "
            f"{args.latency_budget_ms:.0f} ms budget (no batching) => "
            f"{per_instance_rps:.1f} req/s per instance. Dynamic batching typically "
            "improves this 3-10x -- treat the estimate as conservative.",
            f"Average {avg_rps:.1f} req/s; sized for {PEAK_TO_AVERAGE_RATIO:.0f}x "
            f"peak-to-average at {TARGET_UTILIZATION:.0%} target utilization "
            f"=> {instances} instance(s).",
            f"Fleet runs 24/7 ({HOURS_PER_MONTH} hr/month); scale-to-zero or "
            "off-hours downscaling would reduce this.",
        ])

    monthly = instances * hours * gpu["on_demand_price"]
    result = {
        "monthly_cost_usd": round(monthly, 2),
        "gpu": gpu["name"],
        "instances": instances,
        "hours_per_month_per_instance": round(hours, 1),
        "price_per_gpu_hour": gpu["on_demand_price"],
    }
    if args.requests_per_day:
        monthly_requests = args.requests_per_day * DAYS_PER_MONTH
        result["cost_per_1k_requests_usd"] = round(monthly / (monthly_requests / 1000.0), 4)
    _check_vram(args, gpu, assumptions)
    return {"estimate": result, "assumptions": assumptions}


def estimate_llm_api(args) -> dict:
    """Monthly LLM API cost from request volume and per-request token counts."""
    pricing = LLM_PRICING[args.llm_model]
    monthly_requests = args.requests_per_day * DAYS_PER_MONTH
    mtok_in = monthly_requests * args.tokens_in / 1_000_000.0
    mtok_out = monthly_requests * args.tokens_out / 1_000_000.0
    input_cost = mtok_in * pricing["input_per_mtok"]
    output_cost = mtok_out * pricing["output_per_mtok"]
    total = input_cost + output_cost

    assumptions = [
        f"{args.llm_model} list price: ${pricing['input_per_mtok']:.2f}/MTok input, "
        f"${pricing['output_per_mtok']:.2f}/MTok output; re-check the provider "
        "pricing page before quoting.",
        f"{args.requests_per_day:,.0f} requests/day x {DAYS_PER_MONTH} days "
        f"= {monthly_requests:,.0f} requests/month.",
        f"Every request averages {args.tokens_in:,.0f} input + "
        f"{args.tokens_out:,.0f} output tokens.",
        "No prompt caching or batch-API discount applied; caching repeated "
        "system prompts and batch processing can cut input cost substantially.",
    ]

    result = {
        "monthly_cost_usd": round(total, 2),
        "input_cost_usd": round(input_cost, 2),
        "output_cost_usd": round(output_cost, 2),
        "monthly_requests": int(monthly_requests),
        "monthly_input_mtok": round(mtok_in, 2),
        "monthly_output_mtok": round(mtok_out, 2),
        "cost_per_1k_requests_usd": round(total / (monthly_requests / 1000.0), 4),
        "model": args.llm_model,
    }
    return {"estimate": result, "assumptions": assumptions}


def _check_vram(args, gpu: dict, assumptions: list) -> None:
    """Flag when the stated model size does not comfortably fit the GPU."""
    if args.model_size_gb is None:
        return
    # Rule of thumb: weights + KV cache / activations + runtime overhead
    # need roughly 1.5x the model file size in VRAM for inference.
    needed = args.model_size_gb * 1.5
    if needed > gpu["vram_gb"]:
        assumptions.append(
            f"WARNING: a {args.model_size_gb:.0f} GB model needs roughly "
            f"{needed:.0f} GB VRAM (1.5x rule of thumb) but {gpu['name']} has "
            f"{gpu['vram_gb']} GB -- quantize, shard across GPUs, or pick a larger GPU."
        )
    else:
        assumptions.append(
            f"Model size {args.model_size_gb:.0f} GB fits {gpu['name']} "
            f"({gpu['vram_gb']} GB VRAM) with the 1.5x headroom rule of thumb."
        )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_human(workload: str, payload: dict) -> str:
    est = payload["estimate"]
    lines = [
        "",
        "=" * 60,
        f"  MONTHLY COST ESTIMATE -- {workload.upper()} (pre-sales sizing)",
        "=" * 60,
        "",
    ]
    for key, value in est.items():
        label = key.replace("_", " ").capitalize()
        if isinstance(value, float) and "usd" in key:
            lines.append(f"  {label:<34} ${value:,.2f}")
        elif isinstance(value, int) and key == "monthly_requests":
            lines.append(f"  {label:<34} {value:,}")
        else:
            lines.append(f"  {label:<34} {value}")
    lines.extend(["", "-" * 60, "  ASSUMPTIONS", "-" * 60, ""])
    for i, a in enumerate(payload["assumptions"], 1):
        lines.append(f"  {i}. {a}")
    lines.extend(["", "=" * 60, ""])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-sales ML workload cost estimator: monthly cost + stated "
            "assumptions for training, inference, or LLM API workloads."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cost_estimator.py --workload training --gpu-type a100_40gb --instances 4 --hours-per-month 200
  python cost_estimator.py --workload inference --gpu-type l4 --requests-per-day 2000000 --latency-budget-ms 50
  python cost_estimator.py --workload llm-api --llm-model claude-sonnet-5 \\
      --requests-per-day 100000 --tokens-in 1500 --tokens-out 400
        """,
    )
    parser.add_argument(
        "--workload", required=True, choices=["training", "inference", "llm-api"],
        help="Workload type to size",
    )
    parser.add_argument(
        "--gpu-type", choices=list(GPU_CATALOG.keys()), default=None,
        help="GPU type (training/inference); prices shared with cost_analyzer.py",
    )
    parser.add_argument("--instances", type=int, default=None,
                        help="Number of GPUs/instances (training; optional for inference)")
    parser.add_argument("--hours-per-month", type=float, default=None,
                        help="Busy hours per month per instance")
    parser.add_argument("--requests-per-day", type=float, default=None,
                        help="Daily request volume (inference sizing, llm-api)")
    parser.add_argument("--latency-budget-ms", type=float, default=None,
                        help="Per-request latency budget in ms (inference sizing)")
    parser.add_argument("--model-size-gb", type=float, default=None,
                        help="Model artifact size in GB (VRAM fit check)")
    parser.add_argument(
        "--llm-model", choices=list(LLM_PRICING.keys()), default=None,
        help="LLM API model (llm-api workload)",
    )
    parser.add_argument("--tokens-in", type=float, default=None,
                        help="Average input tokens per request (llm-api)")
    parser.add_argument("--tokens-out", type=float, default=None,
                        help="Average output tokens per request (llm-api)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    return parser


def validate_args(parser: argparse.ArgumentParser, args) -> None:
    if args.workload == "training":
        if not args.gpu_type or not args.instances or args.hours_per_month is None:
            parser.error("training requires --gpu-type, --instances, and --hours-per-month")
    elif args.workload == "inference":
        if not args.gpu_type:
            parser.error("inference requires --gpu-type")
        by_hours = args.hours_per_month is not None
        by_traffic = args.requests_per_day is not None and args.latency_budget_ms is not None
        if not by_hours and not by_traffic:
            parser.error(
                "inference requires either --hours-per-month or "
                "--requests-per-day plus --latency-budget-ms"
            )
    elif args.workload == "llm-api":
        if not args.llm_model or args.requests_per_day is None \
                or args.tokens_in is None or args.tokens_out is None:
            parser.error(
                "llm-api requires --llm-model, --requests-per-day, "
                "--tokens-in, and --tokens-out"
            )


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)

    if args.workload == "training":
        payload = estimate_training(args)
    elif args.workload == "inference":
        payload = estimate_inference(args)
    else:
        payload = estimate_llm_api(args)

    payload["workload"] = args.workload
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(format_human(args.workload, payload))


if __name__ == "__main__":
    main()
