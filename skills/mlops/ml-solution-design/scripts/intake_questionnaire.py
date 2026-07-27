#!/usr/bin/env python3
"""
ML Engagement Requirements Intake Questionnaire

Captures customer requirements for an ML/LLM engagement either interactively
or from command-line flags, validates completeness, and writes requirements.json
consumable by architecture_recommender.py. Every unanswered question is reported
as a gap warning (candidate risk-register entry).

Usage:
    # Fully interactive session
    python intake_questionnaire.py --interactive --output ./requirements.json

    # Args-driven (CI-friendly, no prompts)
    python intake_questionnaire.py \
        --business-goal "Reduce churn by 5% via targeted retention offers" \
        --data-type tabular --data-volume-gb 50 --labels-available yes \
        --latency-slo-ms 0 --predictions-per-day 100000 \
        --monthly-budget-usd 2000 --team-k8s-experience no \
        --pii yes --data-residency none --cloud aws \
        --output ./requirements.json

    # Partial answers: unanswered fields become gap warnings
    python intake_questionnaire.py --business-goal "Doc summarization" \
        --data-type text --output ./requirements.json

Dependencies:
    - Python 3.8+ standard library only
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Question schema: (field, prompt, type, choices, gap_severity, gap_message)
# ---------------------------------------------------------------------------

QUESTIONS = [
    {
        "field": "business_goal",
        "prompt": "What business decision does the model drive, and what KPI improves?",
        "type": str,
        "choices": None,
        "severity": "high",
        "gap": "No business goal/KPI defined -- project has no success measure.",
    },
    {
        "field": "kpi_owner",
        "prompt": "Who owns the business KPI (name/role)?",
        "type": str,
        "choices": None,
        "severity": "medium",
        "gap": "No KPI owner identified -- acceptance sign-off will stall.",
    },
    {
        "field": "data_type",
        "prompt": "Primary data type?",
        "type": str,
        "choices": ["tabular", "text", "image", "timeseries", "mixed"],
        "severity": "high",
        "gap": "Data type unknown -- cannot choose model family.",
    },
    {
        "field": "data_volume_gb",
        "prompt": "Approximate data volume in GB?",
        "type": float,
        "choices": None,
        "severity": "medium",
        "gap": "Data volume unknown -- cannot size storage/compute.",
    },
    {
        "field": "labels_available",
        "prompt": "Are labeled examples available?",
        "type": str,
        "choices": ["yes", "no", "partial"],
        "severity": "high",
        "gap": "Label availability unknown -- supervised approach may be infeasible.",
    },
    {
        "field": "data_access_confirmed",
        "prompt": "Is data access already granted (not promised)?",
        "type": str,
        "choices": ["yes", "no"],
        "severity": "high",
        "gap": "Data access not confirmed -- the #1 cause of engagement delay.",
    },
    {
        "field": "latency_slo_ms",
        "prompt": "Per-prediction latency SLO in milliseconds (0 = batch acceptable)?",
        "type": float,
        "choices": None,
        "severity": "high",
        "gap": "No latency SLO -- cannot decide batch vs real-time; 'real-time' must be a number.",
    },
    {
        "field": "predictions_per_day",
        "prompt": "Expected predictions per day (peak day)?",
        "type": float,
        "choices": None,
        "severity": "medium",
        "gap": "Throughput unknown -- cannot size serving tier or estimate cost.",
    },
    {
        "field": "monthly_budget_usd",
        "prompt": "Monthly infrastructure budget in USD?",
        "type": float,
        "choices": None,
        "severity": "medium",
        "gap": "Budget undefined -- architecture options cannot be filtered by cost.",
    },
    {
        "field": "team_k8s_experience",
        "prompt": "Does the operating team have Kubernetes experience?",
        "type": str,
        "choices": ["yes", "no"],
        "severity": "high",
        "gap": "Team skills unknown -- self-hosted stack may be un-operable after handover.",
    },
    {
        "field": "team_oncall_exists",
        "prompt": "Does an on-call rotation exist for production services?",
        "type": str,
        "choices": ["yes", "no"],
        "severity": "medium",
        "gap": "On-call status unknown -- real-time serving needs an owner at 3am.",
    },
    {
        "field": "pii",
        "prompt": "Does the data contain PII/PHI?",
        "type": str,
        "choices": ["yes", "no", "unknown"],
        "severity": "high",
        "gap": "PII status unknown -- compliance requirements cannot be assessed.",
    },
    {
        "field": "data_residency",
        "prompt": "Data residency constraint?",
        "type": str,
        "choices": ["none", "region", "onprem"],
        "severity": "high",
        "gap": "Residency constraint unknown -- may invalidate cloud-managed options.",
    },
    {
        "field": "explainability_required",
        "prompt": "Is model explainability mandated (regulatory/audit)?",
        "type": str,
        "choices": ["yes", "no", "unknown"],
        "severity": "medium",
        "gap": "Explainability requirement unknown -- may rule out black-box/LLM approaches.",
    },
    {
        "field": "cloud",
        "prompt": "Primary cloud provider?",
        "type": str,
        "choices": ["aws", "gcp", "azure", "onprem", "none"],
        "severity": "medium",
        "gap": "Cloud provider unknown -- managed-service options cannot be selected.",
    },
    {
        "field": "deadline",
        "prompt": "Hard deadline (YYYY-MM-DD or 'none')?",
        "type": str,
        "choices": None,
        "severity": "low",
        "gap": "No deadline captured -- scoping and phasing cannot be planned.",
    },
    {
        "field": "event_driven",
        "prompt": "Must predictions fire on events without a request (streaming)?",
        "type": str,
        "choices": ["yes", "no"],
        "severity": "medium",
        "gap": "Event-driven requirement unknown -- streaming vs request/response undecided.",
    },
]


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def flag_name(field):
    """Convert a field name to its CLI flag, e.g. latency_slo_ms -> --latency-slo-ms."""
    return "--" + field.replace("_", "-")


def collect_from_args(args):
    """Read answers already supplied on the command line."""
    answers = {}
    for q in QUESTIONS:
        value = getattr(args, q["field"], None)
        if value is not None:
            answers[q["field"]] = value
    return answers


def collect_interactive(answers):
    """Prompt for any question not already answered. Blank input = skip."""
    print("\nML Solution Intake -- press Enter to skip a question (skips become gap warnings)\n")
    for q in QUESTIONS:
        if q["field"] in answers:
            continue
        prompt = q["prompt"]
        if q["choices"]:
            prompt += " [" + "/".join(q["choices"]) + "]"
        while True:
            raw = input(prompt + ": ").strip()
            if not raw:
                break  # skipped
            if q["choices"] and raw.lower() not in q["choices"]:
                print("  Please answer one of: " + ", ".join(q["choices"]))
                continue
            try:
                answers[q["field"]] = q["type"](raw.lower() if q["choices"] else raw)
            except ValueError:
                print("  Expected a %s value, got '%s'" % (q["type"].__name__, raw))
                continue
            break
    return answers


def find_gaps(answers):
    """Return gap warnings for every unanswered question."""
    gaps = []
    for q in QUESTIONS:
        if q["field"] not in answers:
            gaps.append({
                "field": q["field"],
                "severity": q["severity"],
                "warning": q["gap"],
            })
    return gaps


def derive_signals(answers):
    """Add derived convenience signals used by downstream tooling."""
    signals = {}
    slo = answers.get("latency_slo_ms")
    if slo is not None:
        if answers.get("event_driven") == "yes":
            signals["inference_mode"] = "streaming"
        elif slo == 0:
            signals["inference_mode"] = "batch"
        else:
            signals["inference_mode"] = "realtime"
    ppd = answers.get("predictions_per_day")
    if ppd is not None:
        signals["avg_predictions_per_second"] = round(ppd / 86400.0, 3)
    return signals


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="Capture ML engagement requirements; outputs requirements.json + gap warnings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Each intake question maps to a flag (see below). Unanswered questions "
               "become gap warnings in the output.",
    )
    parser.add_argument("--interactive", action="store_true",
                        help="Prompt for any question not supplied via flags")
    parser.add_argument("--output", default="./requirements.json",
                        help="Path for the requirements JSON (default: ./requirements.json)")
    for q in QUESTIONS:
        kwargs = {"help": q["prompt"], "default": None}
        if q["choices"]:
            kwargs["choices"] = q["choices"]
        elif q["type"] is float:
            kwargs["type"] = float
        parser.add_argument(flag_name(q["field"]), dest=q["field"], **kwargs)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    answers = collect_from_args(args)
    if args.interactive:
        try:
            answers = collect_interactive(answers)
        except (KeyboardInterrupt, EOFError):
            print()
            logger.warning("Interactive session interrupted; saving partial answers")

    if not answers:
        logger.error("No answers provided. Use --interactive or supply flags (see --help).")
        sys.exit(1)

    gaps = find_gaps(answers)
    doc = {
        "schema_version": "1.0",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "answers": answers,
        "derived": derive_signals(answers),
        "gaps": gaps,
    }

    with open(args.output, "w") as f:
        json.dump(doc, f, indent=2)
    logger.info("Requirements written to %s (%d/%d questions answered)",
                args.output, len(answers), len(QUESTIONS))

    if gaps:
        high = [g for g in gaps if g["severity"] == "high"]
        logger.warning("%d gap(s) found (%d high severity) -- add these to the risk register:",
                       len(gaps), len(high))
        for g in gaps:
            logger.warning("  [%s] %s: %s", g["severity"].upper(), g["field"], g["warning"])
    else:
        logger.info("Intake complete: no gaps.")


if __name__ == "__main__":
    main()
