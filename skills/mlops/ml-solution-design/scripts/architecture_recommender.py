#!/usr/bin/env python3
"""
Rule-Based ML Architecture Recommender

Reads a requirements.json produced by intake_questionnaire.py and emits a
recommended architecture: inference mode, hosting model, model family,
a stage-to-skill mapping for this repository, and a rough monthly cost band.

All rules are deterministic and documented in the RULES section below --
no LLM calls, no network access. Treat cost bands as +/-50% order-of-magnitude
estimates to be validated against the cloud pricing calculator.

Usage:
    # Recommend from an intake file
    python architecture_recommender.py --requirements ./requirements.json

    # Write the recommendation to a file
    python architecture_recommender.py --requirements ./requirements.json \
        --output ./recommendation.json

Dependencies:
    - Python 3.8+ standard library only

RULES (summary; each rule is annotated inline):
  R1  event_driven=yes                          -> streaming inference
  R2  latency_slo_ms == 0 or missing            -> batch inference
  R3  latency_slo_ms > 0                        -> real-time inference
  R4  data_residency=onprem                     -> self-hosted (only option)
  R5  team_k8s_experience=no                    -> managed (never hand over K8s to a non-K8s team)
  R6  cloud=aws and not R4                      -> managed AWS by default
  R7  k8s=yes and budget >= platform baseline   -> self-hosted viable; still managed unless residency/portability demands it
  R8  data_type in {text, mixed}                -> LLM or hybrid path
  R9  data_type in {tabular, timeseries}        -> classical ML path
  R10 labels_available=no and data_type=tabular -> flag: supervised infeasible, scope discovery phase
  R11 explainability_required=yes               -> prefer classical (tree + SHAP); flag LLM paths
  R12 cost band = base(inference mode, hosting) + GPU/LLM adders (documented in COST_BANDS)
"""

import argparse
import json
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cost model (USD/month, us-east-1, 2026; order-of-magnitude, +/-50%)
# ---------------------------------------------------------------------------

COST_BANDS = {
    # (inference_mode, hosting) -> (low, high, note)
    ("batch", "managed"): (150, 800, "SageMaker Processing + Batch Transform, pay-per-run"),
    ("batch", "self-hosted"): (1500, 4000, "K8s cluster baseline; batch jobs marginal on top"),
    ("realtime", "managed"): (500, 1500, "2x ml.m5.xlarge endpoint + Model Monitor"),
    ("realtime", "self-hosted"): (1800, 5000, "K8s baseline + serving replicas + observability"),
    ("streaming", "managed"): (800, 2500, "Kinesis + always-on consumers + endpoint"),
    ("streaming", "self-hosted"): (2500, 6000, "Kafka + stream processors + K8s baseline"),
}
LLM_MANAGED_ADDER = (100, 500, "Bedrock tokens at moderate volume (10M in / 2M out per month)")
LLM_SELFHOSTED_ADDER = (750, 4000, "1 GPU node g5.xlarge-g5.12xlarge for vLLM")


# ---------------------------------------------------------------------------
# Stage -> repo skill maps
# ---------------------------------------------------------------------------

CLASSICAL_SKILL_MAP = [
    ("ingest", "data-ingestion"),
    ("validate", "data-validation"),
    ("features", "feature-engineering"),
    ("train", "model-training"),
    ("track", "ml-experiment-tracking"),
    ("test", "ml-testing"),
    ("register", "model-registry"),
    ("serve", "model-serving"),
    ("monitor", "model-monitoring"),
    ("drift", "model-drift-detection"),
]

LLM_SKILL_MAP = [
    ("data prep", "llm-data-preparation"),
    ("grounding", "llm-rag"),
    ("adaptation (optional)", "llm-fine-tuning"),
    ("evaluation", "llm-evaluation"),
    ("deployment", "llm-deployment"),
    ("guardrails", "llm-guardrails"),
    ("observability", "llm-observability"),
    ("cost", "llm-cost-optimization"),
]


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

def decide_inference_mode(answers):
    """Rules R1-R3."""
    if answers.get("event_driven") == "yes":
        return "streaming", "R1: predictions must fire on events without a request"
    slo = answers.get("latency_slo_ms")
    if slo is None or float(slo) == 0:
        return "batch", "R2: no per-prediction latency SLO -- batch is cheapest and simplest"
    return "realtime", "R3: latency SLO of %.0f ms requires request/response serving" % float(slo)


def decide_hosting(answers):
    """Rules R4-R7."""
    if answers.get("data_residency") == "onprem":
        return "self-hosted", "R4: on-prem residency constraint -- managed cloud is not an option"
    if answers.get("team_k8s_experience") == "no":
        return "managed", "R5: operating team has no K8s experience -- do not hand over a platform"
    budget = answers.get("monthly_budget_usd")
    if budget is not None and float(budget) < 1500:
        return "managed", "R7: budget below self-hosted platform baseline (~$1,500/mo)"
    if answers.get("cloud") == "aws":
        return "managed", ("R6: AWS + K8s-capable team -- managed still recommended by default; "
                           "revisit self-hosted at sustained high utilization or for portability")
    return "self-hosted", "R7: non-AWS/on-prem with K8s-capable team and sufficient budget"


def decide_model_family(answers):
    """Rules R8-R11."""
    dtype = answers.get("data_type")
    flags = []
    if dtype in ("text",):
        family, why = "llm", "R8: unstructured text workload -- LLM path (RAG before fine-tuning)"
    elif dtype == "mixed":
        family, why = "hybrid", "R8: mixed data -- LLM extracts structure, classical model scores"
    elif dtype in ("tabular", "timeseries"):
        family, why = "classical", "R9: tabular/timeseries -- gradient boosting beats LLMs here on cost, latency, accuracy"
    elif dtype == "image":
        family, why = "classical", "R9(ext): vision task -- CNN/ViT fine-tune, classical MLOps path"
    else:
        family, why = "unknown", "data_type missing -- rerun intake"
        flags.append("Cannot choose model family without data_type.")

    if answers.get("labels_available") == "no" and family == "classical":
        flags.append("R10: no labels for a supervised task -- scope a labeling/discovery phase first.")
    if answers.get("explainability_required") == "yes" and family in ("llm", "hybrid"):
        flags.append("R11: explainability mandated -- LLM components will face audit pushback; "
                     "keep regulated decisions on the classical path (tree models + SHAP).")
    return family, why, flags


def estimate_cost(inference_mode, hosting, family):
    """Rule R12."""
    low, high, note = COST_BANDS[(inference_mode, hosting)]
    notes = [note]
    if family in ("llm", "hybrid"):
        adder = LLM_MANAGED_ADDER if hosting == "managed" else LLM_SELFHOSTED_ADDER
        low, high = low + adder[0], high + adder[1]
        notes.append(adder[2])
    return {"low_usd": low, "high_usd": high, "assumptions": notes,
            "caveat": "Order-of-magnitude only (+/-50%); validate with the pricing calculator."}


def pick_skill_map(family):
    if family == "llm":
        return {"llm_path": LLM_SKILL_MAP}
    if family == "hybrid":
        return {"classical_path": CLASSICAL_SKILL_MAP, "llm_path": LLM_SKILL_MAP}
    return {"classical_path": CLASSICAL_SKILL_MAP}


def recommend(doc):
    answers = doc.get("answers", {})
    inference_mode, mode_why = decide_inference_mode(answers)
    hosting, hosting_why = decide_hosting(answers)
    family, family_why, flags = decide_model_family(answers)

    stack = "aws-managed" if hosting == "managed" else "oss-on-k8s"
    if family == "hybrid":
        stack += "+llm-hybrid"
    elif family == "llm":
        stack += "+llm"

    result = {
        "recommended_stack": stack,
        "decisions": {
            "inference_mode": {"choice": inference_mode, "rationale": mode_why},
            "hosting": {"choice": hosting, "rationale": hosting_why},
            "model_family": {"choice": family, "rationale": family_why},
        },
        "flags": flags,
        "skill_map": pick_skill_map(family),
        "monthly_cost_band": estimate_cost(inference_mode, hosting, family),
        "unresolved_gaps": doc.get("gaps", []),
    }
    if hosting == "managed" and family in ("llm", "hybrid"):
        result["notes"] = [
            "LLM serving default: Amazon Bedrock, e.g. model id global.anthropic.claude-sonnet-5.",
            "Apply the llm-cost-optimization skill before scaling token volume.",
        ]
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="Recommend an ML architecture from requirements.json (rule-based, no LLM calls).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--requirements", required=True,
                        help="Path to requirements.json from intake_questionnaire.py")
    parser.add_argument("--output", default=None,
                        help="Write recommendation JSON here (default: print to stdout)")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        with open(args.requirements) as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Cannot read requirements file: %s", exc)
        sys.exit(1)

    if "answers" not in doc:
        logger.error("File does not look like intake output (missing 'answers' key).")
        sys.exit(1)

    result = recommend(doc)
    payload = json.dumps(result, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(payload + "\n")
        logger.info("Recommendation written to %s", args.output)
    else:
        print(payload)

    for flag in result["flags"]:
        logger.warning("FLAG: %s", flag)
    gaps = result["unresolved_gaps"]
    if gaps:
        logger.warning("%d intake gap(s) remain -- recommendation confidence is reduced.", len(gaps))


if __name__ == "__main__":
    main()
