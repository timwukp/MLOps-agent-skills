#!/usr/bin/env python3
"""LLM cost optimization toolkit - estimate, route, compress, and report on LLM API costs.

Usage:
    python cost_optimizer.py --action estimate --input prompts.jsonl --models gpt-5,claude-sonnet-5
    python cost_optimizer.py --action route --input prompts.jsonl --budget 5.00
    python cost_optimizer.py --action compress --input prompts.jsonl --output compressed.jsonl
    python cost_optimizer.py --action report --input prompts.jsonl --models gpt-5,gpt-5-mini --output report.json
"""
import argparse
import json
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Model pricing: USD per 1M tokens (verified 2026-07 - re-check provider pages),
# plus a coarse relative quality score (0-1) used only for routing comparisons.
# Model IDs are the real API identifiers, not families.
MODEL_PRICING = {
    "claude-opus-5":    {"input": 5.00,  "output": 25.00, "quality": 0.97},
    "claude-sonnet-5":  {"input": 3.00,  "output": 15.00, "quality": 0.95},
    "claude-haiku-4-5": {"input": 1.00,  "output": 5.00,  "quality": 0.86},
    "gpt-5":            {"input": 1.25,  "output": 10.00, "quality": 0.94},
    "gpt-5-mini":       {"input": 0.25,  "output": 2.00,  "quality": 0.85},
    "gpt-4.1-mini":     {"input": 0.40,  "output": 1.60,  "quality": 0.80},
    "llama-4-scout":    {"input": 0.80,  "output": 2.40,  "quality": 0.78},
}

# Bare verbs like "explain" appear in the simplest questions, so they are weak
# signals. Require either a strong signal or a verb plus a substantial prompt;
# see _task_complexity for how the two tiers are weighted.
_STRONG_COMPLEX_RE = re.compile(
    r"(write\s+code|debug|refactor|implement|prove|derive|step[- ]by[- ]step|"
    r"trade[- ]?offs?|architecture)",
    re.IGNORECASE,
)
_WEAK_COMPLEX_RE = re.compile(
    r"(explain|analyz[es]|analys[es]|compare|contrast|summari[sz]e|translate|evaluate)",
    re.IGNORECASE,
)
_WEAK_SIGNAL_MIN_WORDS = 40
_TOKENIZER_WARNED = set()


def count_tokens(text, model="gpt-5-mini"):
    """Count tokens via tiktoken if available, otherwise a words/0.75 fallback.

    tiktoken only ships OpenAI tokenizers, so Claude/Llama models fall through to
    the heuristic (logged once per model). For exact Claude counts use
    client.messages.count_tokens(model=..., messages=[...]).
    """
    try:
        import tiktoken
    except ImportError:
        return max(1, int(len(text.split()) / 0.75))
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        if model not in _TOKENIZER_WARNED:
            _TOKENIZER_WARNED.add(model)
            logger.warning(
                f"tiktoken has no tokenizer for '{model}' (non-OpenAI model); "
                f"using a words/0.75 estimate. Token counts are approximate.")
        return max(1, int(len(text.split()) / 0.75))
    return len(enc.encode(text))


def estimate_cost(prompts, models=None, avg_output_ratio=1.5):
    """Estimate cost for a batch of prompts across selected models."""
    models = models or list(MODEL_PRICING.keys())
    results = {}
    for model in models:
        pricing = MODEL_PRICING.get(model)
        if pricing is None:
            logger.warning(f"Unknown model '{model}', skipping")
            continue
        total_in = sum(count_tokens(p, model) for p in prompts)
        total_out = int(total_in * avg_output_ratio)
        cost = (total_in * pricing["input"] + total_out * pricing["output"]) / 1_000_000
        results[model] = {
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "cost_usd": round(cost, 6),
        }
    return results


def _task_complexity(prompt):
    """Heuristic complexity score in [0, 1] based on length, vocab, patterns.

    Weights are a hand-tuned starting point, not a calibrated model. Validate
    routing decisions against your own labelled prompts before trusting them.
    """
    words = len(prompt.split())
    score = min(words / 500, 0.4)                       # length, up to 0.4
    unique = len(set(prompt.lower().split()))
    score += min((unique / max(words, 1)) * 0.3, 0.3)   # vocab richness, up to 0.3
    if _STRONG_COMPLEX_RE.search(prompt):
        score += 0.3
    elif _WEAK_COMPLEX_RE.search(prompt) and words >= _WEAK_SIGNAL_MIN_WORDS:
        # "explain"/"compare" on their own describe one-line questions too, so
        # they only count when the prompt itself has some substance.
        score += 0.15
    return min(score, 1.0)


def route_request(prompt, budget_per_request=None, quality_floor=0.0,
                  avg_output_ratio=1.5):
    """Route a prompt to the cheapest model that meets a quality threshold
    derived from task complexity.  Short/simple -> cheap; long/complex -> premium."""
    complexity = _task_complexity(prompt)
    min_quality = max(quality_floor, complexity * 0.95)
    input_tokens = count_tokens(prompt)
    output_tokens = input_tokens * avg_output_ratio
    candidates = []
    for name, info in MODEL_PRICING.items():
        if info["quality"] < min_quality:
            continue
        est = (input_tokens * info["input"] + output_tokens * info["output"]) / 1_000_000
        if budget_per_request and est > budget_per_request:
            continue
        candidates.append((name, est, info["quality"]))
    if not candidates:
        # Nothing fits: fall back to the cheapest model by total estimated cost
        # (input+output), not by input price alone.
        def _total(item):
            info = item[1]
            return (input_tokens * info["input"] + output_tokens * info["output"]) / 1_000_000
        name, info = min(MODEL_PRICING.items(), key=_total)
        est = (input_tokens * info["input"] + output_tokens * info["output"]) / 1_000_000
        return {"model": name, "estimated_cost_usd": round(est, 8),
                "quality": info["quality"], "complexity": round(complexity, 3),
                "reason": "budget_fallback"}
    candidates.sort(key=lambda x: x[1])
    c = candidates[0]
    return {"model": c[0], "estimated_cost_usd": round(c[1], 8),
            "quality": c[2], "complexity": round(complexity, 3), "reason": "optimal"}


def compress_prompt(text, max_tokens=None):
    """Remove redundant whitespace, shorten boilerplate, optionally truncate."""
    out = re.sub(r"[ \t]+", " ", text)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    for pat, repl in [
        (r"You are a helpful assistant that ", ""),
        (r"Please provide a detailed and comprehensive ", "Provide a "),
        (r"Make sure to include all relevant ", "Include relevant "),
        (r"In your response, please ", ""),
    ]:
        out = re.sub(pat, repl, out, flags=re.IGNORECASE)
    if max_tokens:
        cur = count_tokens(out)
        if cur > max_tokens:
            words = out.split()
            out = " ".join(words[:max(1, int(len(words) * max_tokens / cur))])
            logger.info(f"Truncated from {cur} to ~{count_tokens(out)} tokens")
    return out


def batch_vs_realtime(prompts, model="gpt-5-mini", batch_discount=0.50):
    """Compare batch (async, discounted) vs real-time API cost."""
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        return {"error": f"Unknown model: {model}"}
    total_in = sum(count_tokens(p, model) for p in prompts)
    total_out = int(total_in * 1.5)
    realtime = (total_in * pricing["input"] + total_out * pricing["output"]) / 1_000_000
    batch = realtime * (1 - batch_discount)
    return {"model": model, "num_prompts": len(prompts),
            "realtime_cost_usd": round(realtime, 6),
            "batch_cost_usd": round(batch, 6),
            "savings_usd": round(realtime - batch, 6),
            "savings_pct": round(batch_discount * 100, 1)}


def generate_report(prompts, models=None, requests_per_day=None):
    """Cost breakdown report with monthly projections."""
    estimates = estimate_cost(prompts, models)
    rpd = requests_per_day or len(prompts)
    daily_factor = rpd / max(len(prompts), 1)
    report = {"num_prompts_analyzed": len(prompts), "models": {}}
    for model, data in estimates.items():
        monthly = data["cost_usd"] * daily_factor * 30
        report["models"][model] = {
            **data,
            "projected_daily_usd": round(data["cost_usd"] * daily_factor, 4),
            "projected_monthly_usd": round(monthly, 2),
        }
    cheapest = min(report["models"].items(), key=lambda x: x[1]["cost_usd"])
    report["recommendation"] = {"cheapest_model": cheapest[0],
                                "cost_usd": cheapest[1]["cost_usd"]}
    return report


def load_prompts(path):
    """Load prompts from JSONL (expects 'prompt' or 'text' field per line)."""
    prompts = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                prompts.append(obj.get("prompt", obj.get("text", "")))
            except json.JSONDecodeError as exc:
                logger.warning(f"Skipping line {lineno}: {exc}")
    logger.info(f"Loaded {len(prompts)} prompts from {path}")
    return prompts


def write_json(data, path):
    """Write data as formatted JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    logger.info(f"Output written to {path}")


def main():
    parser = argparse.ArgumentParser(description="LLM cost optimization toolkit")
    parser.add_argument("--action", required=True,
                        choices=["estimate", "route", "compress", "report"],
                        help="Action to perform")
    parser.add_argument("--input", required=True, help="Input JSONL file with prompts")
    parser.add_argument("--models", default=None,
                        help="Comma-separated model list (default: all)")
    parser.add_argument("--budget", type=float, default=None,
                        help="Max budget in USD (used by route)")
    parser.add_argument("--output", default=None, help="Output file path")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Token budget for compression")
    parser.add_argument("--requests-per-day", type=int, default=None,
                        help="Daily request volume for report projections")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")] if args.models else None
    prompts = load_prompts(args.input)
    if not prompts:
        logger.error("No prompts found in input file")
        sys.exit(1)

    if args.action == "estimate":
        result = estimate_cost(prompts, models)
        print(json.dumps(result, indent=2))

    elif args.action == "route":
        per_req = args.budget / len(prompts) if args.budget else None
        result = []
        for p in prompts:
            r = route_request(p, budget_per_request=per_req)
            result.append(r)
            logger.info(f"Routed to {r['model']} (complexity={r['complexity']})")
        print(json.dumps(result, indent=2))

    elif args.action == "compress":
        compressed, saved = [], 0
        for p in prompts:
            orig = count_tokens(p)
            c = compress_prompt(p, max_tokens=args.max_tokens)
            new = count_tokens(c)
            saved += orig - new
            compressed.append({"prompt": c, "original_tokens": orig,
                               "compressed_tokens": new})
        # Same shape to stdout and to --output.
        result = {"total_tokens_saved": saved, "prompts": compressed}
        logger.info(f"Total tokens saved: {saved}")
        print(json.dumps(result, indent=2))

    elif args.action == "report":
        result = generate_report(prompts, models,
                                 requests_per_day=args.requests_per_day)
        print(json.dumps(result, indent=2))

    if args.output:
        write_json(result, args.output)


if __name__ == "__main__":
    main()
