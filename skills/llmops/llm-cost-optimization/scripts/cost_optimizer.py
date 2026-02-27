#!/usr/bin/env python3
"""LLM cost optimization toolkit - estimate, route, compress, and report on LLM API costs.

Usage:
    python cost_optimizer.py --action estimate --input prompts.jsonl --models gpt-4o,claude-3.5-sonnet
    python cost_optimizer.py --action route --input prompts.jsonl --budget 5.00
    python cost_optimizer.py --action compress --input prompts.jsonl --output compressed.jsonl
    python cost_optimizer.py --action report --input prompts.jsonl --models gpt-4o,gpt-4o-mini --output report.json
"""
import argparse
import json
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Model pricing: USD per 1M tokens, plus a relative quality score (0-1)
MODEL_PRICING = {
    "gpt-4o":            {"input": 2.50,  "output": 10.00, "quality": 0.95},
    "gpt-4o-mini":       {"input": 0.15,  "output": 0.60,  "quality": 0.82},
    "claude-3.5-sonnet": {"input": 3.00,  "output": 15.00, "quality": 0.96},
    "claude-haiku":      {"input": 0.25,  "output": 1.25,  "quality": 0.78},
    "llama-3":           {"input": 0.05,  "output": 0.08,  "quality": 0.75},
    "mistral":           {"input": 0.10,  "output": 0.30,  "quality": 0.73},
}

_COMPLEX_RE = re.compile(
    r"(explain|analyze|compare|contrast|summarize .{200,}|write code|debug|refactor|translate)",
    re.IGNORECASE,
)


def count_tokens(text, model="gpt-4o"):
    """Count tokens via tiktoken if available, otherwise ~words/0.75 fallback."""
    try:
        import tiktoken
        return len(tiktoken.encoding_for_model(model).encode(text))
    except Exception:
        return max(1, int(len(text.split()) / 0.75))


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
    """Heuristic complexity score in [0, 1] based on length, vocab, patterns."""
    words = len(prompt.split())
    score = min(words / 500, 0.4)
    unique = len(set(prompt.lower().split()))
    score += min((unique / max(words, 1)) * 0.3, 0.3)
    if _COMPLEX_RE.search(prompt):
        score += 0.3
    return min(score, 1.0)


def route_request(prompt, budget_per_request=None, quality_floor=0.0):
    """Route a prompt to the cheapest model that meets a quality threshold
    derived from task complexity.  Short/simple -> cheap; long/complex -> premium."""
    complexity = _task_complexity(prompt)
    min_quality = max(quality_floor, complexity * 0.95)
    input_tokens = count_tokens(prompt)
    candidates = []
    for name, info in MODEL_PRICING.items():
        if info["quality"] < min_quality:
            continue
        est = (input_tokens * info["input"] + input_tokens * 1.5 * info["output"]) / 1_000_000
        if budget_per_request and est > budget_per_request:
            continue
        candidates.append((name, est, info["quality"]))
    if not candidates:
        cheapest = min(MODEL_PRICING.items(), key=lambda x: x[1]["input"])
        return {"model": cheapest[0], "reason": "budget_fallback",
                "complexity": round(complexity, 3)}
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


def batch_vs_realtime(prompts, model="gpt-4o", batch_discount=0.50):
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
        result = compressed
        logger.info(f"Total tokens saved: {saved}")
        print(json.dumps({"total_tokens_saved": saved, "prompts": result}, indent=2))

    elif args.action == "report":
        result = generate_report(prompts, models,
                                 requests_per_day=args.requests_per_day)
        print(json.dumps(result, indent=2))

    if args.output:
        write_json(result, args.output)


if __name__ == "__main__":
    main()
