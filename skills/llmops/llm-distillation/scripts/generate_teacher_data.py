#!/usr/bin/env python3
"""Batch teacher-data generation for sequence-level knowledge distillation via Bedrock Converse.

Reads prompts from JSONL ({"id": ..., "prompt": ..., "answer": optional ground truth}),
calls a Bedrock teacher (default DeepSeek-R1 inference profile) with retry/backoff,
checkpoints completed samples to an append-only JSONL, and supports resume + cost estimation.

Usage:
    python generate_teacher_data.py --action estimate --prompts prompts.jsonl --samples-per-prompt 3
    python generate_teacher_data.py --action generate --prompts prompts.jsonl --output teacher_raw.jsonl
    python generate_teacher_data.py --action generate --prompts prompts.jsonl --output teacher_raw.jsonl \\
        --model-id us.deepseek.r1-v1:0 --temperature 0.6 --max-tokens 8192 --workers 4
"""
import argparse
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "us.deepseek.r1-v1:0"  # cross-region inference profile (R1 requires a profile)
MAX_RETRIES = 6
BACKOFF_CAP_S = 60

# Illustrative mid-2026 rates (USD per 1M tokens) for --action estimate only.
# Always verify current pricing: https://aws.amazon.com/bedrock/pricing/
PRICING_PER_M = {
    "us.deepseek.r1-v1:0": {"input": 1.35, "output": 5.40},
}
DEFAULT_AVG_OUTPUT_TOKENS = 4000  # reasoning traces are long; override with --avg-output-tokens

_write_lock = threading.Lock()


def _get_client():
    """Return a bedrock-runtime client (region from environment/profile - never hardcoded)."""
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        logger.error("Install: pip install boto3"); sys.exit(1)
    # Disable SDK retries: this script owns the backoff policy so attempts are observable.
    return boto3.client("bedrock-runtime", config=Config(retries={"max_attempts": 0}))


def _is_throttle(exc):
    code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
    return code in ("ThrottlingException", "TooManyRequestsException",
                    "ServiceUnavailableException", "ModelNotReadyException")


def call_teacher(client, model_id, prompt, temperature, top_p, max_tokens):
    """Single Converse call with exponential backoff on throttling."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.converse(
                modelId=model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": max_tokens,
                                 "temperature": temperature, "topP": top_p},
            )
            blocks = resp["output"]["message"]["content"]
            # Reasoning models return the trace in a reasoningContent block; others won't.
            reasoning = next((b["reasoningContent"]["reasoningText"]["text"]
                              for b in blocks if "reasoningContent" in b), "")
            answer = next((b["text"] for b in blocks if "text" in b), "")
            return {
                "reasoning": reasoning,
                "answer": answer,
                "stop_reason": resp["stopReason"],
                "input_tokens": resp["usage"]["inputTokens"],
                "output_tokens": resp["usage"]["outputTokens"],
            }
        except Exception as e:  # noqa: BLE001 - classify below, re-raise non-throttles
            if not _is_throttle(e):
                raise
            wait = min(2 ** attempt, BACKOFF_CAP_S)
            logger.warning(f"Throttled (attempt {attempt + 1}/{MAX_RETRIES}), sleeping {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Throttled {MAX_RETRIES} consecutive times - reduce --workers")


def load_prompts(path):
    with open(path, "r", encoding="utf-8") as f:
        prompts = [json.loads(l) for l in f if l.strip()]
    for i, p in enumerate(prompts):
        p.setdefault("id", str(i))
        if "prompt" not in p:
            logger.error(f"Record {p['id']} missing 'prompt' field"); sys.exit(1)
    logger.info(f"Loaded {len(prompts)} prompts from {path}")
    return prompts


def load_completed(output_path):
    """Return set of (prompt_id, sample_idx) already generated - resume support."""
    done = set()
    if Path(output_path).exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done.add((str(r["id"]), int(r.get("sample_idx", 0))))
        logger.info(f"Resume: {len(done)} samples already in {output_path}")
    return done


def _append_record(record, output_path):
    with _write_lock:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate(args):
    client = _get_client()
    prompts = load_prompts(args.prompts)
    done = load_completed(args.output)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    tasks = [(p, k) for p in prompts for k in range(args.samples_per_prompt)
             if (str(p["id"]), k) not in done]
    logger.info(f"{len(tasks)} samples to generate "
                f"({len(prompts)} prompts x {args.samples_per_prompt}, {len(done)} resumed)")

    stats = {"ok": 0, "truncated": 0, "failed": 0, "in_tok": 0, "out_tok": 0}

    def worker(task):
        prompt_rec, sample_idx = task
        result = call_teacher(client, args.model_id, prompt_rec["prompt"],
                              args.temperature, args.top_p, args.max_tokens)
        record = {
            "id": str(prompt_rec["id"]),
            "sample_idx": sample_idx,
            "prompt": prompt_rec["prompt"],
            "reasoning": result["reasoning"],
            "answer": result["answer"],
            "stop_reason": result["stop_reason"],
            "teacher_model_id": args.model_id,
            "temperature": args.temperature,
            "usage": {"input_tokens": result["input_tokens"],
                      "output_tokens": result["output_tokens"]},
        }
        if "answer" in prompt_rec:  # carry ground truth through for verification stage
            record["ground_truth"] = prompt_rec["answer"]
        _append_record(record, args.output)
        return record

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                rec = fut.result()
                stats["ok"] += 1
                stats["in_tok"] += rec["usage"]["input_tokens"]
                stats["out_tok"] += rec["usage"]["output_tokens"]
                if rec["stop_reason"] == "max_tokens":
                    stats["truncated"] += 1  # curation stage will drop these
            except Exception as e:  # noqa: BLE001 - log and continue the batch
                stats["failed"] += 1
                pid, k = futures[fut][0]["id"], futures[fut][1]
                logger.error(f"Prompt {pid} sample {k} failed: {e}")
            if i % 50 == 0 or i == len(tasks):
                logger.info(f"Progress {i}/{len(tasks)} | ok={stats['ok']} "
                            f"truncated={stats['truncated']} failed={stats['failed']} | "
                            f"tokens in/out = {stats['in_tok']}/{stats['out_tok']}")

    price = PRICING_PER_M.get(args.model_id)
    if price:
        cost = (stats["in_tok"] * price["input"] + stats["out_tok"] * price["output"]) / 1e6
        logger.info(f"Approx spend this run: ${cost:.2f} (verify against current pricing)")
    logger.info(f"Done. Output: {args.output}")
    if stats["truncated"]:
        logger.warning(f"{stats['truncated']} samples hit max_tokens - "
                       f"curate_distillation_data.py will reject them")


def estimate(args):
    """Cost estimate BEFORE launching a large run."""
    prompts = load_prompts(args.prompts)
    price = PRICING_PER_M.get(args.model_id)
    if not price:
        logger.error(f"No pricing table entry for {args.model_id}; "
                     f"pass a known model or check the Bedrock pricing page")
        sys.exit(1)
    avg_in = sum(len(p["prompt"]) // 4 for p in prompts) / max(1, len(prompts))  # ~4 chars/token
    n = len(prompts) * args.samples_per_prompt
    cost = n * (avg_in * price["input"] + args.avg_output_tokens * price["output"]) / 1e6
    per_1k = cost / n * 1000
    print(json.dumps({
        "model_id": args.model_id,
        "prompts": len(prompts),
        "samples_per_prompt": args.samples_per_prompt,
        "total_samples": n,
        "avg_input_tokens_est": round(avg_in),
        "avg_output_tokens_assumed": args.avg_output_tokens,
        "cost_per_1k_samples_usd": round(per_1k, 2),
        "total_cost_usd_est": round(cost, 2),
        "note": "Illustrative rates - verify at https://aws.amazon.com/bedrock/pricing/",
    }, indent=2))


def main():
    p = argparse.ArgumentParser(description="Teacher data generation for LLM distillation (Bedrock Converse)")
    p.add_argument("--action", required=True, choices=["generate", "estimate"],
                   help="generate: run teacher calls; estimate: cost estimate only (no API calls)")
    p.add_argument("--prompts", required=True, help="Input JSONL with {'id','prompt'[,'answer']}")
    p.add_argument("--output", default="teacher_raw.jsonl", help="Append-only output JSONL (resume-safe)")
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Bedrock model/inference-profile ID")
    p.add_argument("--temperature", type=float, default=0.6,
                   help="0.5-0.7 for reasoning teachers (never 0 for R1); 0.2-0.3 for plain QA")
    p.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling")
    p.add_argument("--max-tokens", type=int, default=8192,
                   help="Generous cap - truncated reasoning chains are rejected downstream")
    p.add_argument("--samples-per-prompt", type=int, default=1,
                   help="Use 2-4 with ground truth for rejection sampling")
    p.add_argument("--workers", type=int, default=4, help="Concurrent Bedrock calls (mind account quotas)")
    p.add_argument("--avg-output-tokens", type=int, default=DEFAULT_AVG_OUTPUT_TOKENS,
                   help="Assumed output length for --action estimate")
    args = p.parse_args()

    if args.action == "estimate":
        estimate(args)
    else:
        generate(args)


if __name__ == "__main__":
    main()
