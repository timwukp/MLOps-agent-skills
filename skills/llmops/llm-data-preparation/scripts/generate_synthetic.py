#!/usr/bin/env python3
"""Synthetic data generation for LLM training - self-instruct, evol-instruct, and topic-based.

Usage:
    python generate_synthetic.py --action generate --seed-data seeds.jsonl --num-samples 100 --output data.jsonl
    python generate_synthetic.py --action evolve --seed-data simple.jsonl --num-samples 50 --output evolved.jsonl
    python generate_synthetic.py --action filter --seed-data raw.jsonl --output clean.jsonl
    python generate_synthetic.py --action generate --topics "python,ml" --format sharegpt --output out.jsonl
"""
import argparse
import hashlib
import json
import logging
import random
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SELF_INSTRUCT_PROMPT = """You are an expert instruction-data generator.
Given these seed instructions, create {n} NEW diverse instructions that differ in topic, style, and complexity.
For each instruction also generate a high-quality response.
Seed instructions:
{seeds}
Return a JSON object: {{"pairs": [{{"instruction": "...", "response": "..."}}]}}"""

EVOL_INSTRUCT_PROMPT = """Rewrite the following instruction to make it more {strategy}.
Keep the core intent but increase difficulty. Then provide a detailed response.
Original instruction: {instruction}
Return JSON: {{"instruction": "...", "response": "..."}}"""

TOPIC_PROMPT = """Generate {n} diverse instruction-response pairs about: {topic}.
Vary the task type (explain, compare, analyze, code, list, debate, create).
Return a JSON object: {{"pairs": [{{"instruction": "...", "response": "..."}}]}}"""

EVOL_STRATEGIES = [
    "complex by adding constraints or multi-step reasoning",
    "specific by narrowing scope and requiring domain knowledge",
    "creative by requiring an unusual or novel approach",
    "analytical by requiring comparison or critical thinking",
    "practical by grounding it in a real-world scenario",
]


def _get_client(model):
    """Return an OpenAI-compatible client."""
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("Install: pip install openai"); sys.exit(1)
    return OpenAI()


def _call_llm(client, prompt, model="gpt-4o-mini", temperature=0.9):
    """Call the LLM and parse JSON response."""
    try:
        resp = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            temperature=temperature, response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        logger.warning(f"LLM call failed: {e}")
        return None


def generate_self_instruct(seed_examples, num_samples, model, client):
    """Self-instruct: bootstrap new instruction-response pairs from seeds."""
    results, batch = [], min(10, num_samples)
    seed_texts = [ex["instruction"] for ex in seed_examples]
    while len(results) < num_samples:
        sampled = random.sample(seed_texts, min(5, len(seed_texts)))
        data = _call_llm(client, SELF_INSTRUCT_PROMPT.format(
            n=batch, seeds="\n".join(f"- {s}" for s in sampled)), model=model)
        if data:
            pairs = data.get("pairs", data.get("data", []))
            results.extend(pairs)
            seed_texts.extend(p["instruction"] for p in pairs if p.get("instruction"))
            logger.info(f"Generated {len(results)}/{num_samples}")
    return results[:num_samples]


def generate_topic_based(topics, num_samples, model, client):
    """Generate instruction-response pairs per topic."""
    results, per_topic = [], max(1, num_samples // len(topics))
    for topic in topics:
        remaining = per_topic
        while remaining > 0:
            n = min(10, remaining)
            data = _call_llm(client, TOPIC_PROMPT.format(n=n, topic=topic.strip()), model=model)
            if data:
                pairs = data.get("pairs", data.get("data", []))
                for p in pairs:
                    p["topic"] = topic.strip()
                results.extend(pairs)
                remaining -= len(pairs)
            else:
                remaining -= n
    return results[:num_samples]


def evolve_instructions(seed_examples, num_samples, model, client):
    """Evol-instruct: evolve simple instructions into complex ones."""
    results, pool = [], list(seed_examples)
    random.shuffle(pool)
    for ex in pool:
        if len(results) >= num_samples:
            break
        strategy = random.choice(EVOL_STRATEGIES)
        data = _call_llm(client, EVOL_INSTRUCT_PROMPT.format(
            strategy=strategy, instruction=ex["instruction"]), model=model, temperature=0.8)
        if data and data.get("instruction"):
            data["original_instruction"] = ex["instruction"]
            data["evolution_strategy"] = strategy
            results.append(data)
            logger.info(f"Evolved {len(results)}/{num_samples}")
    return results[:num_samples]


def _minhash_sig(text, num_perm=128):
    """Compute MinHash signature for near-duplicate detection."""
    words = text.lower().split()
    shingles = {" ".join(words[i:i+3]) for i in range(len(words)-2)} or {text.lower().strip()}
    sig = []
    for i in range(num_perm):
        min_h = min(int(hashlib.md5(f"{i}:{s}".encode()).hexdigest(), 16) & 0xFFFFFFFF for s in shingles)
        sig.append(min_h)
    return sig


def _jaccard(a, b):
    return sum(x == y for x, y in zip(a, b)) / len(a)


def filter_quality(examples, min_instr_len=5, min_resp_len=20, dedup_threshold=0.8):
    """Apply length filter, exact dedup, near-dup removal (MinHash), and diversity scoring."""
    logger.info(f"Filtering {len(examples)} examples ...")
    # Length filter
    filtered = [ex for ex in examples
                if len(ex.get("instruction", "").split()) >= min_instr_len
                and len(ex.get("response", "").split()) >= min_resp_len]
    logger.info(f"After length filter: {len(filtered)}")
    # Exact dedup
    seen, deduped = set(), []
    for ex in filtered:
        key = ex["instruction"].strip().lower()
        if key not in seen:
            seen.add(key); deduped.append(ex)
    logger.info(f"After exact dedup: {len(deduped)}")
    # Near-dup removal
    sigs, unique = [], []
    for ex in deduped:
        sig = _minhash_sig(ex["instruction"] + " " + ex.get("response", ""))
        if not any(_jaccard(sig, ps) >= dedup_threshold for ps in sigs):
            sigs.append(sig); unique.append(ex)
    logger.info(f"After near-dup removal: {len(unique)}")
    # Diversity score
    trigrams = set()
    for ex in unique:
        w = (ex["instruction"] + " " + ex.get("response", "")).lower().split()
        trigrams.update(tuple(w[i:i+3]) for i in range(len(w)-2))
    total = max(1, sum(len((ex.get("instruction","")+ex.get("response","")).split()) for ex in unique))
    logger.info(f"Trigram diversity: {len(trigrams)/total:.4f}")
    return unique


def format_output(examples, fmt):
    """Convert to alpaca, sharegpt, or chat format."""
    out = []
    for ex in examples:
        instr, resp, sys_msg = ex.get("instruction",""), ex.get("response",""), ex.get("system","")
        if fmt == "alpaca":
            out.append({"instruction": instr, "input": ex.get("input", ""), "output": resp})
        elif fmt == "sharegpt":
            c = ([{"from":"system","value":sys_msg}] if sys_msg else [])
            c += [{"from":"human","value":instr}, {"from":"gpt","value":resp}]
            out.append({"conversations": c})
        else:
            m = ([{"role":"system","content":sys_msg}] if sys_msg else [])
            m += [{"role":"user","content":instr}, {"role":"assistant","content":resp}]
            out.append({"messages": m})
    return out


def write_jsonl(records, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(records)} records to {path}")


def load_seed_data(path):
    with open(path, "r", encoding="utf-8") as f:
        examples = [json.loads(l) for l in f if l.strip()]
    logger.info(f"Loaded {len(examples)} seed examples from {path}")
    return examples


def main():
    p = argparse.ArgumentParser(description="Synthetic data generation for LLM training")
    p.add_argument("--action", required=True, choices=["generate", "evolve", "filter"],
                   help="generate: self-instruct/topic; evolve: evol-instruct; filter: quality filter")
    p.add_argument("--seed-data", help="Path to seed JSONL file")
    p.add_argument("--topics", help="Comma-separated topics for topic-based generation")
    p.add_argument("--model", default="gpt-4o-mini", help="LLM model name (OpenAI-compatible)")
    p.add_argument("--num-samples", type=int, default=50, help="Number of samples to generate")
    p.add_argument("--output", default="synthetic_data.jsonl", help="Output JSONL path")
    p.add_argument("--format", dest="fmt", choices=["alpaca","sharegpt","chat"], default="chat")
    p.add_argument("--min-instr-len", type=int, default=5, help="Min instruction word count")
    p.add_argument("--min-resp-len", type=int, default=20, help="Min response word count")
    p.add_argument("--dedup-threshold", type=float, default=0.8, help="Near-dup Jaccard threshold")
    args = p.parse_args()

    if args.action == "generate":
        if args.topics:
            topics = [t.strip() for t in args.topics.split(",") if t.strip()]
            results = generate_topic_based(topics, args.num_samples, args.model, _get_client(args.model))
        elif args.seed_data:
            seeds = load_seed_data(args.seed_data)
            results = generate_self_instruct(seeds, args.num_samples, args.model, _get_client(args.model))
        else:
            logger.error("Provide --seed-data or --topics"); sys.exit(1)
        results = filter_quality(results, args.min_instr_len, args.min_resp_len, args.dedup_threshold)
        write_jsonl(format_output(results, args.fmt), args.output)
    elif args.action == "evolve":
        if not args.seed_data:
            logger.error("Provide --seed-data for evol-instruct"); sys.exit(1)
        seeds = load_seed_data(args.seed_data)
        results = evolve_instructions(seeds, args.num_samples, args.model, _get_client(args.model))
        results = filter_quality(results, args.min_instr_len, args.min_resp_len, args.dedup_threshold)
        write_jsonl(format_output(results, args.fmt), args.output)
    elif args.action == "filter":
        if not args.seed_data:
            logger.error("Provide --seed-data to filter"); sys.exit(1)
        data = load_seed_data(args.seed_data)
        cleaned = filter_quality(data, args.min_instr_len, args.min_resp_len, args.dedup_threshold)
        write_jsonl(format_output(cleaned, args.fmt), args.output)
    logger.info("Done.")


if __name__ == "__main__":
    main()
