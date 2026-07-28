#!/usr/bin/env python3
"""Curate raw teacher generations into a TRL-ready SFT dataset for knowledge distillation.

Pipeline: format validation -> dedup (exact + MinHash near-dup) -> decontamination vs eval set
-> correctness filter (exact-match on ground truth, or optional LLM judge) -> length filter
vs student context -> train/val split by prompt -> TRL messages-format JSONL.

Input records (from generate_teacher_data.py):
    {"id","sample_idx","prompt","reasoning","answer","stop_reason"[,"ground_truth"]}

Usage:
    python curate_distillation_data.py --input teacher_raw.jsonl --output-dir curated/ \\
        --reasoning-mode keep --student-max-length 8192
    python curate_distillation_data.py --input teacher_raw.jsonl --output-dir curated/ \\
        --reasoning-mode strip --eval-set eval_prompts.jsonl
    python curate_distillation_data.py --input teacher_raw.jsonl --output-dir curated/ \\
        --reasoning-mode keep --verify exact-match --answer-pattern "\\\\boxed\\{([^}]*)\\}"
"""
import argparse
import hashlib
import json
import logging
import random
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NUM_PERM = 128
LSH_BANDS = 16
CHARS_PER_TOKEN = 4  # rough estimate; pass --tokenizer for exact student-token counts


# ---------------------------------------------------------------- IO helpers

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]
    logger.info(f"Loaded {len(records)} records from {path}")
    return records


def write_jsonl(records, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(records)} records to {path}")


# ------------------------------------------------------- stage 1: format validation

def validate_format(records, reasoning_mode):
    """Reject truncated, empty, or malformed generations."""
    kept, rejected = [], {"truncated": 0, "empty_answer": 0, "no_reasoning": 0}
    for r in records:
        if r.get("stop_reason") == "max_tokens":
            rejected["truncated"] += 1      # unterminated chains poison the student
            continue
        if not r.get("answer", "").strip():
            rejected["empty_answer"] += 1
            continue
        if reasoning_mode == "keep" and not r.get("reasoning", "").strip():
            rejected["no_reasoning"] += 1   # can't train reasoning targets without a trace
            continue
        kept.append(r)
    logger.info(f"Format validation: kept {len(kept)}/{len(records)} (rejected: {rejected})")
    return kept


# ------------------------------------------------------- stage 2: deduplication

def _minhash_sig(text, num_perm=NUM_PERM):
    words = text.lower().split()
    shingles = {" ".join(words[i:i + 3]) for i in range(len(words) - 2)} or {text.lower().strip()}
    return [min(int(hashlib.md5(f"{i}:{s}".encode()).hexdigest(), 16) & 0xFFFFFFFF
                for s in shingles) for i in range(num_perm)]


def _jaccard(a, b):
    return sum(x == y for x, y in zip(a, b)) / len(a)


def _bands(sig, bands=LSH_BANDS):
    rows = max(1, len(sig) // bands)
    return [(b, tuple(sig[b * rows:(b + 1) * rows])) for b in range(bands)]


def deduplicate(records, threshold=0.8):
    """Exact dedup on prompt+answer, then MinHash+LSH near-dup removal."""
    seen, exact = set(), []
    for r in records:
        key = (r["prompt"].strip().lower(), r["answer"].strip().lower())
        if key not in seen:
            seen.add(key); exact.append(r)
    logger.info(f"Exact dedup: kept {len(exact)}/{len(records)}")

    buckets, kept_sigs, unique = {}, [], []
    for r in exact:
        sig = _minhash_sig(r["prompt"] + " " + r["answer"])
        candidates = set()
        for bk in _bands(sig):
            candidates.update(buckets.get(bk, ()))
        if any(_jaccard(sig, kept_sigs[i]) >= threshold for i in candidates):
            continue
        idx = len(kept_sigs)
        kept_sigs.append(sig); unique.append(r)
        for bk in _bands(sig):
            buckets.setdefault(bk, []).append(idx)
    logger.info(f"Near-dup removal: kept {len(unique)}/{len(exact)}")
    return unique


# ------------------------------------------------------- stage 3: decontamination

def decontaminate(records, eval_path, ngram=13):
    """Drop training samples sharing any `ngram`-gram with eval prompts (Llama/GPT-4-style)."""
    eval_records = load_jsonl(eval_path)
    eval_ngrams = set()
    for e in eval_records:
        text = (e.get("prompt") or e.get("question") or json.dumps(e)).lower()
        words = text.split()
        eval_ngrams.update(tuple(words[i:i + ngram]) for i in range(len(words) - ngram + 1))
    if not eval_ngrams:
        logger.warning("Eval set produced no n-grams (prompts shorter than n?) - nothing filtered")
        return records

    kept, dropped = [], 0
    for r in records:
        words = r["prompt"].lower().split()
        grams = (tuple(words[i:i + ngram]) for i in range(len(words) - ngram + 1))
        if any(g in eval_ngrams for g in grams):
            dropped += 1
            continue
        kept.append(r)
    logger.info(f"Decontamination ({ngram}-gram vs {eval_path}): "
                f"kept {len(kept)}/{len(records)}, dropped {dropped}")
    if dropped:
        logger.warning(f"{dropped} contaminated prompts removed - "
                       f"eval scores would have been inflated")
    return kept


# ------------------------------------------------------- stage 4: correctness filter

def extract_answer(text, pattern):
    m = re.findall(pattern, text)
    return m[-1].strip() if m else None  # last match: final answer follows the reasoning


def _normalize(ans):
    return re.sub(r"[\s$,]", "", str(ans)).lower()


def verify_exact_match(records, answer_pattern):
    """Verifiable-reward filter: keep samples whose extracted answer matches ground truth.

    Deterministic and un-gameable - always prefer this over an LLM judge when
    ground truth exists. Records without ground_truth pass through with a warning.
    """
    kept, wrong, no_gt = [], 0, 0
    for r in records:
        gt = r.get("ground_truth")
        if gt is None:
            no_gt += 1
            kept.append(r)
            continue
        extracted = extract_answer(r["answer"], answer_pattern) if answer_pattern else r["answer"]
        if extracted is not None and _normalize(extracted) == _normalize(gt):
            kept.append(r)
        else:
            wrong += 1
    logger.info(f"Exact-match verification: kept {len(kept)}/{len(records)} "
                f"(wrong={wrong}, no_ground_truth={no_gt})")
    if no_gt:
        logger.warning(f"{no_gt} records had no ground_truth - passed unverified; "
                       f"consider LLM-judge filtering for those")
    return kept


def verify_llm_judge(records, model_id, threshold=4, sample_audit=50):
    """LLM-judge filter for domains without ground truth.

    Uses Bedrock Converse with a judge model that MUST differ from the teacher
    (a teacher judging its own outputs inflates scores). Hand-audit `sample_audit`
    decisions before trusting at scale.
    """
    try:
        import boto3
    except ImportError:
        logger.error("Install: pip install boto3"); sys.exit(1)
    client = boto3.client("bedrock-runtime")

    prompt_tpl = ("Rate this response for correctness, completeness, and clarity "
                  "on a scale of 1-5. Respond with ONLY the integer.\n\n"
                  "Question: {q}\n\nResponse: {a}")
    kept = []
    for i, r in enumerate(records):
        try:
            resp = client.converse(
                modelId=model_id,
                messages=[{"role": "user",
                           "content": [{"text": prompt_tpl.format(q=r["prompt"], a=r["answer"])}]}],
                inferenceConfig={"maxTokens": 8, "temperature": 0.0},
            )
            text = resp["output"]["message"]["content"][0]["text"]
            m = re.search(r"[1-5]", text)
            score = int(m.group()) if m else 0
        except Exception as e:  # noqa: BLE001 - skip-and-continue keeps the batch alive
            logger.warning(f"Judge call failed for {r['id']}: {e}")
            continue
        r["judge_score"] = score
        if score >= threshold:
            kept.append(r)
        if (i + 1) % 100 == 0:
            logger.info(f"Judged {i + 1}/{len(records)}, kept {len(kept)}")
    logger.info(f"LLM-judge filter (>= {threshold}): kept {len(kept)}/{len(records)}")
    logger.warning(f"Hand-audit ~{sample_audit} judge decisions before trusting this filter; "
                   f"judges reward fluency and can be gamed - see SKILL.md anti-patterns")
    return kept


# ------------------------------------------------------- stage 5: build targets + length filter

def build_target(record, reasoning_mode):
    """Assemble the assistant training target - one canonical serialization."""
    if reasoning_mode == "keep":
        return f"<think>\n{record['reasoning'].strip()}\n</think>\n\n{record['answer'].strip()}"
    return record["answer"].strip()


def _token_len(text, tokenizer):
    if tokenizer is not None:
        return len(tokenizer.encode(text))
    return len(text) // CHARS_PER_TOKEN


def length_filter(records, reasoning_mode, student_max_length, tokenizer_name=None):
    """Drop samples exceeding 90% of the STUDENT's context (student tokenizer, not teacher's)."""
    tokenizer = None
    if tokenizer_name:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            logger.info(f"Length filter using student tokenizer: {tokenizer_name}")
        except Exception as e:  # noqa: BLE001 - degrade to estimate rather than abort
            logger.warning(f"Could not load tokenizer ({e}); using ~{CHARS_PER_TOKEN} chars/token estimate")

    budget = int(student_max_length * 0.9)
    kept, dropped = [], 0
    for r in records:
        target = build_target(r, reasoning_mode)
        if _token_len(r["prompt"], tokenizer) + _token_len(target, tokenizer) <= budget:
            kept.append(r)
        else:
            dropped += 1
    logger.info(f"Length filter (<= {budget} student tokens): kept {len(kept)}/{len(records)}, "
                f"dropped {dropped} over-length (truncated targets teach unterminated rambling)")
    return kept


# ------------------------------------------------------- stage 6: split + format

def split_and_format(records, reasoning_mode, val_ratio=0.05, seed=42):
    """Split BY PROMPT ID (never the same prompt in train and val), emit TRL messages format."""
    by_prompt = {}
    for r in records:
        by_prompt.setdefault(str(r["id"]), []).append(r)
    ids = sorted(by_prompt)
    random.Random(seed).shuffle(ids)
    n_val = max(1, int(len(ids) * val_ratio)) if len(ids) > 1 else 0
    val_ids = set(ids[:n_val])

    def to_trl(r):
        return {"messages": [
            {"role": "user", "content": r["prompt"]},
            {"role": "assistant", "content": build_target(r, reasoning_mode)},
        ]}

    train = [to_trl(r) for pid in ids if pid not in val_ids for r in by_prompt[pid]]
    val = [to_trl(r) for pid in val_ids for r in by_prompt[pid]]
    logger.info(f"Split by prompt: {len(train)} train / {len(val)} val "
                f"({len(ids) - n_val}/{n_val} prompts)")
    return train, val


def main():
    p = argparse.ArgumentParser(description="Curate teacher generations into TRL-ready distillation data")
    p.add_argument("--input", required=True, help="Raw teacher JSONL from generate_teacher_data.py")
    p.add_argument("--output-dir", default="curated", help="Directory for train/val JSONL + report")
    p.add_argument("--reasoning-mode", required=True, choices=["keep", "strip"],
                   help="keep: <think> trace in target (math/code/puzzles); strip: final answer only (simple QA)")
    p.add_argument("--eval-set", help="Eval prompts JSONL for decontamination (strongly recommended)")
    p.add_argument("--ngram", type=int, default=13, help="N-gram size for decontamination overlap")
    p.add_argument("--dedup-threshold", type=float, default=0.8, help="MinHash near-dup Jaccard threshold")
    p.add_argument("--verify", choices=["exact-match", "llm-judge", "none"], default="none",
                   help="Correctness filter: exact-match needs ground_truth; llm-judge needs --judge-model-id")
    p.add_argument("--answer-pattern", default=r"\\boxed\{([^}]*)\}",
                   help="Regex extracting the final answer for exact-match (default: \\boxed{...})")
    p.add_argument("--judge-model-id", help="Bedrock model ID for LLM judge (must differ from teacher)")
    p.add_argument("--judge-threshold", type=int, default=4, help="Minimum judge score (1-5) to keep")
    p.add_argument("--student-max-length", type=int, default=8192,
                   help="Student training context; samples over 90%% of this are dropped")
    p.add_argument("--tokenizer", help="Student HF tokenizer name for exact length counting")
    p.add_argument("--val-ratio", type=float, default=0.05, help="Validation fraction (by prompt)")
    p.add_argument("--seed", type=int, default=42, help="Split shuffle seed")
    args = p.parse_args()

    records = load_jsonl(args.input)
    initial = len(records)

    records = validate_format(records, args.reasoning_mode)
    records = deduplicate(records, args.dedup_threshold)

    if args.eval_set:
        records = decontaminate(records, args.eval_set, args.ngram)
    else:
        logger.warning("No --eval-set given: SKIPPING decontamination. "
                       "Do not report benchmark numbers from a model trained on this data.")

    if args.verify == "exact-match":
        records = verify_exact_match(records, args.answer_pattern)
    elif args.verify == "llm-judge":
        if not args.judge_model_id:
            logger.error("--verify llm-judge requires --judge-model-id"); sys.exit(1)
        records = verify_llm_judge(records, args.judge_model_id, args.judge_threshold)

    records = length_filter(records, args.reasoning_mode, args.student_max_length, args.tokenizer)

    if not records:
        logger.error("No records survived curation - inspect stage logs above"); sys.exit(1)

    train, val = split_and_format(records, args.reasoning_mode, args.val_ratio, args.seed)
    out = Path(args.output_dir)
    write_jsonl(train, out / "train.jsonl")
    write_jsonl(val, out / "val.jsonl")

    report = {
        "input_records": initial,
        "final_records": len(records),
        "yield_rate": round(len(records) / max(1, initial), 3),
        "reasoning_mode": args.reasoning_mode,
        "decontaminated": bool(args.eval_set),
        "verification": args.verify,
        "train": len(train),
        "val": len(val),
        "handoff": "Feed train.jsonl/val.jsonl to llm-fine-tuning (TRL SFTTrainer messages format)",
    }
    write_jsonl([report], out / "curation_report.jsonl")
    logger.info(f"Curation complete: {json.dumps(report)}")


if __name__ == "__main__":
    main()
