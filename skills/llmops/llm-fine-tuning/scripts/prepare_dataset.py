#!/usr/bin/env python3
"""Dataset preparation and validation for LLM fine-tuning.

Usage:
    python prepare_dataset.py --input data.jsonl --output prepared.jsonl --input-format alpaca --output-format chat
    python prepare_dataset.py --input data.csv --output out.jsonl --input-format csv --output-format alpaca --deduplicate
    python prepare_dataset.py --input data.jsonl --input-format sharegpt --preview 5
"""
import argparse, csv, json, logging, sys
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REQUIRED_FIELDS = {
    "alpaca": ["instruction", "output"], "sharegpt": ["conversations"],
    "chat": ["messages"], "completion": ["prompt", "completion"], "csv": [],
}

def load_data(input_path):
    """Load records from CSV, JSONL, JSON, Parquet, or HuggingFace datasets."""
    p = str(input_path)
    if p.endswith(".jsonl"):
        records = []
        with open(p, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed JSON at line {lineno}: {e}")
        return records
    if p.endswith(".json"):
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return data
        raise ValueError("JSON file must contain a top-level array")
    if p.endswith(".csv"):
        with open(p, "r", encoding="utf-8", newline="") as fh:
            return [dict(row) for row in csv.DictReader(fh)]
    if p.endswith(".parquet"):
        try:
            import pandas as pd
        except ImportError:
            logger.error("Install: pip install pandas pyarrow"); sys.exit(1)
        return pd.read_parquet(p).to_dict(orient="records")
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Install: pip install datasets"); sys.exit(1)
    logger.info(f"Treating '{p}' as a HuggingFace dataset identifier")
    return list(load_dataset(p, split="train"))

def record_text(rec):
    """Extract full text of a record for length estimation."""
    if "messages" in rec:
        return " ".join(m.get("content", "") for m in rec["messages"])
    if "conversations" in rec:
        return " ".join(c.get("value", "") for c in rec["conversations"])
    if "instruction" in rec:
        return f"{rec.get('instruction','')} {rec.get('input','')} {rec.get('output','')}"
    if "prompt" in rec:
        return f"{rec.get('prompt','')} {rec.get('completion','')}"
    return rec.get("text", "")

def estimate_tokens(text):
    """Rough token count (words * 1.3)."""
    return int(len(text.split()) * 1.3)

def to_alpaca(rec):
    """Convert any record to Alpaca (instruction/input/output) format."""
    if "instruction" in rec:
        return {"instruction": rec["instruction"], "input": rec.get("input", ""),
                "output": rec.get("output", rec.get("response", ""))}
    if "messages" in rec:
        msgs = rec["messages"]
        return {"instruction": next((m["content"] for m in msgs if m["role"] == "user"), ""),
                "input": "", "output": next((m["content"] for m in msgs if m["role"] == "assistant"), "")}
    if "conversations" in rec:
        c = rec["conversations"]
        return {"instruction": next((x["value"] for x in c if x["from"] == "human"), ""),
                "input": "", "output": next((x["value"] for x in c if x["from"] == "gpt"), "")}
    if "prompt" in rec:
        return {"instruction": rec["prompt"], "input": "", "output": rec.get("completion", "")}
    return {"instruction": rec.get("text", ""), "input": "", "output": ""}

def to_sharegpt(rec):
    """Convert any record to ShareGPT (conversations) format."""
    if "conversations" in rec:
        return {"conversations": rec["conversations"]}
    if "messages" in rec:
        rmap = {"user": "human", "assistant": "gpt", "system": "system"}
        return {"conversations": [{"from": rmap.get(m["role"], m["role"]),
                                    "value": m["content"]} for m in rec["messages"]]}
    alp = to_alpaca(rec)
    hv = alp["instruction"] + ("\n" + alp["input"] if alp["input"] else "")
    return {"conversations": [{"from": "human", "value": hv}, {"from": "gpt", "value": alp["output"]}]}

def to_chat(rec):
    """Convert any record to Chat (messages array) format."""
    if "messages" in rec:
        return {"messages": rec["messages"]}
    if "conversations" in rec:
        rmap = {"human": "user", "gpt": "assistant", "system": "system"}
        return {"messages": [{"role": rmap.get(c["from"], c["from"]),
                              "content": c["value"]} for c in rec["conversations"]]}
    alp = to_alpaca(rec)
    uc = alp["instruction"] + ("\n" + alp["input"] if alp["input"] else "")
    return {"messages": [{"role": "user", "content": uc}, {"role": "assistant", "content": alp["output"]}]}


CONVERTERS = {"alpaca": to_alpaca, "sharegpt": to_sharegpt, "chat": to_chat}

def validate_dataset(records, fmt):
    """Check required fields and empty strings; log aggregate issues."""
    issues = Counter()
    for rec in records:
        for field in REQUIRED_FIELDS.get(fmt, []):
            if field not in rec:
                issues[f"missing '{field}'"] += 1
            elif isinstance(rec[field], str) and not rec[field].strip():
                issues[f"empty '{field}'"] += 1
    if issues:
        logger.warning(f"Validation issues in {len(records)} records:")
        for issue, cnt in issues.most_common():
            logger.warning(f"  {issue}: {cnt}")
    else:
        logger.info("Validation passed - all records look good")

def token_length_analysis(records):
    """Print token-length statistics and identify outliers."""
    lengths = sorted(estimate_tokens(record_text(r)) for r in records)
    if not lengths:
        return
    mean_l = sum(lengths) / len(lengths)
    p95 = lengths[int(len(lengths) * 0.95)]
    logger.info(f"Token stats ({len(lengths)} records): min={lengths[0]}, max={lengths[-1]}, "
                f"mean={mean_l:.0f}, median={lengths[len(lengths)//2]}, p95={p95}")
    outlier_th = mean_l + 3 * max(mean_l, 1)
    outliers = sum(1 for l in lengths if l > outlier_th)
    if outliers:
        logger.info(f"  {outliers} outliers (> {outlier_th:.0f} est. tokens)")

def filter_by_length(records, min_tok=0, max_tok=None):
    """Filter records by estimated token length (min/max)."""
    out = [r for r in records if estimate_tokens(record_text(r)) >= min_tok
           and (max_tok is None or estimate_tokens(record_text(r)) <= max_tok)]
    if len(out) < len(records):
        logger.info(f"Filtered out {len(records) - len(out)} records by token length")
    return out

def deduplicate(records, threshold=0.95):
    """Remove exact and fuzzy duplicates (Jaccard similarity on token sets)."""
    seen_exact, unique, token_sets = set(), [], []
    for rec in records:
        text = record_text(rec).strip()
        if text in seen_exact:
            continue
        seen_exact.add(text)
        tok_set = set(text.lower().split())
        is_dup = False
        for existing in token_sets:
            inter, union = len(tok_set & existing), len(tok_set | existing)
            if union and inter / union >= threshold:
                is_dup = True; break
        if not is_dup:
            unique.append(rec); token_sets.append(tok_set)
    if len(unique) < len(records):
        logger.info(f"Deduplication removed {len(records) - len(unique)}/{len(records)} records")
    return unique

def train_val_split(records, val_ratio=0.1, seed=42):
    """Split records into train and validation sets with configurable ratio."""
    import random
    rng = random.Random(seed)
    idx = list(range(len(records)))
    rng.shuffle(idx)
    val_n = max(1, int(len(records) * val_ratio))
    val_set = set(idx[:val_n])
    train = [records[i] for i in range(len(records)) if i not in val_set]
    val = [records[i] for i in range(len(records)) if i in val_set]
    logger.info(f"Split: {len(train)} train, {len(val)} validation")
    return train, val

def write_jsonl(records, path):
    """Write records as JSONL."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(records)} records to {path}")

def preview_records(records, n=5):
    """Print first N records as a sample dataset preview."""
    print(f"\n--- Preview (first {min(n, len(records))} of {len(records)} records) ---")
    for i, rec in enumerate(records[:n]):
        print(f"\n[{i+1}] {json.dumps(rec, indent=2, ensure_ascii=False)[:600]}")
    print()

def main():
    p = argparse.ArgumentParser(description="Prepare and validate datasets for LLM fine-tuning")
    p.add_argument("--input", required=True, help="Input path (JSONL/JSON/CSV/Parquet) or HF dataset")
    p.add_argument("--output", help="Output JSONL path (omit for dry-run / preview only)")
    p.add_argument("--input-format", choices=["alpaca", "sharegpt", "chat", "csv", "completion"],
                   default="alpaca", help="Input data format (default: alpaca)")
    p.add_argument("--output-format", choices=["alpaca", "sharegpt", "chat"],
                   default="chat", help="Output format (default: chat)")
    p.add_argument("--max-tokens", type=int, default=None, help="Drop records above this token count")
    p.add_argument("--min-tokens", type=int, default=0, help="Drop records below this token count")
    p.add_argument("--val-split", type=float, default=0.0, help="Validation hold-out fraction (0=no split)")
    p.add_argument("--deduplicate", action="store_true", help="Remove exact + fuzzy duplicates")
    p.add_argument("--dedup-threshold", type=float, default=0.95, help="Jaccard threshold (default: 0.95)")
    p.add_argument("--preview", type=int, default=0, help="Print first N records and exit")
    args = p.parse_args()

    logger.info(f"Loading data from {args.input}")
    records = load_data(args.input)
    logger.info(f"Loaded {len(records)} records")
    if not records:
        logger.error("No records loaded - exiting"); sys.exit(1)
    validate_dataset(records, args.input_format)
    converter = CONVERTERS.get(args.output_format)
    if converter is None:
        logger.error(f"Unsupported output format: {args.output_format}"); sys.exit(1)
    records = [converter(r) for r in records]
    logger.info(f"Converted {len(records)} records to '{args.output_format}' format")
    token_length_analysis(records)
    if args.min_tokens or args.max_tokens:
        records = filter_by_length(records, args.min_tokens, args.max_tokens)
    if args.deduplicate:
        records = deduplicate(records, threshold=args.dedup_threshold)
    if args.preview:
        preview_records(records, args.preview); return
    if not args.output:
        logger.info("No --output specified; use --preview or provide an output path"); return
    if args.val_split > 0:
        train, val = train_val_split(records, args.val_split)
        base = Path(args.output)
        write_jsonl(train, str(base.with_stem(base.stem + "_train")))
        write_jsonl(val, str(base.with_stem(base.stem + "_val")))
    else:
        write_jsonl(records, args.output)
    logger.info("Dataset preparation complete")

if __name__ == "__main__":
    main()
