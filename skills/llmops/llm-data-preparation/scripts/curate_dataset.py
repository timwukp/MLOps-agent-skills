#!/usr/bin/env python3
"""Dataset curation and quality control for LLM training data.

Usage:
    python curate_dataset.py --input data.jsonl --action stats
    python curate_dataset.py --input data.jsonl --action deduplicate --output deduped.jsonl
    python curate_dataset.py --input data.jsonl --action filter --min-length 10 --max-length 2048 --output out.jsonl
    python curate_dataset.py --input data.jsonl --action clean --output cleaned.jsonl
    python curate_dataset.py --input data.jsonl --action split --val-ratio 0.1 --test-ratio 0.1 --output splits/
"""
import argparse, csv, hashlib, json, logging, random, re, sys
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PII_PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    "phone_us": re.compile(r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "ip_address": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
}


def load_data(input_path):
    """Load dataset from JSONL, JSON, CSV, Parquet, or HuggingFace datasets."""
    path = Path(input_path)
    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else data.get("data", [data])
    if path.suffix == ".csv":
        with open(path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    if path.suffix == ".parquet":
        try:
            import pandas as pd
            return pd.read_parquet(path).to_dict(orient="records")
        except ImportError:
            logger.error("Install: pip install pandas pyarrow"); sys.exit(1)
    try:
        from datasets import load_dataset
        return [dict(row) for row in load_dataset(input_path, split="train")]
    except Exception as e:
        logger.error(f"Cannot load '{input_path}': {e}"); sys.exit(1)


def write_jsonl(records, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(records)} records to {path}")


def _text_of(rec):
    """Extract main text from a record."""
    if "instruction" in rec:
        return rec["instruction"] + " " + rec.get("response", rec.get("output", ""))
    if "messages" in rec:
        return " ".join(m.get("content", "") for m in rec["messages"])
    return rec.get("text", json.dumps(rec, sort_keys=True))


def _minhash_sig(text, num_perm=128):
    """Compute MinHash signature for near-duplicate detection."""
    words = text.lower().split()
    shingles = {" ".join(words[i:i+3]) for i in range(len(words)-2)} or {text.lower().strip()}
    return [min(int(hashlib.md5(f"{i}:{s}".encode()).hexdigest(), 16) & 0xFFFFFFFF
                for s in shingles) for i in range(num_perm)]


def deduplicate(records, threshold=0.8):
    """Remove exact and near-duplicate records using hashing and MinHash."""
    seen, exact_unique = set(), []
    for r in records:
        h = hashlib.sha256(_text_of(r).strip().lower().encode()).hexdigest()
        if h not in seen:
            seen.add(h); exact_unique.append(r)
    logger.info(f"Exact dedup: {len(records)} -> {len(exact_unique)}")
    sigs, unique = [], []
    for r in exact_unique:
        sig = _minhash_sig(_text_of(r))
        if not any(sum(a == b for a, b in zip(sig, ps)) / len(sig) >= threshold for ps in sigs):
            sigs.append(sig); unique.append(r)
    logger.info(f"Near-dup removal: {len(exact_unique)} -> {len(unique)}")
    return unique


def score_quality(rec):
    """Compute quality scores: instruction/response length, repetition rate, formatting."""
    instr = rec.get("instruction", "")
    resp = rec.get("response", rec.get("output", ""))
    if not instr and not resp:
        t = _text_of(rec); instr, resp = t[:len(t)//2], t[len(t)//2:]
    iw, rw = len(instr.split()), len(resp.split())
    # Perplexity proxy: repeated bigram rate
    words = resp.lower().split()
    if len(words) >= 3:
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        counts = Counter(bigrams)
        rep = sum(c-1 for c in counts.values() if c > 1) / len(bigrams)
    else:
        rep = 0.0
    # Formatting quality
    fmt = 0.5
    if any(m in resp for m in ["```", "- ", "1.", "* ", "## "]): fmt += 0.2
    if len(resp.split("\n")) > 1: fmt += 0.15
    if resp.strip() and resp.strip()[-1] in ".!?)\"'": fmt += 0.15
    fmt = min(fmt, 1.0)
    overall = min(iw/15, 1)*0.2 + min(rw/80, 1)*0.3 + max(0, 1-rep*5)*0.25 + fmt*0.25
    return {"instruction_words": iw, "response_words": rw, "repetition_rate": round(rep, 4),
            "formatting_score": round(fmt, 2), "quality_score": round(overall, 4)}


def detect_language(text):
    """Detect language using langdetect or ASCII heuristic fallback."""
    try:
        from langdetect import detect
        return detect(text[:500])
    except ImportError:
        if not text: return "unknown"
        return "en" if sum(c.isascii() and c.isalpha() for c in text) / max(len(text), 1) > 0.5 else "non-en"
    except Exception:
        return "unknown"


def detect_pii(text):
    return {t: len(p.findall(text)) for t, p in PII_PATTERNS.items() if p.findall(text)}


def remove_pii(text):
    for pii_type, pattern in PII_PATTERNS.items():
        text = pattern.sub(f"[{pii_type.upper()}_REDACTED]", text)
    return text


def filter_records(records, min_length=0, max_length=999999, lang=None):
    """Filter records by word count and optional language."""
    kept = []
    for r in records:
        wc = len(_text_of(r).split())
        if wc < min_length or wc > max_length: continue
        if lang and detect_language(_text_of(r)) != lang: continue
        kept.append(r)
    logger.info(f"Filtered: {len(records)} -> {len(kept)}")
    return kept


def split_dataset(records, val_ratio=0.1, test_ratio=0.1, stratify_key=None, seed=42):
    """Split into train/val/test with optional stratification."""
    random.seed(seed)
    if stratify_key and all(stratify_key in r for r in records):
        groups = {}
        for r in records:
            groups.setdefault(r[stratify_key], []).append(r)
        train, val, test = [], [], []
        for grp in groups.values():
            random.shuffle(grp)
            nt, nv = max(1, int(len(grp)*test_ratio)), max(1, int(len(grp)*val_ratio))
            test.extend(grp[:nt]); val.extend(grp[nt:nt+nv]); train.extend(grp[nt+nv:])
    else:
        s = list(records); random.shuffle(s)
        nt, nv = int(len(s)*test_ratio), int(len(s)*val_ratio)
        test, val, train = s[:nt], s[nt:nt+nv], s[nt+nv:]
    logger.info(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


def compute_stats(records):
    """Compute and print dataset statistics report."""
    if not records:
        logger.warning("No records"); return {}
    wc = [len(_text_of(r).split()) for r in records]
    il = [len(r["instruction"].split()) for r in records if "instruction" in r]
    rl = [len(r.get("response", r.get("output","")).split()) for r in records if "response" in r or "output" in r]
    qv = [score_quality(r)["quality_score"] for r in records]
    lk = next((k for k in ("label","category","topic","class") if any(k in r for r in records)), None)
    cd = dict(Counter(r.get(lk,"N/A") for r in records)) if lk else "N/A"
    sample = random.sample(records, min(200, len(records)))
    pii_n = sum(1 for r in sample if detect_pii(_text_of(r)))
    stats = {
        "total_records": len(records),
        "word_count": {"min": min(wc), "max": max(wc), "mean": round(sum(wc)/len(wc),1)},
        "instruction_words": {"min": min(il, default=0), "max": max(il, default=0),
                              "mean": round(sum(il)/max(len(il),1),1)},
        "response_words": {"min": min(rl, default=0), "max": max(rl, default=0),
                           "mean": round(sum(rl)/max(len(rl),1),1)},
        "quality": {"min": round(min(qv),4), "max": round(max(qv),4), "mean": round(sum(qv)/len(qv),4)},
        "pii_estimate_pct": round(100*pii_n/max(len(sample),1),1),
        "class_distribution": cd,
    }
    print("\n===== Dataset Statistics =====")
    for k, v in stats.items(): print(f"  {k}: {v}")
    print("==============================\n")
    return stats


def clean_records(records, min_quality=0.3):
    """Remove PII and drop low-quality records."""
    cleaned, pii_ct, low_q = [], 0, 0
    for r in records:
        for fld in ("instruction", "response", "output", "text", "input"):
            if fld in r and isinstance(r[fld], str):
                if detect_pii(r[fld]): pii_ct += 1
                r[fld] = remove_pii(r[fld])
        if "messages" in r and isinstance(r["messages"], list):
            for msg in r["messages"]:
                if "content" in msg:
                    if detect_pii(msg["content"]): pii_ct += 1
                    msg["content"] = remove_pii(msg["content"])
        if score_quality(r)["quality_score"] < min_quality:
            low_q += 1; continue
        cleaned.append(r)
    logger.info(f"PII redacted: {pii_ct} fields, dropped {low_q} low-quality, kept {len(cleaned)}/{len(records)}")
    return cleaned


def main():
    p = argparse.ArgumentParser(description="Dataset curation and quality control for LLM training")
    p.add_argument("--input", required=True, help="Input file (JSONL/CSV/Parquet) or HuggingFace dataset")
    p.add_argument("--action", required=True, choices=["deduplicate","filter","split","stats","clean"])
    p.add_argument("--output", help="Output file or directory")
    p.add_argument("--min-length", type=int, default=0, help="Min word count for filter")
    p.add_argument("--max-length", type=int, default=999999, help="Max word count for filter")
    p.add_argument("--lang", help="Keep only this language (e.g. 'en')")
    p.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    p.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio")
    p.add_argument("--stratify-key", help="Key to stratify splits on")
    p.add_argument("--dedup-threshold", type=float, default=0.8, help="MinHash Jaccard threshold")
    p.add_argument("--min-quality", type=float, default=0.3, help="Min quality score for clean")
    args = p.parse_args()
    records = load_data(args.input)
    logger.info(f"Loaded {len(records)} records")
    if args.action == "stats":
        compute_stats(records)
    elif args.action == "deduplicate":
        if not args.output: logger.error("--output required"); sys.exit(1)
        write_jsonl(deduplicate(records, args.dedup_threshold), args.output)
    elif args.action == "filter":
        if not args.output: logger.error("--output required"); sys.exit(1)
        write_jsonl(filter_records(records, args.min_length, args.max_length, args.lang), args.output)
    elif args.action == "split":
        if not args.output: logger.error("--output dir required"); sys.exit(1)
        train, val, test = split_dataset(records, args.val_ratio, args.test_ratio, args.stratify_key)
        out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
        write_jsonl(train, out/"train.jsonl"); write_jsonl(val, out/"val.jsonl"); write_jsonl(test, out/"test.jsonl")
    elif args.action == "clean":
        if not args.output: logger.error("--output required"); sys.exit(1)
        write_jsonl(clean_records(records, args.min_quality), args.output)
    logger.info("Done.")


if __name__ == "__main__":
    main()
