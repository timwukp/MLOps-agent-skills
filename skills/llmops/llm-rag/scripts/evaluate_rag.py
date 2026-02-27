#!/usr/bin/env python3
"""RAG pipeline evaluation - retrieval metrics (Recall@K, Precision@K, MRR, NDCG),
generation metrics (Exact Match, F1, BLEU, ROUGE-L), faithfulness, and context relevance.

Usage:
    python evaluate_rag.py --eval-data eval.jsonl --db chroma_db/ --metrics all --output report.json
    python evaluate_rag.py --eval-data eval.jsonl --db chroma_db/ --metrics retrieval --top-k 10
"""
import argparse, json, logging, math, sys
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def load_eval_data(path):
    """Load evaluation JSONL (fields: question, ground_truth, context (optional))."""
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed JSON at line {lineno}: {e}"); continue
            if "question" not in obj:
                logger.warning(f"Line {lineno} missing 'question', skipping"); continue
            records.append(obj)
    logger.info(f"Loaded {len(records)} evaluation examples from {path}")
    return records

def retrieve_from_chromadb(question, db_path, collection_name="documents",
                           top_k=5, embedding_model="all-MiniLM-L6-v2"):
    """Query a chromadb vector store and return retrieved texts with scores."""
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error("Install: pip install chromadb sentence-transformers"); sys.exit(1)
    model = SentenceTransformer(embedding_model)
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_collection(collection_name)
    emb = model.encode(question).tolist()
    results = collection.query(query_embeddings=[emb], n_results=top_k)
    return [{"text": doc, "score": round(1 - dist, 4)}
            for doc, dist in zip(results["documents"][0], results["distances"][0])]

# --- Retrieval metrics ---

def _norm(text):
    return " ".join(text.lower().split())

def _contains(a, b):
    na, nb = _norm(a), _norm(b)
    return na in nb or nb in na

def recall_at_k(retrieved, relevant):
    """Fraction of relevant passages found in retrieved set."""
    if not relevant:
        return 0.0
    return sum(1 for r in relevant if any(_contains(r, d) for d in retrieved)) / len(relevant)

def precision_at_k(retrieved, relevant):
    """Fraction of retrieved passages that are relevant."""
    if not retrieved:
        return 0.0
    return sum(1 for d in retrieved if any(_contains(d, r) for r in relevant)) / len(retrieved)

def mrr(retrieved, relevant):
    """Mean reciprocal rank of the first relevant result."""
    for rank, doc in enumerate(retrieved, 1):
        if any(_contains(doc, r) for r in relevant):
            return 1.0 / rank
    return 0.0

def ndcg_at_k(retrieved, relevant):
    """Normalised discounted cumulative gain."""
    if not relevant:
        return 0.0
    rel = lambda d: 1.0 if any(_contains(d, r) for r in relevant) else 0.0
    dcg = sum(rel(d) / math.log2(i + 2) for i, d in enumerate(retrieved))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), len(retrieved))))
    return dcg / idcg if idcg > 0 else 0.0

def retrieval_metrics(retrieved_texts, relevant_texts):
    """Compute all retrieval metrics."""
    return {
        "recall@k": round(recall_at_k(retrieved_texts, relevant_texts), 4),
        "precision@k": round(precision_at_k(retrieved_texts, relevant_texts), 4),
        "mrr": round(mrr(retrieved_texts, relevant_texts), 4),
        "ndcg@k": round(ndcg_at_k(retrieved_texts, relevant_texts), 4),
    }

# --- Generation metrics ---

def _tok(text):
    return _norm(text).split()

def exact_match(pred, ref):
    return 1.0 if _norm(pred) == _norm(ref) else 0.0

def f1_score(pred, ref):
    """Token-level F1."""
    pc, rc = Counter(_tok(pred)), Counter(_tok(ref))
    common = sum((pc & rc).values())
    if common == 0:
        return 0.0
    p, r = common / sum(pc.values()), common / sum(rc.values())
    return round(2 * p * r / (p + r), 4)

def bleu_score(pred, ref, max_n=4):
    """BLEU with brevity penalty."""
    pt, rt = _tok(pred), _tok(ref)
    if not pt or not rt:
        return 0.0
    log_avg, orders = 0.0, 0
    for n in range(1, max_n + 1):
        p_ng = Counter(tuple(pt[i:i+n]) for i in range(len(pt) - n + 1))
        r_ng = Counter(tuple(rt[i:i+n]) for i in range(len(rt) - n + 1))
        total = sum(p_ng.values())
        if total == 0:
            continue
        prec = sum((p_ng & r_ng).values()) / total
        if prec > 0:
            log_avg += math.log(prec); orders += 1
    if orders == 0:
        return 0.0
    bp = min(1.0, math.exp(1 - len(rt) / max(len(pt), 1)))
    return round(bp * math.exp(log_avg / orders), 4)

def rouge_l(pred, ref):
    """ROUGE-L F1 via longest common subsequence."""
    pt, rt = _tok(pred), _tok(ref)
    if not pt or not rt:
        return 0.0
    m, n = len(rt), len(pt)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            curr[j] = prev[j-1] + 1 if rt[i-1] == pt[j-1] else max(prev[j], curr[j-1])
        prev = curr
    lcs = prev[n]
    if lcs == 0:
        return 0.0
    p, r = lcs / n, lcs / m
    return round(2 * p * r / (p + r), 4)

def generation_metrics(pred, ref):
    """Compute all generation metrics."""
    return {"exact_match": exact_match(pred, ref), "f1": f1_score(pred, ref),
            "bleu": bleu_score(pred, ref), "rouge_l": rouge_l(pred, ref)}

# --- Faithfulness and context relevance ---

def faithfulness_score(answer, context_texts):
    """Fraction of answer sentences supported by context (>=60% token overlap)."""
    if not answer or not context_texts:
        return 0.0
    ctx_tok = set(_tok(" ".join(context_texts)))
    sents = [s.strip() for s in answer.replace("\n", " ").split(".") if s.strip()]
    if not sents:
        return 0.0
    supported = sum(1 for s in sents for st in [_tok(s)]
                    if st and sum(1 for t in st if t in ctx_tok) / len(st) >= 0.6)
    return round(supported / len(sents), 4)

def context_relevance(question, context_texts):
    """Fraction of retrieved chunks relevant to the question (>=20% query-token overlap)."""
    if not context_texts:
        return 0.0
    qt = set(_tok(question))
    if not qt:
        return 0.0
    return round(sum(1 for c in context_texts if len(qt & set(_tok(c))) / len(qt) >= 0.2)
                 / len(context_texts), 4)

# --- End-to-end evaluation ---

def evaluate_example(ex, db_path, top_k, emb_model, do_ret, do_gen):
    """Evaluate a single example and return per-question scores."""
    question, ground_truth = ex["question"], ex.get("ground_truth", "")
    gt_ctx = ex.get("context", [])
    if isinstance(gt_ctx, str):
        gt_ctx = [gt_ctx]
    result = {"question": question}
    retrieved_texts = []
    if db_path:
        try:
            retrieved_texts = [r["text"] for r in retrieve_from_chromadb(
                question, db_path, top_k=top_k, embedding_model=emb_model)]
        except Exception as e:
            logger.warning(f"Retrieval failed: {e}")
    if do_ret and gt_ctx:
        result["retrieval"] = retrieval_metrics(retrieved_texts, gt_ctx)
    if retrieved_texts:
        result["context_relevance"] = context_relevance(question, retrieved_texts)
    prediction = ex.get("predicted_answer", ground_truth)
    if do_gen and ground_truth:
        result["generation"] = generation_metrics(prediction, ground_truth)
        if retrieved_texts:
            result["faithfulness"] = faithfulness_score(prediction, retrieved_texts)
    return result

def aggregate(results):
    """Compute mean scores across all questions."""
    agg = {}
    for group, keys in [("retrieval", ["recall@k", "precision@k", "mrr", "ndcg@k"]),
                         ("generation", ["exact_match", "f1", "bleu", "rouge_l"])]:
        scores = [r[group] for r in results if group in r]
        if scores:
            for k in keys:
                vals = [s[k] for s in scores]
                agg[f"mean_{k}"] = round(sum(vals) / len(vals), 4)
    for field in ("faithfulness", "context_relevance"):
        vals = [r[field] for r in results if field in r]
        if vals:
            agg[f"mean_{field}"] = round(sum(vals) / len(vals), 4)
    return agg

def print_report(agg, per_q, output_path=None):
    """Print and optionally save the evaluation report."""
    print("\n========== RAG Evaluation Report ==========\n")
    for key, val in sorted(agg.items()):
        print(f"  {key:30s} {val:.4f}")
    print(f"\n  Evaluated {len(per_q)} questions")
    print("============================================\n")
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump({"aggregate": agg, "per_question": per_q}, fh, indent=2, ensure_ascii=False)
        logger.info(f"Report saved to {output_path}")

def main():
    p = argparse.ArgumentParser(description="Evaluate a RAG pipeline")
    p.add_argument("--eval-data", required=True, help="Evaluation dataset (JSONL)")
    p.add_argument("--db", default=None, help="Chromadb persistent directory")
    p.add_argument("--top-k", type=int, default=5, help="Documents to retrieve (default: 5)")
    p.add_argument("--metrics", choices=["retrieval", "generation", "all"],
                   default="all", help="Metric group (default: all)")
    p.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="Embedding model")
    p.add_argument("--output", default=None, help="Save JSON report to this path")
    args = p.parse_args()
    data = load_eval_data(args.eval_data)
    if not data:
        logger.error("No evaluation data loaded - exiting"); sys.exit(1)
    do_ret = args.metrics in ("retrieval", "all")
    do_gen = args.metrics in ("generation", "all")
    per_q = []
    for i, ex in enumerate(data):
        logger.info(f"Evaluating [{i+1}/{len(data)}]: {ex['question'][:80]}")
        per_q.append(evaluate_example(ex, args.db, args.top_k, args.embedding_model, do_ret, do_gen))
    agg = aggregate(per_q)
    print_report(agg, per_q, args.output)
    logger.info("Evaluation complete")

if __name__ == "__main__":
    main()
