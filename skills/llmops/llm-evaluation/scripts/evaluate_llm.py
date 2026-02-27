#!/usr/bin/env python3
"""LLM evaluation pipeline - measure response quality with multiple metrics.

Usage:
    python evaluate_llm.py --data eval.jsonl --metrics bleu rouge exact-match --output results.json
    python evaluate_llm.py --data eval.jsonl --metrics ragas llm-judge --judge-model gpt-4o-mini --output results.json
"""
import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

AVAILABLE_METRICS = ["ragas", "llm-judge", "bleu", "rouge", "bertscore", "exact-match"]


def load_eval_data(data_path):
    """Load evaluation dataset from JSONL. Expected columns: question, answer, context, reference."""
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                s = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed line {i}: {e}")
                continue
            if "question" not in s or "answer" not in s:
                logger.warning(f"Line {i} missing 'question' or 'answer', skipping")
                continue
            samples.append(s)
    logger.info(f"Loaded {len(samples)} evaluation samples from {data_path}")
    return samples


def _aggregate(scores, key):
    """Average non-None values for a metric key."""
    valid = [s[key] for s in scores if s.get(key) is not None]
    return round(sum(valid) / len(valid), 4) if valid else None


def compute_exact_match(samples, **_kw):
    """Exact-match accuracy between answer and reference."""
    scores = []
    for s in samples:
        ref = s.get("reference", "")
        val = (1.0 if s["answer"].strip().lower() == ref.strip().lower() else 0.0) if ref else None
        scores.append({"exact_match": val})
    agg = _aggregate(scores, "exact_match")
    logger.info(f"Exact-match: {agg}")
    return scores, {"exact_match": agg}


def compute_bleu(samples, **_kw):
    """BLEU score using nltk."""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    except ImportError:
        logger.error("Install nltk: pip install nltk")
        return [{"bleu": None}] * len(samples), {"bleu": None}
    smoother = SmoothingFunction().method1
    scores = []
    for s in samples:
        ref = s.get("reference", "")
        if not ref:
            scores.append({"bleu": None}); continue
        val = sentence_bleu([ref.lower().split()], s["answer"].lower().split(), smoothing_function=smoother)
        scores.append({"bleu": round(val, 4)})
    agg = _aggregate(scores, "bleu")
    logger.info(f"BLEU: {agg}")
    return scores, {"bleu": agg}


def compute_rouge(samples, **_kw):
    """ROUGE-1, ROUGE-2, ROUGE-L scores."""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        logger.error("Install rouge-score: pip install rouge-score")
        return [{"rouge1": None}] * len(samples), {"rouge1": None}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = []
    for s in samples:
        ref = s.get("reference", "")
        if not ref:
            scores.append({"rouge1": None, "rouge2": None, "rougeL": None}); continue
        r = scorer.score(ref, s["answer"])
        scores.append({k: round(r[k].fmeasure, 4) for k in ["rouge1", "rouge2", "rougeL"]})
    agg = {k: _aggregate(scores, k) for k in ["rouge1", "rouge2", "rougeL"]}
    logger.info(f"ROUGE: {agg}")
    return scores, agg


def compute_bertscore(samples, **_kw):
    """BERTScore between answer and reference."""
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        logger.error("Install bert-score: pip install bert-score")
        return [{"bertscore_f1": None}] * len(samples), {"bertscore_f1": None}
    pairs = [(i, s["answer"], s["reference"]) for i, s in enumerate(samples) if s.get("reference")]
    if not pairs:
        return [{"bertscore_f1": None}] * len(samples), {"bertscore_f1": None}
    indices, answers, refs = zip(*pairs)
    logger.info(f"Computing BERTScore for {len(answers)} samples...")
    _P, _R, F1 = bert_score_fn(list(answers), list(refs), lang="en", verbose=False)
    scores = [{"bertscore_f1": None}] * len(samples)
    for idx, f1 in zip(indices, F1.tolist()):
        scores[idx] = {"bertscore_f1": round(f1, 4)}
    agg = _aggregate(scores, "bertscore_f1")
    logger.info(f"BERTScore F1: {agg}")
    return scores, {"bertscore_f1": agg}


def compute_ragas(samples, **_kw):
    """RAGAS metrics: faithfulness, answer relevancy, context precision, context recall."""
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset
    except ImportError:
        logger.error("Install ragas: pip install ragas datasets")
        return [{}] * len(samples), {"ragas_error": "ragas not installed"}
    rows = []
    for s in samples:
        ctx = s.get("context", "")
        rows.append({"question": s["question"], "answer": s["answer"],
                      "contexts": [ctx] if ctx else [""], "ground_truth": s.get("reference", "")})
    logger.info("Running RAGAS evaluation...")
    try:
        result = ragas_evaluate(Dataset.from_list(rows),
                                metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        return [{}] * len(samples), {"ragas_error": str(e)}
    keys = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    df = result.to_pandas()
    scores = [{k: round(float(row.get(k, 0)), 4) for k in keys} for _, row in df.iterrows()]
    agg = {k: _aggregate(scores, k) for k in keys}
    logger.info(f"RAGAS aggregate: {agg}")
    return scores, agg


def compute_llm_judge(samples, judge_model="gpt-4o-mini", **_kw):
    """LLM-as-judge: score response quality on a 1-5 scale via OpenAI or Anthropic."""
    client, provider = None, None
    try:
        from openai import OpenAI
        client, provider = OpenAI(), "openai"
    except (ImportError, Exception):
        pass
    if client is None:
        try:
            from anthropic import Anthropic
            client, provider = Anthropic(), "anthropic"
        except (ImportError, Exception):
            pass
    if client is None:
        logger.error("Install openai or anthropic: pip install openai / pip install anthropic")
        return [{"llm_judge_score": None}] * len(samples), {"llm_judge_score": None}

    sys_prompt = ("You are an evaluation judge. Score the answer on a scale of 1-5.\n"
                  "Criteria: accuracy, completeness, relevance, clarity.\n"
                  'Return ONLY JSON: {"score": <int 1-5>, "reason": "<brief>"}')
    scores = []
    for i, s in enumerate(samples):
        user_msg = f"Question: {s['question']}\n"
        if s.get("context"):
            user_msg += f"Context: {s['context']}\n"
        if s.get("reference"):
            user_msg += f"Reference: {s['reference']}\n"
        user_msg += f"Answer: {s['answer']}\n\nEvaluate."
        try:
            if provider == "openai":
                resp = client.chat.completions.create(
                    model=judge_model, temperature=0, max_tokens=200,
                    messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_msg}])
                text = resp.choices[0].message.content.strip()
            else:
                resp = client.messages.create(
                    model=judge_model, system=sys_prompt, temperature=0, max_tokens=200,
                    messages=[{"role": "user", "content": user_msg}])
                text = resp.content[0].text.strip()
            parsed = json.loads(text)
            scores.append({"llm_judge_score": int(parsed["score"]), "llm_judge_reason": parsed.get("reason", "")})
        except Exception as e:
            logger.warning(f"Judge failed on sample {i}: {e}")
            scores.append({"llm_judge_score": None, "llm_judge_reason": str(e)})
        if (i + 1) % 10 == 0:
            logger.info(f"Judged {i + 1}/{len(samples)} samples")
    agg = _aggregate(scores, "llm_judge_score")
    logger.info(f"LLM-judge avg score: {agg}/5")
    return scores, {"llm_judge_score_avg": agg}


METRIC_FUNCTIONS = {
    "exact-match": compute_exact_match, "bleu": compute_bleu, "rouge": compute_rouge,
    "bertscore": compute_bertscore, "ragas": compute_ragas, "llm-judge": compute_llm_judge,
}


def run_evaluation(samples, metrics, judge_model="gpt-4o-mini"):
    """Run selected metrics and merge per-sample and aggregate results."""
    all_per_sample = [{} for _ in samples]
    all_aggregate = {}
    for name in metrics:
        fn = METRIC_FUNCTIONS.get(name)
        if fn is None:
            logger.warning(f"Unknown metric '{name}', skipping"); continue
        logger.info(f"--- Computing: {name} ---")
        per_sample, aggregate = fn(samples, judge_model=judge_model)
        for i, sc in enumerate(per_sample):
            all_per_sample[i].update(sc)
        all_aggregate.update(aggregate)
    return all_per_sample, all_aggregate


def save_results(samples, per_sample_scores, aggregate, output_path):
    """Save evaluation results as JSON with detailed breakdown."""
    detailed = [{"question": s["question"], "answer": s["answer"],
                 "context": s.get("context", ""), "reference": s.get("reference", ""),
                 "scores": sc} for s, sc in zip(samples, per_sample_scores)]
    report = {"num_samples": len(samples), "aggregate_scores": aggregate, "per_sample_results": detailed}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {output_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="LLM evaluation pipeline")
    parser.add_argument("--data", required=True, help="Evaluation dataset (JSONL)")
    parser.add_argument("--metrics", nargs="+", required=True, choices=AVAILABLE_METRICS,
                        help="Metrics to compute")
    parser.add_argument("--judge-model", default="gpt-4o-mini", help="Model for LLM-as-judge")
    parser.add_argument("--output", default="eval_results.json", help="Output JSON path")
    args = parser.parse_args()

    if not Path(args.data).exists():
        logger.error(f"Data file not found: {args.data}"); sys.exit(1)
    samples = load_eval_data(args.data)
    if not samples:
        logger.error("No valid samples loaded"); sys.exit(1)
    per_sample, aggregate = run_evaluation(samples, args.metrics, args.judge_model)
    report = save_results(samples, per_sample, aggregate, args.output)
    print("\n=== Aggregate Scores ===")
    for k, v in report["aggregate_scores"].items():
        print(f"  {k}: {v}")
    print(f"\nDetailed results: {args.output}")


if __name__ == "__main__":
    main()
