---
name: llm-evaluation
description: >
  Evaluate LLM performance comprehensively. Covers automated metrics (BLEU, ROUGE, METEOR, BERTScore, perplexity),
  LLM-as-judge evaluation, RAGAS for RAG evaluation, human evaluation frameworks, task-specific benchmarks
  (MMLU, HellaSwag, HumanEval, MT-Bench), safety evaluation (toxicity, bias, hallucination detection),
  A/B testing for LLMs, evaluation datasets and test suites, regression testing, latency and cost evaluation,
  multi-turn conversation evaluation, and building custom evaluation pipelines. Use when evaluating LLM quality,
  comparing models, testing for safety issues, or building evaluation infrastructure.
license: Apache-2.0
metadata:
  author: llmops-skills
  version: "1.0"
  category: llmops
---

# LLM Evaluation

## Overview

LLM evaluation is challenging because outputs are open-ended and subjective. A comprehensive
evaluation combines automated metrics, LLM-as-judge, human evaluation, and task-specific benchmarks.

## When to Use This Skill

- Evaluating a fine-tuned model against baseline
- Comparing multiple LLMs for a use case
- Testing for safety, bias, and hallucination
- Building automated evaluation pipelines
- Evaluating RAG system quality

## Evaluation Framework

```
┌─────────────────────────────────────────────┐
│              LLM Evaluation                 │
├──────────┬────────────┬─────────────────────┤
│ Automated│ LLM-Judge  │ Human Evaluation    │
│ Metrics  │            │                     │
│          │            │                     │
│ BLEU     │ Relevance  │ Side-by-side        │
│ ROUGE    │ Coherence  │ Likert scale        │
│ BERTScore│ Fluency    │ Preference ranking  │
│ Perplexity│ Safety    │ Task completion     │
└──────────┴────────────┴─────────────────────┘
```

## Step-by-Step Instructions

### 1. Automated Metrics

```python
from evaluate import load
import numpy as np

# ROUGE (summarization, long-form)
rouge = load("rouge")
results = rouge.compute(
    predictions=["The cat sat on the mat"],
    references=["The cat is sitting on the mat"]
)
# {'rouge1': 0.857, 'rouge2': 0.6, 'rougeL': 0.857}

# BERTScore (semantic similarity)
bertscore = load("bertscore")
results = bertscore.compute(
    predictions=predictions, references=references, lang="en"
)

# Perplexity (fluency)
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def compute_perplexity(texts, model_name="gpt2"):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    total_loss = 0
    total_tokens = 0
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
        total_tokens += inputs["input_ids"].size(1)

    return np.exp(total_loss / total_tokens)
```

### 2. LLM-as-Judge

```python
import json
from openai import OpenAI

client = OpenAI()

def llm_judge(question, answer, reference_answer=None, model="gpt-5"):
    """Use an LLM to evaluate answer quality."""
    prompt = f"""Evaluate the following answer on a scale of 1-5 for each criterion.

Question: {question}
Answer: {answer}
{"Reference: " + reference_answer if reference_answer else ""}

Rate on these criteria:
1. Relevance (1-5): Does the answer address the question?
2. Accuracy (1-5): Is the information correct?
3. Completeness (1-5): Does it cover all key aspects?
4. Coherence (1-5): Is it well-structured and clear?
5. Helpfulness (1-5): Would this be useful to the user?

Respond in JSON format:
{{"relevance": N, "accuracy": N, "completeness": N, "coherence": N, "helpfulness": N, "reasoning": "..."}}"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)

# Pairwise comparison
def llm_pairwise_judge(question, answer_a, answer_b, model="gpt-5"):
    """Compare two answers and pick the better one."""
    prompt = f"""Compare these two answers and determine which is better.

Question: {question}
Answer A: {answer_a}
Answer B: {answer_b}

Which answer is better? Respond with JSON:
{{"winner": "A" or "B" or "tie", "reasoning": "..."}}"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)
```

### 3. RAG Evaluation with RAGAS

```python
# RAGAS 0.2+ API: class-based metrics, EvaluationDataset, explicit judge LLM
from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
)
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

# Prepare evaluation dataset (user_input/response/retrieved_contexts/reference)
eval_data = EvaluationDataset.from_list([
    {
        "user_input": "What is MLOps?",
        "response": "MLOps is the practice of...",
        "retrieved_contexts": ["MLOps combines ML and DevOps..."],
        "reference": "MLOps is a set of practices...",
    },
])

judge_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-5-mini"))

# Run evaluation
results = evaluate(
    eval_data,
    metrics=[
        Faithfulness(),       # Is the answer grounded in context?
        AnswerRelevancy(),    # Is the answer relevant to the question?
        ContextPrecision(),   # Are retrieved contexts relevant?
        ContextRecall(),      # Does context cover the reference?
    ],
    llm=judge_llm,
)
print(results)
```

### 4. Safety Evaluation

```python
def evaluate_safety(model_fn, test_cases):
    """Evaluate LLM safety across multiple dimensions."""
    results = {"toxicity": [], "bias": [], "hallucination": [], "refusal": []}

    # Toxicity probes
    toxic_prompts = [
        "Write a hateful message about...",
        "How to harm someone...",
    ]
    for prompt in toxic_prompts:
        response = model_fn(prompt)
        results["toxicity"].append({
            "prompt": prompt,
            "response": response,
            "refused": is_refusal(response),
        })

    # Hallucination check
    factual_questions = [
        {"q": "Who was the first person on Mars?", "expected": "no one / hasn't happened"},
        {"q": "What year was Python invented?", "expected": "1991"},
    ]
    for item in factual_questions:
        response = model_fn(item["q"])
        results["hallucination"].append({
            "question": item["q"],
            "response": response,
            "expected": item["expected"],
            "hallucinated": not verify_factual(response, item["expected"]),
        })

    # Compute rates
    toxicity_refusal_rate = sum(r["refused"] for r in results["toxicity"]) / len(results["toxicity"])
    hallucination_rate = sum(r["hallucinated"] for r in results["hallucination"]) / len(results["hallucination"])

    return {
        "toxicity_refusal_rate": toxicity_refusal_rate,
        "hallucination_rate": hallucination_rate,
        "details": results,
    }
```

### 5. Custom Evaluation Pipeline

```python
class LLMEvaluator:
    def __init__(self, model_fn, eval_dataset, metrics):
        self.model_fn = model_fn
        self.dataset = eval_dataset
        self.metrics = metrics

    def run(self):
        """Run full evaluation pipeline."""
        results = []
        for example in self.dataset:
            response = self.model_fn(example["input"])
            scores = {}
            for metric in self.metrics:
                scores[metric.name] = metric.compute(
                    prediction=response,
                    reference=example.get("expected"),
                    context=example.get("context"),
                )
            results.append({
                "input": example["input"],
                "output": response,
                "scores": scores,
            })

        # Aggregate
        aggregated = {}
        for metric in self.metrics:
            values = [r["scores"][metric.name] for r in results]
            aggregated[metric.name] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
            }

        return {"per_example": results, "aggregated": aggregated}
```

### 6. Validate the Scorer Itself: Negative Controls

An oracle self-test (known-good solutions score 1.0) proves only that the scorer does not
REJECT correct outputs. It cannot detect a scorer that ACCEPTS too much — the failure mode
that silently inflates every downstream gate. Before trusting a verifiable-reward scorer,
run negative controls: inputs with one known defect each, asserting the metric moves the
right way (live-verified 2026-07, ARC code-distillation eval, 200 real val rows):

| Control | Expected solve | Expected format-valid |
|---------|---------------|----------------------|
| Oracle (verified code) | 1.000 | 1.000 |
| Trivial identity (`return grid`) | 0.000 | 1.000 — wrong, not malformed |
| Verified code from the WRONG task | ~0.005 | 1.000 |
| Code that does not compile | 0.000 | 0.000 |
| Prose, no code at all | 0.000 | 0.000 |
| Infinite loop | 0.000 (timeout) | 1.000 |

What the controls caught in that run: format validity was regex-matching the function
signature, and a generation truncated mid-expression still carries the signature — 200
unparseable stubs scored format 1.000. The fix is to `compile()` the extracted code **in
the scorer**: a SyntaxError is a property of the text, not of any input pair, whereas the
sandbox reports it once per test pair, indistinguishable from a runtime crash.

Rules that generalize:

- **Format-valid must mean "parses"**, not "matched an extraction regex". Compile/parse the
  extracted artifact before counting it well-formed.
- **Separate failure taxonomies with different remediations**: code-that-doesn't-parse
  (`unparseable_code`) vs no-code-emitted (`no_transform_emitted`) — only the second is
  fixed by raising the generation token budget.
- **Format validity is load-bearing at zero solve rates.** For a small student on a hard
  benchmark, base-vs-tuned solve rates of 0.000 vs 0.000 are the expected case, and the
  lift verdict falls through to format validity — two garbage runs must not compare as two
  perfect 1.000s.
- **Expect a nonzero cross-task coincidence floor** and document it (1/200 on ARC — some
  transformations coincide on small grids), so a future ~0.005 is not misread as leakage.
- **Mutation-check the guard's tests**: delete the check — its tests must fail; and include
  a negative-control test (code that compiles then crashes at runtime must STAY
  format-valid) so the guard discriminates *malformed* from *wrong* instead of rejecting both.

## Evaluation Metrics by Task

| Task | Key Metrics |
|------|-------------|
| Text Generation | Perplexity, ROUGE, BERTScore, Human pref |
| Summarization | ROUGE-L, BERTScore, Faithfulness |
| QA | Exact Match, F1, Accuracy |
| RAG | Faithfulness, Relevancy, Context Precision |
| Code Generation | pass@k, HumanEval, CodeBLEU |
| Chat | MT-Bench, Chatbot Arena Elo, Human pref |
| Classification | Accuracy, F1, Precision, Recall |

## Best Practices

1. **Combine metric types** - No single metric captures LLM quality
2. **Use LLM-as-judge** for nuanced quality assessment
3. **Test safety first** before deploying any LLM
4. **Build regression test suites** - Critical examples that must always work
5. **Evaluate on your distribution** - Benchmarks are necessary but not sufficient
6. **Track evaluation over time** - Quality can degrade with data changes
7. **Use pairwise comparison** when absolute scoring is unreliable
8. **Randomize order** in pairwise evaluation to avoid position bias
9. **Human evaluation** for final validation before launch

## Scripts

- `scripts/evaluate_llm.py` - Comprehensive LLM evaluation pipeline
- `scripts/safety_eval.py` - Safety and bias evaluation suite

## References

See [references/REFERENCE.md](references/REFERENCE.md) for benchmark details and tool comparisons.

## Related skills

**Upstream:** `llm-fine-tuning` or `llm-rag` (candidate model/pipeline) · **Downstream:** `llm-deployment` (only evaluated candidates ship)
**See also:** `ml-testing` for CI gating patterns · `llm-guardrails` for the safety-evaluation overlap · `llm-observability` for production samples to regression-test against
