# LLM Evaluation Reference Guide

## Evaluation Frameworks Comparison

| Feature | RAGAS | LangSmith | DeepEval | Promptfoo |
|---------|-------|-----------|----------|-----------|
| Primary Focus | RAG evaluation | Tracing + eval | General LLM eval | Prompt testing |
| LLM-as-Judge | Yes | Yes | Yes | Yes |
| Custom Metrics | Yes | Yes | Yes | Yes |
| CI/CD Integration | Limited | Yes | Yes | Excellent |
| Pricing | Open source | Paid (free tier) | Open source | Open source |
| RAG Metrics | Comprehensive | Basic | Good | Configurable |
| Dataset Management | Basic | Advanced | Good | File-based |
| Language | Python | Python/JS | Python | Node.js/YAML |
| Visualization | Basic | Excellent | Good | Web UI + CLI |
| Tracing/Observability | No | Yes (core feature) | Limited | No |

### When to Use Each

- **RAGAS**: RAG-specific evaluation with well-researched metrics. Best when your primary concern is retrieval and generation quality in a RAG pipeline.
- **LangSmith**: Full observability platform with evaluation built in. Best when you need tracing, debugging, and evaluation in one platform.
- **DeepEval**: General-purpose LLM testing with a pytest-like interface. Best for teams that want unit-test-style LLM evaluation in Python.
- **Promptfoo**: Configuration-driven prompt testing and red-teaming. Best for prompt iteration and CI/CD integration with minimal code.

## LLM-as-Judge Patterns

### Pointwise Evaluation
A single LLM scores one output on a rubric (e.g., 1-5 scale).

```
Evaluate the following response on a scale of 1-5 for helpfulness:
[Response]: {model_output}
[Rubric]:
  5 - Directly answers the question with accurate, complete information
  4 - Mostly answers the question with minor gaps
  3 - Partially answers but missing key information
  2 - Tangentially related but does not answer
  1 - Irrelevant or harmful
Score:
```

**Pros**: Simple, fast, cheap (one LLM call per evaluation).
**Cons**: Score calibration drift, positional bias in long outputs.

### Pairwise Evaluation
An LLM compares two outputs and selects the better one.

```
Compare the following two responses and determine which is better:
[Response A]: {output_a}
[Response B]: {output_b}
[Criteria]: accuracy, completeness, clarity
Which response is better? Respond with "A", "B", or "Tie" and explain.
```

**Pros**: More reliable rankings, reduces calibration issues.
**Cons**: 2x cost, position bias (mitigate by swapping order and averaging).

### Reference-Based Evaluation
An LLM compares the output against a gold-standard reference answer.

```
Given the reference answer, evaluate how well the response matches:
[Reference]: {gold_answer}
[Response]: {model_output}
Score the response on faithfulness (1-5) and completeness (1-5).
```

**Pros**: Most objective, anchored scoring.
**Cons**: Requires curated reference answers, expensive to create gold data.

### Best Practices for LLM-as-Judge
- Use the strongest available model as the judge (GPT-4o, Claude Sonnet/Opus)
- Always randomize order in pairwise comparisons to counter position bias
- Include explicit rubrics with concrete examples for each score level
- Run evaluations multiple times (n=3-5) and take the majority vote or average
- Validate LLM judge alignment with human judgments on a calibration set

## Benchmark Suites

| Benchmark | What It Measures | Tasks | Scoring | Limitations |
|-----------|-----------------|-------|---------|-------------|
| MMLU | Broad knowledge | 57 subjects, multiple choice | Accuracy (%) | Saturating for frontier models; multiple choice limits assessment |
| HumanEval | Code generation | 164 Python problems | pass@k | Python only; simple problems |
| MT-Bench | Multi-turn chat | 80 multi-turn questions, 8 categories | LLM judge 1-10 | Judge bias; limited scope |
| AlpacaEval | Instruction following | 805 instructions | Win rate vs reference | Length bias; single-turn only |
| GPQA | Graduate-level QA | Expert-level science questions | Accuracy (%) | Narrow domain focus |
| IFEval | Instruction following | Verifiable format constraints | Strict/loose accuracy | Only tests format compliance |
| BigBench-Hard | Challenging reasoning | 23 hard tasks from BIG-Bench | Accuracy (%) | Some tasks becoming saturated |
| TruthfulQA | Truthfulness | 817 questions across 38 categories | MC accuracy + generation | Static dataset; leakage risk |

### Benchmark Selection Guidelines

- **General capability**: MMLU + GPQA for knowledge, HumanEval + MBPP for code
- **Chat/assistant models**: MT-Bench + AlpacaEval + IFEval
- **Safety**: TruthfulQA + custom red-teaming suites
- **Domain-specific**: Build custom eval sets aligned to your actual use case

## RAG Evaluation Metrics

### Retrieval Metrics

| Metric | What It Measures | Range | Interpretation |
|--------|-----------------|-------|----------------|
| Context Precision | Relevant chunks ranked higher | 0-1 | Higher = better ranking of relevant chunks |
| Context Recall | All relevant info retrieved | 0-1 | Higher = fewer missed relevant chunks |
| Hit Rate (Recall@k) | At least one relevant chunk in top-k | 0-1 | Baseline retrieval effectiveness |
| MRR (Mean Reciprocal Rank) | Rank of first relevant chunk | 0-1 | Higher = relevant info appears sooner |
| NDCG | Graded relevance with position discount | 0-1 | Accounts for both relevance and ranking |

### Generation Metrics

| Metric | What It Measures | Range | Interpretation |
|--------|-----------------|-------|----------------|
| Faithfulness | Answer grounded in retrieved context | 0-1 | Higher = fewer hallucinations from context |
| Answer Relevancy | Answer addresses the question | 0-1 | Higher = more on-topic response |
| Answer Correctness | Factual accuracy vs ground truth | 0-1 | Requires gold-standard answers |
| Answer Similarity | Semantic similarity to reference | 0-1 | Softer version of correctness |

### End-to-End Metrics

| Metric | What It Measures | Range |
|--------|-----------------|-------|
| Answer + Context F1 | Combined retrieval and generation quality | 0-1 |
| Noise Robustness | Performance with irrelevant context injected | 0-1 |
| Counterfactual Robustness | Resistance to misleading context | 0-1 |

## Safety Evaluation

### Toxicity Detection Approaches

| Approach | Tool/Model | Strengths | Weaknesses |
|----------|-----------|-----------|------------|
| Classifier-based | Perspective API, OpenAI Moderation | Fast, consistent | Limited to trained categories |
| LLM-as-judge | GPT-4, Claude | Nuanced, context-aware | Expensive, variable |
| Regex/keyword | Custom rules | Fast, deterministic | Easily bypassed, high false positives |

### Bias Evaluation

- **Demographic parity**: Test outputs across protected groups for equal treatment
- **Stereotype benchmarks**: BBQ, StereoSet, CrowS-Pairs
- **Representation analysis**: Measure sentiment and association patterns across groups
- **Red-teaming**: Adversarial probing for discriminatory outputs

### Hallucination Detection

| Method | Description | Best For |
|--------|-------------|----------|
| Entailment checking | NLI model verifies claims against source | RAG faithfulness |
| Self-consistency | Sample multiple outputs and check agreement | Factual claims |
| Citation verification | Verify each cited source supports the claim | Attributed generation |
| Knowledge probing | Ask factual questions with known answers | Base model assessment |
| Claim decomposition | Break response into atomic claims and verify each | Detailed fact-checking |

## Evaluation Best Practices and Anti-Patterns

### Best Practices

1. **Build domain-specific eval sets**: Generic benchmarks rarely reflect your actual use case. Invest in curating 100-500 high-quality examples from real user queries.
2. **Version your evaluation datasets**: Track changes to eval sets alongside model and prompt changes.
3. **Use multiple metrics**: No single metric captures all quality dimensions. Combine automated metrics with human evaluation.
4. **Establish baselines**: Always compare against a baseline (previous version, competitor, or simple heuristic).
5. **Separate retrieval and generation evaluation**: In RAG systems, diagnose whether failures come from retrieval or generation.
6. **Evaluate on edge cases**: Include adversarial inputs, long contexts, ambiguous queries, and empty/null inputs.

### Anti-Patterns

1. **Evaluating only on easy examples**: This inflates quality perception. Include hard cases.
2. **Using the same model as judge and candidate**: Self-evaluation is biased. Use a stronger or different model family.
3. **Ignoring evaluation cost**: Running GPT-4 as a judge on thousands of examples is expensive. Budget for it or use cheaper proxy metrics for fast iteration.
4. **Optimizing for benchmarks over real use cases**: Benchmark scores can diverge from user satisfaction.
5. **One-time evaluation**: Evaluation should be continuous, not a one-off gate.
6. **Averaging across dissimilar tasks**: Report per-category scores, not just overall averages.

## Statistical Significance in LLM Evaluation

### Why It Matters
LLM outputs are stochastic. A 2% improvement on 50 test examples is likely noise. Statistical testing prevents shipping regressions disguised as improvements.

### Recommended Tests

| Scenario | Test | Implementation |
|----------|------|----------------|
| Comparing accuracy (binary) | McNemar's test | `statsmodels.stats.contingency_tables` |
| Comparing mean scores | Paired t-test or Wilcoxon signed-rank | `scipy.stats.ttest_rel` or `scipy.stats.wilcoxon` |
| Comparing win rates | Binomial test | `scipy.stats.binom_test` |
| Multiple model comparison | Bootstrap confidence intervals | Resample and compute intervals |

### Sample Size Guidelines

- **Minimum 200 examples** for detecting a 5% difference with 80% power
- **500+ examples** recommended for production evaluation suites
- **30+ examples per category** if reporting category-level metrics
- Use power analysis (`statsmodels.stats.power`) to determine exact requirements

### Confidence Intervals with Bootstrap

```python
import numpy as np

def bootstrap_ci(scores, n_bootstrap=10000, ci=0.95):
    bootstrapped_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        bootstrapped_means.append(np.mean(sample))
    lower = np.percentile(bootstrapped_means, (1 - ci) / 2 * 100)
    upper = np.percentile(bootstrapped_means, (1 + ci) / 2 * 100)
    return lower, upper
```

## Further Reading

- [RAGAS Documentation](https://docs.ragas.io/)
- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [DeepEval Documentation](https://docs.confident-ai.com/)
- [Promptfoo Documentation](https://www.promptfoo.dev/docs/)
- [Judging LLM-as-a-Judge (Zheng et al., 2023)](https://arxiv.org/abs/2306.05685)
- [MMLU Benchmark (Hendrycks et al., 2021)](https://arxiv.org/abs/2009.03300)
- [RAGAS: Automated Evaluation of RAG (Es et al., 2023)](https://arxiv.org/abs/2309.15217)
- [HumanEval (Chen et al., 2021)](https://arxiv.org/abs/2107.03374)
- [AlpacaEval Leaderboard](https://tatsu-lab.github.io/alpaca_eval/)
- [Holistic Evaluation of Language Models (HELM)](https://crfm.stanford.edu/helm/)
