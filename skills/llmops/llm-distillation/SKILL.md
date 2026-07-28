---
name: llm-distillation
description: >
  Distill knowledge from large teacher LLMs into small student models via sequence-level
  (response-based) knowledge distillation. Covers teacher selection and licensing checks,
  generating distillation data with Bedrock serverless teachers (DeepSeek-R1, Claude via converse),
  reasoning distillation (keeping <think>/chain-of-thought for math/code/puzzle domains, stripping
  it for simple QA), sampling and temperature choices, batch generation with retry/backoff,
  cost estimation, data curation (dedup, decontamination, verifiable-reward filtering,
  LLM-judge filtering), producing TRL-ready JSONL for SFT/QLoRA, and evaluating distillation
  quality with student-vs-teacher relative gates. Use when creating a small specialized model
  from a large model's outputs, building R1-style distilled reasoners, or preparing
  teacher-generated training data.
license: Apache-2.0
metadata:
  author: llmops-skills
  version: "1.0"
  category: llmops
---

# LLM Distillation

## Overview

Sequence-level knowledge distillation (KD) trains a small student model on the *outputs*
of a large teacher model: generate responses with the teacher, curate them, then fine-tune
the student on the curated pairs with plain SFT. No logits, no special loss — the entire
"distillation" lives in the data. This is how the official DeepSeek-R1-Distill models were
made, and it is the cheapest path to a small model that inherits a large model's behavior
on a narrow domain.

This skill owns the **data side**: teacher generation, curation, and quality gates.
The training job itself (QLoRA/SFT on SageMaker or local GPU) belongs to `llm-fine-tuning` —
this skill's deliverable is a curated JSONL that skill can consume directly.

## When to Use This Skill

- Creating a small, cheap, self-hostable model that mimics a large teacher on a specific domain
- Building an R1-style distilled reasoner for math, code, or structured-puzzle tasks
- Generating teacher-labeled training data when human labels are unavailable
- Cutting inference cost by replacing serverless frontier-model calls with a distilled student
- Deciding whether to keep or strip chain-of-thought in distillation targets

## Distillation Pipeline

```
Select Teacher → Check License → Generate → Curate → Format → [llm-fine-tuning] → Evaluate
      │              │              │          │        │            │               │
  Reasoning vs    MIT/Apache OK  Bedrock    Dedup    TRL        QLoRA SFT      Student vs
  non-reasoning   ToS may forbid converse   Decontam messages   (out of        teacher gates
  teacher         distillation   + retry    Verify   JSONL      scope here)    (RELATIVE)
                                 + backoff  Judge
```

## Step-by-Step Instructions

### 1. Teacher Selection and License Check

Pick the teacher by domain, then verify you are allowed to distill from it **before**
generating a single token:

| Teacher (Bedrock serverless) | Model / profile ID | Distillation allowed? |
|------------------------------|--------------------|------------------------|
| DeepSeek-R1 (reasoning) | `us.deepseek.r1-v1:0` | Yes — MIT license explicitly permits distillation |
| Claude (general) | `global.anthropic.claude-sonnet-5` | Check current Anthropic ToS — historically restricts training competing models |
| Llama 3.x/4 (general) | `us.meta.llama3-3-70b-instruct-v1:0` | Yes with attribution — community license requires "Built with Llama" naming for derivatives |

Rule of thumb: open-weight models with MIT/Apache-2.0 (DeepSeek, Qwen) are safe teachers;
proprietary API models often forbid using outputs to train models — read the provider ToS,
not just the marketing page. Record the teacher ID and license decision in the dataset card.

```bash
# Confirm the inference profile is available in your region (region-agnostic: uses AWS_REGION)
aws bedrock list-inference-profiles \
  --query "inferenceProfileSummaries[?contains(inferenceProfileId, 'deepseek')].inferenceProfileId"
```

### 2. Teacher Generation via Bedrock Converse

```python
import time
import boto3
from botocore.config import Config

# Region comes from AWS_REGION/profile - never hardcode
bedrock = boto3.client("bedrock-runtime",
                       config=Config(retries={"max_attempts": 0}))  # we own retry policy

TEACHER_ID = "us.deepseek.r1-v1:0"  # cross-region inference profile

def generate_teacher_response(prompt, max_retries=6):
    """One teacher call with exponential backoff on throttling."""
    for attempt in range(max_retries):
        try:
            resp = bedrock.converse(
                modelId=TEACHER_ID,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={
                    "maxTokens": 8192,      # reasoning traces are LONG; don't truncate mid-think
                    "temperature": 0.6,     # R1 recommended range 0.5-0.7; 0 causes repetition loops
                    "topP": 0.95,
                },
            )
            blocks = resp["output"]["message"]["content"]
            # R1 via Converse returns reasoning in a reasoningContent block + answer in text
            reasoning = next((b["reasoningContent"]["reasoningText"]["text"]
                              for b in blocks if "reasoningContent" in b), "")
            answer = next((b["text"] for b in blocks if "text" in b), "")
            return {"reasoning": reasoning, "answer": answer,
                    "usage": resp["usage"], "stop": resp["stopReason"]}
        except bedrock.exceptions.ThrottlingException:
            time.sleep(min(2 ** attempt, 60))   # 1,2,4,...60s
    raise RuntimeError(f"Throttled {max_retries} times - lower concurrency")
```

Generation guidance:

- **Prompt format**: give the teacher the *exact task format the student will see at
  inference* (same system prompt, same answer-format instruction, e.g. "put the final
  answer in \boxed{}"). Distillation copies the whole behavior, format included.
- **Temperature**: 0.6-0.7 for reasoning teachers (R1 loops at temperature 0);
  0.2-0.3 for deterministic QA/summarization teachers. Generate 2-4 samples per prompt
  when you have ground truth to filter against (rejection sampling) — keep only correct ones.
- **`stopReason` check**: drop any sample with `stopReason == "max_tokens"` — a truncated
  reasoning chain teaches the student to trail off.
- **Checkpoint every N samples** to JSONL so a throttling storm or spot interruption
  never loses completed generations (see `scripts/generate_teacher_data.py`).

Cost estimation per 1k samples (do this *before* launching a 100k-sample run):

```
cost_per_1k = 1000 * (avg_input_tokens * price_in + avg_output_tokens * price_out) / 1e6
# Example: DeepSeek-R1 on Bedrock ~$1.35/M in, ~$5.40/M out (verify current pricing).
# Reasoning outputs average 2k-6k tokens: 1000 * (300*1.35 + 4000*5.40)/1e6 ≈ $22 per 1k samples.
# 3 samples/prompt for rejection sampling triples that. Budget first.
```

### 3. Reasoning Mode Decision: Keep or Strip the Chain-of-Thought

The single most important design choice in sequence-level KD:

```
What is the student's task domain?
├─ Multi-step reasoning (math, code gen, ARC-style grid puzzles, planning)
│  └─ KEEP the teacher's <think>/reasoning in the training target
│     (this is exactly how R1-Distill-Qwen models were trained; the student
│      learns to reason, not just to answer)
│     Target = "<think>\n{reasoning}\n</think>\n\n{answer}"
├─ Simple QA, summarization, classification, extraction
│  └─ STRIP reasoning, train on final answers only
│     (CoT adds latency and tokens at inference with no accuracy gain on
│      tasks the student can do in one step)
└─ Mixed / unsure
   └─ Run a 200-sample pilot both ways, compare student accuracy AND
      inference token cost; reasoning targets typically cost 5-20x more
      output tokens at inference
```

Two hard constraints when keeping reasoning:

1. **Length vs student context**: if the teacher's reasoning + answer exceeds the student's
   training `max_length`, the sample is truncated and the student learns unterminated
   rambling. Filter out samples longer than ~90% of the student's context window
   (in *student-tokenizer* tokens, not teacher tokens).
2. **Consistent format**: every training target must use the same reasoning delimiters
   (`<think>...</think>`); mixed formats destroy format-validity at inference.

### 4. Data Curation

Apply in this order (each stage feeds the next — see `scripts/curate_distillation_data.py`):

1. **Format validation**: reasoning block present and closed (if reasoning mode), answer
   non-empty, `stopReason` was `end_turn`. Reject malformed samples outright.
2. **Deduplication**: exact dedup on prompt, then MinHash near-dup at threshold 0.8 on
   prompt+answer (reuse the machinery from `llm-data-preparation`).
3. **Decontamination vs eval set**: run n-gram/MinHash overlap between training prompts
   and *every* eval set you will report on. A distilled student that memorized its eval
   set is the most common fake success in KD.
4. **Correctness filtering — prefer verifiable reward**:
   - **Ground truth exists** (math answers, unit tests, puzzle solutions): exact-match /
     execution-based validation. Deterministic, free, and un-gameable — always beats an
     LLM judge when available. Keep only samples where the teacher's final answer verifies.
   - **No ground truth** (open QA, summarization): LLM-judge scoring (1-5 scale, keep >= 4).
     Use a *different* model as judge than the teacher — a teacher judging its own outputs
     inflates scores. Spot-check 50 judge decisions by hand before trusting it at scale.
5. **Train/val split**: 95/5 split *by prompt* (never let the same prompt land in both),
   stratified by task subtype if the domain is heterogeneous.

### 5. Student Training Handoff (Boundary with `llm-fine-tuning`)

**This skill produces data; `llm-fine-tuning` owns the training job.** The contract is a
JSONL in TRL messages format that `SFTTrainer` consumes without modification:

```json
{"messages": [
  {"role": "user", "content": "Solve: If 3x + 7 = 22, what is x? Put the final answer in \\boxed{}."},
  {"role": "assistant", "content": "<think>\n3x + 7 = 22, so 3x = 15, x = 5.\n</think>\n\n\\boxed{5}"}
]}
```

Handoff notes for the fine-tuning side:

- KD data is plain SFT data — QLoRA works unchanged (`llm-fine-tuning` sections 1-3).
  No KL loss, no teacher logits, no special trainer.
- Set the training `max_length` to cover the 95th-percentile sample length measured in
  the *student's* tokenizer; you already filtered outliers in step 3.
- Reasoning distillation benefits from higher LoRA rank (r=64) than plain style
  transfer (r=16) — reasoning is a capability, not a formatting tweak.
- 1 epoch is usually right for >= 50k samples; 2-3 epochs for 5-20k. Overfit shows up
  as the student parroting teacher phrasing on val prompts.

### 6. Evaluating Distillation Quality

Gates are **relative to the teacher**, never absolute — a 1.7B student will not match a
671B teacher's absolute scores, and pretending otherwise makes every distillation "fail":

| Gate | Threshold | How |
|------|-----------|-----|
| Relative solve rate | student >= 0.80 x teacher solve rate on held-out set | Same prompts, same verifier, both models |
| Format validity | >= 0.98 of student outputs parse (closed `</think>`, extractable answer) | Regex/parser over student generations |
| Decontamination proof | 0 eval prompts with >= 13-gram overlap vs training set | Re-run decontamination against final training JSONL |
| No judge-only wins | Verifiable metrics agree with judge direction | If judge says "improved" but exact-match dropped, trust exact-match |

Report both numbers ("teacher 84%, student 71%, ratio 0.85 — pass"), and evaluate at the
same sampling settings you will deploy the student with. Baseline the *un-tuned* student
too: if base student = 45% and distilled = 71%, distillation added 26 points; without the
baseline you cannot attribute the gain.

## Anti-Patterns

1. **Distilling from a forbidden teacher** — check the license/ToS first. DeepSeek-R1 (MIT)
   explicitly allows it; several proprietary providers' terms prohibit training on outputs.
   This is a legal defect in the model, not a data bug you can filter out later.
2. **Training on contaminated data** — skipping decontamination against eval sets produces
   scores that evaporate in production. Decontaminate *before* reporting anything.
3. **Reasoning chains longer than the student's context** — truncated targets teach the
   student to produce unterminated reasoning. Filter by student-tokenizer length.
4. **Judge-gaming** — filtering with the teacher as its own judge, or optimizing until the
   judge score is high while exact-match falls. Verifiable reward > judge, always.
5. **Absolute-performance gates** — demanding the student match the teacher's absolute score
   guarantees failure; gate on the relative ratio and honest baselines.
6. **Temperature 0 with reasoning teachers** — R1-class models degenerate into repetition
   loops at temperature 0; stay in the 0.5-0.7 band the model card recommends.

## Best Practices

1. **License check is step zero** — record teacher ID + license decision in the dataset card
2. **Budget before generating** — estimate cost per 1k samples; reasoning outputs are 2k-6k tokens
3. **Checkpoint generation** — resume-safe JSONL appends, never lose paid teacher tokens
4. **Rejection-sample when ground truth exists** — 2-4 generations per prompt, keep verified ones
5. **Drop truncated generations** — `stopReason == "max_tokens"` samples are poison
6. **Verifiable reward over LLM judge** whenever an exact answer or test suite exists
7. **Different model for judging** than the teacher being distilled
8. **Gate relative, report absolute** — ratio >= 0.80 with both numbers shown
9. **Baseline the un-tuned student** to attribute the distillation gain

## Scripts

- `scripts/generate_teacher_data.py` - Batch teacher generation via Bedrock converse with retry/backoff, checkpointing, and cost estimation
- `scripts/curate_distillation_data.py` - Curation pipeline: format validation, dedup, decontamination, answer verification, judge filtering, TRL-format export

## References

See [references/REFERENCE.md](references/REFERENCE.md) for teacher/license comparisons, reasoning-mode guidance, cost math, and quality-gate details.

## Related skills

**Upstream:** `llm-data-preparation` (seed prompts, dedup machinery) · **Downstream:** `llm-fine-tuning` (trains the student on this skill's JSONL), then `llm-evaluation` (student-vs-teacher gates)
**See also:** `llm-cost-optimization` — distillation is itself a cost lever (replace frontier calls with the student) · `llm-deployment` for serving the distilled student
