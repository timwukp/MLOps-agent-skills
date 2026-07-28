# LLM Distillation Reference Guide

## Distillation Approaches Compared

| Approach | What transfers | Needs teacher logits? | Infra cost | Best for |
|----------|---------------|----------------------|------------|----------|
| Sequence-level KD (this skill) | Output behavior (responses, reasoning traces) | No — API access is enough | Low (API calls + SFT) | Distilling from serverless/API teachers; R1-style reasoners |
| Logit distillation (soft labels) | Full output distribution per token | Yes — white-box teacher | High (teacher forward passes during training) | Same-tokenizer teacher/student pairs, on-prem teachers |
| Feature/hidden-state distillation | Internal representations | Yes — architecture access | High | Research settings, encoder models |
| On-policy KD (e.g. GKD) | Distribution on *student-generated* sequences | Yes | Highest | Closing exposure-bias gap when logits are available |

Sequence-level KD is the only option when the teacher is behind an API (Bedrock serverless),
and it is what produced the official DeepSeek-R1-Distill-Qwen/Llama models: R1 generated
~800k curated samples; students were trained with plain SFT on them — no RL, no logits.

## Teacher Licensing Matrix

Verify at generation time — terms change. State of play as of mid-2026:

| Teacher | License / Terms | Distillation of outputs |
|---------|----------------|------------------------|
| DeepSeek-R1 / V3 | MIT | Explicitly allowed, including commercial use |
| Qwen 3 family | Apache-2.0 | Allowed |
| Llama 3.x / 4 | Meta community license | Allowed; derivative models must display "Built with Llama" and comply with the acceptable-use policy |
| Mistral (open-weight) | Apache-2.0 | Allowed |
| OpenAI API models | Terms of Use | Prohibits using outputs "to develop models that compete with OpenAI" |
| Anthropic API models | Usage terms | Historically restricts training competing models on outputs — check current terms |
| Google Gemini API | Terms | Restricts using outputs for ML model training in competitive contexts — check current terms |

Practical rule: open-weight MIT/Apache teachers are unambiguous; anything served only
behind a proprietary API needs a legal read of the current ToS. Record the decision
(teacher ID, license, date checked) in the dataset card next to the data.

## Bedrock Teacher Access Patterns

### Converse API (preferred — uniform across models)

```python
import boto3
bedrock = boto3.client("bedrock-runtime")  # region from environment

resp = bedrock.converse(
    modelId="us.deepseek.r1-v1:0",
    messages=[{"role": "user", "content": [{"text": prompt}]}],
    inferenceConfig={"maxTokens": 8192, "temperature": 0.6, "topP": 0.95},
)
```

DeepSeek-R1 on Bedrock is served **only through cross-region inference profiles**
(`us.deepseek.r1-v1:0`); the bare model ID will not resolve for on-demand invocation.
The Converse response separates the reasoning trace (`reasoningContent` block) from
the final answer (`text` block), so you never have to regex `<think>` tags out of raw text.

### CLI spot check

```bash
aws bedrock-runtime converse \
  --model-id us.deepseek.r1-v1:0 \
  --messages '[{"role":"user","content":[{"text":"What is 17*23? Put the final answer in \\boxed{}."}]}]' \
  --inference-config '{"maxTokens":4096,"temperature":0.6}' \
  --query '{answer: output.message.content, usage: usage}'
```

### Throughput planning

On-demand Bedrock enforces account-level RPM/TPM quotas per model. For a 100k-sample run:

- Measure your effective quota with a 100-call probe before launching.
- Run 4-8 concurrent workers with exponential backoff (1s -> 60s cap) on
  `ThrottlingException`; more workers past the quota just burns retries.
- Consider Bedrock **batch inference** (async jobs, ~50% of on-demand token price) for
  non-urgent generation at scale — same JSONL-in/JSONL-out shape as this pipeline.

## Sampling Settings by Teacher Type

| Teacher type | Temperature | top_p | Samples/prompt | Notes |
|--------------|------------|-------|----------------|-------|
| Reasoning (R1-class) | 0.5-0.7 (0.6 default) | 0.95 | 2-4 with verification, 1 without | Temperature 0 causes repetition loops; model card explicitly warns against it |
| General instruct (QA/summarization) | 0.2-0.3 | 0.9 | 1 | Deterministic tasks want low-variance targets |
| Diverse style transfer | 0.7-0.9 | 0.95 | 1-2 | Higher variance is the point |

Rejection sampling (generate k, keep verified-correct) is the highest-leverage quality
knob when ground truth exists: it converts teacher compute into data quality. The R1
report used exactly this — generation + rule-based filtering — to build its SFT set.

## Cost Model

```
tokens_per_sample = input_tokens + expected_output_tokens
cost_per_1k_samples = 1000 * (in_tok * price_in + out_tok * price_out) / 1e6
total = cost_per_1k * (num_prompts / 1000) * samples_per_prompt
```

Worked example (verify current pricing before relying on it):

| Item | Value |
|------|-------|
| DeepSeek-R1 on Bedrock (mid-2026) | ~$1.35 / 1M input, ~$5.40 / 1M output |
| Avg input (prompt) | 300 tokens |
| Avg output (reasoning + answer) | 4,000 tokens |
| Cost per 1k samples, 1 sample/prompt | 1000 x (300x1.35 + 4000x5.40)/1e6 ≈ **$22** |
| 50k prompts x 3 samples (rejection sampling) | ≈ **$3,300** |

Reasoning traces dominate cost (output tokens are 4x the price and 10x the volume of
input). Stripping reasoning at *curation* time does not refund generation cost — if you
know the domain doesn't need CoT, use a non-reasoning teacher and pay ~10x less per sample.

## Keep vs Strip Reasoning: Decision Detail

| Signal | Keep `<think>` | Strip to final answer |
|--------|---------------|----------------------|
| Task needs multi-step derivation (math proofs, code synthesis, grid puzzles) | Yes | — |
| Teacher accuracy collapses when forced to answer directly | Yes | — |
| Single-hop lookup / extraction / classification | — | Yes |
| Inference latency/cost budget is tight | — | Yes (CoT = 5-20x output tokens) |
| Student <= 2B params | Pilot both | Small students sometimes imitate CoT *form* without gains — measure |

When keeping reasoning, enforce one canonical serialization in every training target:

```
<think>
{teacher reasoning trace}
</think>

{final answer}
```

and verify at curation time that (a) the close tag is present, (b) total length in
**student-tokenizer tokens** is <= 0.9 x student max_length, (c) the extractable final
answer sits *outside* the think block.

## Curation Order and Why It Matters

1. **Format validation** first — cheap, removes garbage before expensive stages.
2. **Exact + near dedup** — duplicated teacher outputs overweight easy prompts.
3. **Decontamination** — 13-gram overlap or MinHash >= 0.8 between training prompts and
   every reported eval set (same machinery as `llm-data-preparation`; see that skill's
   REFERENCE.md for a worked example). Do this before correctness filtering so you never
   waste verifier/judge budget on samples that must be dropped anyway.
4. **Correctness filter** — verifiable reward (exact match, unit tests) when possible;
   LLM judge (different model than the teacher, threshold >= 4/5, hand-audit 50) otherwise.
5. **Split by prompt** — 95/5, no prompt appears in both train and val.

## Quality Gates for the Distilled Student

| Gate | Threshold | Failure usually means |
|------|-----------|----------------------|
| Relative solve rate (student/teacher) | >= 0.80 | Too little data, domain too broad, or reasoning stripped when it was needed |
| Format-validity rate | >= 0.98 | Mixed reasoning formats in training data, or max_length truncation during SFT |
| Gain over un-tuned student | Positive and material | Distillation added nothing — check data quality before blaming size |
| Decontamination re-check on final JSONL | 0 overlaps | Contamination slipped in via augmentation/merge steps |

Honest-expectations table (typical, math/reasoning domains, order-of-magnitude only):

| Student size | Realistic relative solve rate vs 600B-class teacher |
|--------------|------------------------------------------------------|
| 1.5-2B | 0.5-0.8 on narrow domains; below 0.5 on broad ones |
| 7-8B | 0.8-0.95 on narrow domains |
| 32B+ | 0.9-1.0; can match teacher on-domain |

If a 1.7B student must hit 0.95x a frontier teacher across a broad domain, the fix is
scope reduction or a bigger student — not more epochs.

## Common Failures and Solutions

### Student outputs unterminated reasoning
- **Cause**: training targets truncated by max_length, or `max_tokens`-stopped teacher samples kept
- **Solution**: filter by student-tokenizer length at curation; drop `stopReason != "end_turn"`

### Student mimics reasoning style but accuracy flat
- **Cause**: student too small to use the reasoning, or uncorrected teacher errors in data
- **Solution**: add verifiable-reward filtering; pilot strip-reasoning mode; consider larger student

### Eval scores great, production poor
- **Cause**: contamination between teacher data and eval set
- **Solution**: re-run decontamination on the final JSONL; rebuild eval from post-cutoff data

### Judge-filtered data looks clean, exact-match falls
- **Cause**: judge-gaming — judge rewards fluency/length, not correctness
- **Solution**: verifiable reward wherever ground truth exists; audit judge against 50 hand labels

### Generation run dies at 60% and restarts from zero
- **Cause**: no checkpointing; paid tokens lost
- **Solution**: append-mode JSONL with prompt-ID resume (see `generate_teacher_data.py`)

## Further Reading

- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL (DeepSeek-AI, 2025)](https://arxiv.org/abs/2501.12948) — section on distilled models
- [Sequence-Level Knowledge Distillation (Kim & Rush, 2016)](https://arxiv.org/abs/1606.07947)
- [Distilling the Knowledge in a Neural Network (Hinton et al., 2015)](https://arxiv.org/abs/1503.02531)
- [On-Policy Distillation of Language Models / GKD (Agarwal et al., 2023)](https://arxiv.org/abs/2306.13649)
- [Amazon Bedrock Converse API documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html)
- [Amazon Bedrock Model Distillation](https://docs.aws.amazon.com/bedrock/latest/userguide/model-distillation.html) — managed alternative when teacher and student are both Bedrock models
- [Hugging Face TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) — consumes this skill's messages JSONL
- [DeepSeek-R1 model card (usage recommendations)](https://huggingface.co/deepseek-ai/DeepSeek-R1)
