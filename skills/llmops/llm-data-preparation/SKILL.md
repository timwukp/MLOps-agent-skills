---
name: llm-data-preparation
description: >
  Prepare and curate training data for LLM fine-tuning and alignment. Covers instruction dataset creation,
  data quality assessment, synthetic data generation with LLMs, data annotation workflows (Label Studio,
  Argilla, Prodigy), preference data collection for RLHF/DPO (chosen/rejected pairs), data deduplication
  and decontamination, data formatting (Alpaca, ShareGPT, chat templates), dataset balancing and filtering,
  data augmentation for NLP, PII removal from training data, copyright and licensing considerations, and
  dataset versioning. Use when preparing data for LLM fine-tuning, generating synthetic training data,
  setting up annotation pipelines, or curating preference datasets.
license: Apache-2.0
metadata:
  author: llmops-skills
  version: "1.0"
  category: llmops
---

# LLM Data Preparation

## Overview

High-quality training data is the most important factor in LLM fine-tuning success.
This skill covers creating, curating, and validating datasets for instruction tuning,
alignment, and domain adaptation.

## When to Use This Skill

- Creating instruction-following datasets
- Generating synthetic training data
- Setting up annotation workflows
- Building preference datasets for DPO/RLHF
- Cleaning and deduplicating training data

## Data Preparation Pipeline

```
Raw Sources → Clean → Format → Annotate → Validate → Deduplicate → Split → Version
     │           │        │         │          │           │          │        │
  Documents   Remove   Alpaca/   Label     Quality    Near-dup   Train/  DVC/
  APIs        PII      Chat     Studio    Checks     Detection  Val/Test HF Hub
  Logs        Filter   Format   Argilla   LLM-judge  MinHash
```

## Step-by-Step Instructions

### 1. Synthetic Data Generation

```python
from openai import OpenAI

client = OpenAI()

def generate_instruction_data(topic, num_examples=100, model="gpt-4o"):
    """Generate synthetic instruction-response pairs."""
    examples = []

    system_prompt = f"""You are an expert data generator. Create diverse, high-quality
instruction-response pairs about {topic}. Each pair should be unique and educational.
Vary the difficulty, style, and type of instruction (explain, compare, analyze, create, etc.)."""

    for batch_start in range(0, num_examples, 10):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"""Generate 10 instruction-response pairs about {topic}.
Return as JSON array: [{{"instruction": "...", "response": "..."}}]
Make each unique in style and complexity."""},
            ],
            response_format={"type": "json_object"},
            temperature=0.9,
        )

        batch = json.loads(response.choices[0].message.content)
        examples.extend(batch.get("pairs", batch.get("data", [])))

    return examples

# Generate domain-specific data
mlops_data = generate_instruction_data("MLOps best practices", num_examples=500)
```

### 2. Preference Data for DPO

```python
def generate_preference_pairs(instructions, model="gpt-4o"):
    """Generate chosen/rejected pairs for DPO training."""
    pairs = []

    for instruction in instructions:
        # Generate a good response
        good_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Provide a detailed, accurate, helpful response."},
                {"role": "user", "content": instruction},
            ],
            temperature=0.3,
        ).choices[0].message.content

        # Generate a worse response (less helpful, vaguer)
        bad_response = client.chat.completions.create(
            model="gpt-4o-mini",  # Intentionally weaker model
            messages=[
                {"role": "system", "content": "Provide a brief response."},
                {"role": "user", "content": instruction},
            ],
            temperature=0.9,
        ).choices[0].message.content

        pairs.append({
            "prompt": instruction,
            "chosen": good_response,
            "rejected": bad_response,
        })

    return pairs
```

### 3. Data Annotation with Argilla

```python
import argilla as rg

# Initialize
rg.init(api_url="http://localhost:6900", api_key="admin.apikey")

# Create annotation dataset
dataset = rg.FeedbackDataset(
    fields=[
        rg.TextField(name="instruction"),
        rg.TextField(name="response"),
    ],
    questions=[
        rg.RatingQuestion(name="quality", values=[1, 2, 3, 4, 5],
                         description="Rate the response quality"),
        rg.TextQuestion(name="improved_response",
                       description="Provide an improved response if quality < 4"),
        rg.LabelQuestion(name="category",
                        labels=["correct", "partially_correct", "incorrect", "harmful"]),
    ],
)

# Add records
records = [
    rg.FeedbackRecord(
        fields={"instruction": ex["instruction"], "response": ex["response"]}
    )
    for ex in raw_data
]
dataset.add_records(records)
dataset.push_to_argilla(name="llm-training-data")
```

### 4. Data Quality Assessment

```python
def assess_data_quality(dataset):
    """Comprehensive quality assessment of training data."""
    report = {}

    # Length statistics
    instruction_lengths = [len(ex["instruction"].split()) for ex in dataset]
    response_lengths = [len(ex["response"].split()) for ex in dataset]
    report["length_stats"] = {
        "instruction_mean": np.mean(instruction_lengths),
        "instruction_std": np.std(instruction_lengths),
        "response_mean": np.mean(response_lengths),
        "response_std": np.std(response_lengths),
        "empty_responses": sum(1 for l in response_lengths if l == 0),
        "very_short_responses": sum(1 for l in response_lengths if l < 10),
    }

    # Duplicate detection
    instructions = [ex["instruction"] for ex in dataset]
    report["duplicates"] = {
        "exact_duplicates": len(instructions) - len(set(instructions)),
        "duplicate_ratio": 1 - len(set(instructions)) / len(instructions),
    }

    # Language detection (optional)
    # Toxicity check (optional)
    # Formatting consistency

    # LLM-based quality scoring (sample)
    sample = random.sample(dataset, min(100, len(dataset)))
    quality_scores = []
    for ex in sample:
        score = llm_quality_judge(ex["instruction"], ex["response"])
        quality_scores.append(score)

    report["quality_scores"] = {
        "mean": np.mean(quality_scores),
        "median": np.median(quality_scores),
        "below_threshold": sum(1 for s in quality_scores if s < 3) / len(quality_scores),
    }

    return report

def llm_quality_judge(instruction, response):
    """Score a training example using an LLM judge."""
    prompt = f"""Rate this instruction-response pair on quality (1-5).

Instruction: {instruction}
Response: {response}

Consider: accuracy, helpfulness, completeness, formatting.
Return just the number (1-5):"""

    result = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5,
    )
    try:
        return int(result.choices[0].message.content.strip())
    except ValueError:
        return 3
```

### 5. Deduplication

```python
from datasketch import MinHash, MinHashLSH

def deduplicate_dataset(dataset, threshold=0.8, num_perm=128):
    """Remove near-duplicate examples using MinHash LSH."""
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes = {}

    for i, example in enumerate(dataset):
        text = example["instruction"] + " " + example["response"]
        m = MinHash(num_perm=num_perm)
        for word in text.lower().split():
            m.update(word.encode("utf-8"))
        minhashes[i] = m

        try:
            lsh.insert(str(i), m)
        except ValueError:
            pass  # Duplicate found

    # Find unique examples
    seen = set()
    unique = []
    for i, example in enumerate(dataset):
        if i in seen:
            continue
        result = lsh.query(minhashes[i])
        seen.update(int(r) for r in result)
        unique.append(example)

    print(f"Removed {len(dataset) - len(unique)} near-duplicates")
    return unique
```

### 6. Dataset Formatting

```python
def format_for_chat(examples, tokenizer):
    """Format dataset for chat-style fine-tuning."""
    formatted = []
    for ex in examples:
        messages = []
        if ex.get("system"):
            messages.append({"role": "system", "content": ex["system"]})
        messages.append({"role": "user", "content": ex["instruction"]})
        messages.append({"role": "assistant", "content": ex["response"]})

        text = tokenizer.apply_chat_template(messages, tokenize=False)
        formatted.append({"text": text, "messages": messages})

    return formatted

def format_alpaca(examples):
    """Format to Alpaca format."""
    return [{
        "instruction": ex["instruction"],
        "input": ex.get("input", ""),
        "output": ex["response"],
    } for ex in examples]
```

## Best Practices

1. **Quality > Quantity** - 1K excellent examples > 100K mediocre ones
2. **Diversify instructions** - Vary style, complexity, and task type
3. **Use LLM judges** to filter low-quality examples
4. **Deduplicate aggressively** - Near-duplicates hurt generalization
5. **Remove PII** before training
6. **Version your datasets** - Track what data trained which model
7. **Balance categories** - Don't over-represent any single topic
8. **Include edge cases** - Refusals, "I don't know", multi-step reasoning
9. **Validate with human annotators** on a sample
10. **Check licensing** of source data

## Scripts

- `scripts/generate_synthetic.py` - Synthetic data generation pipeline
- `scripts/curate_dataset.py` - Data curation and quality assessment

## References

See [references/REFERENCE.md](references/REFERENCE.md) for annotation tool comparisons and format guides.
