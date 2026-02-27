---
name: llm-fine-tuning
description: >
  Fine-tune large language models with parameter-efficient methods. Covers full fine-tuning, LoRA, QLoRA, PEFT,
  adapter-based tuning, prefix tuning, prompt tuning, instruction tuning, RLHF (PPO, DPO), SFT (supervised
  fine-tuning), HuggingFace TRL and PEFT libraries, Unsloth, Axolotl, dataset preparation (Alpaca, ShareGPT,
  chat formats), training data quality, evaluation during fine-tuning, hyperparameter selection, memory optimization,
  multi-GPU fine-tuning, merging adapters, and deploying fine-tuned models. Use when fine-tuning LLMs, creating
  custom models, or implementing RLHF/DPO alignment.
license: Apache-2.0
metadata:
  author: llmops-skills
  version: "1.0"
  category: llmops
---

# LLM Fine-Tuning

## Overview

Fine-tuning adapts pre-trained LLMs to specific tasks or domains. Parameter-efficient
methods (LoRA, QLoRA) make this practical on consumer hardware.

## When to Use This Skill

- Adapting an LLM for domain-specific tasks
- Instruction-tuning a base model
- Aligning a model with RLHF or DPO
- Reducing inference costs with a smaller fine-tuned model
- Building custom chat models

## Fine-Tuning Decision Tree

```
Need to customize an LLM?
├─ Small dataset (< 1000 examples)
│  └─ Use few-shot prompting or prompt tuning
├─ Medium dataset (1K-100K examples)
│  ├─ Limited GPU memory (< 24GB)
│  │  └─ QLoRA (4-bit quantized + LoRA)
│  ├─ Moderate GPU (24-80GB)
│  │  └─ LoRA
│  └─ Multiple GPUs
│     └─ Full fine-tuning or LoRA + DeepSpeed
└─ Large dataset (> 100K examples)
   └─ Full fine-tuning with distributed training
```

## Step-by-Step Instructions

### 1. LoRA Fine-Tuning with PEFT

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

# Load base model
model_name = "meta-llama/Llama-3.1-8B"
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Configure LoRA
lora_config = LoraConfig(
    r=16,                      # Rank (4-64, higher = more capacity)
    lora_alpha=32,             # Scaling factor (usually 2x rank)
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Trainable: 0.5% of total parameters

# Training
training_args = TrainingArguments(
    output_dir="./lora-output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    bf16=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    report_to="mlflow",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    max_seq_length=2048,
)

trainer.train()
trainer.save_model("./lora-adapter")
```

### 2. QLoRA (4-bit Quantized LoRA)

```python
from transformers import BitsAndBytesConfig

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

# Apply LoRA on quantized model
model = get_peft_model(model, lora_config)
# Now fine-tuning a 7B model fits in ~6GB VRAM!
```

### 3. Dataset Preparation

```python
# Chat/Instruction format (recommended)
def format_chat(example):
    """Format data for chat-style fine-tuning."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["response"]},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

# Alpaca format
def format_alpaca(example):
    if example.get("input"):
        text = f"""### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"""
    else:
        text = f"""### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"""
    return {"text": text}

# Apply formatting
dataset = dataset.map(format_chat)
```

### 4. DPO (Direct Preference Optimization)

```python
from trl import DPOTrainer, DPOConfig

# Dataset format: {"prompt": ..., "chosen": ..., "rejected": ...}
dpo_config = DPOConfig(
    output_dir="./dpo-output",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-7,
    beta=0.1,           # KL penalty coefficient
    bf16=True,
    max_length=1024,
    max_prompt_length=512,
)

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,      # Uses implicit reference with LoRA
    args=dpo_config,
    train_dataset=preference_dataset,
    processing_class=tokenizer,
    peft_config=lora_config,
)

dpo_trainer.train()
```

### 5. Merging and Deploying Adapters

```python
from peft import PeftModel

# Load base + adapter
base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base_model, "./lora-adapter")

# Merge adapter into base model
merged_model = model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("./merged-model")
tokenizer.save_pretrained("./merged-model")

# Convert to GGUF for llama.cpp (optional)
# python convert_hf_to_gguf.py ./merged-model --outtype q4_k_m
```

### 6. Training Data Quality

```python
def validate_training_data(dataset):
    """Check fine-tuning dataset quality."""
    issues = []

    # Check for empty responses
    empty = sum(1 for ex in dataset if not ex["response"].strip())
    if empty > 0:
        issues.append(f"{empty} examples have empty responses")

    # Check for duplicates
    texts = [ex["instruction"] for ex in dataset]
    dupes = len(texts) - len(set(texts))
    if dupes > 0:
        issues.append(f"{dupes} duplicate instructions")

    # Check length distribution
    lengths = [len(ex["response"].split()) for ex in dataset]
    if np.std(lengths) / np.mean(lengths) > 2:
        issues.append("High variance in response lengths")

    # Check for contamination with eval data
    # (implementation depends on eval set)

    return issues
```

## LoRA Hyperparameter Guide

| Parameter | Range | Impact |
|-----------|-------|--------|
| rank (r) | 4-64 | Higher = more capacity, more memory |
| alpha | 2x rank | Scaling factor, higher = stronger adaptation |
| target_modules | All linear | More modules = better but slower |
| dropout | 0.0-0.1 | Regularization for small datasets |
| learning_rate | 1e-5 to 5e-4 | Higher than full fine-tuning |
| epochs | 1-5 | Watch for overfitting on small data |

## Best Practices

1. **Start with QLoRA** - Test hypothesis before scaling up
2. **Curate data quality** over quantity - 1K high-quality > 10K noisy examples
3. **Use chat template** from the base model's tokenizer
4. **Evaluate during training** - Watch for overfitting
5. **Use DPO over RLHF** when possible - Simpler, more stable
6. **Merge and quantize** for deployment efficiency
7. **Track experiments** with MLflow or W&B
8. **Test on held-out data** that's distinct from training distribution
9. **Start with small rank** (r=8) and increase if needed

## Scripts

- `scripts/finetune_lora.py` - LoRA/QLoRA fine-tuning pipeline
- `scripts/prepare_dataset.py` - Dataset formatting and validation

## References

See [references/REFERENCE.md](references/REFERENCE.md) for method comparisons and guides.
