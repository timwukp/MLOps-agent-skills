#!/usr/bin/env python3
"""LoRA/QLoRA fine-tuning pipeline for LLMs.

Usage:
    python finetune_lora.py --model meta-llama/Llama-3.1-8B --data train.jsonl --output ./lora-output
    python finetune_lora.py --model meta-llama/Llama-3.1-8B --data train.jsonl --qlora --output ./qlora-output
"""
import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_dataset(data_path, val_split=0.1):
    """Load and split dataset."""
    from datasets import load_dataset, Dataset

    if data_path.endswith(".jsonl") or data_path.endswith(".json"):
        dataset = load_dataset("json", data_files=data_path, split="train")
    elif data_path.endswith(".csv"):
        dataset = load_dataset("csv", data_files=data_path, split="train")
    elif data_path.endswith(".parquet"):
        dataset = load_dataset("parquet", data_files=data_path, split="train")
    else:
        dataset = load_dataset(data_path, split="train")

    split = dataset.train_test_split(test_size=val_split, seed=42)
    logger.info(f"Dataset: {len(split['train'])} train, {len(split['test'])} val")
    return split["train"], split["test"]


def format_dataset(dataset, tokenizer, max_seq_length=2048):
    """Format dataset for chat-style fine-tuning."""

    def format_example(example):
        messages = []
        if "system" in example and example["system"]:
            messages.append({"role": "system", "content": example["system"]})

        if "instruction" in example:
            messages.append({"role": "user", "content": example["instruction"]})
            messages.append({"role": "assistant", "content": example.get("response", example.get("output", ""))})
        elif "messages" in example:
            messages = example["messages"]
        elif "text" in example:
            return {"text": example["text"]}

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    return dataset.map(format_example, remove_columns=dataset.column_names)


def setup_model(model_name, use_qlora=False):
    """Load model with optional quantization."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"device_map": "auto", "trust_remote_code": True}

    if use_qlora:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    return model, tokenizer


def setup_lora(model, rank=16, alpha=32, target_modules=None, dropout=0.05):
    """Apply LoRA to model."""
    from peft import LoraConfig, get_peft_model, TaskType

    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"]

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, config)
    trainable, total = model.get_nb_trainable_parameters()
    logger.info(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    return model


def train(model, tokenizer, train_dataset, val_dataset, output_dir, **kwargs):
    """Run training with SFTTrainer."""
    from transformers import TrainingArguments
    from trl import SFTTrainer

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=kwargs.get("epochs", 3),
        per_device_train_batch_size=kwargs.get("batch_size", 4),
        gradient_accumulation_steps=kwargs.get("grad_accum", 4),
        learning_rate=kwargs.get("lr", 2e-4),
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=kwargs.get("eval_steps", 100),
        save_strategy="steps",
        save_steps=kwargs.get("save_steps", 100),
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        max_grad_norm=1.0,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        max_seq_length=kwargs.get("max_seq_length", 2048),
    )

    logger.info("Starting training...")
    result = trainer.train()
    logger.info(f"Training complete. Loss: {result.training_loss:.4f}")

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")

    return result


def main():
    parser = argparse.ArgumentParser(description="LoRA/QLoRA fine-tuning for LLMs")
    parser.add_argument("--model", required=True, help="Base model name or path")
    parser.add_argument("--data", required=True, help="Training data path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--qlora", action="store_true", help="Use QLoRA (4-bit quantization)")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--val-split", type=float, default=0.1)

    args = parser.parse_args()

    # Load dataset
    train_data, val_data = load_dataset(args.data, args.val_split)

    # Setup model
    model, tokenizer = setup_model(args.model, use_qlora=args.qlora)

    # Format dataset
    train_data = format_dataset(train_data, tokenizer, args.max_seq_length)
    val_data = format_dataset(val_data, tokenizer, args.max_seq_length)

    # Apply LoRA
    model = setup_lora(model, rank=args.rank, alpha=args.alpha)

    # Train
    train(model, tokenizer, train_data, val_data, args.output,
          epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
          max_seq_length=args.max_seq_length)


if __name__ == "__main__":
    main()
