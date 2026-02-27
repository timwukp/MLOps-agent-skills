# LLM Fine-Tuning Reference Guide

## Fine-Tuning Methods Comparison

| Method | Trainable Params | Memory Usage | Quality | Training Speed | Best For |
|--------|-----------------|--------------|---------|----------------|----------|
| Full Fine-Tuning | 100% | Very High | Highest | Slowest | Unlimited compute, max quality |
| LoRA | 0.1-1% | Low | High | Fast | General PEFT, production use |
| QLoRA | 0.1-1% | Very Low | High | Moderate | Limited GPU memory |
| DoRA | 0.1-1% | Low | Higher than LoRA | Moderate | Improved directional adaptation |
| Prefix Tuning | <0.1% | Very Low | Moderate | Very Fast | Task-specific prompting |
| Adapter Layers | 1-5% | Low-Moderate | High | Moderate | Multi-task learning |

### Method Details

**Full Fine-Tuning**: Updates all model weights. Requires multiple GPUs for models above 7B. Risk of catastrophic forgetting is highest. Use when you have abundant compute and need maximum task performance.

**LoRA (Low-Rank Adaptation)**: Injects trainable low-rank decomposition matrices into attention layers. The original weights remain frozen. Merging adapters back into the base model incurs zero inference latency overhead.

**QLoRA**: Combines 4-bit NormalFloat quantization of the base model with LoRA adapters trained in BF16. Enables fine-tuning 65B models on a single 48GB GPU. Slight quality loss compared to LoRA due to quantization noise.

**DoRA (Weight-Decomposed Low-Rank Adaptation)**: Decomposes pre-trained weights into magnitude and direction components, applying LoRA only to the directional component. Consistently outperforms LoRA at the same rank across multiple benchmarks.

**Prefix Tuning**: Prepends trainable continuous vectors to the key and value matrices at every transformer layer. Extremely parameter-efficient but less expressive than LoRA.

**Adapter Layers**: Inserts small bottleneck layers between existing transformer blocks. Good for multi-task scenarios where different adapters serve different tasks.

## PEFT Parameter Guide

### LoRA / QLoRA Hyperparameters

| Parameter | Typical Range | Recommendation | Impact |
|-----------|--------------|----------------|--------|
| Rank (r) | 4-256 | 16-64 for most tasks | Higher = more capacity, more memory |
| Alpha | 8-512 | 2x rank (e.g., r=32, alpha=64) | Scaling factor; higher alpha = stronger adaptation |
| Target Modules | varies | q_proj, v_proj minimum; add k_proj, o_proj, gate_proj, up_proj, down_proj for more capacity | More modules = better quality, more memory |
| Dropout | 0.0-0.1 | 0.05 | Regularization; increase if overfitting |
| Bias | none/all/lora_only | "none" | Training bias terms; "none" is most stable |

### Choosing Rank

- **r=8**: Simple classification, sentiment tasks
- **r=16-32**: Instruction following, chat format adaptation
- **r=64**: Complex reasoning, code generation, domain-specific knowledge
- **r=128-256**: Approaching full fine-tuning capacity; diminishing returns beyond 128

### Target Module Selection by Architecture

```
# LLaMA / Mistral
target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# GPT-NeoX / Pythia
target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]

# Phi-2/3
target_modules = ["q_proj", "v_proj", "k_proj", "dense", "fc1", "fc2"]
```

## Training Data Format Comparison

| Format | Structure | Ecosystem | Best For |
|--------|-----------|-----------|----------|
| Alpaca | instruction/input/output JSON | Widely supported | Single-turn instruction tuning |
| ShareGPT | conversations array with from/value | Axolotl, FastChat | Multi-turn chat fine-tuning |
| ChatML | `<\|im_start\|>role\n content<\|im_end\|>` | OpenAI, Mistral | Production chat models |
| Completion | Plain text with EOS tokens | All frameworks | Continued pre-training, base models |

### Format Examples

**Alpaca**:
```json
{"instruction": "Summarize the text", "input": "Long article...", "output": "Summary..."}
```

**ShareGPT**:
```json
{"conversations": [
  {"from": "human", "value": "Explain quantum computing"},
  {"from": "gpt", "value": "Quantum computing uses..."}
]}
```

**ChatML**:
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Explain quantum computing<|im_end|>
<|im_start|>assistant
Quantum computing uses...<|im_end|>
```

## Hardware Requirements

### LoRA Fine-Tuning (BF16 base + BF16 adapters)

| Model Size | GPU Memory Required | Recommended GPUs | Batch Size (per GPU) |
|-----------|--------------------|--------------------|---------------------|
| 7B | 18-24 GB | 1x A100 40GB, 1x RTX 4090 | 4-8 |
| 13B | 32-40 GB | 1x A100 40GB | 2-4 |
| 70B | 160+ GB | 4x A100 80GB | 1-2 |

### QLoRA Fine-Tuning (4-bit base + BF16 adapters)

| Model Size | GPU Memory Required | Recommended GPUs | Batch Size (per GPU) |
|-----------|--------------------|--------------------|---------------------|
| 7B | 6-10 GB | 1x RTX 3090/4090 | 4-8 |
| 13B | 12-18 GB | 1x RTX 4090, 1x A100 40GB | 2-4 |
| 70B | 40-48 GB | 1x A100 80GB | 1-2 |

Note: Gradient checkpointing reduces memory by approximately 30% at the cost of 20-25% slower training.

## Training Hyperparameters Guide

| Parameter | Recommended Range | Notes |
|-----------|------------------|-------|
| Learning Rate | 1e-5 to 2e-4 | Use 2e-4 for LoRA/QLoRA; 1e-5 to 5e-5 for full fine-tuning |
| LR Scheduler | cosine | Cosine with warmup is standard; constant with warmup also works |
| Warmup Ratio | 0.03-0.1 | 3-10% of total steps |
| Batch Size (effective) | 32-128 | Use gradient accumulation to reach target |
| Epochs | 1-3 | More than 3 epochs risks overfitting; 1 epoch often sufficient for large datasets |
| Max Sequence Length | 512-4096 | Match your use case; longer = more memory |
| Weight Decay | 0.0-0.01 | Light regularization; 0.0 is common for LoRA |
| Gradient Clipping | 1.0 | Standard max norm clipping |
| BF16/FP16 | BF16 preferred | FP16 can cause instability with some architectures |

## DPO vs RLHF vs SFT Comparison

| Aspect | SFT | RLHF | DPO |
|--------|-----|------|-----|
| Training Data | Demonstrations | Preference pairs + reward model | Preference pairs |
| Complexity | Low | Very High | Moderate |
| Compute Cost | Low | High (reward model + PPO) | Moderate |
| Stability | Stable | Prone to reward hacking | Generally stable |
| Alignment Quality | Baseline | Strong | Strong |
| Components Needed | 1 model | 4 models (actor, critic, reward, ref) | 2 models (policy, ref) |

### Training Pipeline Order

1. **SFT first**: Always start with supervised fine-tuning to teach format and basic task capability
2. **Preference alignment second**: Apply DPO or RLHF on the SFT checkpoint to align with human preferences
3. **DPO is preferred** for most teams due to simplicity; RLHF if you have the infrastructure and expertise

### DPO Hyperparameters

- **Beta**: 0.1 (standard), 0.05 (more aggressive optimization), 0.5 (more conservative)
- **Loss type**: sigmoid (standard DPO), hinge (robust to noisy preferences), IPO (bounded optimization)

## Common Fine-Tuning Failures and Solutions

### Loss Not Decreasing
- **Cause**: Learning rate too low, data formatting issues, or incorrect padding
- **Solution**: Verify data tokenization manually. Check that labels are not masked where they should be trained. Increase learning rate by 2-5x.

### Loss Spikes or NaN
- **Cause**: Learning rate too high, FP16 overflow, or corrupted data samples
- **Solution**: Switch to BF16. Reduce learning rate. Add gradient clipping at 1.0. Inspect data for extremely long or malformed examples.

### Overfitting (Train loss drops, eval loss increases)
- **Cause**: Too many epochs, dataset too small, or rank too high
- **Solution**: Reduce epochs to 1-2. Increase dropout to 0.1. Lower LoRA rank. Add more diverse training data.

### Catastrophic Forgetting
- **Cause**: Learning rate too high or training for too many steps on narrow data
- **Solution**: Lower learning rate. Use LoRA instead of full fine-tuning. Mix in general instruction data (5-10% of training set).

### Model Outputs Gibberish After Fine-Tuning
- **Cause**: Mismatched chat template, wrong padding token, or tokenizer issues
- **Solution**: Ensure pad_token is set (use eos_token if no dedicated pad token). Verify the chat template matches inference. Check that special tokens are not trained on unintentionally.

### Poor Multi-Turn Performance
- **Cause**: Training only on single-turn data or masking conversation history
- **Solution**: Use multi-turn format (ShareGPT/ChatML). Ensure loss is computed on all assistant turns, not just the last one.

## Further Reading

- [Hugging Face PEFT Documentation](https://huggingface.co/docs/peft)
- [LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314)
- [DoRA: Weight-Decomposed Low-Rank Adaptation (Liu et al., 2024)](https://arxiv.org/abs/2402.09353)
- [DPO: Direct Preference Optimization (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290)
- [Axolotl Fine-Tuning Framework](https://github.com/OpenAccess-AI-Collective/axolotl)
- [Unsloth: 2x Faster Fine-Tuning](https://github.com/unslothai/unsloth)
- [LLaMA-Factory: Unified Fine-Tuning Framework](https://github.com/hiyouga/LLaMA-Factory)
- [Hugging Face TRL (Transformer Reinforcement Learning)](https://huggingface.co/docs/trl)
- [RLHF: Training Language Models to Follow Instructions (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155)
