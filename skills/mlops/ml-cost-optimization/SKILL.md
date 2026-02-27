---
name: ml-cost-optimization
description: "ML cost optimization skill covering GPU cost analysis, training cost reduction, inference cost optimization, spot instance strategies, model compression (quantization, pruning, knowledge distillation), mixed precision training, gradient accumulation, resource right-sizing, compute management, auto-scaling, scale-to-zero, batch inference, ONNX Runtime, storage tiering, artifact lifecycle, data processing efficiency, cost tracking dashboards, FinOps for ML, cost-aware experiment design, shared compute scheduling with Slurm and Kubernetes, and cloud vs on-prem cost comparison."
license: Apache-2.0
metadata:
  author: mlops-skills
  version: "1.0"
---

# ML Cost Optimization

## Overview

Machine learning workloads are among the most expensive compute tasks in modern engineering organizations. A single large model training run can cost tens of thousands of dollars, and poorly optimized inference pipelines can silently drain budgets. This skill provides actionable guidance for understanding, measuring, and reducing ML costs across the entire lifecycle.

For deep-dive content on storage optimization, data processing, cost tracking, FinOps practices, resource right-sizing, shared compute scheduling, and cloud vs on-prem comparisons, see `references/REFERENCE.md`.

---

## 1. ML Cost Components

Every ML project incurs costs across four major categories:

### 1.1 Compute Costs (60-80% of total ML spend)

- **Training compute**: GPU/TPU hours for model training and hyperparameter tuning
- **Inference compute**: GPU/CPU resources for serving predictions
- **Experimentation compute**: Interactive notebooks, development runs, failed experiments
- **Data processing compute**: ETL, feature engineering, preprocessing pipelines

### 1.2 Storage Costs

- **Training data**: Raw datasets, preprocessed features, augmented data
- **Model artifacts**: Checkpoints (every epoch), final models, compressed variants
- **Experiment metadata**: Logs, metrics, TensorBoard files, profiling data
- **Container images**: Large Docker images with ML frameworks (often 5-15 GB each)

### 1.3 Data Transfer Costs

- **Cross-region transfers**: Moving training data to GPU-available regions
- **Multi-cloud transfers**: Hybrid setups with data in one cloud, compute in another
- **API serving egress**: Inference API responses leaving the cloud network

### 1.4 Tooling and Platform Costs

- **Managed ML platforms**: SageMaker, Vertex AI, Databricks, etc.
- **Experiment tracking**: Weights & Biases, Neptune, Comet (per-seat or per-usage)
- **Monitoring and orchestration**: Model monitoring, Airflow, Kubeflow overhead

---

## 2. GPU Selection and Sizing

Choosing the right GPU is one of the highest-leverage cost decisions. The wrong choice can lead to 3-10x cost overruns.

### 2.1 GPU Quick Comparison

| GPU | VRAM | FP16 TFLOPS | On-Demand $/hr | Best For |
|-----|------|-------------|----------------|----------|
| T4 | 16 GB | 65 | $0.35-0.53 | Inference, small training |
| V100 | 16/32 GB | 125 | $0.74-2.48 | General training |
| A10G | 24 GB | 62.5 | $0.75-1.50 | Inference, fine-tuning |
| A100 | 40/80 GB | 312 | $1.10-4.10 | Large model training |
| H100 | 80 GB | 990 | $2.50-8.00 | LLM training, large-scale |
| L4 | 24 GB | 121 | $0.35-0.81 | Cost-effective inference |

See `references/REFERENCE.md` for detailed per-cloud pricing and GPU memory requirements by model size.

### 2.2 GPU Sizing Guidelines

- **Small models (< 500M params)**: T4 or V100 16 GB for training; T4 or L4 for inference
- **Medium models (500M-5B params)**: A100 40 GB for training; A10G or quantized model on T4 for inference
- **Large models (> 5B params)**: A100 80 GB or H100 for training; A100 or quantized model on A10G for inference

### 2.3 Cost/Performance Analysis

Do not blindly choose the cheapest GPU. Calculate cost-efficiency:

```
Cost Efficiency = (Throughput in samples/sec) / ($/hr)
Time-to-Result Cost = (Total training hours) * ($/hr per GPU) * (Number of GPUs)
```

A faster GPU that completes training in half the time may cost less total than a cheaper GPU running twice as long.

---

## 3. Spot/Preemptible Instance Strategies

Spot instances offer 60-90% cost savings but require fault-tolerant training.

### 3.1 When to Use Spot Instances

**Good candidates**: Hyperparameter search, long-running training with checkpointing, batch inference with retry logic, data preprocessing pipelines.

**Poor candidates**: Real-time inference endpoints, short jobs (< 30 min), jobs with expensive uncacheable initialization.

### 3.2 Fault-Tolerant Training

```python
import signal, sys, time, os

class SpotInstanceCheckpointer:
    """Handles graceful checkpointing on spot termination (2-min warning)."""

    def __init__(self, checkpoint_dir, checkpoint_interval_minutes=15):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval_minutes * 60
        self.last_checkpoint_time = 0
        signal.signal(signal.SIGTERM, self._handle_termination)
        signal.signal(signal.SIGINT, self._handle_termination)

    def _handle_termination(self, signum, frame):
        print(f"Termination signal received ({signum}). Saving checkpoint...")
        self.save_checkpoint(emergency=True)
        sys.exit(0)

    def save_checkpoint(self, model, optimizer, epoch, step, loss, emergency=False):
        import torch
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch, "step": step, "loss": loss,
        }
        path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch{epoch}_step{step}.pt")
        torch.save(checkpoint, path)
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, "checkpoint_latest.pt"))
        self.last_checkpoint_time = time.time()

    def should_checkpoint(self):
        return (time.time() - self.last_checkpoint_time) >= self.checkpoint_interval
```

### 3.3 Cloud Provider Notes

- **AWS**: Spot Fleet with diversified allocation across instance types and AZs. 2-minute interruption notice.
- **GCP**: Preemptible VMs (24-hour max) or Spot VMs. Implement shutdown scripts.
- **Azure**: Spot VMs with stop-deallocate eviction policy. Use Azure Batch for managed orchestration.

---

## 4. Training Cost Optimization

### 4.1 Mixed Precision Training

Mixed precision (FP16/BF16) reduces memory by ~50% and increases throughput 1.5-3x on Tensor Core GPUs.

```python
import torch
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for batch in dataloader:
    optimizer.zero_grad()
    with autocast(dtype=torch.float16):
        outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"])
        loss = criterion(outputs, batch["labels"])
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Cost impact**: ~40% savings in GPU-hours. Use BF16 on A100/H100 (more numerically stable); FP16 on V100/T4 (requires loss scaling).

### 4.2 Gradient Accumulation

Simulate larger batch sizes without more GPU memory, enabling use of smaller (cheaper) GPUs.

```python
accumulation_steps = 8  # Effective batch size = per_GPU_batch * accumulation_steps
for i, batch in enumerate(dataloader):
    with autocast(dtype=torch.float16):
        loss = model(batch) / accumulation_steps
    scaler.scale(loss).backward()
    if (i + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

**Cost impact**: Use 1x A100 40 GB with 4x accumulation instead of 4x A100 80 GB. ~75% GPU cost savings at ~10-20% slower wall-clock time.

### 4.3 Knowledge Distillation

Train a smaller "student" model to mimic a larger "teacher" model. The student can be 5-20x smaller while retaining 90-98% of performance.

```python
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, labels, temperature=4.0, alpha=0.7):
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature ** 2)
    hard_loss = F.cross_entropy(student_logits, labels)
    return alpha * soft_loss + (1 - alpha) * hard_loss
```

**Cost impact**: A distilled BERT-tiny serves inference at 1/10th the cost of BERT-large with 85-95% accuracy on many tasks.

### 4.4 Progressive Resizing

Start training on smaller inputs, then gradually increase. Common in computer vision. Early epochs on small images are 4-16x faster (quadratic scaling with resolution). Total training cost drops 30-50%.

### 4.5 Early Stopping

Avoid wasting compute on training that has plateaued. Training often converges 20-40% before the planned epoch budget. Implement patience-based stopping that monitors validation loss and halts when no improvement is seen for N consecutive evaluations.

---

## 5. Inference Cost Optimization

Inference often exceeds training cost over the model's lifetime. A model trained once but served millions of times makes inference optimization critical.

### 5.1 Model Quantization

Reduce model precision from FP32 to INT8 or INT4 for 2-8x size reduction and increased throughput.

- **Dynamic quantization**: Weights quantized ahead of time, activations on-the-fly. No calibration data needed. Best for CPU inference.
- **Static quantization (PTQ)**: Both weights and activations quantized. Better accuracy. Needs calibration data.
- **Quantization-Aware Training (QAT)**: Simulate quantization during training. Best accuracy, highest effort.

```python
import torch

# Dynamic quantization (simplest approach)
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

See `references/REFERENCE.md` for detailed quantization method comparison tables.

### 5.2 Model Pruning

- **Unstructured pruning**: Zero out individual weights by magnitude. High sparsity (90%+) but needs sparse hardware for speedup.
- **Structured pruning**: Remove entire neurons/channels/attention heads. Directly reduces size and compute on any hardware.

```python
import torch.nn.utils.prune as prune

# Unstructured: remove 50% of weights by magnitude
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name="weight", amount=0.5)

# Structured: remove 30% of output channels
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.ln_structured(module, name="weight", amount=0.3, n=1, dim=0)
```

### 5.3 ONNX Runtime Optimization

Export models to ONNX format for 1.5-3x inference speedup:

```python
import torch
import onnxruntime as ort

# Export to ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model, dummy_input, "model.onnx",
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=17,
)

# Run optimized inference
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession("model.onnx", session_options)
result = session.run(None, {"input": input_array})
```

### 5.4 Batch vs Real-Time Inference

| Aspect | Batch Inference | Real-Time Inference |
|--------|----------------|-------------------|
| Latency | Minutes to hours | Milliseconds |
| GPU utilization | 80-100% | Often 10-40% |
| Best for | Reports, recommendations, ETL | APIs, user-facing features |

**Cost tip**: If latency > 1 second is tolerable, dynamic batching can improve GPU utilization from 15% to 80%+.

### 5.5 Auto-Scaling and Scale-to-Zero

- Configure HPA to scale inference pods based on request rate
- Scale to zero for low-traffic endpoints (but plan for 30-120s cold start for large models)
- Keep a warm pool of 1 replica during business hours; scale to zero only overnight
- Serverless ML options: AWS Lambda (up to 10 GB), Google Cloud Run (GPU preview), Azure Container Apps

---

## Quick Reference: Cost Optimization Checklist

### Before Training
- [ ] Estimate cost with cost estimator (see `scripts/cost_analyzer.py`)
- [ ] Choose appropriate GPU (do not default to the largest)
- [ ] Enable mixed precision (AMP)
- [ ] Configure gradient accumulation if batch size is a bottleneck
- [ ] Set up checkpointing for spot instance resilience
- [ ] Implement early stopping
- [ ] Start with a data subset for initial experiments

### During Training
- [ ] Monitor GPU utilization (target > 70%)
- [ ] Track cost per epoch
- [ ] Use spot instances with checkpointing
- [ ] Kill experiments that are clearly not converging

### Before Deployment
- [ ] Apply quantization (INT8 at minimum)
- [ ] Evaluate pruning and knowledge distillation
- [ ] Export to ONNX Runtime for CPU inference
- [ ] Calculate cost per prediction

### In Production
- [ ] Configure auto-scaling with scale-to-zero
- [ ] Use dynamic batching for throughput
- [ ] Monitor GPU utilization and right-size monthly
- [ ] Set up cost alerts and budgets
- [ ] Tag all resources for cost attribution

See `references/REFERENCE.md` for the full phase-by-phase checklist with 70+ items.

---

## Scripts

- **`scripts/cost_analyzer.py`**: Analyze and estimate ML training and inference costs, including GPU cost comparison, spot instance savings calculation, and experiment cost projection.
- **`scripts/model_compress.py`**: Model compression utilities including quantization (dynamic, static, QAT), pruning (unstructured, structured), and ONNX Runtime export with benchmarking.

---

## References

- **`references/REFERENCE.md`**: Comprehensive reference covering:
  - Detailed GPU pricing tables (per-cloud provider)
  - GPU memory requirements by model size
  - Quantization, pruning, and distillation method comparison tables
  - Cost estimation formulas (training, inference, spot savings, TCO)
  - FinOps maturity model for ML teams (3 levels with metrics)
  - Full cost optimization checklist (7 phases, 70+ items)
  - Cloud provider cost comparison (compute, storage, managed platforms)
  - Cloud vs on-prem vs hybrid decision guide
  - Storage cost optimization (tiered storage, artifact lifecycle, smart checkpointing)
  - Data processing optimization (sampling, caching, efficient loading)
  - Cost tracking and budgeting (tagging, CostTracker, BudgetGuard)
  - Cost-aware experiment design (budgeting, prioritization)
  - Resource right-sizing (GPU monitoring, decision matrix)
  - Shared compute scheduling (Kubernetes, Slurm)
  - Monthly cost benchmarks by workload type
