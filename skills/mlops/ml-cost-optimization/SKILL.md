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

FP16 TFLOPS below is the **dense** Tensor-Core figure. Vendor slides usually quote
the 2x **sparse** number (H100 SXM: 989 sparse vs 494 dense) — sparse throughput
requires a 2:4-sparsified model and is not what a normal run achieves. $/hr is AWS
us-east-1 on-demand **per GPU** (multi-GPU instance price / GPU count).

| GPU | VRAM | FP16 TFLOPS (dense) | FP16 TFLOPS (sparse) | AWS On-Demand $/GPU-hr | Best For |
|-----|------|---------------------|----------------------|------------------------|----------|
| T4 | 16 GB | 65 | 130 | $0.53 (g4dn.xlarge) | Inference, small training |
| V100 16GB | 16 GB | 125 | n/a (no sparsity) | $3.06 (p3.2xlarge) | General training |
| V100 32GB | 32 GB | 125 | n/a | $3.90 (p3dn.24xlarge / 8) | Large batch training |
| A10G | 24 GB | 62.5 | 125 | $1.01 (g5.xlarge) | Inference, fine-tuning |
| L4 | 24 GB | 121 | 242 | $0.81 (g6.xlarge) | Cost-effective inference |
| L40S | 48 GB | 366 | 733 | $1.86 (g6e.xlarge) | Mid-size training, no NVLink |
| A100 40GB | 40 GB | 312 | 624 | $4.10 (p4d.24xlarge / 8) | Large model training |
| A100 80GB | 80 GB | 312 | 624 | $5.12 (p4de.24xlarge / 8) | Very large models |
| H100 80GB SXM | 80 GB | 494 | 989 | $12.29 (p5.48xlarge / 8) | LLM training, large-scale |
| H200 141GB SXM | 141 GB | 494 | 989 | ~$11 (p5e/p5en / 8) | Memory-bound LLM work |

Spot is a live market: recent AWS spot has run roughly $0.16 (T4), $0.92 (V100 16GB),
$1.30-1.80 (A100), and $6+ (H100) per GPU-hour, and H100/A100 spot capacity is
frequently unavailable in the big three clouds. Sub-$3 H100 rates come from
neoclouds (Lambda, CoreWeave, RunPod), not AWS/GCP/Azure. Also note that quoted
accelerator-only prices (e.g. GCP's ~$0.35/hr T4) exclude the attached VM, which
roughly doubles the effective rate.

See `references/REFERENCE.md` for detailed per-cloud pricing and GPU memory requirements by model size.

### 2.2 GPU Sizing Guidelines

- **Small models (< 500M params)**: T4 or L4 for training; T4 or L4 for inference
- **Medium models (500M-5B params)**: L40S or A100 40 GB for training; A10G/L4, or a quantized model on T4, for inference
- **Large models (> 5B params)**: A100 80 GB, H100, or H200 for training; A100, or a quantized model on A10G/L40S, for inference
- **Very large / MI300X, B200 and GB200**: newer parts (Blackwell B200/GB200, AMD MI300X 192 GB) are entering GA with limited regional availability; price them directly with the provider rather than assuming a ratio to H100

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
import os, signal, sys, threading, time
import urllib.request

class SpotInstanceCheckpointer:
    """
    Graceful checkpointing on spot interruption.

    The signal handler takes no arguments beyond (signum, frame), so it cannot
    receive model/optimizer/epoch/step/loss. Register the live training state on
    the instance (via track_state) and have the handler read it — a handler that
    calls save_checkpoint(emergency=True) with the 5 required positional args
    missing raises TypeError at the exact moment you need the checkpoint.
    """

    def __init__(self, checkpoint_dir, checkpoint_interval_minutes=15):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval_minutes * 60
        self.last_checkpoint_time = 0
        self._state = None          # set by track_state() each step
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        signal.signal(signal.SIGTERM, self._handle_termination)
        signal.signal(signal.SIGINT, self._handle_termination)

    def track_state(self, model, optimizer, epoch, step, loss):
        """Call once per step (cheap: stores references, not copies)."""
        self._state = dict(model=model, optimizer=optimizer,
                           epoch=epoch, step=step, loss=float(loss))

    def _handle_termination(self, signum, frame):
        print(f"Termination signal received ({signum}). Saving checkpoint...", flush=True)
        if self._state is None:
            print("No training state registered; nothing to save.", flush=True)
        else:
            self.save_checkpoint(emergency=True, **self._state)
        sys.exit(0)

    def save_checkpoint(self, model, optimizer, epoch, step, loss, emergency=False):
        import torch
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch, "step": step, "loss": loss, "emergency": emergency,
        }
        path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch{epoch}_step{step}.pt")
        torch.save(checkpoint, path)
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, "checkpoint_latest.pt"))
        self.last_checkpoint_time = time.time()

    def should_checkpoint(self):
        return (time.time() - self.last_checkpoint_time) >= self.checkpoint_interval
```

**Bare EC2 does not send SIGTERM.** A spot interruption on plain EC2 is published
only as instance metadata (and an EventBridge event) — nothing signals your
process, so a SIGTERM-only design silently loses the last checkpoint. You must
poll IMDS:

```python
IMDS_TOKEN_URL = "http://169.254.169.254/latest/api/token"
SPOT_ACTION_URL = "http://169.254.169.254/latest/meta-data/spot/instance-action"

def _imds_token():
    req = urllib.request.Request(
        IMDS_TOKEN_URL, method="PUT",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
    )
    return urllib.request.urlopen(req, timeout=2).read().decode()

def spot_interruption_pending():
    """True once AWS has scheduled this instance for termination (~2 min notice)."""
    try:
        req = urllib.request.Request(
            SPOT_ACTION_URL, headers={"X-aws-ec2-metadata-token": _imds_token()},
        )
        urllib.request.urlopen(req, timeout=2).read()
        return True          # 200 => interruption scheduled
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False     # 404 => no interruption scheduled (normal case)
        raise
    except Exception:
        return False         # network hiccup: treat as no notice

def watch_for_interruption(checkpointer, poll_seconds=5):
    def _loop():
        while True:
            if spot_interruption_pending():
                checkpointer._handle_termination("spot-interruption", None)
            time.sleep(poll_seconds)
    threading.Thread(target=_loop, daemon=True).start()
```

Where SIGTERM *is* delivered: **ECS** (`stopTimeout` grace), **EKS/Kubernetes**
(pod `terminationGracePeriodSeconds`, typically driven by the AWS Node
Termination Handler), and **SageMaker** managed spot training (which also handles
checkpoint sync to S3 for you). On GCP, preemption triggers an ACPI G2 soft-off
that surfaces as SIGTERM to the shutdown script; on Azure, use Scheduled Events
(`http://169.254.169.254/metadata/scheduledevents`).

### 3.3 Cloud Provider Notes

- **AWS**: Spot Fleet with diversified allocation across instance types and AZs. 2-minute notice via IMDS `spot/instance-action` + EventBridge; SIGTERM only under ECS/EKS/SageMaker.
- **GCP**: Preemptible VMs (24-hour max) or Spot VMs. ~30-second notice; implement shutdown scripts (SIGTERM is delivered).
- **Azure**: Spot VMs with stop-deallocate eviction policy. ~30-second notice via Scheduled Events; use Azure Batch for managed orchestration.

---

## 4. Training Cost Optimization

### 4.1 Mixed Precision Training

Mixed precision (FP16/BF16) reduces memory by ~50% and increases throughput 1.5-3x on Tensor Core GPUs.

```python
import torch
# torch.cuda.amp.autocast/GradScaler are deprecated since PyTorch 2.4 —
# use the device-generic torch.amp API.
from torch.amp import autocast, GradScaler

scaler = GradScaler("cuda")
for batch in dataloader:
    optimizer.zero_grad(set_to_none=True)
    with autocast("cuda", dtype=torch.bfloat16):   # bfloat16 on A100/H100; float16 on V100/T4
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
    with autocast("cuda", dtype=torch.bfloat16):
        outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"])
        # model(...) returns logits, not a loss — you must apply the criterion.
        loss = criterion(outputs, batch["labels"]) / accumulation_steps
    scaler.scale(loss).backward()
    if (i + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
```

**What accumulation does and does not buy you.** It reduces *activation* memory
only, by shrinking the per-step micro-batch. Weights, gradients, and optimizer
state are resident in full on every device regardless of accumulation, so
accumulation cannot make a model fit that does not fit at batch size 1. A 13B model
in mixed precision needs roughly 13B x 2 B (weights) + 13B x 2 B (grads) + 13B x 8 B
(Adam m/v in FP32) ~ 156 GB of state — no amount of accumulation puts that on one
40 GB A100.

**Cost impact**: accumulation lets you trade wall-clock for GPU count when
activations are the binding constraint (for example, keeping a 24 GB A10G instead
of moving to a 40 GB A100 for a mid-size vision model). Expect wall-clock to scale
roughly with the number of micro-steps — 4x accumulation on one GPU replacing 4
data-parallel GPUs is about 4x slower, not 10-20% slower, since you removed the
parallelism rather than the work. For models whose *state* does not fit, use
ZeRO/FSDP sharding (DeepSpeed ZeRO-3, `torch.distributed.fsdp`) or CPU/NVMe
offload instead.

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
# torch.quantization is the legacy alias; the maintained namespace is torch.ao.quantization.
import torch.ao.quantization as tq

# Dynamic quantization (simplest approach; CPU inference)
quantized_model = tq.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

The eager-mode API above requires you to place QuantStub/DeQuantStub and fuse
modules by hand for static quantization. For new work prefer a graph-based flow:
FX graph mode (`torch.ao.quantization.quantize_fx`) or PT2 export quantization
(`torch.ao.quantization.quantize_pt2e` with `torch.export`), which insert
observers automatically.

See `references/REFERENCE.md` for detailed quantization method comparison tables.

### 5.2 Model Pruning

- **Unstructured pruning**: Zero out individual weights by magnitude. High sparsity (90%+) but needs sparse hardware for speedup.
- **Structured pruning**: Remove entire neurons/channels/attention heads. Directly reduces size and compute on any hardware.

```python
import torch.nn.utils.prune as prune

# Unstructured: zero 50% of weights by magnitude
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name="weight", amount=0.5)

# Structured: zero 30% of output channels
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.ln_structured(module, name="weight", amount=0.3, n=1, dim=0)
```

Two traps in this API:

1. **Masking is not compression.** `prune.*` adds a mask plus a `weight_orig`
   parameter; shapes, FLOPs, and (until `prune.remove()`) file size are unchanged
   — the checkpoint actually gets bigger. Even after `prune.remove()` you hold
   zeros in a dense tensor. Realizing a structured-pruning speedup requires
   rebuilding the layers at the reduced width; realizing an unstructured speedup
   requires NVIDIA Ampere+ 2:4 semi-structured sparsity plus a sparsity-aware
   runtime (TensorRT).
2. **Iterative pruning needs a cumulative schedule.** `amount` is the fraction of
   *all* candidate weights to zero, and already-zeroed weights are the
   smallest-magnitude ones — so calling it repeatedly with a per-step amount
   re-selects the same zeros and sparsity plateaus at the first step's level
   (~37% when you asked for 90%). Pass the cumulative target each step:

```python
target, steps = 0.9, 5
for step in range(steps):
    prune.global_unstructured(
        params_to_prune, pruning_method=prune.L1Unstructured,
        amount=target * (step + 1) / steps,     # cumulative, not per-step
    )
    finetune(model)
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
- Serverless ML options: AWS Lambda (up to 10 GB memory, CPU only), Google Cloud Run with NVIDIA L4 GPUs (generally available since 2025, scales to zero), Azure Container Apps (serverless GPU)

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
- **`scripts/model_compress.py`**: Model compression utilities including quantization (dynamic, static, QAT), pruning (unstructured, structured, iterative), and benchmarking. The CLI subcommands load a state_dict into a built-in demo architecture selected by `--arch` (`mlp` | `cnn` | `transformer_block`); import the functions directly for your own model classes. Note that `compare` reads two FP32 state_dicts of the same architecture, so it cannot load the packed-INT8 output of `quantize` — benchmark quantized models in-process with `compare_models()`.

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
