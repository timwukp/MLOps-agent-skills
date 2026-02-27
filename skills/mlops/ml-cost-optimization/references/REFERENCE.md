# ML Cost Optimization Reference

## GPU Comparison Table

### Cloud GPU Instances (Approximate Pricing, 2025)

| GPU | VRAM | FP16 TFLOPS | Architecture | AWS On-Demand ($/hr) | AWS Spot ($/hr) | GCP On-Demand ($/hr) | Azure On-Demand ($/hr) | Best For |
|-----|------|-------------|-------------|---------------------|-----------------|---------------------|----------------------|----------|
| NVIDIA T4 | 16 GB | 65 | Turing | $0.53 (g4dn.xlarge) | $0.16 | $0.35 (n1+T4) | $0.53 (NC4as_T4_v3) | Inference, small models |
| NVIDIA V100 16GB | 16 GB | 125 | Volta | $3.06 (p3.2xlarge) | $0.92 | $2.48 (a2+V100) | $3.06 (NC6s_v3) | General training |
| NVIDIA V100 32GB | 32 GB | 125 | Volta | $3.06 (p3.2xlarge) | $0.92 | $2.48 | $3.06 | Large batch training |
| NVIDIA A10G | 24 GB | 62.5 | Ampere | $1.01 (g5.xlarge) | $0.35 | N/A | N/A | Inference, fine-tuning |
| NVIDIA A100 40GB | 40 GB | 312 | Ampere | $3.67 (p4d 1-GPU) | $1.10 | $2.94 (a2-highgpu-1g) | $3.40 (NC24ads_A100) | Large model training |
| NVIDIA A100 80GB | 80 GB | 312 | Ampere | $4.10 (p4de 1-GPU) | $1.50 | $3.67 (a2-ultragpu-1g) | $3.67 | Very large models |
| NVIDIA H100 80GB | 80 GB | 990 | Hopper | $8.00+ (p5 1-GPU) | $2.50 | $5.07 (a3-highgpu-1g) | $7.35 (ND96isr_H100) | LLM training, peak perf |
| NVIDIA L4 | 24 GB | 121 | Ada Lovelace | $0.81 (g6.xlarge) | $0.28 | $0.35 (g2-standard-4) | N/A | Cost-effective inference |

**Notes**:
- Prices are approximate and vary by region. Check cloud provider pricing pages for current rates.
- Spot/preemptible pricing fluctuates based on demand.
- Multi-GPU instances offer bulk discounts (e.g., 8x A100 on p4d.24xlarge is ~$32/hr total).

### GPU Memory Requirements by Model Size

| Model Parameters | FP32 Memory | FP16 Memory | INT8 Memory | Recommended GPU |
|-----------------|-------------|-------------|-------------|----------------|
| 100M | ~0.4 GB | ~0.2 GB | ~0.1 GB | T4 (16 GB) |
| 350M (BERT-large) | ~1.4 GB | ~0.7 GB | ~0.35 GB | T4 (16 GB) |
| 1B | ~4 GB | ~2 GB | ~1 GB | V100 16GB or A10G |
| 3B | ~12 GB | ~6 GB | ~3 GB | V100 32GB or A100 40GB |
| 7B (LLaMA-7B) | ~28 GB | ~14 GB | ~7 GB | A100 40GB |
| 13B (LLaMA-13B) | ~52 GB | ~26 GB | ~13 GB | A100 80GB |
| 30B | ~120 GB | ~60 GB | ~30 GB | 2x A100 80GB |
| 70B (LLaMA-70B) | ~280 GB | ~140 GB | ~70 GB | 4x A100 80GB or 2x H100 |

**Training memory rule of thumb** (with Adam optimizer):
- FP32 training: ~4x model parameter memory (model + gradients + optimizer states)
- Mixed precision: ~2-3x model parameter memory
- Activation memory adds an additional 1-5x depending on batch size and sequence length

---

## Compression Technique Comparison

### Quantization Methods

| Method | Precision | Accuracy Loss | Size Reduction | Speedup (CPU) | Speedup (GPU) | Calibration Data | Retraining |
|--------|-----------|---------------|---------------|----------------|---------------|-----------------|------------|
| Dynamic (INT8) | INT8 weights, FP32 activations | 0.1-1% | 2-4x | 1.5-2x | Minimal | Not needed | No |
| Static PTQ (INT8) | INT8 weights + activations | 0.5-2% | 2-4x | 2-3x | 1.5-2x | 100-1000 samples | No |
| QAT (INT8) | INT8 weights + activations | 0.1-0.5% | 2-4x | 2-3x | 1.5-2x | Full training set | Yes |
| GPTQ (INT4) | INT4 weights, FP16 activations | 0.5-3% | 4-8x | N/A | 2-4x | 128 samples | No |
| AWQ (INT4) | INT4 weights, FP16 activations | 0.3-2% | 4-8x | N/A | 2-4x | Small calibration | No |
| GGML/GGUF (Q4_K_M) | Mixed 4-6 bit | 0.5-2% | 4-6x | 2-4x (CPU) | N/A | Not needed | No |
| FP16 / BF16 | Half precision | < 0.1% | 2x | Minimal | 1.5-3x | Not needed | No |

**When to use which**:
- **Dynamic INT8**: Default starting point. Free accuracy, easy to apply.
- **Static INT8**: When dynamic is not fast enough and you have calibration data.
- **QAT**: When accuracy is critical and you can afford retraining.
- **GPTQ/AWQ**: For LLMs that must fit in limited GPU memory.
- **GGML/GGUF**: For running LLMs on CPU/edge devices.
- **FP16/BF16**: Always use for training on modern GPUs. Use BF16 on A100/H100.

### Pruning Methods

| Method | Type | Sparsity Range | Accuracy Impact | Real Speedup | Hardware Support |
|--------|------|----------------|-----------------|-------------|-----------------|
| Magnitude (L1) unstructured | Unstructured | 50-95% | Low at 50%, moderate at 90% | Minimal without sparse kernels | NVIDIA Ampere+ (2:4 sparsity) |
| Random unstructured | Unstructured | 50-80% | Moderate | Minimal | Same as above |
| Structured (channel) | Structured | 20-60% | Moderate | Direct (smaller model) | All hardware |
| Structured (attention head) | Structured | 20-50% | Low-moderate | Direct | All hardware |
| Movement pruning | Unstructured | 70-95% | Low (with fine-tuning) | Requires sparse kernels | NVIDIA Ampere+ |
| Lottery Ticket | Unstructured | 80-95% | Can match original | Requires sparse kernels | NVIDIA Ampere+ |

**Key insight**: Unstructured pruning achieves high sparsity but requires specialized hardware/software for actual speedup. Structured pruning gives smaller models that run faster on any hardware.

### Knowledge Distillation Variants

| Variant | Description | Compression | Accuracy Retained | Use Case |
|---------|-------------|-------------|-------------------|----------|
| Logit distillation | Student matches teacher's output distribution | 5-20x | 90-98% | General classification |
| Feature distillation | Student matches teacher's intermediate representations | 5-15x | 92-99% | When internal features matter |
| Attention transfer | Student matches teacher's attention maps | 3-10x | 93-98% | Transformer models |
| Self-distillation | Model distills knowledge into a smaller version of itself | 2-5x | 95-99% | Iterative model improvement |
| Task-specific distillation | Distill on task data, not general | 5-20x | 93-98% | Domain-specific deployment |

---

## Cost Estimation Formulas

### Training Cost

```
Training Cost ($) = GPU_Hours * Price_Per_GPU_Hour * Num_GPUs + Storage_Cost + Transfer_Cost

GPU_Hours = (Model_Params / Base_Throughput) * Dataset_Size * Num_Epochs / Speedup_Factor

Speedup_Factor = GPU_Throughput_Factor * (1.5 if mixed_precision else 1.0)

Storage_Cost = Dataset_GB * 2.0 * Storage_Price_Per_GB_Month * Duration_Months

Transfer_Cost = Data_Moved_GB * Egress_Price_Per_GB
```

### Inference Cost

```
Inference_Cost_Per_Month ($) = Instance_Hours_Per_Month * Price_Per_Hour

Instance_Hours_Per_Month =
    (Avg_Requests_Per_Second * Avg_Latency_Seconds / Batch_Size)
    / Max_Concurrent_Requests_Per_Instance
    * Hours_Per_Month

Cost_Per_1000_Predictions = (Inference_Cost_Per_Month / Monthly_Predictions) * 1000

Break_Even_Optimization_Cost =
    (Current_Cost_Per_Prediction - Optimized_Cost_Per_Prediction) * Monthly_Volume * Payback_Months
```

### Cost Allocation Framework

```
Total ML Cost = Training Cost + Inference Cost + Platform Cost + Data Cost

Training Cost = Sum over experiments of:
    (GPU hours * GPU $/hr) + (Storage GB * Storage $/GB/mo * months) + Data transfer

Inference Cost = Sum over models of:
    (Instance hours * Instance $/hr) + (Requests * per-request overhead)

Cost per Prediction = Total Inference Cost / Total Predictions Served
Cost per Experiment = Total Experiment Cost / Number of Experiments
Cost per Model Update = Retraining Cost + Validation Cost + Deployment Cost
```

### Spot Instance Savings

```
Expected_Spot_Cost = Spot_Price * Expected_Hours * (1 + Interruption_Overhead)

Interruption_Overhead = Avg_Interruptions * (Checkpoint_Time + Restart_Time) / Total_Job_Time

Net_Savings = On_Demand_Cost - Expected_Spot_Cost
Savings_Percentage = Net_Savings / On_Demand_Cost * 100
```

### Total Cost of Ownership (On-Prem)

```
Annual_TCO = Hardware_Depreciation + Power_Cost + Cooling_Cost + Space_Cost +
             Staff_Cost + Network_Cost + Maintenance_Cost + Software_Licenses

Hardware_Depreciation = Total_Hardware_Cost / Useful_Life_Years

Power_Cost = Num_GPUs * TDP_kW * PUE * Hours_Per_Year * Electricity_$/kWh

# Example: 8x H100 cluster
# Hardware: $250,000 (server + GPUs + networking)
# Depreciation: $250,000 / 4 years = $62,500/year
# Power: 8 * 0.7 kW * 1.3 PUE * 8760 hrs * $0.10/kWh = $6,377/year
# Staff: $200,000/year (partial FTE for 1 admin)
# Effective hourly rate: ($62,500 + $6,377 + $50,000) / (8 GPUs * 8760 hrs) = $1.70/GPU-hr
# Compare to: H100 on-demand at $6-8/hr, spot at $2.50/hr
```

---

## FinOps Maturity Model for ML

### Level 1: Inform (Visibility)

**Goal**: Understand where money is being spent.

| Practice | Implementation | Tools |
|----------|---------------|-------|
| Resource tagging | Tag all ML resources with team, project, environment | Cloud tagging, Terraform |
| Cost reporting | Monthly cost reports by team/project | AWS Cost Explorer, GCP Billing, Azure Cost Management |
| Billing alerts | Alerts at 80% and 100% of budget | Cloud provider alerts |
| Usage tracking | Log GPU hours per experiment | MLflow, W&B, custom logging |

**Key metrics**:
- Total ML spend per month
- Spend by team
- Spend by project
- GPU utilization rate

### Level 2: Optimize (Efficiency)

**Goal**: Reduce waste and improve unit economics.

| Practice | Implementation | Expected Savings |
|----------|---------------|-----------------|
| Spot instances for training | Fault-tolerant training with checkpointing | 60-70% on training compute |
| Mixed precision training | Enable AMP on all training jobs | 30-50% on training time |
| Right-sizing | Monthly GPU utilization review, resize under-utilized instances | 20-40% on compute |
| Auto-scaling inference | HPA with scale-to-zero for low-traffic models | 40-80% on inference compute |
| Model compression | Quantize all production models (INT8 minimum) | 50-75% on inference compute |
| Storage tiering | Lifecycle policies: hot -> warm -> cold -> delete | 40-60% on storage |
| Artifact cleanup | Auto-delete old checkpoints, retain best N only | 50-70% on storage |

**Key metrics**:
- Cost per GPU hour (effective rate)
- GPU utilization rate (target > 70%)
- Cost per experiment
- Cost per prediction
- Storage cost per project

### Level 3: Operate (Continuous Optimization)

**Goal**: Bake cost optimization into the ML development lifecycle.

| Practice | Implementation | Impact |
|----------|---------------|--------|
| Cost-aware CI/CD | Reject model versions that increase inference cost by > X% | Prevents cost regression |
| Budget-constrained HPO | Set cost budget as HPO constraint | Eliminates runaway experiments |
| Automated right-sizing | Auto-recommend instance changes based on usage | Continuous optimization |
| Cost anomaly detection | Alert on unusual spending patterns | Early detection of issues |
| Reserved capacity planning | Quarterly review of commitment purchases | 30-60% on baseline compute |
| Chargeback/showback | Bill teams for actual ML resource usage | Accountability |

**Key metrics**:
- Cost per unit of model performance (e.g., $/accuracy-point)
- Cost efficiency trend (month-over-month)
- Percentage of workloads on spot/reserved
- Time from cost anomaly to resolution
- Budget adherence rate

---

## Cost Optimization Checklist by Phase

### Phase 1: Data Preparation

- [ ] Store raw data in cheapest appropriate storage tier
- [ ] Use columnar formats (Parquet, ORC) for tabular data -- 50-80% smaller than CSV
- [ ] Implement data versioning to avoid duplicate storage (DVC, lakeFS)
- [ ] Cache preprocessed features to avoid recomputation
- [ ] Use data sampling (5-10%) for initial exploration and debugging
- [ ] Minimize cross-region data transfers -- keep data and compute co-located
- [ ] Set retention policies: delete raw intermediate data after processing
- [ ] Compress large datasets (gzip, snappy, zstd)

### Phase 2: Experimentation

- [ ] Start with smallest viable model and dataset sample
- [ ] Use progressive scaling: 5% data -> 25% data -> 100% data
- [ ] Estimate experiment cost before launching (use cost_analyzer.py)
- [ ] Set GPU utilization alert threshold (alert if < 50% for > 30 min)
- [ ] Use early stopping to avoid wasted training epochs
- [ ] Implement experiment deduplication: check if similar run already exists
- [ ] Use notebooks on CPU/small GPU for exploration, GPU only for training
- [ ] Shut down idle notebook instances (auto-shutdown after 30-60 min)
- [ ] Tag all experiments for cost attribution

### Phase 3: Training

- [ ] Enable mixed precision training (FP16/BF16)
- [ ] Use gradient accumulation to reduce GPU memory requirements
- [ ] Configure spot instances with robust checkpointing
- [ ] Set training budget limits with automatic termination
- [ ] Use the smallest GPU that fits the workload
- [ ] Monitor GPU utilization during training (target > 70%)
- [ ] Enable gradient checkpointing for large models to trade compute for memory
- [ ] Use efficient data loading (prefetch, num_workers, pin_memory)
- [ ] Schedule large training jobs during off-peak hours for better spot availability
- [ ] Save only top-N checkpoints, delete the rest

### Phase 4: Evaluation and Validation

- [ ] Run validation on CPU if latency is not critical
- [ ] Cache validation predictions to avoid recomputation during analysis
- [ ] Use sampled evaluation for large test sets during development
- [ ] Automate evaluation pipelines to avoid manual GPU usage

### Phase 5: Model Optimization (Pre-Deployment)

- [ ] Apply dynamic quantization (INT8) as baseline -- free accuracy, smaller model
- [ ] Benchmark quantized model -- measure latency, throughput, accuracy
- [ ] Evaluate static quantization if dynamic is insufficient
- [ ] Consider structured pruning (20-40%) for additional size reduction
- [ ] Evaluate knowledge distillation for high-traffic production models
- [ ] Export to ONNX Runtime for optimized CPU inference
- [ ] Profile model to identify bottleneck layers
- [ ] Test model on target deployment hardware

### Phase 6: Deployment

- [ ] Configure auto-scaling with appropriate min/max replicas
- [ ] Implement scale-to-zero for low-traffic endpoints
- [ ] Use dynamic batching to maximize GPU utilization
- [ ] Set request timeout to prevent runaway inference
- [ ] Use appropriate instance type: GPU for heavy models, CPU for quantized/small models
- [ ] Implement request queuing with backpressure
- [ ] Configure health checks and circuit breakers
- [ ] Use model caching to avoid repeated loading

### Phase 7: Production Operations

- [ ] Monitor inference cost per prediction daily
- [ ] Set up cost anomaly alerts
- [ ] Review GPU utilization weekly -- right-size if consistently < 60%
- [ ] Rotate storage tiers: move old artifacts to cold storage monthly
- [ ] Review reserved instance / savings plan coverage quarterly
- [ ] Update cost dashboards and share with stakeholders monthly
- [ ] Run cost optimization review quarterly
- [ ] Track cost-per-prediction trend over time

---

## Cloud Provider Cost Comparison for ML Workloads

### Compute Cost Comparison (Single GPU, On-Demand)

| Workload | AWS | GCP | Azure | Notes |
|----------|-----|-----|-------|-------|
| T4 inference | $0.53/hr (g4dn.xl) | $0.35/hr (n1+T4) | $0.53/hr (NC4as) | GCP cheapest for T4 |
| A100 40GB training | $3.67/hr (p4d share) | $2.94/hr (a2-highgpu) | $3.40/hr (NC24ads) | GCP cheapest for A100 |
| H100 training | $8.00+/hr (p5) | $5.07/hr (a3-highgpu) | $7.35/hr (ND96isr) | GCP cheapest for H100 |
| 8x A100 cluster | $32.77/hr (p4d.24xl) | $23.51/hr (a2-megagpu-16g) | $27.20/hr | GCP leads for large clusters |

### Managed ML Platform Comparison

| Service | AWS SageMaker | GCP Vertex AI | Azure ML | Databricks |
|---------|---------------|---------------|----------|------------|
| Training markup | ~20% over EC2 | ~15% over GCE | ~15% over VMs | ~30% over raw compute |
| Notebook instances | $0.05-16/hr | $0.04-12/hr | $0.05-15/hr | $0.07-20/hr |
| Real-time inference | EC2 price + markup | GCE price + markup | VM price + markup | N/A (use other tools) |
| Batch inference | Per-job pricing | Per-job pricing | Per-job pricing | Per-DPU pricing |
| HPO | Included | Included | Included | Via MLflow |
| Experiment tracking | Included | Included | Included | MLflow included |
| Model registry | Included | Included | Included | MLflow included |

### Storage Cost Comparison

| Tier | AWS S3 | GCP GCS | Azure Blob | Best Choice |
|------|--------|---------|-----------|-------------|
| Hot / Standard | $0.023/GB/mo | $0.020/GB/mo | $0.018/GB/mo | Azure cheapest |
| Warm / Nearline | $0.0125/GB/mo | $0.010/GB/mo | $0.010/GB/mo | GCP/Azure tied |
| Cold / Coldline | $0.004/GB/mo | $0.004/GB/mo | $0.002/GB/mo | Azure cheapest |
| Archive | $0.00099/GB/mo | $0.0012/GB/mo | $0.00099/GB/mo | AWS/Azure tied |

### Savings Plans and Commitments

| Type | AWS | GCP | Azure |
|------|-----|-----|-------|
| 1-year commitment | 30-40% savings | 20-37% CUD | 20-35% RI |
| 3-year commitment | 50-60% savings | 45-55% CUD | 50-65% RI |
| Spot / Preemptible | 60-90% savings | 60-80% savings | 60-80% savings |
| Sustained use discount | None (use SPs) | Auto 10-30% | None (use RIs) |

### Cost Optimization Features by Provider

| Feature | AWS | GCP | Azure |
|---------|-----|-----|-------|
| Spot termination notice | 2 min | 30 sec | 30 sec |
| Auto-scaling for ML | SageMaker auto-scaling | Vertex AI auto-scaling | Azure ML auto-scaling |
| Scale-to-zero inference | SageMaker Serverless | Cloud Run (GPU preview) | Container Apps |
| GPU sharing / MIG | P5 MIG, SageMaker multi-model | A100 MIG on GKE | AKS with MIG |
| Spot training support | SageMaker managed spot | Vertex AI preemptible | Azure ML low-priority |
| Cost anomaly detection | AWS Cost Anomaly Detection | GCP Budget alerts | Azure Cost Alerts |
| Billing export | CUR to S3 | BigQuery export | Cost Management export |

### Cloud vs On-Prem vs Hybrid Decision Guide

**When Cloud Wins**:
- **Bursty workloads**: Occasional large training runs separated by idle periods
- **Rapid experimentation**: Need access to many GPU types for benchmarking
- **Small teams**: Cannot justify hiring infrastructure engineers
- **Scale-to-zero inference**: Low-traffic models that do not need 24/7 uptime

**When On-Prem Wins**:
- **Sustained utilization > 60%**: Break-even typically at 12-18 months vs cloud on-demand
- **Data locality requirements**: Massive datasets that are expensive to transfer
- **Regulatory constraints**: Data sovereignty, air-gapped environments
- **Predictable workloads**: Consistent GPU demand with minimal variation

**Hybrid Strategy**:
```
Baseline load (predictable): On-prem or reserved instances
  -> Covers 60-70% of average compute needs
  -> 1-3 year commitments for 40-60% savings

Burst capacity (variable): Cloud spot/preemptible instances
  -> Handles peaks, deadlines, large experiments
  -> 60-90% savings vs on-demand

Development/experimentation: Cloud on-demand (small instances)
  -> Flexibility for quick iterations
  -> Shut down when not in use
```

**TCO Factors** (do not compare GPU $/hr alone):
- Hardware depreciation (3-5 years for GPUs)
- Power and cooling (PUE of 1.2-1.6)
- Datacenter space
- Network infrastructure
- System administration staffing
- Hardware failures and replacements
- Software licensing
- Opportunity cost of capital

---

## Quick Cost Reference Cards

### "I need to train a model" Decision Tree

```
1. How big is the model?
   - < 500M params -> T4 or V100 (16 GB)
   - 500M - 5B params -> A100 40GB
   - 5B - 15B params -> A100 80GB
   - 15B+ params -> Multi-GPU A100 80GB or H100

2. How long will training take?
   - < 1 hour -> On-demand instance (not worth spot overhead)
   - 1-8 hours -> Spot with checkpointing every 15 min
   - 8+ hours -> Spot with checkpointing every 10 min

3. Is this exploration or final training?
   - Exploration -> Use 5-10% of data, smallest viable GPU, aggressive early stopping
   - Validation -> Use 25% of data, right-sized GPU
   - Final -> Full data, right-sized GPU, full epoch budget

4. Can you use mixed precision?
   - Almost always yes. Enable AMP. Save 30-50% on training time.
```

### "I need to deploy a model" Decision Tree

```
1. What is the latency requirement?
   - < 10ms -> GPU inference, optimized model
   - 10-100ms -> GPU or optimized CPU (ONNX Runtime)
   - 100ms-1s -> CPU inference with quantized model
   - > 1s -> Batch inference

2. What is the traffic pattern?
   - Constant high traffic -> Reserved instances, auto-scaling
   - Variable traffic -> Auto-scaling with min replicas
   - Bursty/low traffic -> Scale-to-zero, serverless
   - Batch/periodic -> Batch inference jobs

3. How do I reduce inference cost?
   - Step 1: Quantize to INT8 (2-4x savings, < 1% accuracy loss)
   - Step 2: Use ONNX Runtime (1.5-3x speedup on CPU)
   - Step 3: Implement dynamic batching (2-5x throughput)
   - Step 4: Consider distilled model for 5-20x cost reduction
   - Step 5: Auto-scale aggressively, scale to zero when idle
```

### Monthly Cost Benchmarks (Typical Workloads)

| Workload | Minimal Setup | Typical Setup | Enterprise Setup |
|----------|--------------|---------------|-----------------|
| Single model training (weekly) | $50-200 | $200-1,000 | $1,000-10,000 |
| HPO (50 trials, monthly) | $100-500 | $500-5,000 | $5,000-50,000 |
| Real-time inference (1 model) | $50-400 | $400-2,000 | $2,000-20,000 |
| Batch inference (daily) | $30-100 | $100-500 | $500-5,000 |
| Experiment tracking + storage | $20-50 | $50-200 | $200-1,000 |
| **Total monthly ML spend** | **$250-1,250** | **$1,250-8,700** | **$8,700-86,000** |

---

## Storage Cost Optimization

### Tiered Storage Strategy

```
Hot tier  (SSD, $0.08-0.23/GB/mo): Active training data, current model, latest checkpoints
Warm tier (HDD, $0.01-0.05/GB/mo): Recent experiment artifacts (last 30 days)
Cold tier (Archive, $0.004/GB/mo): Historical data, old model versions, compliance archives
```

### Artifact Lifecycle Management

```python
# Example: S3 lifecycle policy for ML artifacts
lifecycle_policy = {
    "Rules": [
        {
            "ID": "checkpoints-cleanup",
            "Filter": {"Prefix": "checkpoints/"},
            "Status": "Enabled",
            "Transitions": [
                {"Days": 7, "StorageClass": "STANDARD_IA"},
                {"Days": 30, "StorageClass": "GLACIER"},
            ],
            "Expiration": {"Days": 90},
        },
        {
            "ID": "keep-best-models",
            "Filter": {"Prefix": "models/production/"},
            "Status": "Enabled",
            "Transitions": [
                {"Days": 365, "StorageClass": "GLACIER"},
            ],
            # No expiration -- keep production models indefinitely
        },
        {
            "ID": "tensorboard-logs",
            "Filter": {"Prefix": "logs/tensorboard/"},
            "Status": "Enabled",
            "Expiration": {"Days": 30},
        },
    ]
}
```

### Model Checkpoint Management

Do not save every epoch. Use intelligent checkpointing:

```python
class SmartCheckpointer:
    """Only keep the best N checkpoints, delete others to save storage."""

    def __init__(self, checkpoint_dir, max_checkpoints=3, metric="val_loss", mode="min"):
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.mode = mode
        self.checkpoints = []  # List of (metric_value, path)

    def save_if_improved(self, model, metric_value, epoch):
        import os, torch
        path = os.path.join(self.checkpoint_dir, f"model_epoch{epoch}.pt")
        torch.save(model.state_dict(), path)
        self.checkpoints.append((metric_value, path))

        # Sort: best first
        reverse = self.mode == "max"
        self.checkpoints.sort(key=lambda x: x[0], reverse=reverse)

        # Remove worst checkpoints beyond max_checkpoints
        while len(self.checkpoints) > self.max_checkpoints:
            _, old_path = self.checkpoints.pop()
            if os.path.exists(old_path):
                os.remove(old_path)
                print(f"Removed old checkpoint: {old_path}")
```

---

## Data Processing Cost Optimization

### Smart Sampling

Not every experiment needs the full dataset. Use progressive data scaling:

```python
def progressive_data_strategy(dataset_size, experiment_phase):
    """
    Return fraction of data to use based on experiment phase.

    - exploration: 5-10% of data (test hypotheses quickly)
    - validation: 25-50% (confirm approach works)
    - final: 100% (production training)
    """
    fractions = {
        "exploration": 0.05,
        "validation": 0.25,
        "final": 1.0,
    }
    return int(dataset_size * fractions.get(experiment_phase, 1.0))
```

### Data Caching

Cache preprocessed data to avoid recomputing:

```python
import hashlib, pickle, os

def cached_preprocess(raw_data, preprocess_fn, cache_dir="/tmp/ml_cache"):
    """Cache preprocessed data based on hash of raw data and function."""
    os.makedirs(cache_dir, exist_ok=True)
    data_hash = hashlib.md5(pickle.dumps(raw_data[:100])).hexdigest()[:12]
    fn_hash = hashlib.md5(preprocess_fn.__code__.co_code).hexdigest()[:8]
    cache_path = os.path.join(cache_dir, f"preprocessed_{data_hash}_{fn_hash}.pkl")

    if os.path.exists(cache_path):
        print(f"Loading cached preprocessed data from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    result = preprocess_fn(raw_data)
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    return result
```

### Efficient Data Loading

```python
# PyTorch DataLoader optimization
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=4,          # Parallel data loading
    pin_memory=True,        # Faster GPU transfer
    prefetch_factor=2,      # Prefetch batches
    persistent_workers=True, # Keep workers alive between epochs
)
```

---

## Cost Tracking and Budgeting

### Tagging Strategy

Every ML resource should be tagged for cost attribution:

```
Required tags:
  - team: "ml-platform", "nlp-research", "cv-team"
  - project: "fraud-detection-v2", "recommendation-engine"
  - environment: "development", "staging", "production"
  - experiment_id: "exp-2024-001"
  - cost_center: "engineering-ml"

Optional tags:
  - model: "bert-base", "resnet50"
  - phase: "training", "inference", "data-processing"
  - owner: "alice@example.com"
```

### Cost Tracking Implementation

```python
import time
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class CostTracker:
    """Track costs for an ML experiment or job."""
    experiment_id: str
    gpu_type: str
    gpu_cost_per_hour: float
    num_gpus: int = 1
    storage_gb: float = 0.0
    storage_cost_per_gb_month: float = 0.023  # S3 standard
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    extra_costs: dict = field(default_factory=dict)

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()

    @property
    def gpu_hours(self):
        if not self.start_time:
            return 0
        end = self.end_time or time.time()
        return ((end - self.start_time) / 3600) * self.num_gpus

    @property
    def compute_cost(self):
        return self.gpu_hours * self.gpu_cost_per_hour

    @property
    def storage_cost(self):
        return self.storage_gb * self.storage_cost_per_gb_month

    @property
    def total_cost(self):
        return self.compute_cost + self.storage_cost + sum(self.extra_costs.values())

    def report(self):
        return {
            "experiment_id": self.experiment_id,
            "gpu_type": self.gpu_type,
            "gpu_hours": round(self.gpu_hours, 2),
            "compute_cost": round(self.compute_cost, 2),
            "storage_cost": round(self.storage_cost, 2),
            "extra_costs": self.extra_costs,
            "total_cost": round(self.total_cost, 2),
        }
```

### Budget Alerts

Set budget thresholds and receive alerts before overspending:

```python
class BudgetGuard:
    """Stop experiments that exceed budget."""

    def __init__(self, max_budget_usd, warning_threshold=0.8):
        self.max_budget = max_budget_usd
        self.warning_threshold = warning_threshold
        self.alerted = False

    def check(self, cost_tracker):
        current = cost_tracker.total_cost
        if current >= self.max_budget:
            raise RuntimeError(
                f"Budget exceeded: ${current:.2f} >= ${self.max_budget:.2f}. "
                f"Stopping experiment {cost_tracker.experiment_id}."
            )
        if current >= self.max_budget * self.warning_threshold and not self.alerted:
            print(f"WARNING: {current / self.max_budget * 100:.0f}% of budget used "
                  f"(${current:.2f} / ${self.max_budget:.2f})")
            self.alerted = True
```

---

## Cost-Aware Experiment Design

### Experiment Cost Budgeting

Before running an experiment, estimate the cost:

```python
def estimate_experiment_cost(
    model_params_millions,
    dataset_size_gb,
    num_epochs,
    gpu_type="a100_40gb",
    num_hpo_trials=0,
):
    """
    Rough cost estimate for a training experiment.
    """
    gpu_costs = {
        "t4": {"price": 0.35, "throughput_factor": 1.0},
        "v100": {"price": 1.50, "throughput_factor": 2.5},
        "a100_40gb": {"price": 3.50, "throughput_factor": 8.0},
        "a100_80gb": {"price": 4.00, "throughput_factor": 9.0},
        "h100": {"price": 6.00, "throughput_factor": 15.0},
    }

    gpu = gpu_costs[gpu_type]
    # Rough estimate: 1 epoch over 1 GB data with 100M params takes ~1 hour on T4
    base_hours = (model_params_millions / 100) * dataset_size_gb * num_epochs
    adjusted_hours = base_hours / gpu["throughput_factor"]
    training_cost = adjusted_hours * gpu["price"]

    # HPO multiplier
    hpo_cost = training_cost * num_hpo_trials * 0.3  # HPO trials are usually shorter

    # Storage (rough: 2x dataset size for artifacts)
    storage_cost = dataset_size_gb * 2 * 0.023

    total = training_cost + hpo_cost + storage_cost
    return {
        "estimated_gpu_hours": round(adjusted_hours, 1),
        "training_cost": round(training_cost, 2),
        "hpo_cost": round(hpo_cost, 2),
        "storage_cost": round(storage_cost, 2),
        "total_estimated_cost": round(total, 2),
    }
```

### Prioritization Framework

Rank experiments by expected value per dollar:

```
Priority Score = (Expected Metric Improvement * Business Value) / Estimated Cost
```

Run cheap experiments first. Only invest in expensive runs after validating the approach on smaller scale.

---

## Resource Right-Sizing

### GPU Utilization Monitoring

```python
import subprocess
import csv
from io import StringIO

def get_gpu_utilization():
    """Query current GPU utilization using nvidia-smi."""
    result = subprocess.run(
        ["nvidia-smi",
         "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    gpus = []
    for line in csv.reader(StringIO(result.stdout)):
        if len(line) >= 6:
            gpus.append({
                "index": int(line[0].strip()),
                "name": line[1].strip(),
                "gpu_util_pct": float(line[2].strip()),
                "mem_util_pct": float(line[3].strip()),
                "mem_used_gb": float(line[4].strip()) / 1024,
                "mem_total_gb": float(line[5].strip()) / 1024,
            })
    return gpus

def right_sizing_recommendation(avg_gpu_util, avg_mem_util, current_gpu):
    """Suggest right-sizing based on utilization."""
    if avg_gpu_util < 30 and avg_mem_util < 40:
        return "DOWNSIZE: GPU is significantly underutilized. Consider a smaller instance."
    elif avg_gpu_util < 50 and avg_mem_util < 60:
        return "REVIEW: GPU may be oversized. Profile your workload for right-sizing."
    elif avg_gpu_util > 90 or avg_mem_util > 90:
        return "UPGRADE or OPTIMIZE: GPU is near capacity. Consider larger GPU or model optimization."
    else:
        return "OPTIMAL: GPU utilization is in a healthy range."
```

### Right-Sizing Decision Matrix

| GPU Utilization | Memory Utilization | Action |
|----------------|-------------------|--------|
| < 30% | < 40% | Downsize GPU (e.g., A100 -> V100 or T4) |
| 30-60% | < 60% | Consider smaller GPU or batch more work |
| 60-85% | 50-85% | Good fit, no changes needed |
| > 85% | > 85% | Consider upgrading or optimizing model |
| < 30% | > 80% | Memory-bound: use GPU with more VRAM but fewer cores |

---

## Shared Compute Resources and Scheduling

### Kubernetes for ML Workloads

```yaml
# GPU resource request with limits
apiVersion: v1
kind: Pod
metadata:
  name: training-job
  labels:
    app: ml-training
    cost-center: ml-team
spec:
  containers:
    - name: trainer
      image: training:latest
      resources:
        requests:
          nvidia.com/gpu: 1
          memory: "32Gi"
          cpu: "8"
        limits:
          nvidia.com/gpu: 1
          memory: "64Gi"
          cpu: "16"
  tolerations:
    - key: "nvidia.com/gpu"
      operator: "Exists"
      effect: "NoSchedule"
  nodeSelector:
    gpu-type: "a100"
  # Priority class for fair scheduling
  priorityClassName: training-medium
```

### Job Scheduling Best Practices

- Use **priority classes**: production inference > scheduled retraining > ad-hoc experiments
- Implement **preemption**: Lower-priority jobs yield GPUs to higher-priority ones
- Set **resource quotas** per team/namespace to prevent any single team from consuming all GPUs
- Use **gang scheduling** for distributed training (all pods start together or none do)
- **Time-based scheduling**: Run large training jobs during off-peak hours (nights/weekends) when GPU demand is lower

### Slurm Integration

For on-prem or hybrid clusters with Slurm:

```bash
#!/bin/bash
#SBATCH --job-name=model-training
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out

# Load modules
module load cuda/12.1
module load python/3.11

# Activate environment
source /home/$USER/envs/ml/bin/activate

# Run distributed training
torchrun --nproc_per_node=2 train.py \
    --model bert-large \
    --batch_size 32 \
    --epochs 10 \
    --mixed_precision bf16
```
