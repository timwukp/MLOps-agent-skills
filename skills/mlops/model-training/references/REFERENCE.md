# Model Training Reference Guide

## Training Framework Comparison

| Framework    | Type           | Best For                        | Scalability | Ease of Use | GPU Support |
|-------------|----------------|----------------------------------|-------------|-------------|-------------|
| scikit-learn | Traditional ML | Tabular data, prototyping       | Low         | High        | Limited     |
| XGBoost      | Gradient Boost | Structured/tabular data         | Medium      | High        | Yes         |
| LightGBM     | Gradient Boost | Large tabular datasets          | Medium      | High        | Yes         |
| CatBoost     | Gradient Boost | Categorical-heavy tabular data  | Medium      | High        | Yes         |
| PyTorch      | Deep Learning  | Research, NLP, custom models    | High        | Medium      | Full        |
| TensorFlow   | Deep Learning  | Production DL, mobile/edge      | High        | Medium      | Full        |

### When to Choose What

- **scikit-learn**: Start here for tabular data. Fast iteration, excellent preprocessing pipelines.
- **XGBoost**: Default for Kaggle-style tabular problems. Strong regularization, handles missing values.
- **LightGBM**: Preferred when datasets exceed 100K rows. Faster training via histogram-based splitting.
- **CatBoost**: Best out-of-the-box performance with categorical features (no manual encoding needed).
- **PyTorch**: Preferred for research, transformer-based models, and custom training loops.
- **TensorFlow**: Preferred for production pipelines with TFX, mobile deployment via TFLite, and TPU training.

## Hyperparameter Optimization Comparison

| Feature             | Optuna           | Ray Tune          | Hyperopt         |
|--------------------|------------------|-------------------|------------------|
| Search Algorithms  | TPE, CMA-ES, Grid, Random | All (wraps Optuna, Hyperopt, etc.) | TPE, Random, Adaptive TPE |
| Pruning            | Built-in (Median, Hyperband) | Schedulers (ASHA, PBT, HyperBand) | No native pruning |
| Distributed        | Via RDB storage  | Native distributed | Via MongoDB      |
| Dashboard          | Optuna Dashboard | TensorBoard integration | No built-in     |
| Framework Agnostic | Yes              | Yes               | Yes              |
| Learning Curve     | Low              | Medium            | Low              |

### Configuration Examples

```python
# Optuna example
import optuna

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    n_layers = trial.suggest_int("n_layers", 1, 5)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    # ... build and train model ...
    return validation_accuracy

study = optuna.create_study(direction="maximize", pruner=optuna.pruners.HyperbandPruner())
study.optimize(objective, n_trials=100, timeout=3600)
```

```python
# Ray Tune example
from ray import tune
from ray.tune.schedulers import ASHAScheduler

search_space = {
    "lr": tune.loguniform(1e-5, 1e-1),
    "batch_size": tune.choice([32, 64, 128, 256]),
    "n_layers": tune.randint(1, 6),
}

scheduler = ASHAScheduler(max_t=100, grace_period=10, reduction_factor=2)
tuner = tune.Tuner(train_fn, param_space=search_space,
    tune_config=tune.TuneConfig(scheduler=scheduler, num_samples=50))
results = tuner.fit()
```

## Distributed Training Strategies

### PyTorch Distributed Data Parallel (DDP)

- Replicates the model on each GPU; splits data across GPUs.
- Synchronizes gradients via all-reduce after each backward pass.
- Best for: models that fit in a single GPU memory.

```python
# DDP launch
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
    --master_addr="192.168.1.1" --master_port=29500 train.py
```

### Fully Sharded Data Parallel (FSDP)

- Shards model parameters, gradients, and optimizer states across GPUs.
- Trades communication overhead for memory efficiency.
- Best for: large models (1B+ parameters) that do not fit on a single GPU.

### DeepSpeed ZeRO Stages

| Stage   | What is Sharded                        | Memory Savings | Communication Overhead |
|---------|----------------------------------------|----------------|----------------------|
| ZeRO-1  | Optimizer states                       | ~4x            | Low                  |
| ZeRO-2  | Optimizer states + gradients           | ~8x            | Medium               |
| ZeRO-3  | Optimizer states + gradients + params  | Linear scaling  | High                 |
| ZeRO-Infinity | ZeRO-3 + NVMe offloading        | Maximum         | Highest              |

### Horovod

- Framework-agnostic distributed training (TensorFlow, PyTorch, MXNet).
- Uses ring-allreduce for gradient synchronization.
- Simpler API than native DDP but less flexible.
- Best for: multi-framework teams, Spark integration via Horovod Spark.

## Mixed Precision Training Guide

### bf16 vs fp16

| Aspect            | fp16 (float16)          | bf16 (bfloat16)        |
|-------------------|-------------------------|------------------------|
| Exponent bits     | 5                       | 8                      |
| Mantissa bits     | 10                      | 7                      |
| Dynamic range     | Limited (needs scaling) | Same as fp32           |
| Precision         | Higher precision        | Lower precision        |
| Loss scaling      | Required                | Not required           |
| Hardware          | All modern GPUs         | A100+, TPUs            |
| Stability         | Can overflow/underflow  | Very stable            |

### When to Use

- **fp16**: Use when training on V100 or older GPUs. Always pair with dynamic loss scaling.
- **bf16**: Preferred on A100/H100/H200 GPUs and TPUs. No loss scaling needed, fewer NaN issues.
- **tf32**: Automatic on Ampere+ GPUs for matmul operations. No code changes required.

```python
# PyTorch AMP with fp16
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    with autocast(dtype=torch.float16):
        loss = model(batch)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# PyTorch AMP with bf16 (no scaler needed)
with autocast(dtype=torch.bfloat16):
    loss = model(batch)
loss.backward()
optimizer.step()
```

## GPU Selection Guide

| GPU            | VRAM   | Best For                          | Approx. Max Model Size (full fine-tune) |
|----------------|--------|-----------------------------------|-----------------------------------------|
| T4             | 16 GB  | Inference, small model training   | ~500M params                            |
| A10G           | 24 GB  | Medium model fine-tuning          | ~1B params                              |
| V100           | 16/32 GB | General training                | ~1-2B params                            |
| A100           | 40/80 GB | Large model training, bf16      | ~3-7B params                            |
| H100           | 80 GB  | LLM training, fastest throughput  | ~7B params                              |
| H200           | 141 GB | Largest single-GPU capacity       | ~13B params                             |

*Note*: With techniques like LoRA/QLoRA, effective model sizes are 4-8x larger than listed.

## Training Debugging Checklist

### Loss Not Decreasing

1. Verify data loading is correct (visualize a batch, check labels match inputs).
2. Confirm the learning rate is not too low (try 10x higher) or too high (try 10x lower).
3. Check for vanishing gradients: print gradient norms per layer.
4. Ensure the loss function matches the task (cross-entropy for classification, MSE for regression).
5. Try overfitting a single batch first to confirm the model can learn.
6. Verify weight initialization is appropriate for the architecture.

### Overfitting

1. Increase training data (augmentation, synthetic data).
2. Add regularization: dropout, weight decay (L2), label smoothing.
3. Reduce model capacity (fewer layers, smaller hidden dimensions).
4. Apply early stopping based on validation loss.
5. Use data augmentation appropriate for the domain.

### Gradient Issues

- **Exploding gradients**: Apply gradient clipping (`torch.nn.utils.clip_grad_norm_`), reduce learning rate.
- **Vanishing gradients**: Use residual connections, batch normalization, LSTM/GRU over vanilla RNN.
- **NaN loss**: Check for division by zero, log(0), reduce learning rate, enable anomaly detection (`torch.autograd.set_detect_anomaly(True)`).

## Early Stopping and Learning Rate Scheduling

### Early Stopping Best Practices

```python
# PyTorch early stopping pattern
best_val_loss = float("inf")
patience_counter = 0
patience = 10

for epoch in range(max_epochs):
    train_loss = train_one_epoch(model, train_loader)
    val_loss = evaluate(model, val_loader)

    if val_loss < best_val_loss - min_delta:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
```

- Set `patience` to 5-20 epochs depending on convergence speed.
- Use `min_delta` (e.g., 1e-4) to avoid stopping on noise.
- Always save the best checkpoint, not the last one.

### Learning Rate Scheduling Strategies

| Scheduler         | When to Use                                 | Configuration Tip                        |
|-------------------|---------------------------------------------|------------------------------------------|
| StepLR            | Simple baseline, fixed decay schedule       | Decay by 0.1 every 30 epochs            |
| CosineAnnealing   | Most deep learning tasks                   | Set T_max to total training epochs       |
| OneCycleLR        | Fast convergence, super-convergence        | Set max_lr via LR range test             |
| ReduceOnPlateau   | When you cannot predict convergence pattern | patience=5, factor=0.5                   |
| WarmupCosine       | Transformer models, large batch training   | Warmup for 5-10% of total steps         |
| Linear warmup+decay| LLM fine-tuning                           | Standard for Hugging Face Trainer        |

### Configuration Example

```python
# Cosine annealing with warmup
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

warmup = LinearLR(optimizer, start_factor=0.01, total_iters=500)
cosine = CosineAnnealingLR(optimizer, T_max=total_steps - 500, eta_min=1e-6)
scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[500])
```

## Further Reading

- [PyTorch Distributed Training Guide](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [DeepSpeed Documentation](https://www.deepspeed.ai/getting-started/)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Ray Tune User Guide](https://docs.ray.io/en/latest/tune/index.html)
- [Mixed Precision Training (NVIDIA)](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/)
- [Hugging Face Training Arguments Reference](https://huggingface.co/docs/transformers/main_classes/trainer)
- [Google Best Practices for ML Engineering](https://developers.google.com/machine-learning/guides/rules-of-ml)
- [fastai Learning Rate Finder](https://docs.fast.ai/callback.schedule.html)
