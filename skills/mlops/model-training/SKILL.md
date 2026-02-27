---
name: model-training
description: >
  Design and run ML model training pipelines with PyTorch, TensorFlow, scikit-learn, XGBoost, LightGBM, and
  HuggingFace. Covers hyperparameter tuning (Optuna, Ray Tune, Bayesian optimization), distributed training
  (DDP, DeepSpeed, FSDP, Horovod), mixed precision training (AMP, bf16), learning rate scheduling, early stopping,
  checkpointing, cross-validation, GPU memory optimization (gradient accumulation, gradient checkpointing),
  reproducibility, and config-driven training pipeline design. Use when training models, tuning hyperparameters,
  setting up distributed training, or optimizing training performance.
license: Apache-2.0
metadata:
  author: mlops-skills
  version: "1.0"
  category: mlops
---

# Model Training

## Overview

Model training transforms data and algorithms into predictive models. This skill covers
production-grade training pipelines with reproducibility, scalability, and optimization.

## When to Use This Skill

- Training new ML models from scratch
- Tuning hyperparameters for optimal performance
- Scaling training to multiple GPUs/nodes
- Optimizing training speed and memory usage
- Setting up reproducible training pipelines

## Step-by-Step Instructions

### 1. Config-Driven Training

```yaml
# config/train_config.yaml
experiment:
  name: "product-classifier-v2"
  seed: 42
  tracking_uri: "http://mlflow:5000"

data:
  train_path: "data/train.parquet"
  val_path: "data/val.parquet"
  test_path: "data/test.parquet"

model:
  type: "random_forest"  # or xgboost, pytorch, lightgbm
  params:
    n_estimators: 100
    max_depth: 10
    min_samples_split: 5

training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
  early_stopping:
    patience: 5
    metric: "val_loss"
    mode: "min"
  checkpoint:
    save_top_k: 3
    metric: "val_f1"
```

```python
import yaml
from pathlib import Path

def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)

def train_from_config(config_path):
    config = load_config(config_path)
    set_seed(config["experiment"]["seed"])
    data = load_data(config["data"])
    model = build_model(config["model"])
    train(model, data, config["training"])
```

### 2. PyTorch Training Loop

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

def train_pytorch(model, train_loader, val_loader, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0.01),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"] * len(train_loader)
    )
    scaler = GradScaler()  # Mixed precision
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(config["epochs"]):
        # Training
        model.train()
        train_loss = 0
        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            with autocast():  # Mixed precision forward pass
                output = model(X)
                loss = criterion(output, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                with autocast():
                    output = model(X)
                    val_loss += criterion(output, y).item()

        val_loss /= len(val_loader)
        scheduler.step()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= config["early_stopping"]["patience"]:
                print(f"Early stopping at epoch {epoch}")
                break
```

### 3. Hyperparameter Tuning with Optuna

```python
import optuna

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }

    model = xgb.XGBClassifier(**params)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_weighted")

    return scores.mean()

# Run optimization
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
)
study.optimize(objective, n_trials=100, timeout=3600)

print(f"Best params: {study.best_params}")
print(f"Best F1: {study.best_value:.4f}")
```

### 4. Distributed Training (PyTorch DDP)

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup_ddp(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def train_ddp(rank, world_size, model, dataset, config):
    setup_ddp(rank, world_size)

    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=config["batch_size"], sampler=sampler)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

    for epoch in range(config["epochs"]):
        sampler.set_epoch(epoch)
        for X, y in loader:
            X, y = X.to(rank), y.to(rank)
            loss = model(X, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if rank == 0:  # Save only on main process
            torch.save(model.module.state_dict(), f"checkpoint_epoch_{epoch}.pt")

    dist.destroy_process_group()

# Launch: torchrun --nproc_per_node=4 train.py
```

### 5. HuggingFace Trainer

```python
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    TrainingArguments, Trainer
)

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=True,
    gradient_accumulation_steps=2,
    dataloader_num_workers=4,
    report_to="mlflow",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
```

### 6. scikit-learn / XGBoost / LightGBM

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold
import xgboost as xgb
import lightgbm as lgb

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=100, max_depth=6, learning_rate=0.1,
    tree_method="hist", device="cuda"  # GPU training
)
scores = cross_val_score(xgb_model, X, y, cv=cv, scoring="f1_weighted")

# LightGBM
lgb_model = lgb.LGBMClassifier(
    n_estimators=100, max_depth=6, learning_rate=0.1,
    device="gpu"
)
```

### 7. GPU Memory Optimization

```python
# Gradient accumulation (simulate larger batch size)
accumulation_steps = 4
for i, (X, y) in enumerate(loader):
    loss = model(X, y) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# Gradient checkpointing (trade compute for memory)
model.gradient_checkpointing_enable()

# Mixed precision training
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    output = model(X)
    loss = criterion(output, y)
```

### 8. Learning Rate Schedules

| Schedule | Best For |
|----------|----------|
| CosineAnnealing | General training, transformers |
| OneCycleLR | Fast convergence, super-convergence |
| StepLR | Simple decay at fixed intervals |
| ReduceLROnPlateau | Adaptive decay on validation metric |
| Warmup + Cosine | Pre-trained model fine-tuning |

## Best Practices

1. **Start with a baseline** - Simple model, default hyperparameters
2. **Use config files** - Never hardcode hyperparameters
3. **Set seeds everywhere** - Python, NumPy, PyTorch, CUDA
4. **Log everything** - Parameters, metrics, environment, code hash
5. **Use mixed precision** - Almost free speedup on modern GPUs
6. **Gradient accumulation** before requesting more GPUs
7. **Early stopping** to avoid overfitting and wasted compute
8. **Cross-validate** for robust evaluation
9. **Profile before optimizing** - Find actual bottlenecks first
10. **Save checkpoints** regularly for crash recovery

## Scripts

- `scripts/train_model.py` - Config-driven multi-framework training
- `scripts/distributed_train.py` - PyTorch DDP distributed training

## References

See [references/REFERENCE.md](references/REFERENCE.md) for framework comparisons and tuning guides.
