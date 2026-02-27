#!/usr/bin/env python3
"""
PyTorch Distributed Data-Parallel (DDP) Training Script

Trains image-classification models with DDP, mixed-precision (AMP),
gradient accumulation, learning-rate warmup, and cosine scheduling.

Usage:
    # Single-GPU (for debugging)
    python distributed_train.py --data ./imagenet --model resnet18 --epochs 10 \
        --batch-size 64 --lr 0.001 --output ./checkpoints

    # Multi-GPU on one node via torchrun
    torchrun --nproc_per_node=4 distributed_train.py --data ./imagenet \
        --model resnet50 --epochs 90 --batch-size 64 --lr 0.1 \
        --mixed-precision --output ./checkpoints

    # Multi-node (2 nodes x 4 GPUs)
    torchrun --nnodes=2 --nproc_per_node=4 --rdzv_backend=c10d \
        --rdzv_endpoint=master:29500 distributed_train.py --data ./imagenet \
        --model resnet50 --epochs 90 --batch-size 64 --lr 0.1 \
        --mixed-precision --gradient-accumulation 4 --output ./checkpoints

Dependencies:
    - Python 3.8+, torch >= 2.0, torchvision
    - Optional: timm (for additional model architectures)
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------

def _import_torch():
    import torch
    return torch


def _import_torchvision():
    import torchvision
    import torchvision.transforms as T
    return torchvision, T


# ---------------------------------------------------------------------------
# DDP utilities
# ---------------------------------------------------------------------------

def setup_ddp() -> Tuple[int, int]:
    """Initialise the distributed process group and return (local_rank, world_size).

    When launched via ``torchrun`` the environment variables LOCAL_RANK,
    RANK, and WORLD_SIZE are set automatically.  For single-GPU runs these
    default to 0/1 so the script still works.
    """
    torch = _import_torch()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        logger.info("DDP initialised: rank=%d  world_size=%d", local_rank, world_size)
    else:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        logger.info("Running in single-process mode (no DDP)")

    return local_rank, world_size


def cleanup_ddp():
    torch = _import_torch()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def is_main_process() -> bool:
    return int(os.environ.get("RANK", 0)) == 0


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def build_dataset(data_dir: str, split: str = "train"):
    """Build an ImageFolder dataset with standard augmentation.

    Expected directory layout::

        data_dir/
          train/
            class_a/  img1.jpg ...
            class_b/  img2.jpg ...
          val/
            class_a/ ...
    """
    _, T = _import_torchvision()
    import torchvision

    if split == "train":
        transform = T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    path = os.path.join(data_dir, split)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Dataset split directory not found: {path}")

    dataset = torchvision.datasets.ImageFolder(path, transform=transform)
    logger.info("Loaded %s split: %d images, %d classes", split, len(dataset), len(dataset.classes))
    return dataset


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(name: str, num_classes: int):
    """Return a torchvision or timm model."""
    torch = _import_torch()
    torchvision, _ = _import_torchvision()

    name_lower = name.lower()
    if name_lower == "resnet18":
        model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    elif name_lower == "resnet50":
        model = torchvision.models.resnet50(weights=None, num_classes=num_classes)
    elif name_lower == "custom":
        # Lightweight custom CNN for quick experiments
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1), torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(128, num_classes),
        )
    else:
        try:
            import timm
            model = timm.create_model(name_lower, pretrained=False, num_classes=num_classes)
            logger.info("Loaded model '%s' from timm", name_lower)
        except ImportError:
            raise ValueError(
                f"Unknown model '{name}'. Install timm for extra architectures."
            )

    param_count = sum(p.numel() for p in model.parameters())
    logger.info("Model '%s': %.2fM parameters", name, param_count / 1e6)
    return model


# ---------------------------------------------------------------------------
# LR scheduler helpers
# ---------------------------------------------------------------------------

def build_scheduler(optimizer, args, steps_per_epoch: int):
    """Build a learning rate scheduler with optional linear warmup."""
    torch = _import_torch()
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = int(args.warmup_epochs * steps_per_epoch)

    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)

    if warmup_steps > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_steps,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps],
        )
    else:
        scheduler = cosine

    return scheduler


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model, loader, criterion, optimizer, scheduler, scaler, device,
    epoch: int, grad_accum: int, use_amp: bool,
) -> Dict[str, float]:
    """Run one training epoch, return loss and accuracy."""
    torch = _import_torch()
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad(set_to_none=True)

    for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets) / grad_accum

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        running_loss += loss.item() * grad_accum * images.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return {"loss": epoch_loss, "accuracy": epoch_acc}


def validate(model, loader, criterion, device, use_amp: bool) -> Dict[str, float]:
    """Run validation, return loss and accuracy."""
    torch = _import_torch()
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return {
        "loss": running_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, metrics, path):
    torch = _import_torch()
    state = {
        "epoch": epoch,
        "model_state_dict": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "metrics": metrics,
    }
    torch.save(state, path)
    logger.info("Checkpoint saved: %s (epoch %d)", path, epoch)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None):
    torch = _import_torch()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    logger.info("Resumed from checkpoint: %s (epoch %d)", path, ckpt["epoch"])
    return ckpt["epoch"]


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace):
    torch = _import_torch()

    local_rank, world_size = setup_ddp()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    use_amp = args.mixed_precision and torch.cuda.is_available()

    # Datasets and loaders
    train_ds = build_dataset(args.data, "train")
    val_ds = build_dataset(args.data, "val")
    num_classes = len(train_ds.classes)

    train_sampler = (
        torch.utils.data.distributed.DistributedSampler(train_ds, shuffle=True)
        if world_size > 1 else None
    )
    val_sampler = (
        torch.utils.data.distributed.DistributedSampler(val_ds, shuffle=False)
        if world_size > 1 else None
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        sampler=val_sampler, num_workers=4, pin_memory=True,
    )

    # Model
    model = build_model(args.model, num_classes).to(device)
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # Optimizer, scheduler, scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation)
    scheduler = build_scheduler(optimizer, args, steps_per_epoch)
    scaler = torch.amp.GradScaler(enabled=use_amp)
    criterion = torch.nn.CrossEntropyLoss()

    # Resume from checkpoint
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler, scaler) + 1

    os.makedirs(args.output, exist_ok=True)
    history = []
    best_val_acc = 0.0

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler,
            device, epoch, args.gradient_accumulation, use_amp,
        )
        val_metrics = validate(model, val_loader, criterion, device, use_amp)

        epoch_time = time.time() - epoch_start
        record = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 6),
            "train_acc": round(train_metrics["accuracy"], 4),
            "val_loss": round(val_metrics["loss"], 6),
            "val_acc": round(val_metrics["accuracy"], 4),
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_s": round(epoch_time, 1),
        }
        history.append(record)

        if is_main_process():
            logger.info(
                "Epoch %d/%d  train_loss=%.4f  train_acc=%.4f  val_loss=%.4f  val_acc=%.4f  lr=%.2e  (%.1fs)",
                epoch + 1, args.epochs,
                train_metrics["loss"], train_metrics["accuracy"],
                val_metrics["loss"], val_metrics["accuracy"],
                optimizer.param_groups[0]["lr"], epoch_time,
            )

            # Save latest checkpoint
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, val_metrics,
                os.path.join(args.output, "checkpoint_latest.pt"),
            )

            # Save best checkpoint
            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                save_checkpoint(
                    model, optimizer, scheduler, scaler, epoch, val_metrics,
                    os.path.join(args.output, "checkpoint_best.pt"),
                )

    # Save training report (rank 0 only)
    if is_main_process():
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "dataset": args.data,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "world_size": world_size,
            "mixed_precision": use_amp,
            "gradient_accumulation": args.gradient_accumulation,
            "best_val_accuracy": round(best_val_acc, 4),
            "history": history,
        }
        report_path = os.path.join(args.output, "training_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Training report saved to %s", report_path)
        logger.info("Best validation accuracy: %.4f", best_val_acc)

    cleanup_ddp()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PyTorch DDP distributed training script.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data", required=True, help="Root directory of the dataset (train/ and val/ sub-dirs)")
    parser.add_argument("--model", default="resnet18", help="Model architecture: resnet18, resnet50, custom, or any timm name")
    parser.add_argument("--epochs", type=int, default=90, help="Total training epochs (default: 90)")
    parser.add_argument("--batch-size", type=int, default=64, help="Per-GPU batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate (default: 1e-3)")
    parser.add_argument("--warmup-epochs", type=float, default=5.0, help="Linear warmup epochs (default: 5)")
    parser.add_argument("--output", default="./checkpoints", help="Directory for checkpoints and report")
    parser.add_argument("--mixed-precision", action="store_true", help="Enable AMP mixed-precision training")
    parser.add_argument("--gradient-accumulation", type=int, default=1, help="Gradient accumulation steps (default: 1)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not os.path.isdir(args.data):
        logger.error("Data directory not found: %s", args.data)
        sys.exit(1)

    try:
        run(args)
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        cleanup_ddp()
        sys.exit(130)
    except Exception:
        logger.exception("Training failed")
        cleanup_ddp()
        sys.exit(1)


if __name__ == "__main__":
    main()
