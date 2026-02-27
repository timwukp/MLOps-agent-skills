#!/usr/bin/env python3
"""
Model Compression Toolkit

Provides quantization, pruning, and knowledge distillation utilities for
reducing model size and inference cost. Includes benchmarking to compare
original vs. compressed model performance.

Usage:
    python model_compress.py quantize --model model.pt --method dynamic --output quantized.pt
    python model_compress.py prune --model model.pt --method magnitude --sparsity 0.5 --output pruned.pt
    python model_compress.py distill --teacher teacher.pt --student student.pt --data data/ --epochs 10
    python model_compress.py benchmark --original model.pt --compressed compressed.pt --data data/
    python model_compress.py compare --original model.pt --compressed compressed.pt
    python model_compress.py demo

Dependencies:
    - Python 3.8+
    - torch >= 2.0
    - numpy

Optional:
    - onnx, onnxruntime (for ONNX export benchmarks)
    - torchvision (for demo models)
"""

import argparse
import json
import os
import sys
import time
import tempfile
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.utils.prune as prune
    import torch.quantization as quantization
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data classes for results
# ---------------------------------------------------------------------------

@dataclass
class CompressionResult:
    """Result of a model compression operation."""
    method: str
    original_size_mb: float
    compressed_size_mb: float
    compression_ratio: float
    original_params: int
    compressed_params: int
    sparsity: float
    notes: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class BenchmarkResult:
    """Result of a model benchmark."""
    model_name: str
    size_mb: float
    num_params: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_samples_per_sec: float
    peak_memory_mb: float

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_model_size_mb(model: "nn.Module") -> float:
    """Calculate model size in MB by saving to a temporary file."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=True) as f:
        torch.save(model.state_dict(), f.name)
        size_bytes = os.path.getsize(f.name)
    return size_bytes / (1024 * 1024)


def count_parameters(model: "nn.Module") -> int:
    """Count total parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def count_nonzero_parameters(model: "nn.Module") -> int:
    """Count non-zero parameters in a model."""
    total = 0
    for p in model.parameters():
        total += torch.count_nonzero(p).item()
    return total


def calculate_sparsity(model: "nn.Module") -> float:
    """Calculate the fraction of zero-valued parameters."""
    total = count_parameters(model)
    nonzero = count_nonzero_parameters(model)
    if total == 0:
        return 0.0
    return 1.0 - (nonzero / total)


def create_sample_model(model_type: str = "mlp") -> "nn.Module":
    """Create a sample model for demonstration purposes."""
    if model_type == "mlp":
        model = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
    elif model_type == "cnn":
        model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 10),
        )
    elif model_type == "transformer_block":
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=512, batch_first=True,
        )
        model = nn.Sequential(
            nn.Linear(256, 256),
            nn.TransformerEncoder(encoder_layer, num_layers=4),
            nn.Linear(256, 10),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'mlp', 'cnn', or 'transformer_block'.")

    return model


def create_sample_input(model_type: str = "mlp", batch_size: int = 32) -> "torch.Tensor":
    """Create appropriate sample input for a model type."""
    if model_type == "mlp":
        return torch.randn(batch_size, 784)
    elif model_type == "cnn":
        return torch.randn(batch_size, 3, 32, 32)
    elif model_type == "transformer_block":
        return torch.randn(batch_size, 16, 256)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def quantize_dynamic(model: "nn.Module", dtype=None) -> Tuple["nn.Module", CompressionResult]:
    """
    Apply dynamic quantization to a model.

    Dynamic quantization quantizes weights statically and activations dynamically
    at inference time. No calibration data required. Best suited for LSTM and
    Transformer models, and generally effective for CPU inference.
    """
    if dtype is None:
        dtype = torch.qint8

    original_size = get_model_size_mb(model)
    original_params = count_parameters(model)

    # Dynamic quantization targets Linear and LSTM layers
    quantized = torch.quantization.quantize_dynamic(
        model, {nn.Linear, nn.LSTM, nn.GRU}, dtype=dtype,
    )

    compressed_size = get_model_size_mb(quantized)
    compressed_params = count_parameters(quantized)
    sparsity = calculate_sparsity(quantized)

    result = CompressionResult(
        method="dynamic_quantization",
        original_size_mb=round(original_size, 2),
        compressed_size_mb=round(compressed_size, 2),
        compression_ratio=round(original_size / max(compressed_size, 0.01), 2),
        original_params=original_params,
        compressed_params=compressed_params,
        sparsity=round(sparsity, 4),
        notes=f"Dynamic quantization to {dtype}. No calibration needed.",
    )

    return quantized, result


def quantize_static(
    model: "nn.Module",
    calibration_data: Optional[List] = None,
    num_calibration_batches: int = 100,
) -> Tuple["nn.Module", CompressionResult]:
    """
    Apply post-training static quantization.

    Static quantization quantizes both weights and activations. Requires
    calibration data to determine activation ranges. Generally provides
    better accuracy than dynamic quantization.
    """
    original_size = get_model_size_mb(model)
    original_params = count_parameters(model)

    # Prepare model for static quantization
    model_prepared = model.cpu().eval()

    # Fuse common patterns (Conv+BN+ReLU, Linear+ReLU)
    # This is model-specific; here we handle Sequential models generically
    model_prepared.qconfig = quantization.get_default_qconfig("x86")
    quantization.prepare(model_prepared, inplace=True)

    # Calibration pass
    if calibration_data is not None:
        with torch.no_grad():
            for i, data in enumerate(calibration_data):
                if i >= num_calibration_batches:
                    break
                model_prepared(data)
    else:
        # Use random data for calibration if none provided
        print("  No calibration data provided; using random inputs for calibration.")
        with torch.no_grad():
            for _ in range(num_calibration_batches):
                dummy = torch.randn(1, 784)  # Assumes MLP input shape
                try:
                    model_prepared(dummy)
                except Exception:
                    break

    # Convert to quantized model
    quantized = quantization.convert(model_prepared, inplace=False)

    compressed_size = get_model_size_mb(quantized)
    compressed_params = count_parameters(quantized)
    sparsity = calculate_sparsity(quantized)

    result = CompressionResult(
        method="static_quantization",
        original_size_mb=round(original_size, 2),
        compressed_size_mb=round(compressed_size, 2),
        compression_ratio=round(original_size / max(compressed_size, 0.01), 2),
        original_params=original_params,
        compressed_params=compressed_params,
        sparsity=round(sparsity, 4),
        notes=f"Static quantization with {num_calibration_batches} calibration batches.",
    )

    return quantized, result


class QuantizationAwareTrainingWrapper:
    """
    Helper for Quantization-Aware Training (QAT).

    QAT inserts fake quantization operations during training so the model
    learns to be robust to quantization error. Provides the best accuracy
    among quantization methods but requires retraining.

    Usage:
        qat = QuantizationAwareTrainingWrapper(model, qconfig="x86")
        prepared_model = qat.prepare()
        # ... train prepared_model as usual ...
        quantized_model = qat.convert()
    """

    def __init__(self, model: "nn.Module", qconfig: str = "x86"):
        self.original_model = model
        self.qconfig = qconfig
        self.prepared_model = None
        self.original_size = get_model_size_mb(model)
        self.original_params = count_parameters(model)

    def prepare(self) -> "nn.Module":
        """Prepare model for QAT by inserting fake quantization modules."""
        model = self.original_model.cpu().train()
        model.qconfig = quantization.get_default_qat_qconfig(self.qconfig)
        self.prepared_model = quantization.prepare_qat(model, inplace=False)
        print("  Model prepared for QAT. Train as usual, then call .convert().")
        return self.prepared_model

    def convert(self) -> Tuple["nn.Module", CompressionResult]:
        """Convert QAT model to fully quantized model."""
        if self.prepared_model is None:
            raise RuntimeError("Call .prepare() and train the model first.")

        self.prepared_model.eval()
        quantized = quantization.convert(self.prepared_model, inplace=False)

        compressed_size = get_model_size_mb(quantized)
        compressed_params = count_parameters(quantized)

        result = CompressionResult(
            method="quantization_aware_training",
            original_size_mb=round(self.original_size, 2),
            compressed_size_mb=round(compressed_size, 2),
            compression_ratio=round(self.original_size / max(compressed_size, 0.01), 2),
            original_params=self.original_params,
            compressed_params=compressed_params,
            sparsity=round(calculate_sparsity(quantized), 4),
            notes="Quantization-Aware Training. Best accuracy among quantization methods.",
        )

        return quantized, result


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------

def prune_magnitude_unstructured(
    model: "nn.Module",
    sparsity: float = 0.5,
    make_permanent: bool = True,
) -> Tuple["nn.Module", CompressionResult]:
    """
    Apply global unstructured magnitude pruning.

    Removes the smallest-magnitude weights across the entire model.
    High sparsity ratios (70-95%) are achievable, but actual speedup
    requires sparse computation support (e.g., sparse CUDA kernels).
    """
    import copy
    original_size = get_model_size_mb(model)
    original_params = count_parameters(model)

    pruned_model = copy.deepcopy(model)

    # Collect all parameters to prune
    parameters_to_prune = []
    for name, module in pruned_model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            parameters_to_prune.append((module, "weight"))

    if not parameters_to_prune:
        return pruned_model, CompressionResult(
            method="magnitude_unstructured_pruning",
            original_size_mb=round(original_size, 2),
            compressed_size_mb=round(original_size, 2),
            compression_ratio=1.0,
            original_params=original_params,
            compressed_params=original_params,
            sparsity=0.0,
            notes="No prunable layers found.",
        )

    # Global unstructured pruning by L1 magnitude
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=sparsity,
    )

    # Make pruning permanent (remove re-parametrization)
    if make_permanent:
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)

    actual_sparsity = calculate_sparsity(pruned_model)
    compressed_size = get_model_size_mb(pruned_model)

    result = CompressionResult(
        method="magnitude_unstructured_pruning",
        original_size_mb=round(original_size, 2),
        compressed_size_mb=round(compressed_size, 2),
        compression_ratio=round(original_size / max(compressed_size, 0.01), 2),
        original_params=original_params,
        compressed_params=count_nonzero_parameters(pruned_model),
        sparsity=round(actual_sparsity, 4),
        notes=(
            f"Target sparsity: {sparsity:.0%}, actual: {actual_sparsity:.1%}. "
            f"Note: file size may not decrease without sparse storage format."
        ),
    )

    return pruned_model, result


def prune_structured(
    model: "nn.Module",
    sparsity: float = 0.3,
    make_permanent: bool = True,
) -> Tuple["nn.Module", CompressionResult]:
    """
    Apply structured pruning (remove entire neurons/channels).

    Structured pruning removes entire output channels (for Conv2d) or
    output neurons (for Linear layers). This directly reduces model
    dimensions and provides real speedups without sparse hardware.
    """
    import copy
    original_size = get_model_size_mb(model)
    original_params = count_parameters(model)

    pruned_model = copy.deepcopy(model)

    pruned_layers = 0
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name="weight", amount=sparsity, n=1, dim=0)
            if make_permanent:
                prune.remove(module, "weight")
            pruned_layers += 1
        elif isinstance(module, nn.Linear):
            prune.ln_structured(module, name="weight", amount=sparsity, n=1, dim=0)
            if make_permanent:
                prune.remove(module, "weight")
            pruned_layers += 1

    actual_sparsity = calculate_sparsity(pruned_model)
    compressed_size = get_model_size_mb(pruned_model)

    result = CompressionResult(
        method="structured_pruning",
        original_size_mb=round(original_size, 2),
        compressed_size_mb=round(compressed_size, 2),
        compression_ratio=round(original_size / max(compressed_size, 0.01), 2),
        original_params=original_params,
        compressed_params=count_nonzero_parameters(pruned_model),
        sparsity=round(actual_sparsity, 4),
        notes=(
            f"Structured pruning at {sparsity:.0%} on {pruned_layers} layers. "
            f"Directly reduces compute without sparse kernel support."
        ),
    )

    return pruned_model, result


def iterative_pruning(
    model: "nn.Module",
    target_sparsity: float = 0.9,
    steps: int = 5,
    finetune_fn=None,
) -> Tuple["nn.Module", List[CompressionResult]]:
    """
    Gradually prune model to target sparsity in multiple steps.

    Iterative pruning with fine-tuning between steps generally preserves
    accuracy much better than one-shot pruning at high sparsity levels.
    """
    import copy
    results = []
    current_model = copy.deepcopy(model)

    sparsity_per_step = 1.0 - (1.0 - target_sparsity) ** (1.0 / steps)

    for step in range(steps):
        current_sparsity = calculate_sparsity(current_model)
        step_target = 1.0 - (1.0 - current_sparsity) * (1.0 - sparsity_per_step)

        print(f"  Pruning step {step+1}/{steps}: target sparsity {step_target:.1%}")

        current_model, result = prune_magnitude_unstructured(
            current_model, sparsity=sparsity_per_step, make_permanent=True,
        )
        results.append(result)

        # Fine-tune if a function is provided
        if finetune_fn is not None:
            print(f"  Fine-tuning after step {step+1}...")
            finetune_fn(current_model)

    return current_model, results


# ---------------------------------------------------------------------------
# Knowledge Distillation
# ---------------------------------------------------------------------------

class DistillationTrainer:
    """
    Knowledge distillation trainer.

    Trains a smaller student model to mimic a larger teacher model's outputs.
    The student learns from both the soft targets (teacher's probability
    distribution) and the hard targets (ground truth labels).
    """

    def __init__(
        self,
        teacher: "nn.Module",
        student: "nn.Module",
        temperature: float = 4.0,
        alpha: float = 0.7,
        learning_rate: float = 1e-3,
    ):
        self.teacher = teacher.eval()
        self.student = student
        self.temperature = temperature
        self.alpha = alpha
        self.optimizer = None
        self.learning_rate = learning_rate

        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False

    def distillation_loss(self, student_logits, teacher_logits, labels):
        """Combined distillation + classification loss."""
        import torch.nn.functional as F

        # Soft target loss (KL divergence between softened distributions)
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction="batchmean",
        ) * (self.temperature ** 2)

        # Hard target loss (standard cross-entropy)
        hard_loss = F.cross_entropy(student_logits, labels)

        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss

    def train_epoch(self, dataloader, device="cpu"):
        """Train student for one epoch."""
        self.student.train()
        self.teacher.eval()

        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.student.parameters(), lr=self.learning_rate,
            )

        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                inputs, labels = batch[0].to(device), batch[1].to(device)
            else:
                inputs = batch.to(device)
                labels = torch.zeros(inputs.size(0), dtype=torch.long, device=device)

            self.optimizer.zero_grad()

            with torch.no_grad():
                teacher_logits = self.teacher(inputs)
            student_logits = self.student(inputs)

            loss = self.distillation_loss(student_logits, teacher_logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def train(self, dataloader, epochs: int = 10, device: str = "cpu"):
        """Full distillation training loop."""
        self.teacher.to(device)
        self.student.to(device)

        history = []
        for epoch in range(epochs):
            avg_loss = self.train_epoch(dataloader, device)
            history.append({"epoch": epoch + 1, "loss": round(avg_loss, 6)})
            print(f"  Epoch {epoch+1}/{epochs}  loss={avg_loss:.6f}")

        teacher_size = get_model_size_mb(self.teacher)
        student_size = get_model_size_mb(self.student)

        print(f"\n  Teacher size: {teacher_size:.2f} MB ({count_parameters(self.teacher):,} params)")
        print(f"  Student size: {student_size:.2f} MB ({count_parameters(self.student):,} params)")
        print(f"  Compression:  {teacher_size / student_size:.1f}x smaller")

        return history


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

def benchmark_model(
    model: "nn.Module",
    input_tensor: "torch.Tensor",
    model_name: str = "model",
    num_warmup: int = 10,
    num_iterations: int = 100,
    device: str = "cpu",
) -> BenchmarkResult:
    """
    Benchmark a model's inference latency, throughput, and memory usage.
    """
    model = model.to(device).eval()
    input_tensor = input_tensor.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_tensor)

    # Synchronize if on GPU
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark latency
    latencies = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = model(input_tensor)
            if device != "cpu" and torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms

    latencies.sort()
    avg_latency = sum(latencies) / len(latencies)
    p50 = latencies[int(len(latencies) * 0.50)]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]

    batch_size = input_tensor.shape[0]
    throughput = (batch_size / (avg_latency / 1000))  # samples/sec

    # Memory usage
    peak_memory = 0
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(input_tensor)
        peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        # Rough estimate for CPU: model size + input size + output size
        peak_memory = get_model_size_mb(model) + (
            input_tensor.nelement() * input_tensor.element_size() / (1024 * 1024)
        )

    return BenchmarkResult(
        model_name=model_name,
        size_mb=round(get_model_size_mb(model), 2),
        num_params=count_parameters(model),
        avg_latency_ms=round(avg_latency, 3),
        p50_latency_ms=round(p50, 3),
        p95_latency_ms=round(p95, 3),
        p99_latency_ms=round(p99, 3),
        throughput_samples_per_sec=round(throughput, 1),
        peak_memory_mb=round(peak_memory, 2),
    )


def compare_models(
    original: "nn.Module",
    compressed: "nn.Module",
    input_tensor: "torch.Tensor",
    original_name: str = "original",
    compressed_name: str = "compressed",
    device: str = "cpu",
) -> Dict:
    """Benchmark and compare original vs compressed model."""
    print(f"\n  Benchmarking '{original_name}'...")
    orig_bench = benchmark_model(original, input_tensor, original_name, device=device)

    print(f"  Benchmarking '{compressed_name}'...")
    comp_bench = benchmark_model(compressed, input_tensor, compressed_name, device=device)

    speedup = orig_bench.avg_latency_ms / max(comp_bench.avg_latency_ms, 0.001)
    size_reduction = orig_bench.size_mb / max(comp_bench.size_mb, 0.001)

    comparison = {
        "original": orig_bench.to_dict(),
        "compressed": comp_bench.to_dict(),
        "improvement": {
            "speedup": round(speedup, 2),
            "size_reduction": round(size_reduction, 2),
            "latency_reduction_pct": round(
                (1 - comp_bench.avg_latency_ms / max(orig_bench.avg_latency_ms, 0.001)) * 100, 1
            ),
            "size_reduction_pct": round(
                (1 - comp_bench.size_mb / max(orig_bench.size_mb, 0.001)) * 100, 1
            ),
            "throughput_improvement_pct": round(
                (comp_bench.throughput_samples_per_sec / max(orig_bench.throughput_samples_per_sec, 0.001) - 1) * 100, 1
            ),
        },
    }

    return comparison


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_compression_result(result: CompressionResult) -> str:
    lines = [
        "",
        "=" * 55,
        f"  COMPRESSION RESULT: {result.method}",
        "=" * 55,
        "",
        f"  Original size:      {result.original_size_mb:.2f} MB",
        f"  Compressed size:    {result.compressed_size_mb:.2f} MB",
        f"  Compression ratio:  {result.compression_ratio:.2f}x",
        f"  Original params:    {result.original_params:,}",
        f"  Remaining params:   {result.compressed_params:,}",
        f"  Sparsity:           {result.sparsity:.1%}",
        f"  Notes:              {result.notes}",
        "",
        "=" * 55,
    ]
    return "\n".join(lines)


def format_benchmark_result(result: BenchmarkResult) -> str:
    lines = [
        "",
        "-" * 50,
        f"  BENCHMARK: {result.model_name}",
        "-" * 50,
        f"  Size:                {result.size_mb:.2f} MB",
        f"  Parameters:          {result.num_params:,}",
        f"  Avg latency:         {result.avg_latency_ms:.3f} ms",
        f"  P50 latency:         {result.p50_latency_ms:.3f} ms",
        f"  P95 latency:         {result.p95_latency_ms:.3f} ms",
        f"  P99 latency:         {result.p99_latency_ms:.3f} ms",
        f"  Throughput:          {result.throughput_samples_per_sec:.1f} samples/sec",
        f"  Peak memory:         {result.peak_memory_mb:.2f} MB",
        "-" * 50,
    ]
    return "\n".join(lines)


def format_comparison(comparison: Dict) -> str:
    orig = comparison["original"]
    comp = comparison["compressed"]
    imp = comparison["improvement"]

    lines = [
        "",
        "=" * 65,
        "  MODEL COMPARISON: ORIGINAL vs COMPRESSED",
        "=" * 65,
        "",
        f"  {'Metric':<30} {'Original':>14} {'Compressed':>14}",
        f"  {'-'*28:<30} {'--------':>14} {'----------':>14}",
        f"  {'Size (MB)':<30} {orig['size_mb']:>13.2f} {comp['size_mb']:>13.2f}",
        f"  {'Parameters':<30} {orig['num_params']:>14,} {comp['num_params']:>14,}",
        f"  {'Avg latency (ms)':<30} {orig['avg_latency_ms']:>13.3f} {comp['avg_latency_ms']:>13.3f}",
        f"  {'P95 latency (ms)':<30} {orig['p95_latency_ms']:>13.3f} {comp['p95_latency_ms']:>13.3f}",
        f"  {'Throughput (samples/sec)':<30} {orig['throughput_samples_per_sec']:>13.1f} {comp['throughput_samples_per_sec']:>13.1f}",
        f"  {'Peak memory (MB)':<30} {orig['peak_memory_mb']:>13.2f} {comp['peak_memory_mb']:>13.2f}",
        "",
        "-" * 65,
        "  IMPROVEMENTS",
        "-" * 65,
        "",
        f"  Speedup:              {imp['speedup']:.2f}x",
        f"  Size reduction:       {imp['size_reduction']:.2f}x ({imp['size_reduction_pct']:.1f}%)",
        f"  Latency reduction:    {imp['latency_reduction_pct']:.1f}%",
        f"  Throughput increase:  {imp['throughput_improvement_pct']:.1f}%",
        "",
        "=" * 65,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def run_demo():
    """Run a comprehensive demo of all compression techniques."""
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch is required. Install with: pip install torch")
        sys.exit(1)

    print("\n" + "=" * 65)
    print("  MODEL COMPRESSION DEMO")
    print("=" * 65)

    # Create sample model
    model_type = "mlp"
    model = create_sample_model(model_type)
    sample_input = create_sample_input(model_type, batch_size=32)

    original_size = get_model_size_mb(model)
    original_params = count_parameters(model)
    print(f"\n  Original model: {original_size:.2f} MB, {original_params:,} parameters")

    results = []

    # 1. Dynamic Quantization
    print("\n--- Dynamic Quantization ---")
    q_model, q_result = quantize_dynamic(model)
    print(format_compression_result(q_result))
    results.append(q_result)

    # 2. Unstructured Pruning at various sparsity levels
    for sparsity in [0.3, 0.5, 0.7, 0.9]:
        print(f"\n--- Unstructured Pruning (sparsity={sparsity:.0%}) ---")
        p_model, p_result = prune_magnitude_unstructured(model, sparsity=sparsity)
        print(format_compression_result(p_result))
        results.append(p_result)

    # 3. Structured Pruning
    print("\n--- Structured Pruning (30%) ---")
    sp_model, sp_result = prune_structured(model, sparsity=0.3)
    print(format_compression_result(sp_result))
    results.append(sp_result)

    # 4. Benchmark: original vs quantized vs pruned
    print("\n--- Benchmarking ---")
    comparison = compare_models(
        model, q_model, sample_input,
        original_name="Original FP32",
        compressed_name="Dynamic INT8",
    )
    print(format_comparison(comparison))

    # 5. Summary table
    print("\n" + "=" * 65)
    print("  COMPRESSION SUMMARY")
    print("=" * 65)
    print(f"\n  {'Method':<35} {'Size (MB)':>10} {'Ratio':>8} {'Sparsity':>10}")
    print(f"  {'-'*33:<35} {'--------':>10} {'-----':>8} {'--------':>10}")
    print(f"  {'Original':<35} {original_size:>9.2f} {'1.00x':>8} {'0.0%':>10}")
    for r in results:
        ratio_str = f"{r.compression_ratio:.2f}x"
        sparsity_str = f"{r.sparsity:.1%}"
        print(f"  {r.method:<35} {r.compressed_size_mb:>9.2f} {ratio_str:>8} {sparsity_str:>10}")
    print()

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Model Compression Toolkit: quantization, pruning, and distillation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run comprehensive demo
  python model_compress.py demo

  # Quantize a saved model
  python model_compress.py quantize --model model.pt --method dynamic --output quantized.pt

  # Prune a model
  python model_compress.py prune --model model.pt --method magnitude --sparsity 0.5 --output pruned.pt

  # Compare original and compressed models
  python model_compress.py compare --original model.pt --compressed quantized.pt

  # Benchmark a model
  python model_compress.py benchmark --model model.pt --iterations 200
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Demo command
    subparsers.add_parser("demo", help="Run comprehensive compression demo")

    # Quantize command
    quant = subparsers.add_parser("quantize", help="Quantize a model")
    quant.add_argument("--model", type=str, required=True, help="Path to saved model (.pt)")
    quant.add_argument("--method", type=str, default="dynamic",
                       choices=["dynamic", "static"],
                       help="Quantization method (default: dynamic)")
    quant.add_argument("--output", type=str, required=True, help="Output path for quantized model")
    quant.add_argument("--json", action="store_true", help="Output result as JSON")

    # Prune command
    prn = subparsers.add_parser("prune", help="Prune a model")
    prn.add_argument("--model", type=str, required=True, help="Path to saved model (.pt)")
    prn.add_argument("--method", type=str, default="magnitude",
                     choices=["magnitude", "structured"],
                     help="Pruning method (default: magnitude)")
    prn.add_argument("--sparsity", type=float, default=0.5, help="Target sparsity (default: 0.5)")
    prn.add_argument("--output", type=str, required=True, help="Output path for pruned model")
    prn.add_argument("--json", action="store_true", help="Output result as JSON")

    # Distill command
    dist = subparsers.add_parser("distill", help="Setup knowledge distillation")
    dist.add_argument("--teacher", type=str, required=True, help="Path to teacher model (.pt)")
    dist.add_argument("--student", type=str, required=True, help="Path to student model (.pt)")
    dist.add_argument("--temperature", type=float, default=4.0, help="Distillation temperature")
    dist.add_argument("--alpha", type=float, default=0.7, help="Distillation loss weight")
    dist.add_argument("--epochs", type=int, default=10, help="Training epochs")
    dist.add_argument("--output", type=str, help="Output path for distilled student model")

    # Benchmark command
    bench = subparsers.add_parser("benchmark", help="Benchmark model inference")
    bench.add_argument("--model", type=str, required=True, help="Path to model (.pt)")
    bench.add_argument("--batch-size", type=int, default=32, help="Batch size")
    bench.add_argument("--iterations", type=int, default=100, help="Benchmark iterations")
    bench.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    bench.add_argument("--json", action="store_true", help="Output as JSON")

    # Compare command
    comp = subparsers.add_parser("compare", help="Compare original and compressed model")
    comp.add_argument("--original", type=str, required=True, help="Path to original model")
    comp.add_argument("--compressed", type=str, required=True, help="Path to compressed model")
    comp.add_argument("--batch-size", type=int, default=32, help="Batch size")
    comp.add_argument("--device", type=str, default="cpu", help="Device")
    comp.add_argument("--json", action="store_true", help="Output as JSON")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch is required. Install with: pip install torch")
        sys.exit(1)

    if args.command == "demo":
        run_demo()

    elif args.command == "quantize":
        print(f"  Loading model from {args.model}...")
        # Load model -- expects a full model save or state dict
        # For demonstration, we create a sample model and load weights
        model = create_sample_model("mlp")
        state_dict = torch.load(args.model, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)

        if args.method == "dynamic":
            quantized, result = quantize_dynamic(model)
        else:
            quantized, result = quantize_static(model)

        torch.save(quantized.state_dict(), args.output)
        print(f"  Quantized model saved to {args.output}")

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(format_compression_result(result))

    elif args.command == "prune":
        print(f"  Loading model from {args.model}...")
        model = create_sample_model("mlp")
        state_dict = torch.load(args.model, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)

        if args.method == "magnitude":
            pruned, result = prune_magnitude_unstructured(model, sparsity=args.sparsity)
        else:
            pruned, result = prune_structured(model, sparsity=args.sparsity)

        torch.save(pruned.state_dict(), args.output)
        print(f"  Pruned model saved to {args.output}")

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(format_compression_result(result))

    elif args.command == "distill":
        print("  Knowledge Distillation Setup")
        print(f"  Teacher: {args.teacher}")
        print(f"  Student: {args.student}")
        print(f"  Temperature: {args.temperature}, Alpha: {args.alpha}")
        print()
        print("  To use distillation in your training script:")
        print()
        print("    from model_compress import DistillationTrainer")
        print()
        print("    trainer = DistillationTrainer(")
        print(f"        teacher=teacher_model,")
        print(f"        student=student_model,")
        print(f"        temperature={args.temperature},")
        print(f"        alpha={args.alpha},")
        print("    )")
        print(f"    history = trainer.train(dataloader, epochs={args.epochs})")
        print()
        print("  The DistillationTrainer class handles:")
        print("  - Freezing teacher weights")
        print("  - KL divergence loss on soft targets")
        print("  - Cross-entropy loss on hard targets")
        print("  - Combined loss with configurable weighting")

    elif args.command == "benchmark":
        print(f"  Loading model from {args.model}...")
        model = create_sample_model("mlp")
        state_dict = torch.load(args.model, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)

        sample_input = create_sample_input("mlp", batch_size=args.batch_size)
        result = benchmark_model(
            model, sample_input,
            model_name=os.path.basename(args.model),
            num_iterations=args.iterations,
            device=args.device,
        )

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(format_benchmark_result(result))

    elif args.command == "compare":
        print(f"  Loading original from {args.original}...")
        original = create_sample_model("mlp")
        original.load_state_dict(
            torch.load(args.original, map_location="cpu", weights_only=True)
        )

        print(f"  Loading compressed from {args.compressed}...")
        compressed = create_sample_model("mlp")
        compressed.load_state_dict(
            torch.load(args.compressed, map_location="cpu", weights_only=True)
        )

        sample_input = create_sample_input("mlp", batch_size=args.batch_size)
        comparison = compare_models(
            original, compressed, sample_input,
            original_name=os.path.basename(args.original),
            compressed_name=os.path.basename(args.compressed),
            device=args.device,
        )

        if args.json:
            print(json.dumps(comparison, indent=2))
        else:
            print(format_comparison(comparison))

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
