#!/usr/bin/env python3
"""Batch inference / scoring pipeline with progress tracking and error handling.

Usage:
    python batch_inference.py --input data.csv --model-path model.joblib --framework sklearn
    python batch_inference.py --input data.parquet --model-path model.onnx --framework onnx \
        --batch-size 512 --output preds.parquet --include-probabilities
    python batch_inference.py --input data.json --model-path model.pt --framework pytorch \
        --output results.csv --batch-size 256
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data I/O
# ---------------------------------------------------------------------------

def load_dataframe(path: str):
    """Load a DataFrame from CSV, Parquet, or JSON."""
    import pandas as pd
    path = str(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    elif path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".json") or path.endswith(".jsonl"):
        return pd.read_json(path, lines=path.endswith(".jsonl"))
    else:
        raise ValueError(f"Unsupported input format: {path}")


def save_dataframe(df, path: str):
    """Write a DataFrame to CSV, Parquet, or JSON."""
    path = str(path)
    if path.endswith(".parquet"):
        df.to_parquet(path, index=False)
    elif path.endswith(".csv"):
        df.to_csv(path, index=False)
    elif path.endswith(".json") or path.endswith(".jsonl"):
        df.to_json(path, orient="records", lines=path.endswith(".jsonl"))
    else:
        raise ValueError(f"Unsupported output format: {path}")
    logger.info("Saved %d rows to %s", len(df), path)

# ---------------------------------------------------------------------------
# Model loaders (lazy imports)
# ---------------------------------------------------------------------------

def load_model(path: str, framework: str):
    """Load model for the given framework."""
    if framework == "sklearn":
        import joblib
        logger.info("Loading sklearn model from %s", path)
        return joblib.load(path)
    elif framework == "onnx":
        import onnxruntime as ort
        logger.info("Loading ONNX model from %s", path)
        return ort.InferenceSession(path)
    elif framework == "pytorch":
        import torch
        logger.info("Loading PyTorch model from %s", path)
        model = torch.jit.load(path)
        model.eval()
        return model
    else:
        raise ValueError(f"Unsupported framework: {framework}")

# ---------------------------------------------------------------------------
# Per-framework prediction functions
# ---------------------------------------------------------------------------

def _predict_sklearn(model, batch, include_probs: bool):
    import numpy as np
    preds = model.predict(batch).tolist()
    probs = None
    if include_probs and hasattr(model, "predict_proba"):
        probs = model.predict_proba(batch).tolist()
    return preds, probs


def _predict_onnx(session, batch, include_probs: bool):
    import numpy as np
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: np.array(batch, dtype=np.float32)})
    preds = outputs[0].tolist()
    probs = outputs[1].tolist() if include_probs and len(outputs) > 1 else None
    return preds, probs


def _predict_pytorch(model, batch, include_probs: bool):
    import torch
    tensor = torch.tensor(batch, dtype=torch.float32)
    with torch.no_grad():
        output = model(tensor)
    preds = output.numpy().tolist()
    probs = None
    if include_probs:
        try:
            probs = torch.nn.functional.softmax(output, dim=-1).numpy().tolist()
        except Exception:
            pass
    return preds, probs


PREDICT_FN = {
    "sklearn": _predict_sklearn,
    "onnx": _predict_onnx,
    "pytorch": _predict_pytorch,
}

# ---------------------------------------------------------------------------
# Batch inference loop
# ---------------------------------------------------------------------------

def run_batch_inference(df, model, framework: str, batch_size: int, include_probs: bool):
    """Run inference over the dataframe in batches. Returns predictions, probabilities, and latencies."""
    import numpy as np
    from tqdm import tqdm

    n = len(df)
    all_preds: list = []
    all_probs: list = []
    latencies: list[float] = []
    failed_batches = 0
    predict = PREDICT_FN[framework]
    values = df.values

    for start in tqdm(range(0, n, batch_size), desc="Inference", unit="batch"):
        end = min(start + batch_size, n)
        batch = values[start:end]
        try:
            t0 = time.perf_counter()
            preds, probs = predict(model, batch, include_probs)
            elapsed = time.perf_counter() - t0

            all_preds.extend(preds)
            if probs is not None:
                all_probs.extend(probs)
            latencies.append(elapsed)
        except Exception:
            failed_batches += 1
            batch_len = end - start
            logger.warning("Batch %d-%d failed, filling with None", start, end)
            all_preds.extend([None] * batch_len)
            if include_probs:
                all_probs.extend([None] * batch_len)

    if failed_batches:
        logger.warning("%d / %d batches failed", failed_batches, (n + batch_size - 1) // batch_size)

    return all_preds, all_probs if all_probs else None, latencies, failed_batches

# ---------------------------------------------------------------------------
# Performance statistics
# ---------------------------------------------------------------------------

def compute_stats(total: int, latencies: list[float], failed: int):
    """Return a dict of throughput / latency stats."""
    import numpy as np
    total_time = sum(latencies) if latencies else 0.0
    throughput = total / total_time if total_time > 0 else 0.0
    arr = np.array(latencies) * 1000 if latencies else np.array([0.0])  # ms
    return {
        "total_predictions": total,
        "failed_batches": failed,
        "total_time_s": round(total_time, 3),
        "throughput_pred_per_sec": round(throughput, 1),
        "latency_p50_ms": round(float(np.percentile(arr, 50)), 2),
        "latency_p95_ms": round(float(np.percentile(arr, 95)), 2),
        "latency_p99_ms": round(float(np.percentile(arr, 99)), 2),
    }

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args):
    """Execute the full batch inference pipeline."""
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error("Model path not found: %s", model_path)
        sys.exit(1)

    output_path = args.output or str(input_path.with_stem(input_path.stem + "_predictions").with_suffix(input_path.suffix))

    logger.info("Loading data from %s", input_path)
    df = load_dataframe(str(input_path))
    logger.info("Loaded %d rows x %d cols", len(df), len(df.columns))

    model = load_model(str(model_path), args.framework)

    logger.info("Running inference (batch_size=%d, probs=%s)", args.batch_size, args.include_probabilities)
    preds, probs, latencies, failed = run_batch_inference(
        df, model, args.framework, args.batch_size, args.include_probabilities,
    )

    import pandas as pd
    result = df.copy()
    result["prediction"] = preds
    if probs is not None:
        result["probabilities"] = [json.dumps(p) if p is not None else None for p in probs]

    save_dataframe(result, output_path)

    stats = compute_stats(len(preds), latencies, failed)
    logger.info("Performance stats: %s", json.dumps(stats, indent=2))
    return stats

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Batch inference / scoring pipeline")
    parser.add_argument("--input", required=True, help="Path to input data (CSV/Parquet/JSON)")
    parser.add_argument("--model-path", required=True, help="Path to model file")
    parser.add_argument(
        "--framework",
        required=True,
        choices=["sklearn", "onnx", "pytorch"],
        help="Model framework",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size (default 256)")
    parser.add_argument("--output", default=None, help="Output path (default: <input>_predictions.<ext>)")
    parser.add_argument(
        "--include-probabilities",
        action="store_true",
        help="Include prediction probabilities in output",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
