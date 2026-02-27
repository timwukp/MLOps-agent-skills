#!/usr/bin/env python3
"""FastAPI model serving with A/B testing, caching, and Prometheus metrics.

Usage:
    python serve_model.py --model-path model.joblib --framework sklearn
    python serve_model.py --model-path model.onnx --framework onnx --port 8080
    python serve_model.py --model-path model.joblib --framework sklearn \
        --model-b-path model_v2.joblib --ab-ratio 0.2
"""
import argparse
import hashlib
import json
import logging
import signal
import sys
import time
from functools import lru_cache
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model loaders -- heavy imports are deferred until actually needed
# ---------------------------------------------------------------------------

def load_sklearn_model(path: str):
    """Load a scikit-learn model via joblib."""
    import joblib
    logger.info("Loading sklearn model from %s", path)
    return joblib.load(path)


def load_onnx_model(path: str):
    """Load an ONNX model via onnxruntime."""
    import onnxruntime as ort
    logger.info("Loading ONNX model from %s", path)
    sess = ort.InferenceSession(path)
    return sess


def load_pytorch_model(path: str):
    """Load a TorchScript model."""
    import torch
    logger.info("Loading PyTorch model from %s", path)
    model = torch.jit.load(path)
    model.eval()
    return model


def load_tf_model(path: str):
    """Load a TensorFlow SavedModel directory."""
    import tensorflow as tf
    logger.info("Loading TensorFlow SavedModel from %s", path)
    return tf.saved_model.load(path)


LOADERS = {
    "sklearn": load_sklearn_model,
    "onnx": load_onnx_model,
    "pytorch": load_pytorch_model,
    "tf": load_tf_model,
}

# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def predict_sklearn(model, features: list[list[float]]):
    import numpy as np
    return model.predict(np.array(features)).tolist()


def predict_onnx(session, features: list[list[float]]):
    import numpy as np
    input_name = session.get_inputs()[0].name
    result = session.run(None, {input_name: np.array(features, dtype=np.float32)})
    return result[0].tolist()


def predict_pytorch(model, features: list[list[float]]):
    import torch
    tensor = torch.tensor(features, dtype=torch.float32)
    with torch.no_grad():
        output = model(tensor)
    return output.numpy().tolist()


def predict_tf(model, features: list[list[float]]):
    import numpy as np
    import tensorflow as tf
    tensor = tf.constant(features, dtype=tf.float32)
    output = model(tensor)
    return output.numpy().tolist()


PREDICTORS = {
    "sklearn": predict_sklearn,
    "onnx": predict_onnx,
    "pytorch": predict_pytorch,
    "tf": predict_tf,
}

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def build_app(args):
    """Construct the FastAPI application with all endpoints."""
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, PlainTextResponse
    from pydantic import BaseModel, Field

    app = FastAPI(title="Model Serving API", version="1.0.0")

    # ---- State ----
    state = {
        "model_a": LOADERS[args.framework](args.model_path),
        "model_b": None,
        "framework": args.framework,
        "ab_ratio": args.ab_ratio,
        "request_count": 0,
        "error_count": 0,
        "latencies": [],
        "startup_time": time.time(),
    }
    if args.model_b_path:
        state["model_b"] = LOADERS[args.framework](args.model_b_path)
        logger.info("A/B testing enabled -- %.0f%% traffic to model B", args.ab_ratio * 100)

    # ---- Pydantic schemas ----
    class PredictRequest(BaseModel):
        features: list[list[float]] = Field(..., min_length=1, description="2-D feature matrix")
        model_hint: str | None = Field(None, description="Force 'A' or 'B' model selection")

    class PredictResponse(BaseModel):
        predictions: list
        model_used: str
        latency_ms: float

    class BatchPredictRequest(BaseModel):
        instances: list[list[float]] = Field(..., min_length=1)

    class HealthResponse(BaseModel):
        status: str
        uptime_seconds: float
        framework: str

    # ---- Prediction cache ----
    @lru_cache(maxsize=1024)
    def _cached_predict(features_hash: str, features_json: str, model_label: str):
        features = json.loads(features_json)
        model = state["model_a"] if model_label == "A" else state["model_b"]
        return PREDICTORS[state["framework"]](model, features)

    def _select_model(request_id: str, hint: str | None) -> str:
        if hint and hint.upper() in ("A", "B"):
            return hint.upper()
        if state["model_b"] is None:
            return "A"
        digest = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
        return "B" if (digest % 100) < (state["ab_ratio"] * 100) else "A"

    # ---- Middleware: logging ----
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        elapsed = (time.time() - start) * 1000
        logger.info("%s %s %d %.1fms", request.method, request.url.path, response.status_code, elapsed)
        return response

    # ---- Endpoints ----
    @app.post("/predict", response_model=PredictResponse)
    async def predict(req: PredictRequest, request: Request):
        state["request_count"] += 1
        request_id = request.headers.get("X-Request-ID", str(state["request_count"]))
        model_label = _select_model(request_id, req.model_hint)

        if model_label == "B" and state["model_b"] is None:
            raise HTTPException(status_code=400, detail="Model B is not loaded")

        start = time.time()
        try:
            features_json = json.dumps(req.features)
            features_hash = hashlib.sha256(features_json.encode()).hexdigest()
            preds = _cached_predict(features_hash, features_json, model_label)
        except Exception as exc:
            state["error_count"] += 1
            logger.exception("Prediction failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        latency_ms = (time.time() - start) * 1000
        state["latencies"].append(latency_ms)
        return PredictResponse(predictions=preds, model_used=model_label, latency_ms=round(latency_ms, 2))

    @app.post("/predict/batch")
    async def predict_batch(req: BatchPredictRequest):
        state["request_count"] += 1
        start = time.time()
        try:
            model = state["model_a"]
            preds = PREDICTORS[state["framework"]](model, req.instances)
        except Exception as exc:
            state["error_count"] += 1
            logger.exception("Batch prediction failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        latency_ms = (time.time() - start) * 1000
        state["latencies"].append(latency_ms)
        return {"predictions": preds, "count": len(preds), "latency_ms": round(latency_ms, 2)}

    @app.get("/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(
            status="healthy",
            uptime_seconds=round(time.time() - state["startup_time"], 1),
            framework=state["framework"],
        )

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics():
        import numpy as np
        lines = []
        lines.append(f'model_serving_requests_total {state["request_count"]}')
        lines.append(f'model_serving_errors_total {state["error_count"]}')
        lats = state["latencies"] or [0]
        arr = np.array(lats)
        lines.append(f"model_serving_latency_p50_ms {float(np.percentile(arr, 50)):.2f}")
        lines.append(f"model_serving_latency_p95_ms {float(np.percentile(arr, 95)):.2f}")
        lines.append(f"model_serving_latency_p99_ms {float(np.percentile(arr, 99)):.2f}")
        lines.append(f"model_serving_uptime_seconds {time.time() - state['startup_time']:.1f}")
        error_rate = state["error_count"] / max(state["request_count"], 1)
        lines.append(f"model_serving_error_rate {error_rate:.4f}")
        lines.append(f"model_serving_cache_size {_cached_predict.cache_info().currsize}")
        lines.append(f"model_serving_cache_hits {_cached_predict.cache_info().hits}")
        return "\n".join(lines) + "\n"

    # ---- Graceful shutdown ----
    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info(
            "Shutting down -- served %d requests, %d errors",
            state["request_count"],
            state["error_count"],
        )

    return app

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Serve an ML model via FastAPI")
    parser.add_argument("--model-path", required=True, help="Path to model file or directory")
    parser.add_argument(
        "--framework",
        required=True,
        choices=["sklearn", "onnx", "pytorch", "tf"],
        help="Model framework",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default 8000)")
    parser.add_argument("--workers", type=int, default=1, help="Uvicorn worker count")
    parser.add_argument("--model-b-path", default=None, help="Path to model B for A/B testing")
    parser.add_argument(
        "--ab-ratio",
        type=float,
        default=0.0,
        help="Fraction of traffic routed to model B (0.0-1.0)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if not Path(args.model_path).exists():
        logger.error("Model path does not exist: %s", args.model_path)
        sys.exit(1)
    if args.model_b_path and not Path(args.model_b_path).exists():
        logger.error("Model B path does not exist: %s", args.model_b_path)
        sys.exit(1)

    app = build_app(args)

    import uvicorn

    # Register SIGTERM for graceful shutdown
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    logger.info("Starting server on %s:%d (workers=%d)", args.host, args.port, args.workers)
    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers, log_level="info")


if __name__ == "__main__":
    main()
