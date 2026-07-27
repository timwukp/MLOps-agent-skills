---
name: model-serving
description: >
  Deploy and serve ML models in production. Covers REST API serving (FastAPI, Flask), gRPC, BentoML, NVIDIA Triton,
  TorchServe, TF Serving, Seldon Core, KServe on Kubernetes, batch inference (Spark, Ray, Dask), model optimization
  for serving (ONNX Runtime, TensorRT, quantization, pruning), A/B testing, canary and blue-green deployments,
  shadow deployments, request batching, auto-scaling, containerization with Docker, health checks, input validation,
  Prometheus metrics, caching, multi-model serving, and model ensembles. Use when deploying models, building
  inference APIs, or optimizing serving performance.
license: Apache-2.0
metadata:
  author: mlops-skills
  version: "1.0"
  category: mlops
---

# Model Serving

## Overview

Model serving exposes trained models as production services for real-time or batch predictions.
This skill covers serving patterns from simple REST APIs to enterprise-grade serving infrastructure.

## When to Use This Skill

- Deploying a trained model as an API endpoint
- Setting up batch inference pipelines
- Optimizing inference latency and throughput
- Implementing A/B testing or canary deployments
- Containerizing models for Kubernetes deployment

## Step-by-Step Instructions

### 1. FastAPI Model Server

```python
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
import joblib
import numpy as np
from typing import List, Optional
import time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

app = FastAPI(title="ML Model Server", version="1.0")

# Load model at startup
model = joblib.load("model.joblib")

# Metrics
PREDICTION_COUNT = Counter("predictions_total", "Total predictions", ["status"])
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency")

class PredictionRequest(BaseModel):
    features: List[float] = Field(..., min_length=1)

class PredictionResponse(BaseModel):
    prediction: float
    probability: Optional[List[float]] = None
    model_version: str = "v1.0"
    latency_ms: float

# Note: `def` (not `async def`) so FastAPI runs the blocking model.predict()
# in its threadpool instead of blocking the event loop.
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    start = time.time()
    try:
        X = np.array(request.features).reshape(1, -1)
        prediction = float(model.predict(X)[0])
        probability = model.predict_proba(X)[0].tolist() if hasattr(model, "predict_proba") else None

        latency = (time.time() - start) * 1000
        PREDICTION_COUNT.labels(status="success").inc()
        PREDICTION_LATENCY.observe(latency / 1000)

        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            latency_ms=round(latency, 2),
        )
    except Exception as e:
        PREDICTION_COUNT.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/metrics")
async def metrics():
    # generate_latest() returns bytes; wrap in Response with the Prometheus
    # content type (a JSONResponse would mangle the exposition format)
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

### 2. BentoML Service

```python
# BentoML 1.2+ service API (the 1.1 runner / bentoml.io API is deprecated)
import bentoml
import numpy as np

# Save model to BentoML store (once, after training)
bentoml.sklearn.save_model("my_model", model)

# service.py
@bentoml.service(resources={"cpu": "2"}, traffic={"timeout": 30})
class PredictionService:
    model_ref = bentoml.models.get("my_model:latest")

    def __init__(self):
        self.model = bentoml.sklearn.load_model(self.model_ref)

    @bentoml.api
    def predict(self, input_array: np.ndarray) -> dict:
        result = self.model.predict(input_array)
        return {"prediction": result.tolist()}

# Serve, build, and containerize
# bentoml serve service:PredictionService
# bentoml build
# bentoml containerize prediction_service:latest
```

### 3. Batch Inference

```python
import pandas as pd
import joblib
import pyarrow.parquet as pq

def batch_predict(model_path, input_path, output_path, batch_size=10000):
    """Run batch inference on large datasets."""
    model = joblib.load(model_path)

    # Stream record batches so only one chunk is in memory at a time
    # (pd.read_parquet would load the entire file up front)
    parquet_file = pq.ParquetFile(input_path)
    total_rows = parquet_file.metadata.num_rows

    results = []
    processed = 0
    for record_batch in parquet_file.iter_batches(batch_size=batch_size):
        chunk = record_batch.to_pandas()
        features = chunk.drop(columns=["id"], errors="ignore")
        predictions = model.predict(features)
        probabilities = model.predict_proba(features) if hasattr(model, "predict_proba") else None

        result = chunk[["id"]].copy() if "id" in chunk.columns else chunk.iloc[:, :0].copy()
        result["prediction"] = predictions
        if probabilities is not None:
            result["probability"] = probabilities.max(axis=1)
        results.append(result)

        processed += len(chunk)
        print(f"Processed {processed}/{total_rows}")

    output = pd.concat(results, ignore_index=True)
    output.to_parquet(output_path)
    print(f"Saved {len(output)} predictions to {output_path}")
```

### 4. Model Optimization for Serving

```python
# ONNX Runtime - fastest cross-platform inference
import onnxruntime as ort

session = ort.InferenceSession("model.onnx", providers=["CUDAExecutionProvider"])
input_name = session.get_inputs()[0].name
result = session.run(None, {input_name: input_data})

# Quantization - reduce model size and speed up CPU inference
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    "model.onnx", "model_quantized.onnx",
    weight_type=QuantType.QInt8
)

# TensorRT - NVIDIA GPU optimization
import tensorrt as trt
# Convert ONNX to TensorRT engine for maximum GPU throughput
```

### 5. Docker Containerization

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model.joblib .
COPY server.py .

EXPOSE 8000

# python:*-slim images do not ship curl; use the stdlib instead
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)" || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### 6. Kubernetes Deployment with A/B Testing

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-server-v2
  labels:
    app: model-server
    version: v2
spec:
  replicas: 3
  selector:
    matchLabels:
      app: model-server
      version: v2
  template:
    metadata:
      labels:
        app: model-server
        version: v2
    spec:
      containers:
      - name: model-server
        image: model-server:v2
        ports:
        - containerPort: 8000
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
          limits:
            cpu: "2000m"
            memory: "4Gi"
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: model-server-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: model-server-v2
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### 7. Request Batching

```python
import asyncio
import numpy as np

class BatchPredictor:
    def __init__(self, model, max_batch_size=32, max_wait_ms=50):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue = asyncio.Queue()
        self.task = None

    async def predict(self, features):
        future = asyncio.get_running_loop().create_future()
        await self.queue.put((features, future))
        # Single long-lived worker avoids the arm/re-arm race that can
        # strand items enqueued while a one-shot task is finishing
        if self.task is None:
            self.task = asyncio.create_task(self._worker())
        return await future

    async def _worker(self):
        while True:
            batch = [await self.queue.get()]  # block until work arrives
            await asyncio.sleep(self.max_wait_ms / 1000)
            while not self.queue.empty() and len(batch) < self.max_batch_size:
                batch.append(self.queue.get_nowait())

            features = np.array([item[0] for item in batch])
            try:
                predictions = self.model.predict(features)
                for (_, future), pred in zip(batch, predictions):
                    future.set_result(pred)
            except Exception as exc:
                for _, future in batch:
                    future.set_exception(exc)
```

## Deployment Patterns

| Pattern | Use When | Risk |
|---------|----------|------|
| Blue-Green | Zero-downtime deployment | High resource cost (2x) |
| Canary | Gradual rollout with monitoring | Slow rollout |
| Shadow | Validate without user impact | No real feedback |
| A/B Test | Compare model performance | Needs traffic splitting |

## Best Practices

1. **Always add health checks** - Readiness and liveness probes
2. **Use ONNX Runtime** for cross-framework inference optimization
3. **Implement request batching** for GPU models
4. **Set resource limits** in Kubernetes to prevent OOM kills
5. **Monitor latency percentiles** (p50, p95, p99), not just averages
6. **Version your API** alongside model versions
7. **Input validation** at the API layer before inference
8. **Auto-scale** based on latency and CPU/GPU utilization
9. **Cache repeated predictions** for identical inputs
10. **Load test** before production deployment

## Scripts

- `scripts/serve_model.py` - FastAPI model server with metrics
- `scripts/batch_inference.py` - Batch prediction pipeline

## References

See [references/REFERENCE.md](references/REFERENCE.md) for framework comparisons and K8s manifests.

## Related skills

**Upstream:** `ml-cicd` (deploy pipeline) delivering a `model-registry` artifact packaged per `skills/mlops/model-registry/references/ARTIFACT_CONTRACT.md` · **Downstream:** `model-monitoring` (prediction logs, latency and error metrics)
**See also:** `feature-store` for online feature lookups at request time · `llm-deployment` for LLM-specific serving stacks
