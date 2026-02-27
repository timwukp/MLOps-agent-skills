# Model Serving Reference Guide

## Serving Framework Comparison

| Feature           | FastAPI        | BentoML          | Triton           | TF Serving       | TorchServe       | Seldon Core      |
|-------------------|---------------|------------------|------------------|------------------|------------------|------------------|
| Framework Support | Any           | Any              | TF, PyTorch, ONNX| TensorFlow only  | PyTorch only     | Any              |
| Batching          | Manual        | Adaptive         | Dynamic          | Built-in         | Built-in         | Built-in         |
| GPU Support       | Manual        | Yes              | Best-in-class    | Yes              | Yes              | Yes              |
| Model Ensemble    | Manual        | Service graph    | Ensemble pipeline| No               | Workflow DAG     | Inference graph  |
| gRPC Support      | No            | Yes              | Yes              | Yes              | Yes              | Yes              |
| Learning Curve    | Low           | Low              | High             | Medium           | Medium           | High             |
| Best For          | Prototypes    | General purpose  | High-perf GPU    | TF ecosystem     | PyTorch ecosystem| K8s-native ML    |

### When to Choose What

- **FastAPI**: Quick prototypes, simple models, full control over preprocessing.
- **BentoML**: General-purpose, batteries-included solution with easy packaging.
- **Triton**: Maximum GPU utilization, multi-model serving, dynamic batching critical.
- **TF Serving**: TensorFlow-only shops needing proven production stability.
- **TorchServe**: PyTorch-only shops needing model versioning and A/B testing.
- **Seldon Core**: Kubernetes-native teams needing complex inference graphs.

## Deployment Patterns

### Real-Time (Synchronous)

```
Client --> API Gateway --> Model Service --> Response  (target: < 100ms)
```

Use for fraud detection, recommendations, search ranking. Key metrics: p50/p95/p99 latency, RPS.

### Batch Inference

```
Scheduler (cron) --> Read Data --> Model Inference --> Write Results
```

Use for nightly scoring, report generation. Tools: Spark MLlib, Ray Batch, AWS Batch.

### Streaming

```
Event Stream (Kafka) --> Stream Processor (Flink) --> Model --> Output Stream
```

Use for real-time anomaly detection, IoT processing. Key concern: exactly-once semantics.

### Edge Deployment

```
Cloud Model --> Optimize (quantize, prune) --> Edge Runtime (TFLite, ONNX Mobile) --> Device
```

Use for mobile apps, IoT, offline-capable inference. Key metrics: model size, on-device latency.

## A/B Testing and Canary Deployment

### Canary Deployment Phases

| Phase   | Traffic to New | Duration    | Rollback Criteria                |
|---------|---------------|-------------|----------------------------------|
| Phase 1 | 1%            | 1 hour      | Error rate increase > 0.1%       |
| Phase 2 | 5%            | 4 hours     | Latency p99 increase > 20%       |
| Phase 3 | 25%           | 24 hours    | Business metric degradation > 2% |
| Phase 4 | 50%           | 24-48 hours | Statistical significance reached |
| Phase 5 | 100%          | Permanent   | Full rollout                     |

### Shadow Deployment (Dark Launch)

Route 100% of traffic to both old and new models. Serve only old model predictions. Compare new model results offline. Zero production risk; useful for pre-A/B validation.

## Model Optimization for Inference

### ONNX Runtime

```python
import onnxruntime as ort

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession("model.onnx", sess_options,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
result = session.run(None, {"input": input_data})
```

### Quantization Strategies

| Method              | Accuracy Impact | Speed Gain | When to Use                       |
|--------------------|----------------|------------|-----------------------------------|
| FP32 (baseline)    | None           | 1x         | Development, debugging            |
| FP16 / BF16        | Minimal        | 1.5-2x    | Default for GPU serving           |
| INT8 (post-train)  | Small (1-3%)   | 2-4x      | CPU serving, latency-sensitive    |
| INT8 (QAT)         | Minimal        | 2-4x      | When PTQ accuracy is unacceptable |
| INT4 (GPTQ/AWQ)    | Moderate       | 3-6x      | LLM serving, memory-constrained   |

**TensorRT**: NVIDIA proprietary, best GPU inference performance. Supports INT8/FP16 with calibration, layer fusion, kernel auto-tuning. Up to 6x speedup over native PyTorch.

## Kubernetes Deployment Patterns

### Standard Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-server
spec:
  replicas: 3
  template:
    spec:
      containers:
        - name: model-server
          image: model-server:v1.2.0
          resources:
            requests: { cpu: "2", memory: "4Gi" }
            limits: { cpu: "4", memory: "8Gi", nvidia.com/gpu: "1" }
          readinessProbe:
            httpGet: { path: /health, port: 8080 }
            initialDelaySeconds: 30
          livenessProbe:
            httpGet: { path: /health, port: 8080 }
            initialDelaySeconds: 60
```

### KServe

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: fraud-detector
spec:
  predictor:
    model:
      modelFormat: { name: sklearn }
      storageUri: "s3://models/fraud-detector/v2.1.0"
      resources:
        requests: { cpu: "1", memory: "2Gi" }
```

**Knative (Serverless)**: Scale-to-zero for cost savings on bursty/low-traffic workloads. Cold start latency (10-60s for large models) is the main trade-off.

## Auto-Scaling Strategies

| Strategy      | Metric                        | Best For                  | Target Example          |
|--------------|-------------------------------|---------------------------|-------------------------|
| CPU-based    | CPU utilization               | CPU-bound models          | 70% CPU                 |
| GPU-based    | GPU utilization (DCGM)        | GPU-served models         | 60% GPU                 |
| Request-based| Requests per second           | Predictable workloads     | 100 RPS per pod         |
| Latency-based| p95 response time             | Latency-sensitive services| p95 < 100ms            |
| Queue-based  | Pending request queue depth   | Bursty traffic            | Queue > 10              |

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
spec:
  scaleTargetRef: { apiVersion: apps/v1, kind: Deployment, name: model-server }
  minReplicas: 2
  maxReplicas: 20
  metrics:
    - type: Pods
      pods:
        metric: { name: http_requests_per_second }
        target: { type: AverageValue, averageValue: "100" }
  behavior:
    scaleUp: { stabilizationWindowSeconds: 60 }
    scaleDown: { stabilizationWindowSeconds: 300 }
```

## Latency Optimization Checklist

### Model Level

- [ ] Convert to ONNX or TensorRT for optimized inference runtime
- [ ] Apply quantization (FP16 for GPU, INT8 for CPU)
- [ ] Use model distillation if accuracy budget allows
- [ ] Enable dynamic batching on the serving framework

### Infrastructure Level

- [ ] Place service close to clients (edge or regional deployment)
- [ ] Use gRPC instead of REST for reduced serialization overhead
- [ ] Pre-warm model containers (load model at startup, not first request)
- [ ] Enable connection pooling and HTTP/2 keep-alive

### Application Level

- [ ] Cache frequent predictions (Redis, in-memory LRU)
- [ ] Implement request coalescing for duplicate concurrent requests
- [ ] Use async I/O for preprocessing and postprocessing
- [ ] Precompute features where possible (feature store vs. on-the-fly)

### Monitoring

- [ ] Profile end-to-end latency breakdown (network, preprocess, inference, postprocess)
- [ ] Monitor p50, p95, p99 separately (not just averages)
- [ ] Set latency budgets per component
- [ ] Alert on latency regressions after model updates

## Common Pitfalls

1. **No health checks**: Always implement readiness and liveness probes; without them, traffic hits unready pods.
2. **Cold start surprises**: Large models take seconds to load. Pre-warm replicas and set adequate initial delays.
3. **Ignoring tail latency**: p99 is often 5-10x of p50. Optimize for the tail, not the median.
4. **Shared GPU without limits**: Multiple models on one GPU without memory limits leads to OOM crashes.
5. **No graceful shutdown**: Implement SIGTERM handling to drain in-flight requests before termination.

## Further Reading

- [BentoML Documentation](https://docs.bentoml.com/)
- [Triton Inference Server Guide](https://docs.nvidia.com/deeplearning/triton-inference-server/)
- [KServe Documentation](https://kserve.github.io/website/)
- [TorchServe Documentation](https://pytorch.org/serve/)
- [ONNX Runtime Performance Tuning](https://onnxruntime.ai/docs/performance/)
- [Seldon Core Documentation](https://docs.seldon.io/projects/seldon-core/en/latest/)
- [Kubernetes HPA Documentation](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
