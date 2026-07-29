---
name: llm-deployment
description: >
  Deploy and serve large language models efficiently. Covers vLLM, Text Generation Inference (TGI), Ollama,
  llama.cpp, ONNX Runtime, TensorRT-LLM, model quantization (GPTQ, AWQ, GGUF, INT4/INT8), KV-cache optimization,
  continuous batching, speculative decoding, PagedAttention, tensor parallelism, pipeline parallelism, API gateway
  setup, streaming responses, multi-model serving, GPU memory management, auto-scaling for LLMs, Docker/Kubernetes
  deployment, and cost-performance optimization. Use when deploying LLMs, optimizing inference speed, reducing
  serving costs, or setting up LLM infrastructure.
license: Apache-2.0
metadata:
  author: llmops-skills
  version: "1.0"
  category: llmops
---

# LLM Deployment

## Overview

LLM deployment requires specialized infrastructure for efficient inference - continuous batching,
KV-cache management, and model parallelism are essential for production serving.

## When to Use This Skill

- Deploying an LLM for production inference
- Optimizing LLM inference speed and throughput
- Setting up self-hosted LLM infrastructure
- Choosing between serving frameworks
- Reducing LLM serving costs

## Step-by-Step Instructions

### 1. vLLM Serving (Recommended for Production)

```python
# Install: pip install vllm

# Python API
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=1,      # Number of GPUs
    gpu_memory_utilization=0.9,
    max_model_len=4096,
    quantization="awq",          # Optional quantization
)

params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
)

outputs = llm.generate(["What is MLOps?"], params)
print(outputs[0].outputs[0].text)
```

```bash
# OpenAI-compatible API server
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096 \
    --port 8000

# Use with OpenAI SDK
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [{"role": "user", "content": "What is MLOps?"}],
        "temperature": 0.7,
        "max_tokens": 512
    }'
```

### 2. Text Generation Inference (TGI)

```bash
# Docker deployment
docker run --gpus all -p 8080:80 \
    -v /data/models:/data \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id meta-llama/Llama-3.1-8B-Instruct \
    --quantize awq \
    --max-input-tokens 2048 \
    --max-total-tokens 4096 \
    --max-batch-prefill-tokens 4096
```

```python
# Python client
from huggingface_hub import InferenceClient

client = InferenceClient("http://localhost:8080")
response = client.text_generation(
    "What is MLOps?",
    max_new_tokens=512,
    temperature=0.7,
    stream=True,
)
for token in response:
    print(token, end="", flush=True)
```

### 3. Ollama (Local Development)

```bash
# Start the server first (the desktop app does this automatically), then pull
ollama serve &
ollama pull qwen3:8b

# API usage
curl http://localhost:11434/api/chat -d '{
    "model": "qwen3:8b",
    "messages": [{"role": "user", "content": "What is MLOps?"}],
    "stream": false
}'
```

```python
# Python client
import ollama

response = ollama.chat(
    model="qwen3:8b",
    messages=[{"role": "user", "content": "What is MLOps?"}],
)
print(response["message"]["content"])
```

### 4. Quantization for Deployment

```python
# Weight quantization with llm-compressor (AutoAWQ is deprecated; vLLM's supported path)
# pip install llmcompressor
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

recipe = QuantizationModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
oneshot(model=model_path, recipe=recipe, output_dir="model-w4a16")

# GGUF for llama.cpp / Ollama (two steps: convert to f16, then quantize)
# python convert_hf_to_gguf.py model_path --outtype f16 --outfile model-f16.gguf
# llama-quantize model-f16.gguf model-Q4_K_M.gguf Q4_K_M
```

**Quantization Comparison:**

| Method | Bits | Quality Loss | Speed | Memory | Best For |
|--------|------|-------------|-------|--------|----------|
| FP16 | 16 | None | Baseline | Baseline | Maximum quality |
| INT8 | 8 | Minimal | 1.5x | 0.5x | Production serving |
| AWQ | 4 | Very low | 2x | 0.25x | vLLM production |
| GPTQ | 4 | Low | 2x | 0.25x | General 4-bit |
| GGUF Q4_K_M | 4 | Low | CPU-friendly | 0.25x | Ollama / llama.cpp |
| GGUF Q2_K | 2 | Moderate | CPU-friendly | 0.125x | Extreme compression |

### 5. Docker + Kubernetes Deployment

```dockerfile
# Dockerfile for vLLM
# The base image's ENTRYPOINT already launches the OpenAI API server;
# CMD supplies its arguments. Note: exec-form CMD does NOT expand env vars,
# so values are inlined here rather than referenced via ${...}.
FROM vllm/vllm-openai:latest

CMD ["--model", "meta-llama/Llama-3.1-8B-Instruct", \
     "--tensor-parallel-size", "1", \
     "--max-model-len", "4096", \
     "--port", "8000"]
```

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-server
spec:
  replicas: 2
  selector:
    matchLabels:
      app: llm-server
  template:
    spec:
      containers:
      - name: vllm
        image: llm-server:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
          requests:
            nvidia.com/gpu: 1
            memory: "24Gi"
        env:
        - name: MODEL_NAME
          value: "meta-llama/Llama-3.1-8B-Instruct"
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 120  # Model loading time
          periodSeconds: 10
```

### 6. Streaming Response Pattern

```python
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI

app = FastAPI()
client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="unused")

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    async def generate():
        stream = await client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=[{"role": "user", "content": request.message}],
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield f"data: {json.dumps({'content': chunk.choices[0].delta.content})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

### 7. SageMaker LMI Endpoints (DJL + vLLM)

Lessons from a live SageMaker deployment (fine-tuned Qwen3-1.7B on an LMI/vLLM container,
verified 2026-07 us-east-1). The feedback loop for a bad endpoint config is 30-60 minutes per
attempt, so validation-left-shift matters more here than anywhere.

**Ship `serving.properties` at the tarball ROOT — it is the canonical LMI config.**
Env-var-only configuration makes DJL scan the tarball root for engine detection; if the model
lives in a subdirectory (e.g. `merged/`), registration fails with
`Failed to detect engine of the model: /opt/ml/model` and the server restart-loops even though
the vLLM engine initialized successfully (observed: 22 successful engine inits, zero
successful registrations).

```properties
# serving.properties — at the tarball root, not inside the model subdir
engine=Python
option.model_id=/opt/ml/model/merged
option.rolling_batch=vllm
option.dtype=fp16
option.max_model_len=14336
option.gpu_memory_utilization=0.9
option.max_rolling_batch_size=8
```

**Validate container env vars against the container's documented config BEFORE
create-endpoint.** `SERVING_LOAD_MODELS` is a local-serving option; on SageMaker it makes DJL
parse the literal string as a model URL and crash-loop. On lmi15 (vLLM-native) images,
`OPTION_ROLLING_BATCH=disable` routes to the legacy HF handler, which fails at init
(`'list' object has no attribute 'keys'`); the supported path is `rolling_batch=vllm`.

**A `Creating` endpoint can be neither deleted nor updated** (live
`ValidationException: Cannot update in-progress endpoint`) — a bad config's punishment is
waiting for `Failed`, 30-60 minutes of billing. Smoke-test new configs cheaply first, and
ALWAYS record the endpoint name plus a teardown instruction to S3 *before* `create-endpoint`
so it can never become an unaccounted orphan.

**Watch train/serve version skew.** Training with transformers 5.x writes
`tokenizer_config.json` fields (e.g. `extra_special_tokens` as a list) that older container
transformers crash on (`AttributeError: 'list' object has no attribute 'keys'` inside
`AutoTokenizer.from_pretrained`). Either match the serving container's transformers family or
post-process the artifact.

**Long generations need the streaming API.** Synchronous `InvokeEndpoint` has a hard 60s
timeout — a long chain-of-thought generation (roughly >2k tokens on small instances) cannot
complete synchronously. Use `invoke_endpoint_with_response_stream`, which is
inactivity-bounded with no wall-clock ceiling. Also budget tokens: effective
`max_new_tokens = max_model_len − prompt tokens`.

**Plan teardown for least-privilege roles.** Roles often grant `CreateModel`/`DeleteEndpoint`
but NOT `DeleteModel`/`DeleteEndpointConfig`/`List*` — plan teardown around known-name
deletion and track created resource names as you go. Scope deletions strictly by name prefix:
a naive delete-all would have hit an unrelated 2-year-old production endpoint.

## Serving Framework Comparison

| Feature | vLLM | TGI | Ollama | llama.cpp |
|---------|------|-----|--------|-----------|
| Performance | Excellent | Excellent | Good | Good |
| GPU support | Yes | Yes | Yes | Optional |
| CPU support | Limited | No | Yes | Excellent |
| Quantization | AWQ, GPTQ | AWQ, GPTQ, BnB | GGUF | GGUF |
| OpenAI compat | Yes | Partial | Yes | Yes |
| Multi-GPU | Yes | Yes | Yes (layer split) | Yes (layer split) |
| Production ready | Yes | Yes | Dev/small | Dev/edge |

## Best Practices

1. **Use vLLM** for production GPU serving (PagedAttention, continuous batching)
2. **Quantize appropriately** - FP8 on H100-class GPUs, 4-bit (AWQ/GPTQ) when memory-bound on older hardware
3. **Set appropriate max_model_len** - Don't over-allocate KV cache
4. **Use streaming** for better user experience
5. **Monitor GPU utilization** and batch sizes
6. **Separate API gateway** from inference server
7. **Pre-download models** in Docker build, not at runtime
8. **Load test** with realistic prompt length distribution

## Scripts

- `scripts/deploy_vllm.py` - vLLM deployment setup and management
- `scripts/benchmark_inference.py` - LLM inference benchmarking

## References

See [references/REFERENCE.md](references/REFERENCE.md) for framework comparisons and deployment patterns.

## Related skills

**Upstream:** `llm-evaluation` (cleared candidate) · **Downstream:** `llm-guardrails` (wrap the live endpoint) and `llm-observability` (traces, token metrics)
**See also:** `model-serving` for classic-ML serving patterns that carry over · `llm-agent-orchestration` builds the application layer on top of this endpoint
