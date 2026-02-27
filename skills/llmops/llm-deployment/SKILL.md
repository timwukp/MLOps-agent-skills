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
# Install and run
ollama pull llama3.1:8b
ollama serve

# API usage
curl http://localhost:11434/api/chat -d '{
    "model": "llama3.1:8b",
    "messages": [{"role": "user", "content": "What is MLOps?"}],
    "stream": false
}'
```

```python
# Python client
import ollama

response = ollama.chat(
    model="llama3.1:8b",
    messages=[{"role": "user", "content": "What is MLOps?"}],
)
print(response["message"]["content"])
```

### 4. Quantization for Deployment

```python
# AWQ quantization (recommended for vLLM)
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4}
model.quantize(tokenizer, quant_config=quant_config)
model.save_quantized("model-awq")

# GGUF for llama.cpp / Ollama
# python convert_hf_to_gguf.py model_path --outtype q4_k_m --outfile model.gguf
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
FROM vllm/vllm-openai:latest

ENV MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
ENV TENSOR_PARALLEL_SIZE=1
ENV MAX_MODEL_LEN=4096

CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "${MODEL_NAME}", \
     "--tensor-parallel-size", "${TENSOR_PARALLEL_SIZE}", \
     "--max-model-len", "${MAX_MODEL_LEN}", \
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

## Serving Framework Comparison

| Feature | vLLM | TGI | Ollama | llama.cpp |
|---------|------|-----|--------|-----------|
| Performance | Excellent | Excellent | Good | Good |
| GPU support | Yes | Yes | Yes | Optional |
| CPU support | Limited | No | Yes | Excellent |
| Quantization | AWQ, GPTQ | AWQ, GPTQ, BnB | GGUF | GGUF |
| OpenAI compat | Yes | Partial | Yes | Yes |
| Multi-GPU | Yes | Yes | No | Limited |
| Production ready | Yes | Yes | Dev/small | Dev/edge |

## Best Practices

1. **Use vLLM** for production GPU serving (PagedAttention, continuous batching)
2. **Quantize to AWQ 4-bit** for best quality/speed tradeoff
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
