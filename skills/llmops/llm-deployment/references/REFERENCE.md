# LLM Deployment Reference Guide

## Inference Engines Comparison

| Feature | vLLM | TGI | Ollama | llama.cpp | TensorRT-LLM |
|---------|------|-----|--------|-----------|---------------|
| Primary Use | Production serving | Production serving | Local/dev | Local/edge | Max throughput |
| PagedAttention | Yes | Yes | Via llama.cpp | No (uses KV cache) | Yes |
| Continuous Batching | Yes | Yes | No | No | Yes |
| Tensor Parallelism | Yes | Yes | No | Limited | Yes |
| Quantization Support | AWQ, GPTQ, FP8 | AWQ, GPTQ, bitsandbytes | GGUF | GGUF, all formats | FP8, INT8, INT4 |
| OpenAI-Compatible API | Yes | Yes (Messages API) | Yes | Yes (via server) | Yes (via Triton) |
| Speculative Decoding | Yes | Yes | No | Yes | Yes |
| Multi-LoRA Serving | Yes | Yes | Limited | No | Limited |
| Ease of Setup | Moderate | Moderate | Very Easy | Easy | Complex |
| GPU Required | Yes | Yes | Optional | Optional | Yes (NVIDIA only) |
| License | Apache 2.0 | Apache 2.0 | MIT | MIT | Apache 2.0 |

### Engine Selection Guide

- **vLLM**: Default choice for production GPU serving. Best throughput for concurrent requests.
- **TGI**: Hugging Face ecosystem integration. Good for HF model hub workflows.
- **Ollama**: Best for local development and prototyping. One-command setup.
- **llama.cpp**: Best for CPU inference and edge deployment. Runs on Apple Silicon.
- **TensorRT-LLM**: Maximum throughput on NVIDIA GPUs. Complex setup but best performance.

## Quantization Methods Comparison

| Method | Bits | Quality (vs FP16) | Speed | Memory Savings | Calibration | Ecosystem |
|--------|------|-------------------|-------|---------------|-------------|-----------|
| AWQ | 4-bit | ~99% | Fast | ~75% | Yes (small dataset) | vLLM, TGI, llama.cpp |
| GPTQ | 4-bit | ~98% | Fast | ~75% | Yes (larger dataset) | vLLM, TGI, llama.cpp |
| GGUF (Q4_K_M) | 4-bit mixed | ~98% | Moderate | ~75% | No | llama.cpp, Ollama |
| GGUF (Q5_K_M) | 5-bit mixed | ~99% | Moderate | ~69% | No | llama.cpp, Ollama |
| bitsandbytes (NF4) | 4-bit | ~97% | Slower | ~75% | No | HF Transformers, TGI |
| FP8 (E4M3) | 8-bit | ~99.5% | Very Fast | ~50% | Optional | TensorRT-LLM, vLLM |

**Selection**: Production NVIDIA GPU = AWQ 4-bit. Max quality = FP8 on Hopper/Ada. Local/edge = GGUF Q4_K_M. Fine-tuning = bitsandbytes NF4.

## Hardware Sizing Guide

### GPU Memory Requirements (Inference, weights only)

| Model Size | FP16 | FP8 | 4-bit (AWQ/GPTQ) | GGUF Q4_K_M |
|-----------|------|-----|-------------------|-------------|
| 7B | 14 GB | 7 GB | 4 GB | 4.5 GB |
| 13B | 26 GB | 13 GB | 7.5 GB | 8 GB |
| 34B | 68 GB | 34 GB | 20 GB | 21 GB |
| 70B | 140 GB | 70 GB | 38 GB | 40 GB |
| 405B | 810 GB | 405 GB | 220 GB | 230 GB |

Add 1-4 GB for KV cache per concurrent request depending on sequence length.

### GPU Recommendations by Use Case

| Use Case | Model Size | Recommended GPU(s) | Quantization |
|----------|-----------|--------------------|----|
| Development/testing | 7-13B | 1x RTX 4090 (24GB) | 4-bit |
| Small production | 7-13B | 1x A10G (24GB) or L4 (24GB) | AWQ 4-bit |
| Medium production | 70B | 2x A100 80GB or 4x A10G | AWQ 4-bit + TP |
| High throughput | 70B | 4-8x A100 80GB or H100 | FP8 + TP |
| Frontier models | 405B | 8x H100 80GB | FP8 + TP |

### CPU/Apple Silicon Inference (llama.cpp)

| Hardware | 7B Q4 | 13B Q4 | 70B Q4 | Tokens/sec (7B Q4) |
|----------|--------|--------|--------|---------------------|
| M1 Pro (16GB) | Yes | Yes | No | ~15-20 t/s |
| M2 Max (32GB) | Yes | Yes | No | ~25-35 t/s |
| M3 Max (48GB) | Yes | Yes | Yes (Q3) | ~30-40 t/s |

## KV Cache Optimization and PagedAttention

### KV Cache Basics

The KV cache stores key-value pairs for all previous tokens in every attention layer, growing linearly with sequence length and batch size.

**KV cache per token** = 2 x num_layers x num_kv_heads x head_dim x bytes_per_element

Example for LLaMA 70B (FP16): 2 x 80 x 8 x 128 x 2 = ~320 KB/token. A batch of 32 at 4096 tokens = ~40 GB.

### PagedAttention (vLLM)

Manages KV cache like virtual memory pages, eliminating waste from pre-allocation. Benefits: near-zero memory waste (from ~60-80% to <4%), 2-4x higher throughput, and prefix caching support.

### Prefix Caching

When requests share system prompts or prefixes, vLLM reuses the KV cache for the shared prefix. Enable with `--enable-prefix-caching`. Beneficial for long system prompts, RAG, and chat applications.

## Deployment Architectures

**Single GPU**: 7-13B models, low-moderate traffic. Simple but single point of failure.

**Multi-GPU Tensor Parallelism**: 70B+ models. `--tensor-parallel-size N`. Requires NVLink interconnect on same node.

**Multi-Instance + Load Balancing**: Horizontal scaling. Fault tolerance and linear throughput scaling. No shared KV cache across instances.

**Multi-Node (Pipeline + Tensor Parallel)**: 405B+ models. Combine `--tensor-parallel-size` with `--pipeline-parallel-size`. Requires high-bandwidth interconnect.

## Kubernetes Deployment Patterns for LLMs

### Key Considerations

1. **GPU scheduling**: NVIDIA device plugin with `nvidia.com/gpu` resource limits
2. **Health checks**: Model-loading-aware readiness probes (loading takes minutes)
3. **Autoscaling**: Scale on GPU utilization or request queue depth, not CPU
4. **Persistent storage**: Mount model weights from PVC or S3/GCS
5. **Node affinity**: Pin LLM pods to GPU node pools

### Example Deployment (vLLM)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-server
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        args: ["--model", "meta-llama/Llama-3.1-70B-Instruct",
               "--tensor-parallel-size", "2", "--enable-prefix-caching"]
        resources:
          limits:
            nvidia.com/gpu: 2
            memory: "200Gi"
        readinessProbe:
          httpGet: { path: /health, port: 8000 }
          initialDelaySeconds: 300
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-a100-80gb
```

## Cost Comparison: Self-Hosted vs API Providers

### Cost per 1M Tokens (approximate, as of 2025)

| Option | Input | Output | Notes |
|--------|-------|--------|-------|
| GPT-4o (OpenAI) | $2.50 | $10.00 | Managed, no infra overhead |
| Claude 3.5 Sonnet | $3.00 | $15.00 | Managed, no infra overhead |
| Llama 3.1 70B (Groq) | $0.59 | $0.79 | Managed, fast inference |
| Llama 3.1 70B (self-hosted, A100) | ~$0.20-0.40 | ~$0.60-1.20 | Requires DevOps expertise |
| Llama 3.1 8B (self-hosted, L4) | ~$0.03-0.05 | ~$0.10-0.20 | High-volume simple tasks |

### Break-Even Analysis

Self-hosting becomes cost-effective when: token volume exceeds 50-100M/month for 70B models (10-20M/month for 7-13B), you maintain >50% GPU utilization, and you have DevOps capacity for GPU infrastructure.

### Hidden Costs of Self-Hosting

GPU provisioning, model updates, monitoring infrastructure, on-call engineering, networking/storage costs, and idle GPU cost during low-traffic periods.

## Further Reading

- [vLLM Documentation](https://docs.vllm.ai/)
- [Text Generation Inference Documentation](https://huggingface.co/docs/text-generation-inference)
- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)
- [TensorRT-LLM Documentation](https://nvidia.github.io/TensorRT-LLM/)
- [PagedAttention (Kwon et al., 2023)](https://arxiv.org/abs/2309.06180)
- [AWQ: Activation-aware Weight Quantization (Lin et al., 2023)](https://arxiv.org/abs/2306.00978)
- [GPTQ: Accurate Post-Training Quantization (Frantar et al., 2023)](https://arxiv.org/abs/2210.17323)
- [Ollama Documentation](https://ollama.com/)
- [NVIDIA Triton Inference Server](https://developer.nvidia.com/triton-inference-server)
