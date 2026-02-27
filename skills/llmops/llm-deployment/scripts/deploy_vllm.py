#!/usr/bin/env python3
"""vLLM deployment and benchmarking toolkit.

Usage:
    python deploy_vllm.py --action serve --model meta-llama/Llama-3.1-8B-Instruct --tp 1
    python deploy_vllm.py --action docker-compose --model meta-llama/Llama-3.1-8B-Instruct --tp 2 --quantization awq
    python deploy_vllm.py --action k8s-manifest --model meta-llama/Llama-3.1-8B-Instruct --tp 4
    python deploy_vllm.py --action benchmark --model meta-llama/Llama-3.1-8B-Instruct --num-requests 50 --concurrency 8
    python deploy_vllm.py --action health-check --port 8000
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BENCHMARK_PROMPTS = [
    "Explain the concept of distributed systems in plain language.",
    "Write a Python function to compute the nth Fibonacci number using memoization.",
    "Summarize the key principles of object-oriented programming.",
    "What are the trade-offs between SQL and NoSQL databases?",
    "Describe how transformers work in natural language processing.",
]


def serve(args):
    """Launch a vLLM OpenAI-compatible API server."""
    import subprocess
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model, "--tensor-parallel-size", str(args.tp),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-model-len", str(args.max_model_len),
        "--port", str(args.port), "--host", "0.0.0.0",
    ]
    if args.quantization and args.quantization != "none":
        cmd.extend(["--quantization", args.quantization])
    logger.info(f"Starting vLLM server: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        logger.error("vLLM is not installed. Run: pip install vllm")
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        logger.error(f"vLLM server exited with code {exc.returncode}")
        sys.exit(exc.returncode)


def generate_docker_compose(args):
    """Generate a docker-compose.yaml for production vLLM deployment."""
    quant_env = f"\n      - QUANTIZATION={args.quantization}" if args.quantization not in (None, "none") else ""
    compose = (
        'version: "3.8"\n\nservices:\n  vllm:\n    image: vllm/vllm-openai:latest\n'
        f'    runtime: nvidia\n    ports:\n      - "{args.port}:8000"\n'
        f"    environment:\n      - MODEL={args.model}\n"
        f"      - TENSOR_PARALLEL_SIZE={args.tp}\n"
        f"      - GPU_MEMORY_UTILIZATION={args.gpu_memory_utilization}\n"
        f"      - MAX_MODEL_LEN={args.max_model_len}{quant_env}\n"
        "      - HF_TOKEN=${HF_TOKEN:-}\n"
        "    volumes:\n      - huggingface-cache:/root/.cache/huggingface\n"
        "    command: >\n"
        "      --model ${MODEL} --tensor-parallel-size ${TENSOR_PARALLEL_SIZE}\n"
        "      --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION}\n"
        "      --max-model-len ${MAX_MODEL_LEN} --host 0.0.0.0 --port 8000\n"
        "    deploy:\n      resources:\n        reservations:\n          devices:\n"
        f"            - driver: nvidia\n              count: {args.tp}\n"
        "              capabilities: [gpu]\n"
        '    healthcheck:\n      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]\n'
        "      interval: 30s\n      timeout: 10s\n      retries: 5\n      start_period: 120s\n"
        "    restart: unless-stopped\n\nvolumes:\n  huggingface-cache:\n"
    )
    out_path = Path(args.output_dir) / "docker-compose.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(compose)
    logger.info(f"docker-compose.yaml written to {out_path}")


def generate_k8s_manifest(args):
    """Generate Kubernetes Deployment + Service + HPA manifests."""
    quant_arg = f'            - "--quantization={args.quantization}"\n' if args.quantization not in (None, "none") else ""
    manifest = (
        "---\napiVersion: apps/v1\nkind: Deployment\nmetadata:\n"
        "  name: vllm-server\n  labels:\n    app: vllm-server\nspec:\n"
        "  replicas: 1\n  selector:\n    matchLabels:\n      app: vllm-server\n"
        "  template:\n    metadata:\n      labels:\n        app: vllm-server\n"
        "    spec:\n      containers:\n        - name: vllm\n"
        "          image: vllm/vllm-openai:latest\n"
        "          ports:\n            - containerPort: 8000\n"
        f'          args:\n            - "--model={args.model}"\n'
        f'            - "--tensor-parallel-size={args.tp}"\n'
        f'            - "--gpu-memory-utilization={args.gpu_memory_utilization}"\n'
        f'            - "--max-model-len={args.max_model_len}"\n'
        '            - "--host=0.0.0.0"\n            - "--port=8000"\n'
        f"{quant_arg}"
        "          env:\n            - name: HF_TOKEN\n"
        "              valueFrom:\n                secretKeyRef:\n"
        "                  name: hf-token\n                  key: token\n"
        "                  optional: true\n"
        "          resources:\n"
        f'            limits:\n              nvidia.com/gpu: "{args.tp}"\n'
        f'            requests:\n              nvidia.com/gpu: "{args.tp}"\n'
        "          readinessProbe:\n            httpGet:\n              path: /health\n"
        "              port: 8000\n            initialDelaySeconds: 120\n"
        "            periodSeconds: 15\n"
        "          livenessProbe:\n            httpGet:\n              path: /health\n"
        "              port: 8000\n            initialDelaySeconds: 180\n"
        "            periodSeconds: 30\n"
        "          volumeMounts:\n            - name: cache\n"
        "              mountPath: /root/.cache/huggingface\n"
        "      volumes:\n        - name: cache\n          emptyDir: {}\n"
        "---\napiVersion: v1\nkind: Service\nmetadata:\n  name: vllm-service\nspec:\n"
        "  selector:\n    app: vllm-server\n  ports:\n"
        "    - protocol: TCP\n      port: 80\n      targetPort: 8000\n"
        "  type: ClusterIP\n"
        "---\napiVersion: autoscaling/v2\nkind: HorizontalPodAutoscaler\n"
        "metadata:\n  name: vllm-hpa\nspec:\n  scaleTargetRef:\n"
        "    apiVersion: apps/v1\n    kind: Deployment\n    name: vllm-server\n"
        "  minReplicas: 1\n  maxReplicas: 4\n  metrics:\n    - type: Pods\n"
        "      pods:\n        metric:\n          name: vllm_requests_running\n"
        '        target:\n          type: AverageValue\n          averageValue: "8"\n'
    )
    out_path = Path(args.output_dir) / "vllm-k8s.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(manifest)
    logger.info(f"Kubernetes manifests written to {out_path}")


def _send_request(url, model, prompt, max_tokens=256):
    """Send a single streaming chat completion request and collect timing info."""
    import urllib.request
    payload = json.dumps({
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"{url}/v1/chat/completions", data=payload,
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    first_token_time = None
    total_tokens = 0
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            for line in resp:
                decoded = line.decode("utf-8").strip()
                if not decoded.startswith("data: "):
                    continue
                data_str = decoded[len("data: "):]
                if data_str == "[DONE]":
                    break
                delta = json.loads(data_str).get("choices", [{}])[0].get("delta", {})
                if delta.get("content"):
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    total_tokens += 1
    except Exception as exc:
        return {"error": str(exc)}
    end = time.perf_counter()
    ttft = (first_token_time - start) if first_token_time else None
    elapsed = end - start
    return {
        "ttft": round(ttft, 4) if ttft else None,
        "total_tokens": total_tokens,
        "elapsed": round(elapsed, 4),
        "tps": round(total_tokens / elapsed, 2) if elapsed > 0 else 0,
    }


def _percentile(data, p):
    """Return the p-th percentile from a sorted list."""
    idx = int(len(data) * p / 100)
    return data[min(idx, len(data) - 1)]


def benchmark(args):
    """Send concurrent requests to the vLLM server and measure performance."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    url = f"http://localhost:{args.port}"
    logger.info(f"Benchmarking {url} | requests={args.num_requests} concurrency={args.concurrency}")
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(_send_request, url, args.model, BENCHMARK_PROMPTS[i % len(BENCHMARK_PROMPTS)])
            for i in range(args.num_requests)
        ]
        for fut in as_completed(futures):
            results.append(fut.result())
    successes = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]
    if not successes:
        logger.error(f"All {len(errors)} requests failed.")
        return
    ttfts = sorted(r["ttft"] for r in successes if r["ttft"] is not None)
    tps_vals = sorted(r["tps"] for r in successes)
    latencies = sorted(r["elapsed"] for r in successes)
    report = {
        "total_requests": args.num_requests, "successful": len(successes),
        "failed": len(errors), "concurrency": args.concurrency,
        "ttft_mean": round(sum(ttfts) / len(ttfts), 4) if ttfts else None,
        "ttft_p50": round(_percentile(ttfts, 50), 4) if ttfts else None,
        "ttft_p95": round(_percentile(ttfts, 95), 4) if ttfts else None,
        "tps_mean": round(sum(tps_vals) / len(tps_vals), 2),
        "tps_p50": round(_percentile(tps_vals, 50), 2),
        "latency_mean": round(sum(latencies) / len(latencies), 4),
        "latency_p95": round(_percentile(latencies, 95), 4),
        "throughput_rps": round(len(successes) / max(latencies), 2) if latencies else 0,
    }
    print(json.dumps(report, indent=2))
    logger.info("Benchmark complete")


def health_check(args):
    """Check the vLLM server health endpoint."""
    import urllib.request
    url = f"http://localhost:{args.port}/health"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            status, body = resp.status, resp.read().decode()
        if status == 200:
            logger.info(f"Health check passed (HTTP {status}): {body}")
        else:
            logger.warning(f"Health check returned HTTP {status}: {body}")
            sys.exit(1)
    except Exception as exc:
        logger.error(f"Health check failed: {exc}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="vLLM deployment and benchmarking toolkit")
    parser.add_argument("--action", required=True,
                        choices=["serve", "docker-compose", "k8s-manifest", "benchmark", "health-check"])
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="Model name or path")
    parser.add_argument("--tp", type=int, default=1, help="Tensor-parallel size (number of GPUs)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--quantization", choices=["awq", "gptq", "none"], default="none")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-requests", type=int, default=20, help="Benchmark request count")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent requests for benchmark")
    parser.add_argument("--output-dir", default=".", help="Output directory for generated files")
    args = parser.parse_args()
    actions = {
        "serve": serve, "docker-compose": generate_docker_compose,
        "k8s-manifest": generate_k8s_manifest, "benchmark": benchmark,
        "health-check": health_check,
    }
    actions[args.action](args)


if __name__ == "__main__":
    main()
