#!/usr/bin/env python3
"""LLM inference benchmarking across multiple backends (vLLM, TGI, Ollama, OpenAI).

Usage:
    python benchmark_inference.py --backend vllm --url http://localhost:8000 --num-requests 50
    python benchmark_inference.py --backend ollama --model llama3.1 --concurrency 8 --output results/ --chart
"""
import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_PROMPTS = [
    "Explain how neural networks learn through backpropagation.",
    "Write a Python class that implements a binary search tree.",
    "Compare microservice and monolithic architectures.",
    "Describe the CAP theorem and its implications for distributed databases.",
    "Summarize the main ideas behind reinforcement learning from human feedback.",
    "Explain how garbage collection works in the JVM.",
]
DEFAULT_URLS = {
    "vllm": "http://localhost:8000", "tgi": "http://localhost:8080",
    "ollama": "http://localhost:11434", "openai": "https://api.openai.com",
}


def load_prompts(prompts_path):
    """Load prompts from a JSONL file. Each line should have a 'prompt' or 'text' field."""
    path = Path(prompts_path)
    if not path.exists():
        logger.error(f"Prompts file not found: {prompts_path}"); sys.exit(1)
    prompts = [json.loads(l).get("prompt", json.loads(l).get("text", ""))
               for l in open(path) if l.strip()]
    logger.info(f"Loaded {len(prompts)} prompts from {prompts_path}")
    return prompts


def _build_result(start, first_token_time, token_count):
    """Build a standardized timing result dict."""
    elapsed = time.perf_counter() - start
    ttft = (first_token_time - start) if first_token_time else None
    return {
        "ttft": round(ttft, 4) if ttft is not None else None, "tokens": token_count,
        "latency": round(elapsed, 4), "tps": round(token_count / elapsed, 2) if elapsed > 0 else 0.0,
    }


def _http_post(url, payload, extra_headers=None):
    """Send HTTP POST and return response context manager."""
    import urllib.request
    headers = {"Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    return urllib.request.urlopen(req, timeout=180)


def _request_openai_compat(url, model, prompt, max_tokens):
    """Streaming chat request to OpenAI-compatible endpoint (vLLM / OpenAI)."""
    extra = {}
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        extra["Authorization"] = f"Bearer {api_key}"
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "stream": True}
    start, ft, tc = time.perf_counter(), None, 0
    with _http_post(f"{url}/v1/chat/completions", payload, extra) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            d = line[6:]
            if d == "[DONE]":
                break
            if json.loads(d).get("choices", [{}])[0].get("delta", {}).get("content"):
                if ft is None:
                    ft = time.perf_counter()
                tc += 1
    return _build_result(start, ft, tc)


def _request_tgi(url, model, prompt, max_tokens):
    """Streaming request to a Text Generation Inference endpoint."""
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_tokens, "details": True},
               "stream": True}
    start, ft, tc = time.perf_counter(), None, 0
    with _http_post(f"{url}/generate_stream", payload) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            d = line[5:].strip()
            if d and json.loads(d).get("token", {}).get("text"):
                if ft is None:
                    ft = time.perf_counter()
                tc += 1
    return _build_result(start, ft, tc)


def _request_ollama(url, model, prompt, max_tokens):
    """Streaming request to an Ollama endpoint."""
    payload = {"model": model, "prompt": prompt, "stream": True,
               "options": {"num_predict": max_tokens}}
    start, ft, tc = time.perf_counter(), None, 0
    with _http_post(f"{url}/api/generate", payload) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            chunk = json.loads(line)
            if chunk.get("response"):
                if ft is None:
                    ft = time.perf_counter()
                tc += 1
            if chunk.get("done"):
                break
    return _build_result(start, ft, tc)


BACKEND_DISPATCH = {
    "vllm": _request_openai_compat, "openai": _request_openai_compat,
    "tgi": _request_tgi, "ollama": _request_ollama,
}


def _pct(data, p):
    return data[min(int(len(data) * p / 100), len(data) - 1)] if data else None


def _stats(data):
    """Compute mean, p50, p95, p99 for a sorted numeric list."""
    if not data:
        return {"mean": None, "p50": None, "p95": None, "p99": None}
    return {"mean": round(sum(data) / len(data), 4), "p50": round(_pct(data, 50), 4),
            "p95": round(_pct(data, 95), 4), "p99": round(_pct(data, 99), 4)}


def run_benchmark(backend, url, model, prompts, num_requests, concurrency, max_tokens, warmup=2):
    """Execute the benchmark: warmup phase then measured concurrent run."""
    fn = BACKEND_DISPATCH[backend]
    if warmup > 0:
        logger.info(f"Running {warmup} warmup request(s)...")
        for i in range(warmup):
            try:
                fn(url, model, prompts[i % len(prompts)], max_tokens)
            except Exception as e:
                logger.warning(f"Warmup {i + 1} failed: {e}")
    logger.info(f"Benchmarking {backend} at {url} | model={model} | "
                f"requests={num_requests} | concurrency={concurrency}")
    results, wall_start = [], time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(fn, url, model, prompts[i % len(prompts)], max_tokens)
                for i in range(num_requests)]
        for f in as_completed(futs):
            try:
                results.append(f.result())
            except Exception as e:
                results.append({"error": str(e)})
    wall = time.perf_counter() - wall_start
    ok = [r for r in results if "error" not in r]
    errs = [r for r in results if "error" in r]
    if not ok:
        logger.error(f"All {len(errs)} requests failed. First: {errs[0]['error']}"); return None
    return {
        "backend": backend, "url": url, "model": model,
        "num_requests": num_requests, "concurrency": concurrency,
        "max_tokens": max_tokens, "successful": len(ok), "failed": len(errs),
        "wall_time_sec": round(wall, 3), "throughput_rps": round(len(ok) / wall, 2),
        "total_tokens_generated": sum(r["tokens"] for r in ok),
        "ttft": _stats(sorted(r["ttft"] for r in ok if r["ttft"] is not None)),
        "tps": _stats(sorted(r["tps"] for r in ok)),
        "latency": _stats(sorted(r["latency"] for r in ok)),
    }


def save_results(report, output_dir, save_chart=False):
    """Save benchmark report to JSON and optionally generate a matplotlib chart."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    json_path = out / f"benchmark_{report['backend']}_{ts}.json"
    json_path.write_text(json.dumps(report, indent=2))
    logger.info(f"Results saved to {json_path}")
    if not save_chart:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping chart"); return
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (key, title, yl), col in zip(
        axes, [("ttft", "TTFT (s)", "Seconds"), ("tps", "Tokens/s", "Tokens/s"),
               ("latency", "Latency (s)", "Seconds")], ["#4c72b0", "#55a868", "#c44e52"]):
        m = report[key]
        if m["mean"] is not None:
            ax.bar(["mean", "p50", "p95", "p99"],
                   [m["mean"], m["p50"], m["p95"], m["p99"]], color=col)
        ax.set_title(title); ax.set_ylabel(yl)
    fig.suptitle(f"{report['backend']} / {report['model']}", fontsize=14)
    plt.tight_layout()
    fig.savefig(out / f"benchmark_{report['backend']}_{ts}.png", dpi=150)
    plt.close(fig)


def print_report(report):
    """Print a formatted summary table to stdout."""
    sep = "=" * 60
    print(f"\n{sep}\n  Benchmark: {report['backend']}  |  {report['model']}\n{sep}")
    print(f"  Requests: {report['successful']} OK / {report['failed']} fail  |  "
          f"Wall: {report['wall_time_sec']}s  |  {report['throughput_rps']} req/s")
    print(f"  Tokens: {report['total_tokens_generated']}  |  Concurrency: {report['concurrency']}")
    print(f"{'-' * 60}\n  {'Metric':<15} {'Mean':>10} {'p50':>10} {'p95':>10} {'p99':>10}")
    for key, u in [("ttft", "s"), ("tps", "t/s"), ("latency", "s")]:
        m = report[key]
        v = [f"{m[k]}{u}" if m[k] is not None else "n/a" for k in ("mean", "p50", "p95", "p99")]
        print(f"  {key.upper():<15} {v[0]:>10} {v[1]:>10} {v[2]:>10} {v[3]:>10}")
    print(sep)


def main():
    p = argparse.ArgumentParser(description="LLM inference benchmarking across multiple backends")
    p.add_argument("--backend", required=True, choices=["vllm", "tgi", "ollama", "openai"])
    p.add_argument("--url", default=None, help="Server URL (auto-detected per backend if omitted)")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--prompts", default=None, help="JSONL file with prompts")
    p.add_argument("--num-requests", type=int, default=20)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--warmup", type=int, default=2, help="Warmup requests before measuring")
    p.add_argument("--output", default=None, help="Output directory for results JSON and chart")
    p.add_argument("--chart", action="store_true", help="Generate matplotlib bar chart")
    args = p.parse_args()
    url = args.url or DEFAULT_URLS.get(args.backend)
    prompts = load_prompts(args.prompts) if args.prompts else DEFAULT_PROMPTS
    report = run_benchmark(backend=args.backend, url=url, model=args.model, prompts=prompts,
                           num_requests=args.num_requests, concurrency=args.concurrency,
                           max_tokens=args.max_tokens, warmup=args.warmup)
    if report is None:
        sys.exit(1)
    print_report(report)
    if args.output:
        save_results(report, args.output, save_chart=args.chart)
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
