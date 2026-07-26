# LLM Cost Optimization Reference

## Model Pricing Comparison

Pricing verified 2026-07 (USD per 1M tokens). Always verify current rates with
providers - these change several times a year. Cache reads are 0.1x input and the
Batch API is 50% off on both OpenAI and Anthropic.

| Model | Input (per 1M) | Output (per 1M) | Context Window | Relative Quality | Best For |
|---|---|---|---|---|---|
| **claude-opus-5** | $5.00 | $25.00 | 200K | Very high | Hardest reasoning, agentic work |
| **claude-sonnet-5** | $3.00 | $15.00 | 200K | High | Long-context, analysis, coding |
| **claude-haiku-4-5** | $1.00 | $5.00 | 200K | Medium-High | Fast, cost-effective |
| **claude-fable-5** | $10.00 | $50.00 | 200K | Frontier | Highest-difficulty tasks |
| **gpt-5** (legacy, EOL 2026-12-11) | $1.25 | $10.00 | 400K | High | General reasoning |
| **gpt-5-mini** | $0.25 | $2.00 | 400K | Medium-High | General tasks, cost-effective |
| **gpt-4.1** | $2.00 | $8.00 | 1M | Medium-High | Very long context |
| **gpt-4.1-mini** | $0.40 | $1.60 | 1M | Medium | High volume, long context |
| **Llama 4 Scout / Maverick** (Bedrock) | ~$0.80 | ~$2.40 | 10M / 1M | Medium-High | Open weights, long context |
| **DeepSeek V3.x** | ~$0.62 | ~$1.85 | 128K | Medium-High | Cost-sensitive reasoning |
| **Llama 4 / Qwen3** (self-hosted) | infra only* | infra only* | Varies | Medium-High | Data privacy, very high volume |

*Self-hosted costs are infrastructure-only and depend on GPU type and utilization;
below roughly 40-50% sustained utilization a hosted API is usually cheaper.

Retired - do not build against these: Claude 3.x (Opus/Sonnet/Haiku), Claude 3.5
Sonnet/Haiku, Gemini 1.5 Pro/Flash. `gpt-4o` remains available as a legacy model
at $2.50/$10.00 but is not a good default for new work.

### Cost Calculation Example

```python
def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = {
        "claude-opus-5":    {"input": 5.00, "output": 25.00},
        "claude-sonnet-5":  {"input": 3.00, "output": 15.00},
        "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
        "gpt-5":            {"input": 1.25, "output": 10.00},
        "gpt-5-mini":       {"input": 0.25, "output": 2.00},
    }
    p = pricing[model]   # KeyError is intentional: never price an unknown model at $0
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000

# Example: 1000 requests averaging 2000 input + 500 output tokens
for model in ["claude-sonnet-5", "claude-haiku-4-5", "gpt-5", "gpt-5-mini"]:
    daily = estimate_cost(model, 2000 * 1000, 500 * 1000)
    print(f"{model}: ${daily:.2f}/day, ${daily * 30:.2f}/month")

# Output:
# claude-sonnet-5:  $13.50/day, $405.00/month
# claude-haiku-4-5: $4.50/day,  $135.00/month
# gpt-5:            $7.50/day,  $225.00/month
# gpt-5-mini:       $1.50/day,  $45.00/month
```

## Cost Optimization Strategies Overview

| Strategy | Effort | Savings Potential | Quality Impact | Best When |
|---|---|---|---|---|
| **Model Routing** | Medium | 40-70% | Minimal if done well | Mixed task complexity |
| **Semantic Caching** | Medium | 20-60% | None (exact reproduction) | Repetitive queries |
| **Provider Prompt Caching** | Very low | up to 90% on the cached prefix | None | Stable system prompt / tools / long docs |
| **Prompt Compression** | Low-Medium | 15-40% | Low-Medium risk | Long prompts, RAG contexts |
| **Batch API** | Low | 50% (OpenAI and Anthropic) | None | Non-real-time workloads |
| **Prompt Optimization** | Low | 10-30% | May improve quality | Verbose or unoptimized prompts |
| **Fine-Tuning** | High | 30-60% | Can improve or degrade | Repetitive, specialized tasks |
| **Self-Hosting** | Very High | 50-80% at scale | Depends on model | Very high volume, data privacy |

## Semantic Caching

### Architecture

```
User Query --> Embedding --> Similarity Search in Cache
                                |
                         [Cache Hit?]
                        /            \
                      Yes             No
                      |               |
               Return Cached    Call LLM --> Store in Cache
               Response         Return Response
```

### GPTCache Implementation

```python
from gptcache import cache
from gptcache.adapter import openai
from gptcache.embedding import Onnx
from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation

# Initialize cache with vector similarity
onnx = Onnx()
cache_base = CacheBase("sqlite")
vector_base = VectorBase("faiss", dimension=onnx.dimension)
data_manager = get_data_manager(cache_base, vector_base)

cache.init(
    embedding_func=onnx.to_embeddings,
    data_manager=data_manager,
    similarity_evaluation=SearchDistanceEvaluation(),
)
cache.set_openai_key()

# GPTCache's openai adapter mirrors the SDK surface it was written against.
# openai.ChatCompletion.create is the removed v0 API; on openai>=1.0 use the
# client-style adapter:
from gptcache.adapter.openai import ChatCompletion

response = ChatCompletion.create(
    model="gpt-5-mini",
    messages=[{"role": "user", "content": "What is machine learning?"}]
)
```

GPTCache is lightly maintained; check that its adapter matches your installed
`openai` version, or wire your own cache around the SDK (as
`scripts/cache_manager.py` does) rather than relying on monkey-patched clients.

### Custom Semantic Cache

```python
import time
import numpy as np
from openai import OpenAI

class SemanticCache:
    def __init__(self, similarity_threshold=0.95, ttl_seconds=3600):
        self.threshold = similarity_threshold
        self.ttl = ttl_seconds
        self.embeddings = []   # List of (embedding, response, timestamp)
        self.client = OpenAI()

    def _embed(self, text: str) -> np.ndarray:
        resp = self.client.embeddings.create(
            input=text, model="text-embedding-3-small"
        )
        return np.array(resp.data[0].embedding)

    def get(self, query: str):
        query_emb = self._embed(query)
        best_score, best_response = 0, None
        for emb, response, ts in self.embeddings:
            if time.time() - ts > self.ttl:
                continue
            score = np.dot(query_emb, emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(emb)
            )
            if score > best_score:
                best_score, best_response = score, response
        if best_score >= self.threshold:
            return best_response
        return None

    def set(self, query: str, response: str):
        emb = self._embed(query)
        self.embeddings.append((emb, response, time.time()))
```

### Cache Invalidation Strategies

- **TTL-based**: Expire entries after a fixed time period. Simple but may serve stale data.
- **Event-based**: Invalidate when underlying data sources change (e.g., knowledge base update).
- **Version-based**: Tag cache entries with prompt/model versions; invalidate on version change.
- **Hybrid**: Use short TTL combined with event-based invalidation for critical data.

## Prompt Compression Techniques

### Token Reduction Strategies

```python
# 1. Remove redundant whitespace and formatting
import re

def compress_whitespace(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()

# 2. LLMLingua-style selective compression
# Removes low-information-density tokens from prompts
from llmlingua import PromptCompressor

compressor = PromptCompressor(
    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
)

compressed = compressor.compress_prompt(
    original_prompt,
    rate=0.5,            # Target 50% compression
    force_tokens=["?"],  # Always keep question marks
)
# Typical result: 50% fewer tokens with <5% quality loss
```

### Context Pruning for RAG

```python
# Only include the most relevant chunks, not all retrieved documents
def prune_context(query: str, documents: list, max_tokens: int = 2000) -> str:
    """Select and truncate documents to fit within token budget."""
    ranked = reranker.rerank(query, documents)
    context = ""
    for doc in ranked:
        if count_tokens(context + doc.text) > max_tokens:
            break
        context += doc.text + "\n\n"
    return context
```

### Summarization-Based Compression

```python
# For very long contexts, summarize before including in prompt
def summarize_for_context(documents: list, max_summary_tokens: int = 500) -> str:
    full_text = "\n".join([d.text for d in documents])
    summary = llm.invoke(
        f"Summarize the following in under {max_summary_tokens} tokens, "
        f"preserving all key facts and numbers:\n\n{full_text}"
    )
    return summary
```

## Model Routing Strategies

### Complexity-Based Routing

```python
# Route queries to different models based on estimated complexity
from enum import Enum

class Complexity(Enum):
    SIMPLE = "simple"       # FAQ, greetings, simple lookups
    MODERATE = "moderate"   # Summarization, basic analysis
    COMPLEX = "complex"     # Multi-step reasoning, code generation

def classify_complexity(query: str) -> Complexity:
    """Use a small classifier or heuristics to estimate query complexity."""
    # Option 1: Rule-based heuristics
    if len(query.split()) < 10 and "?" in query:
        return Complexity.SIMPLE
    # Option 2: Small LLM classifier
    result = fast_llm.invoke(
        f"Classify this query as simple, moderate, or complex: {query}"
    )
    return Complexity(result.strip().lower())

MODEL_MAP = {
    Complexity.SIMPLE: "gpt-5-mini",
    Complexity.MODERATE: "gpt-5-mini",
    Complexity.COMPLEX: "gpt-5",
}

def route_request(query: str) -> str:
    complexity = classify_complexity(query)
    model = MODEL_MAP[complexity]
    return call_llm(model=model, prompt=query)
```

### Cascade Routing (Fallback Pattern)

```python
# Try cheap model first; fall back to expensive model if quality is low
def cascade_route(query: str, quality_threshold: float = 0.8) -> str:
    # Step 1: Try cheap model
    response = call_llm(model="gpt-5-mini", prompt=query)

    # Step 2: Evaluate quality
    quality = evaluate_response(query, response)

    if quality >= quality_threshold:
        return response  # Cheap model was sufficient

    # Step 3: Fall back to expensive model
    return call_llm(model="gpt-5", prompt=query)
```

### Task-Based Routing

| Task Type | Recommended Model | Rationale |
|---|---|---|
| Simple Q&A / FAQ | gpt-5-mini, claude-haiku-4-5 | Low complexity, high volume |
| Summarization | gpt-5-mini, claude-haiku-4-5 | Good enough quality at lower cost |
| Code Generation | claude-sonnet-5, gpt-5 | Requires strong reasoning |
| Complex Analysis | claude-opus-5, claude-sonnet-5 | Multi-step reasoning needed |
| Translation | gpt-5-mini, Mistral Large 3 | Specialized capability |
| Classification | Fine-tuned small model | Highest cost efficiency |
| Embedding | text-embedding-3-small | Dedicated embedding model |

## Batch API vs Real-Time Cost Analysis

### OpenAI Batch API

```python
# Batch API: 50% discount, 24-hour turnaround
import json
from openai import OpenAI

client = OpenAI()

# 1. Prepare batch file (JSONL)
requests = []
for i, prompt in enumerate(prompts):
    requests.append({
        "custom_id": f"req_{i}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-5-mini",
            "messages": [{"role": "user", "content": prompt}]
        }
    })

with open("batch_input.jsonl", "w") as f:
    for req in requests:
        f.write(json.dumps(req) + "\n")

# 2. Upload and create batch
batch_file = client.files.create(file=open("batch_input.jsonl", "rb"), purpose="batch")
batch = client.batches.create(
    input_file_id=batch_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h"
)

# 3. Check status and retrieve results
status = client.batches.retrieve(batch.id)
if status.status == "completed":
    results = client.files.content(status.output_file_id)
```

### When to Use Batch vs Real-Time

| Criterion | Real-Time API | Batch API |
|---|---|---|
| **Latency Requirement** | < 30 seconds | Hours acceptable |
| **Cost** | Full price | 50% discount |
| **Use Cases** | Chat, interactive apps | Evals, data processing, reports |
| **Rate Limits** | Standard limits | Separate (often higher) limits |
| **Error Handling** | Immediate retry | Retry on batch completion |
| **Volume** | Any | Best for 1000+ requests |

## FinOps for LLM Applications

### Cost Monitoring Framework

```python
# Cost tracking middleware for LLM applications
class LLMCostTracker:
    def __init__(self):
        self.daily_costs = {}
        self.budgets = {
            "development": 50.0,
            "staging": 100.0,
            "production": 500.0
        }

    def track(self, model, input_tokens, output_tokens, env, feature):
        cost = estimate_cost(model, input_tokens, output_tokens)
        key = f"{env}:{feature}:{date.today()}"
        self.daily_costs[key] = self.daily_costs.get(key, 0) + cost

        # Check budget
        env_total = sum(v for k, v in self.daily_costs.items()
                        if k.startswith(f"{env}:") and str(date.today()) in k)
        if env_total > self.budgets[env]:
            self.alert(f"Budget exceeded for {env}: ${env_total:.2f}")

    def report(self, env: str) -> dict:
        """Generate cost report by feature for an environment."""
        return {k: v for k, v in self.daily_costs.items() if env in k}
```

### Cost Optimization Checklist

1. **Audit current usage**: Understand which models, features, and users drive cost.
2. **Right-size models**: Use the cheapest model that meets quality requirements for each task.
3. **Implement caching**: Add semantic caching for repetitive or similar queries.
4. **Optimize prompts**: Shorten system prompts, reduce few-shot examples, compress context.
5. **Use batch APIs**: Move non-real-time workloads (evals, reports, preprocessing) to batch.
6. **Set budgets and alerts**: Implement per-team, per-environment, and per-feature budgets.
7. **Monitor continuously**: Track cost per request, cost per user, and cost trends daily.
8. **Review monthly**: Reassess model choices and optimization strategies as pricing changes.

## Best Practices

1. **Measure before optimizing**: Establish cost baselines and quality benchmarks first.
2. **Optimize prompts first**: The cheapest token is one you never send. Trim verbose prompts.
3. **Cache aggressively**: Most applications have significant query overlap.
4. **Route intelligently**: Use small models for simple tasks; reserve large models for complexity.
5. **Use prompt caching**: Take advantage of provider prompt caching for repeated system prompts.
6. **Batch when possible**: Move offline workloads to batch APIs for 50% savings.
7. **Track at the feature level**: Aggregate cost by feature, not just by model or total.

## Common Pitfalls

- Defaulting to the most expensive model for all tasks without evaluating cheaper alternatives.
- Implementing caching without proper invalidation, serving stale responses.
- Compressing prompts so aggressively that output quality degrades noticeably.
- Not accounting for embedding costs when implementing semantic caching (a
  semantic-cache hit is cheap, not free - it still costs one embedding call).
- Breaking provider prompt caching by compressing or reordering the prefix, or by
  injecting a timestamp/request id ahead of the stable content.
- Setting cache similarity thresholds too low, returning irrelevant cached results.
- Ignoring the cost of the routing classifier itself in model routing setups.
- Failing to re-evaluate cost strategies when providers update pricing.
- Over-engineering optimization for low-volume applications where costs are already minimal.

## Further Reading

- [OpenAI Pricing Page](https://openai.com/pricing)
- [Anthropic Pricing Page](https://www.anthropic.com/pricing)
- [OpenAI Batch API Documentation](https://platform.openai.com/docs/guides/batch)
- [LLMLingua Prompt Compression](https://github.com/microsoft/LLMLingua)
- [GPTCache Documentation](https://gptcache.readthedocs.io/)
- [Martian Model Router](https://docs.withmartian.com/)
- [Not Diamond Model Routing](https://notdiamond.ai/)
- [FinOps Foundation](https://www.finops.org/)
- [Chip Huyen - Building LLM Applications for Production](https://huyenchip.com/2023/04/11/llm-engineering.html)
