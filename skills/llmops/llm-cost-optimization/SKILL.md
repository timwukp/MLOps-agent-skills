---
name: llm-cost-optimization
description: >
  Optimize costs for LLM applications. Covers token optimization (prompt compression, caching, context pruning),
  model routing (expensive vs cheap models), semantic caching, prompt caching, response caching, model selection
  strategy (GPT-4o vs GPT-4o-mini vs Claude vs open-source), batch API usage, fine-tuned small models vs large
  models, cost monitoring and budgeting, rate limit management, embedding cost optimization, and ROI analysis
  for LLM features. Use when reducing LLM API costs, implementing caching, choosing cost-effective models,
  or building cost-aware LLM architectures.
license: Apache-2.0
metadata:
  author: llmops-skills
  version: "1.0"
  category: llmops
---

# LLM Cost Optimization

## Overview

LLM API costs can grow quickly at scale. This skill covers practical strategies to reduce
costs while maintaining output quality - from prompt optimization to intelligent model routing.

## When to Use This Skill

- Reducing LLM API costs in production
- Choosing between models for different tasks
- Implementing caching strategies
- Building cost-aware LLM architectures
- Budgeting for LLM workloads

## Cost Optimization Strategies

```
High Impact:
  ┌──────────────────────────────────┐
  │ 1. Model Routing (40-70% save)   │
  │ 2. Caching (30-60% save)         │
  │ 3. Prompt Optimization (20-40%)  │
  │ 4. Batch API (50% save)          │
  │ 5. Fine-tuned small model (80%)  │
  └──────────────────────────────────┘
```

## Step-by-Step Instructions

### 1. Model Routing

```python
class ModelRouter:
    """Route requests to the most cost-effective model."""

    def __init__(self):
        self.models = {
            "simple": {"name": "gpt-4o-mini", "cost_per_1m_in": 0.15, "cost_per_1m_out": 0.60},
            "complex": {"name": "gpt-4o", "cost_per_1m_in": 2.50, "cost_per_1m_out": 10.00},
            "reasoning": {"name": "o3-mini", "cost_per_1m_in": 1.10, "cost_per_1m_out": 4.40},
        }

    def classify_complexity(self, query):
        """Classify query complexity to select the right model."""
        # Simple heuristics (replace with a small classifier in production)
        query_lower = query.lower()
        if any(w in query_lower for w in ["summarize", "extract", "classify", "translate"]):
            return "simple"
        if any(w in query_lower for w in ["analyze", "reason", "compare", "evaluate", "complex"]):
            return "complex"
        if len(query.split()) < 20:
            return "simple"
        return "complex"

    def route(self, query):
        complexity = self.classify_complexity(query)
        model = self.models[complexity]
        return model["name"]

router = ModelRouter()
model = router.route("Summarize this document")  # -> gpt-4o-mini
model = router.route("Analyze the architectural tradeoffs between...")  # -> gpt-4o
```

### 2. Semantic Caching

```python
import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer

class SemanticCache:
    """Cache LLM responses using semantic similarity."""

    def __init__(self, similarity_threshold=0.95):
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.cache = []  # (embedding, query, response)
        self.threshold = similarity_threshold
        self.hits = 0
        self.misses = 0

    def get(self, query):
        query_embedding = self.encoder.encode(query)

        for cached_emb, cached_query, cached_response in self.cache:
            similarity = np.dot(query_embedding, cached_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(cached_emb)
            )
            if similarity >= self.threshold:
                self.hits += 1
                return cached_response

        self.misses += 1
        return None

    def set(self, query, response):
        embedding = self.encoder.encode(query)
        self.cache.append((embedding, query, response))

    @property
    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0
```

### 3. Prompt Compression

```python
import tiktoken

def compress_prompt(prompt, model="gpt-4o", target_reduction=0.3):
    """Reduce prompt token count while preserving meaning."""
    enc = tiktoken.encoding_for_model(model)
    original_tokens = len(enc.encode(prompt))

    strategies = []

    # 1. Remove redundant whitespace
    compressed = " ".join(prompt.split())
    strategies.append(("whitespace", compressed))

    # 2. Abbreviate common phrases
    abbreviations = {
        "for example": "e.g.",
        "that is": "i.e.",
        "in other words": "i.e.",
        "please provide": "provide",
        "I would like you to": "",
        "Could you please": "",
        "Make sure to": "",
    }
    for verbose, short in abbreviations.items():
        compressed = compressed.replace(verbose, short)
    strategies.append(("abbreviate", compressed))

    # 3. Remove filler words
    fillers = ["basically", "actually", "really", "very", "quite", "just", "simply"]
    for filler in fillers:
        compressed = compressed.replace(f" {filler} ", " ")
    strategies.append(("fillers", compressed))

    final_tokens = len(enc.encode(compressed))
    return {
        "original_tokens": original_tokens,
        "compressed_tokens": final_tokens,
        "reduction": 1 - final_tokens / original_tokens,
        "compressed_prompt": compressed,
    }
```

### 4. Batch API Usage

```python
# OpenAI Batch API - 50% cost reduction
import json

def create_batch_request(requests, output_file="batch_output.jsonl"):
    """Create a batch of LLM requests for 50% cost savings."""
    batch_lines = []
    for i, req in enumerate(requests):
        batch_lines.append(json.dumps({
            "custom_id": f"request-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": req["messages"],
                "temperature": req.get("temperature", 0.7),
            },
        }))

    # Write batch file
    input_file = "batch_input.jsonl"
    with open(input_file, "w") as f:
        f.write("\n".join(batch_lines))

    # Submit batch
    client = OpenAI()
    batch_file = client.files.create(file=open(input_file, "rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    return batch.id
```

### 5. Cost-Effective Architecture

```python
class CostAwareLLMPipeline:
    def __init__(self, cache, router):
        self.cache = cache
        self.router = router
        self.client = OpenAI()

    def __call__(self, query, context=None):
        # Step 1: Check cache
        cached = self.cache.get(query)
        if cached:
            return {"response": cached, "source": "cache", "cost": 0}

        # Step 2: Route to appropriate model
        model = self.router.route(query)

        # Step 3: Compress prompt
        messages = [{"role": "user", "content": query}]
        if context:
            # Only include most relevant context
            relevant = self.select_top_context(query, context, max_tokens=2000)
            messages.insert(0, {"role": "system", "content": relevant})

        # Step 4: Generate
        response = self.client.chat.completions.create(
            model=model, messages=messages
        )

        result = response.choices[0].message.content

        # Step 5: Cache response
        self.cache.set(query, result)

        cost = self.calculate_cost(model, response.usage)
        return {"response": result, "source": model, "cost": cost}
```

## Model Cost Comparison

| Model | Input $/1M | Output $/1M | Speed | Quality |
|-------|-----------|------------|-------|---------|
| GPT-4o | $2.50 | $10.00 | Fast | Excellent |
| GPT-4o-mini | $0.15 | $0.60 | Very fast | Good |
| Claude Sonnet 4 | $3.00 | $15.00 | Fast | Excellent |
| Claude Haiku 4.5 | $0.80 | $4.00 | Very fast | Good |
| Llama 3.1 8B (self-hosted) | ~$0.05 | ~$0.05 | Fast | Good |
| Llama 3.1 70B (self-hosted) | ~$0.30 | ~$0.30 | Medium | Very good |

## Best Practices

1. **Route 80% of traffic** to the cheapest adequate model
2. **Cache aggressively** - Most LLM apps have high query overlap
3. **Use batch API** for non-real-time workloads (50% savings)
4. **Compress prompts** - Remove verbosity, use abbreviations
5. **Set token limits** - Don't generate more than needed
6. **Monitor cost per feature** not just total spend
7. **Fine-tune small models** for high-volume, narrow tasks
8. **Use embeddings wisely** - Cache embeddings, use smaller models
9. **Budget alerts** before costs become a problem

## Scripts

- `scripts/cost_optimizer.py` - LLM cost optimization pipeline
- `scripts/cache_manager.py` - Semantic and exact caching system

## References

See [references/REFERENCE.md](references/REFERENCE.md) for pricing comparisons and ROI calculators.
