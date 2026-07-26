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

All prices below are USD per 1M tokens, verified 2026-07. Provider pricing moves;
re-check the pricing pages before committing to a budget.

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
            "simple": {"name": "gpt-5-mini", "cost_per_1m_in": 0.25, "cost_per_1m_out": 2.00},
            "complex": {"name": "gpt-5", "cost_per_1m_in": 1.25, "cost_per_1m_out": 10.00},
            "reasoning": {"name": "claude-opus-5", "cost_per_1m_in": 5.00, "cost_per_1m_out": 25.00},
        }
        self.reasoning_markers = ["prove", "derive", "step by step", "step-by-step",
                                  "plan a migration", "root cause"]

    def classify_complexity(self, query):
        """Classify query complexity to select the right model."""
        # Simple heuristics (replace with a small classifier in production).
        # Order matters: the most specific tier is checked first, otherwise the
        # reasoning tier is unreachable dead code.
        query_lower = query.lower()
        if any(w in query_lower for w in self.reasoning_markers):
            return "reasoning"
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
model = router.route("Summarize this document")  # -> gpt-5-mini
model = router.route("Analyze the architectural tradeoffs between...")  # -> gpt-5
model = router.route("Derive the closed form step by step")  # -> claude-opus-5
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

def compress_prompt(prompt, model="gpt-5-mini"):
    """Reduce prompt token count while preserving meaning.

    Returns the fully compressed prompt plus a per-stage token count so you can
    see which stage actually paid off.
    """
    enc = tiktoken.encoding_for_model(model)
    original_tokens = len(enc.encode(prompt))
    if original_tokens == 0:
        return {"original_tokens": 0, "compressed_tokens": 0,
                "reduction": 0.0, "compressed_prompt": prompt, "stages": {}}

    stages = {}

    # 1. Remove redundant whitespace
    compressed = " ".join(prompt.split())
    stages["whitespace"] = len(enc.encode(compressed))

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
    stages["abbreviate"] = len(enc.encode(compressed))

    # 3. Remove filler words
    fillers = ["basically", "actually", "really", "very", "quite", "just", "simply"]
    for filler in fillers:
        compressed = compressed.replace(f" {filler} ", " ")
    stages["fillers"] = len(enc.encode(compressed))

    final_tokens = stages["fillers"]
    return {
        "original_tokens": original_tokens,
        "compressed_tokens": final_tokens,
        "reduction": 1 - final_tokens / original_tokens,
        "compressed_prompt": compressed,
        "stages": stages,
    }
```

Caveat: compression changes the prompt prefix, which invalidates provider prompt
caching. If a long system prompt is already being cache-read at 0.1x, rewriting it
costs more than the tokens it saves - compress the variable tail, not the cached
prefix.

### 4. Batch API Usage

```python
# OpenAI Batch API - 50% cost reduction
import json
from openai import OpenAI

def create_batch_request(requests, input_file="batch_input.jsonl",
                         model="gpt-5-mini"):
    """Create a batch of LLM requests for 50% cost savings.

    custom_id is how you re-associate results with inputs: batch output lines
    come back in arbitrary order, so never rely on positional matching. Keep the
    id -> original-request map on your side.
    """
    batch_lines = []
    for i, req in enumerate(requests):
        batch_lines.append(json.dumps({
            "custom_id": f"request-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": req["messages"],
                "temperature": req.get("temperature", 0.7),
            },
        }))

    with open(input_file, "w") as f:
        f.write("\n".join(batch_lines))

    # Submit batch
    client = OpenAI()
    with open(input_file, "rb") as fh:
        batch_file = client.files.create(file=fh, purpose="batch")
    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    return batch.id
```

### 5. Cost-Effective Architecture

```python
# Per-1M-token prices for the models this pipeline can route to.
PRICING = {
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5": {"input": 1.25, "output": 10.00},
    "claude-opus-5": {"input": 5.00, "output": 25.00},
}
# Cost of one embedding call for the semantic-cache lookup, per 1M tokens.
EMBEDDING_COST_PER_1M = 0.02

class CostAwareLLMPipeline:
    def __init__(self, cache, router, embedder=None):
        self.cache = cache
        self.router = router
        self.embedder = embedder
        self.client = OpenAI()

    @staticmethod
    def calculate_cost(model, usage):
        p = PRICING.get(model)
        if p is None:
            raise KeyError(f"No pricing for '{model}'")
        return (usage.prompt_tokens * p["input"]
                + usage.completion_tokens * p["output"]) / 1_000_000

    @staticmethod
    def select_top_context(query, chunks, max_tokens=2000, chars_per_token=4):
        """Keep the highest-scoring chunks that fit the context budget.

        Replace the scorer with your retriever's own relevance score; this
        keyword overlap is only a placeholder.
        """
        q_terms = set(query.lower().split())
        ranked = sorted(
            chunks,
            key=lambda c: len(q_terms & set(c.lower().split())),
            reverse=True,
        )
        budget, kept = max_tokens * chars_per_token, []
        for chunk in ranked:
            if len(chunk) > budget:
                break
            kept.append(chunk)
            budget -= len(chunk)
        return "\n\n".join(kept)

    def __call__(self, query, context=None):
        # Step 1: Check cache. A semantic-cache hit is cheap, not free: it still
        # costs one embedding call. Only an exact-hash cache is truly $0.
        lookup_cost = 0.0
        if self.embedder is not None:
            lookup_cost = len(query.split()) * 1.3 * EMBEDDING_COST_PER_1M / 1_000_000
        cached = self.cache.get(query)
        if cached:
            return {"response": cached, "source": "cache", "cost": lookup_cost}

        # Step 2: Route to appropriate model
        model = self.router.route(query)

        # Step 3: Trim the context to a token budget
        messages = [{"role": "user", "content": query}]
        if context:
            relevant = self.select_top_context(query, context, max_tokens=2000)
            messages.insert(0, {"role": "system", "content": relevant})

        # Step 4: Generate
        response = self.client.chat.completions.create(
            model=model, messages=messages
        )

        result = response.choices[0].message.content

        # Step 5: Cache response
        self.cache.set(query, result)

        cost = self.calculate_cost(model, response.usage) + lookup_cost
        return {"response": result, "source": model, "cost": cost}
```

### 6. Provider Prompt Caching

Prompt caching is usually the cheapest win available, and it is orthogonal to the
semantic cache above: the provider caches a *prefix* of your request server-side.

- Cache reads cost 0.1x the normal input rate (Anthropic and OpenAI both); an
  Anthropic cache *write* costs 1.25x, so a prefix must be reused at least twice
  to pay for itself.
- It is a **prefix match**: everything before the cache breakpoint must be
  byte-identical. Put the stable material (system prompt, tool definitions, long
  documents, few-shot examples) first and all variable material last.
- Anything that mutates the prefix - a timestamp, a request id, a compressed
  system prompt, a reordered tool list - invalidates the entry and forces a
  full-price write.
- It interacts badly with prompt compression: compress the variable tail, and
  leave the cached prefix byte-stable.

```python
# Anthropic: mark the end of the cacheable prefix
response = client.messages.create(
    model="claude-sonnet-5",
    system=[{
        "type": "text",
        "text": long_stable_instructions,
        "cache_control": {"type": "ephemeral"},   # everything above is cached
    }],
    messages=[{"role": "user", "content": variable_user_turn}],
    max_tokens=1024,
)
u = response.usage
print(u.cache_creation_input_tokens, u.cache_read_input_tokens)
```

## Model Cost Comparison

Verified 2026-07. Cache reads are 0.1x input on both Anthropic and OpenAI; the
Batch API is 50% off on both.

| Model | Input $/1M | Output $/1M | Speed | Quality |
|-------|-----------|------------|-------|---------|
| claude-opus-5 | $5.00 | $25.00 | Medium | Excellent |
| claude-sonnet-5 | $3.00 | $15.00 | Fast | Excellent |
| claude-haiku-4-5 | $1.00 | $5.00 | Very fast | Good |
| claude-fable-5 | $10.00 | $50.00 | Medium | Frontier |
| gpt-5 (legacy, EOL 2026-12-11) | $1.25 | $10.00 | Fast | Very good |
| gpt-5-mini | $0.25 | $2.00 | Very fast | Good |
| gpt-4.1-mini | $0.40 | $1.60 | Very fast | Good |
| Llama 4 Scout (Bedrock) | ~$0.80 | ~$2.40 | Fast | Good |
| Llama 4 / Qwen3 (self-hosted) | infra only | infra only | Varies | Good |

`claude-haiku-4-5` is the current small Claude and is priced $1.00/$5.00 - not
$0.80/$4.00 and not the $0.25/$1.25 of the retired Claude 3 Haiku. Claude Sonnet 4.6
is previous-generation ($3/$15, still available); the whole Claude 3.x line is
retired - do not build against it. Note the alias `claude-sonnet-4-6` has no date
suffix.

## Best Practices

1. **Turn on provider prompt caching first** - 0.1x reads for a stable prefix is
   the highest return per line of code changed
2. **Route 80% of traffic** to the cheapest adequate model
3. **Cache aggressively** - Most LLM apps have high query overlap
4. **Use batch API** for non-real-time workloads (50% savings)
5. **Compress prompts** - Remove verbosity, use abbreviations (but not the cached prefix)
6. **Set token limits** - Don't generate more than needed
7. **Monitor cost per feature** not just total spend
8. **Fine-tune small models** for high-volume, narrow tasks
9. **Use embeddings wisely** - Cache embeddings, use smaller models
10. **Budget alerts** before costs become a problem

## Scripts

- `scripts/cost_optimizer.py` - LLM cost optimization pipeline
- `scripts/cache_manager.py` - Semantic and exact caching system

## References

See [references/REFERENCE.md](references/REFERENCE.md) for pricing comparisons and ROI calculators.
