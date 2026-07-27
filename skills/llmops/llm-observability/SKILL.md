---
name: llm-observability
description: >
  Monitor and observe LLM applications in production. Covers token usage tracking and cost monitoring, latency
  monitoring (TTFT, TPS, E2E), LangSmith tracing, LangFuse integration, Phoenix/Arize for LLM observability,
  prompt/completion logging, conversation tracking, quality metrics over time, error rate monitoring, rate limit
  tracking, model comparison dashboards, feedback collection, A/B test analysis, hallucination rate monitoring,
  and LLM-specific alerting. Use when monitoring LLM applications, tracking costs, debugging quality issues,
  or setting up LLM observability infrastructure.
license: Apache-2.0
metadata:
  author: llmops-skills
  version: "1.0"
  category: llmops
---

# LLM Observability

## Overview

LLM observability tracks the behavior, quality, cost, and performance of LLM applications
in production - going beyond traditional monitoring to understand WHY outputs are good or bad.

Tested with: `langfuse>=3`, `langsmith>=0.4`, `arize-phoenix>=8`, `openai>=1.0`.

## When to Use This Skill

- Setting up monitoring for LLM applications
- Tracking token usage and costs
- Debugging quality issues in production
- Building dashboards for LLM performance
- Implementing feedback loops

## Key LLM Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| TTFT | Time to First Token | < 500ms |
| TPS | Tokens Per Second | > 30 |
| E2E Latency | End-to-end response time | < 3s |
| Token Cost | Cost per request | Budget-dependent |
| Error Rate | API failures / total | < 1% |
| Quality Score | LLM-judge or human rating | > 4/5 |
| Hallucination Rate | Ungrounded claims | < 5% |

## Step-by-Step Instructions

### 1. Token Usage and Cost Tracking

```python
import pandas as pd
from datetime import datetime, timezone

class LLMCostTracker:
    # Pricing per 1M tokens, verified 2026-07. Re-check the provider pricing
    # pages before trusting these; model aliases are unversioned on purpose.
    PRICING = {
        "gpt-5": {"input": 1.25, "output": 10.00},
        "gpt-5-mini": {"input": 0.25, "output": 2.00},
        "claude-opus-5": {"input": 5.00, "output": 25.00},
        "claude-sonnet-5": {"input": 3.00, "output": 15.00},
        "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    }

    def __init__(self):
        self.records = []

    def track(self, model, input_tokens, output_tokens, metadata=None):
        if model not in self.PRICING:
            # Never silently price an unknown model at $0.
            raise KeyError(f"No pricing for '{model}'; add it to PRICING")
        pricing = self.PRICING[model]
        cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost_usd": round(cost, 6),
            "metadata": metadata or {},
        }
        self.records.append(record)
        return record

    def daily_report(self):
        df = pd.DataFrame(self.records)
        # Named aggregation: aggregating the grouping key ("model") itself is
        # fragile/deprecated in pandas. Count a real column instead.
        return df.groupby("model").agg(
            input_tokens=("input_tokens", "sum"),
            output_tokens=("output_tokens", "sum"),
            cost_usd=("cost_usd", "sum"),
            request_count=("total_tokens", "count"),
        )
```

Token counting: `tiktoken` is OpenAI-only - use `tiktoken.encoding_for_model(model)`
rather than hardcoding `cl100k_base` (current OpenAI models use `o200k_base`).
For Claude, tiktoken is categorically wrong; call
`client.messages.count_tokens(model=..., messages=[...])` instead. Best of all,
read the `usage` block the provider returns rather than estimating.

### 2. LangSmith Tracing

```python
import os
# LANGSMITH_* is the current naming; the LANGCHAIN_* variables are legacy
# aliases. Current keys are prefixed lsv2_pt_ (personal) / lsv2_sk_ (service).
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "lsv2_pt_..."
os.environ["LANGSMITH_PROJECT"] = "my-llm-app"

from langchain_openai import ChatOpenAI

# Automatic tracing with LangChain - no callback wiring needed
llm = ChatOpenAI(model="gpt-5-mini")
response = llm.invoke("What is MLOps?")  # Automatically traced

# Manual tracing for plain SDK calls
from langsmith import traceable

@traceable(run_type="llm", name="custom-llm-call")
def my_llm_function(query: str) -> str:
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": query}],
    )
    return response.choices[0].message.content
```

### 3. LangFuse Integration

Langfuse v3 is an OpenTelemetry-based rewrite: `langfuse.decorators` and
`langfuse_context` were removed. Import `observe` from the top-level package and
get the singleton client with `get_client()`.

```python
from langfuse import observe, get_client

langfuse = get_client()

@observe()
def my_rag_pipeline(query: str):
    # Nested spans are explicit context managers in v3
    with langfuse.start_as_current_span(name="retrieval") as span:
        docs = retriever.invoke(query)
        span.update(output={"n_docs": len(docs)})

    with langfuse.start_as_current_generation(
        name="generation", model="gpt-5-mini"
    ) as gen:
        response = llm.invoke(format_prompt(query, docs))
        gen.update(output=response)

    # Attach a score to the enclosing trace
    langfuse.update_current_trace(
        metadata={"n_docs": len(docs)},
    )
    langfuse.score_current_trace(name="relevance", value=0.9,
                                 comment="Highly relevant response")
    return response

langfuse.flush()  # required before a short-lived process exits
```

### 4. Latency Monitoring

```python
import time
from dataclasses import dataclass

@dataclass
class LLMLatencyMetrics:
    ttft_ms: float        # Time to First Token
    tps: float            # Tokens Per Second
    total_ms: float       # Total response time
    input_tokens: int
    output_tokens: int

def measure_streaming_latency(client, messages, model="gpt-5-mini"):
    """Measure detailed latency metrics for streaming responses."""
    start = time.time()
    first_token_time = None
    chunk_count = 0
    usage = None

    stream = client.chat.completions.create(
        model=model, messages=messages, stream=True, stream_options={"include_usage": True}
    )

    for chunk in stream:
        # With include_usage=True the final chunk carries usage and no choices.
        if chunk.usage:
            usage = chunk.usage
        if chunk.choices and chunk.choices[0].delta.content:
            if first_token_time is None:
                first_token_time = time.time()
            chunk_count += 1

    end = time.time()
    ttft = (first_token_time - start) * 1000 if first_token_time else 0
    total = (end - start) * 1000
    generation_time = (end - first_token_time) if first_token_time else 0

    # A chunk is not a token - use the authoritative usage block; fall back to
    # the chunk count only when usage is unavailable.
    output_tokens = usage.completion_tokens if usage else chunk_count
    input_tokens = usage.prompt_tokens if usage else 0
    tps = output_tokens / generation_time if generation_time > 0 else 0

    return LLMLatencyMetrics(
        ttft_ms=round(ttft, 1),
        tps=round(tps, 1),
        total_ms=round(total, 1),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
```

### 5. Feedback Collection

```python
from collections import Counter
from datetime import datetime, timezone
from statistics import mean

class FeedbackCollector:
    def __init__(self, storage):
        self.storage = storage

    def log_interaction(self, trace_id, query, response, metadata=None):
        self.storage.save({
            "trace_id": trace_id,
            "query": query,
            "response": response,
            # datetime.utcnow() is deprecated (naive UTC); use aware timestamps.
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata,
            "feedback": None,
        })

    def add_feedback(self, trace_id, rating, comment=None):
        """Add user feedback to an interaction."""
        record = self.storage.get(trace_id)
        record["feedback"] = {
            "rating": rating,        # 1-5 or thumbs up/down
            "comment": comment,
            "feedback_at": datetime.now(timezone.utc).isoformat(),
        }
        self.storage.update(record)

    def quality_report(self, start_date, end_date):
        """Generate quality report from feedback."""
        records = self.storage.query(start_date=start_date, end_date=end_date)
        with_feedback = [r for r in records if r["feedback"]]

        ratings = [r["feedback"]["rating"] for r in with_feedback]
        return {
            "total_interactions": len(records),
            "feedback_rate": len(with_feedback) / max(len(records), 1),
            "avg_rating": mean(ratings) if ratings else None,
            "rating_distribution": dict(Counter(ratings)),
        }
```

### 6. Alerting for LLM Applications

```yaml
# llm_alerts.yaml
alerts:
  - name: high_error_rate
    metric: llm_error_rate
    condition: "> 0.05"
    window: 15m
    severity: critical
    channels: [slack, pagerduty]

  - name: high_latency
    metric: llm_ttft_p95
    condition: "> 2000"  # ms
    window: 10m
    severity: warning
    channels: [slack]

  - name: cost_spike
    metric: llm_daily_cost
    condition: "> 500"  # USD
    window: 1d
    severity: warning
    channels: [slack, email]

  - name: quality_degradation
    metric: llm_quality_score_avg
    condition: "< 3.5"
    window: 1h
    severity: warning
    channels: [slack]

  - name: rate_limit_approaching
    metric: llm_rate_limit_usage
    condition: "> 0.8"
    window: 5m
    severity: warning
    channels: [slack]
```

## Best Practices

1. **Track every LLM call** - Tokens, latency, cost, model version
2. **Log prompts and completions** for debugging (with PII redaction)
3. **Monitor TTFT separately** from total latency
4. **Set cost budgets** with alerts before they're exceeded
5. **Collect user feedback** - Thumbs up/down at minimum
6. **Sample for quality evaluation** - LLM-judge on random subset
7. **Track by feature/use case** not just globally
8. **Monitor rate limits** and implement backoff
9. **Compare models** side-by-side when evaluating switches

## Scripts

- `scripts/llm_monitor.py` - LLM monitoring and cost tracking
- `scripts/quality_tracker.py` - Quality metrics and feedback collection

## References

See [references/REFERENCE.md](references/REFERENCE.md) for platform comparisons and dashboard templates.

## Related skills

**Upstream:** `llm-deployment` + `llm-guardrails` (traces, token counts, violation events) · **Downstream:** `llm-cost-optimization` (usage telemetry that drives routing/caching) and `llm-evaluation` (production samples for regression evals)
**See also:** `model-observability` for the classic-ML counterpart
