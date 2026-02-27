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
import tiktoken
from datetime import datetime

class LLMCostTracker:
    # Pricing per 1M tokens (as of 2024, check for updates)
    PRICING = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
        "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    }

    def __init__(self):
        self.records = []

    def track(self, model, input_tokens, output_tokens, metadata=None):
        pricing = self.PRICING.get(model, {"input": 0, "output": 0})
        cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

        record = {
            "timestamp": datetime.utcnow().isoformat(),
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
        return df.groupby(["model"]).agg({
            "input_tokens": "sum",
            "output_tokens": "sum",
            "cost_usd": "sum",
            "model": "count",
        }).rename(columns={"model": "request_count"})
```

### 2. LangSmith Tracing

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls__..."
os.environ["LANGCHAIN_PROJECT"] = "my-llm-app"

from langchain_openai import ChatOpenAI
from langchain.callbacks import LangChainTracer

# Automatic tracing with LangChain
llm = ChatOpenAI(model="gpt-4o")
response = llm.invoke("What is MLOps?")  # Automatically traced

# Manual tracing
from langsmith import traceable

@traceable(run_type="llm", name="custom-llm-call")
def my_llm_function(query: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": query}],
    )
    return response.choices[0].message.content
```

### 3. LangFuse Integration

```python
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context

langfuse = Langfuse()

@observe()
def my_rag_pipeline(query: str):
    # Retrieval step (automatically traced)
    langfuse_context.update_current_observation(name="retrieval")
    docs = retriever.invoke(query)

    # Generation step
    langfuse_context.update_current_observation(name="generation")
    response = llm.invoke(format_prompt(query, docs))

    # Log quality score
    langfuse_context.score_current_trace(
        name="relevance",
        value=0.9,
        comment="Highly relevant response",
    )

    return response
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

def measure_streaming_latency(client, messages, model="gpt-4o"):
    """Measure detailed latency metrics for streaming responses."""
    start = time.time()
    first_token_time = None
    output_tokens = 0

    stream = client.chat.completions.create(
        model=model, messages=messages, stream=True, stream_options={"include_usage": True}
    )

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            if first_token_time is None:
                first_token_time = time.time()
            output_tokens += 1

    end = time.time()
    ttft = (first_token_time - start) * 1000 if first_token_time else 0
    total = (end - start) * 1000
    generation_time = (end - first_token_time) if first_token_time else 0
    tps = output_tokens / generation_time if generation_time > 0 else 0

    return LLMLatencyMetrics(
        ttft_ms=round(ttft, 1),
        tps=round(tps, 1),
        total_ms=round(total, 1),
        input_tokens=0,  # From usage
        output_tokens=output_tokens,
    )
```

### 5. Feedback Collection

```python
class FeedbackCollector:
    def __init__(self, storage):
        self.storage = storage

    def log_interaction(self, trace_id, query, response, metadata=None):
        self.storage.save({
            "trace_id": trace_id,
            "query": query,
            "response": response,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata,
            "feedback": None,
        })

    def add_feedback(self, trace_id, rating, comment=None):
        """Add user feedback to an interaction."""
        record = self.storage.get(trace_id)
        record["feedback"] = {
            "rating": rating,        # 1-5 or thumbs up/down
            "comment": comment,
            "feedback_at": datetime.utcnow().isoformat(),
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
            "avg_rating": np.mean(ratings) if ratings else None,
            "rating_distribution": dict(pd.Series(ratings).value_counts()),
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
