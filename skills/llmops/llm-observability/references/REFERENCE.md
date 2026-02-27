# LLM Observability Reference

## Observability Platforms Comparison

| Feature | LangSmith | Langfuse | Phoenix (Arize) | Helicone | PromptLayer |
|---|---|---|---|---|---|
| **License** | Proprietary | Open-source (MIT) | Open-source (Apache 2.0) | Open-source + Cloud | Proprietary |
| **Self-Hosted** | No | Yes | Yes | Yes | No |
| **Tracing** | Full chain tracing | Full chain tracing | Full chain tracing | Request-level | Request-level |
| **Evaluation** | Built-in evals | Custom evals | Built-in evals | Limited | Template versioning |
| **LangChain Integration** | Native | SDK callback | SDK callback | Proxy | SDK wrapper |
| **OpenAI Integration** | Via wrapper | SDK wrapper | SDK callback | Proxy (1-line) | SDK wrapper |
| **Cost Tracking** | Yes | Yes | Yes | Yes (primary focus) | Yes |
| **Prompt Management** | Yes (Hub) | Yes | No | No | Yes (primary focus) |
| **Feedback Collection** | Yes | Yes | Yes | Limited | No |
| **Alerting** | Limited | Webhooks | Built-in | Webhooks | No |
| **Pricing** | Free tier + paid | Free self-hosted | Free self-hosted | Free tier + paid | Free tier + paid |
| **Best For** | LangChain-heavy stacks | Self-hosted, flexible | ML-native teams | Simple proxy logging | Prompt versioning |

## Key Metrics for LLM Applications

### Latency Metrics

| Metric | Definition | Target Range | How to Measure |
|---|---|---|---|
| **TTFT** (Time to First Token) | Time from request to first streamed token | < 500ms (chat), < 2s (complex) | Timestamp delta at client |
| **TPS** (Tokens Per Second) | Output generation throughput | 30-80 TPS (cloud APIs) | Token count / generation time |
| **End-to-End Latency** | Total time from user input to final response | < 3s (chat), < 30s (agents) | Client-side measurement |
| **Inter-Token Latency** | Average time between consecutive tokens | < 30ms | Streaming timestamp deltas |
| **Queue Wait Time** | Time spent waiting before processing begins | < 100ms | Server-side instrumentation |

### Token and Cost Metrics

```python
# Token usage tracking structure
token_metrics = {
    "prompt_tokens": 1250,
    "completion_tokens": 340,
    "total_tokens": 1590,
    "cached_tokens": 800,          # Prompt cache hits
    "cost_usd": 0.0042,
    "cost_per_1k_tokens": 0.0026,
    "model": "gpt-4o-mini",
    "request_type": "chat_completion"
}
```

### Quality and Reliability Metrics

| Metric | Description | Collection Method |
|---|---|---|
| **Error Rate** | Percentage of failed requests (timeouts, 5xx, malformed) | Automated logging |
| **Retry Rate** | Percentage of requests requiring retries | Client instrumentation |
| **Hallucination Rate** | Percentage of outputs containing fabricated facts | Automated eval + human review |
| **Relevance Score** | How well the output addresses the user query | LLM-as-judge or embedding similarity |
| **User Satisfaction** | Thumbs up/down or explicit ratings | Feedback UI |
| **Guardrail Trigger Rate** | Percentage of requests hitting safety guardrails | Guardrail logging |

## Tracing LLM Chains and Agents

### OpenTelemetry for LLMs (OpenLLMetry)

```python
# Using OpenLLMetry (Traceloop) for automatic instrumentation
from traceloop.sdk import Traceloop

Traceloop.init(
    app_name="my-llm-app",
    api_endpoint="http://localhost:4318",  # OTLP endpoint
    disable_batch=False
)

# Automatic instrumentation for OpenAI, LangChain, LlamaIndex, etc.
# All LLM calls are now traced automatically
```

### LangSmith Tracing

```python
# LangSmith native tracing with LangChain
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls_..."
os.environ["LANGCHAIN_PROJECT"] = "my-project"

# All LangChain calls are automatically traced
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o")
result = llm.invoke("Explain quantum computing")
# Trace visible at smith.langchain.com
```

### Langfuse Tracing

```python
# Langfuse manual tracing for custom pipelines
from langfuse import Langfuse

langfuse = Langfuse()

trace = langfuse.trace(name="rag-pipeline", user_id="user_123")

# Trace retrieval step
span = trace.span(name="retrieval", input={"query": user_query})
docs = retriever.get_relevant_documents(user_query)
span.end(output={"num_docs": len(docs)})

# Trace generation step
generation = trace.generation(
    name="llm-call",
    model="gpt-4o",
    input=[{"role": "user", "content": augmented_prompt}],
    model_parameters={"temperature": 0.1}
)
response = llm.invoke(augmented_prompt)
generation.end(
    output=response.content,
    usage={"input": 1250, "output": 340}
)

# Add quality score
trace.score(name="relevance", value=0.92)
```

### Trace Hierarchy for Agent Systems

```
Trace: "customer-support-agent"
  |-- Span: "intent-classification" (15ms)
  |-- Span: "tool-selection" (120ms)
  |   |-- Generation: "llm-reasoning" (model=gpt-4o, tokens=450)
  |-- Span: "tool-execution: search_kb" (340ms)
  |   |-- Span: "embedding-query" (45ms)
  |   |-- Span: "vector-search" (120ms)
  |   |-- Span: "reranking" (80ms)
  |-- Span: "response-generation" (890ms)
  |   |-- Generation: "llm-synthesis" (model=gpt-4o, tokens=1200)
  |-- Span: "guardrail-check" (65ms)
```

## Token Usage Tracking and Cost Attribution

### Multi-Dimensional Cost Attribution

```python
# Cost attribution schema
cost_record = {
    "timestamp": "2026-02-27T10:30:00Z",
    "trace_id": "trace_abc123",
    "model": "gpt-4o",
    "tokens": {"input": 2400, "output": 600},
    "cost_usd": 0.021,
    # Attribution dimensions
    "team": "customer-support",
    "product": "chatbot-v2",
    "feature": "order-lookup",
    "environment": "production",
    "user_tier": "enterprise",
    "request_type": "rag_query"
}

# Aggregation queries for cost dashboards
# - Cost per team per day
# - Cost per feature per request
# - Cost per user tier
# - Cost trend over time by model
```

### Budget Alerting

```python
# Simple budget tracking with alerting thresholds
class BudgetTracker:
    def __init__(self, daily_limit_usd=100, alert_threshold=0.8):
        self.daily_limit = daily_limit_usd
        self.alert_threshold = alert_threshold
        self.daily_spend = 0.0

    def record_cost(self, cost_usd: float) -> dict:
        self.daily_spend += cost_usd
        utilization = self.daily_spend / self.daily_limit

        if utilization >= 1.0:
            return {"action": "block", "msg": "Daily budget exhausted"}
        elif utilization >= self.alert_threshold:
            return {"action": "alert", "msg": f"Budget {utilization:.0%} used"}
        return {"action": "allow"}
```

## Quality Monitoring: Feedback Loops

### Automated Evaluation Pipeline

```python
# LLM-as-judge for automated quality scoring
evaluation_prompt = """
Rate the following response on a scale of 1-5 for each criterion:

Question: {question}
Context: {context}
Response: {response}

Criteria:
1. Relevance: Does the response address the question?
2. Faithfulness: Is the response supported by the context?
3. Completeness: Does the response fully answer the question?
4. Clarity: Is the response clear and well-structured?

Return JSON: {"relevance": N, "faithfulness": N, "completeness": N, "clarity": N}
"""

# Run evaluations on a sample of production traffic
# Store scores alongside traces for monitoring
```

### Feedback Collection Patterns

```
Implicit Feedback:
  - Response regeneration (user clicked "regenerate") --> negative signal
  - Copy/paste of response --> positive signal
  - Session abandonment after response --> negative signal
  - Follow-up question on same topic --> ambiguous signal

Explicit Feedback:
  - Thumbs up/down on response --> direct quality signal
  - Star rating (1-5) --> granular quality signal
  - Free-text correction --> high-value training signal
  - Report button (harmful/wrong) --> safety signal
```

## Debugging LLM Applications: Common Failure Patterns

### Failure Pattern Taxonomy

| Pattern | Symptoms | Root Cause | Debugging Approach |
|---|---|---|---|
| **Context Overflow** | Truncated or incoherent responses | Input exceeds context window | Check token counts in traces |
| **Retrieval Miss** | Correct format but wrong facts | RAG retriever returned irrelevant docs | Inspect retrieval spans, check embeddings |
| **Prompt Drift** | Gradual quality degradation | Prompt template changes or data shift | Compare prompt versions, A/B test |
| **Tool Misuse** | Agent calls wrong tool or wrong params | Unclear tool descriptions or edge cases | Review tool-call traces, refine descriptions |
| **Infinite Loop** | Agent keeps retrying same action | Missing exit conditions in agent logic | Trace agent step count, add max iterations |
| **Rate Limiting** | Sporadic timeouts and 429 errors | Exceeding provider rate limits | Monitor request rate, implement backoff |
| **Cache Poisoning** | Stale or wrong cached responses | Semantic cache returning incorrect hits | Review cache hit/miss ratio, adjust threshold |

### Debugging Workflow

```
1. Identify the failing trace via error rate dashboard or user report.
2. Open the trace and inspect each span chronologically.
3. For retrieval issues: check the query, retrieved documents, and relevance scores.
4. For generation issues: check the full prompt sent to the LLM.
5. For agent issues: check the sequence of tool calls and reasoning steps.
6. For latency issues: identify the slowest span in the trace.
7. Reproduce with the exact inputs in a development environment.
8. Fix, deploy, and monitor the same metric for improvement.
```

## Dashboard Design for LLM Applications

### Recommended Dashboard Panels

**Operational Health (top row)**
- Request volume (time series, per model)
- Error rate (time series, with threshold line)
- P50/P95/P99 latency (time series)
- Active users (counter)

**Cost and Usage (second row)**
- Daily cost (bar chart, stacked by model)
- Token usage breakdown (input vs output vs cached)
- Cost per request trend
- Budget utilization gauge

**Quality (third row)**
- Average quality scores (relevance, faithfulness)
- Feedback distribution (thumbs up vs down over time)
- Guardrail trigger rate (time series by category)
- Hallucination rate trend

**Debugging (expandable section)**
- Recent error traces (table with links)
- Slowest traces in last hour
- Most expensive traces in last hour
- Failed tool calls breakdown

### Alerting Rules

```yaml
alerts:
  - name: high_error_rate
    condition: error_rate > 5%
    window: 5m
    severity: critical
    channel: pagerduty

  - name: latency_degradation
    condition: p95_latency > 10s
    window: 10m
    severity: warning
    channel: slack

  - name: daily_budget_threshold
    condition: daily_cost > $80
    severity: warning
    channel: slack

  - name: quality_drop
    condition: avg_relevance_score < 3.5
    window: 1h
    severity: warning
    channel: slack

  - name: hallucination_spike
    condition: hallucination_rate > 10%
    window: 30m
    severity: critical
    channel: pagerduty
```

## Best Practices

1. **Trace everything**: Instrument all LLM calls, retrieval steps, and tool executions from day one.
2. **Use structured logging**: Ensure all log entries include trace IDs, model names, and token counts.
3. **Sample evaluations**: Run automated quality evaluations on a representative sample (5-10%) of production traffic.
4. **Set budgets early**: Implement cost tracking and budget alerts before scaling to production.
5. **Correlate metrics**: Link quality scores to specific trace attributes to find patterns.
6. **Version prompts**: Track which prompt version produced which quality scores.
7. **Retain traces**: Keep production traces for at least 30 days for debugging and evaluation.

## Common Pitfalls

- Logging only errors and ignoring successful requests (miss quality degradation).
- Not tracking token usage per request (impossible to attribute costs).
- Over-aggregating metrics (losing per-request detail needed for debugging).
- Ignoring TTFT in favor of total latency (poor streaming user experience).
- Not correlating user feedback with specific traces.
- Storing raw prompts containing PII in observability platforms without redaction.
- Building dashboards without actionable alerting rules.

## Further Reading

- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [Langfuse Documentation](https://langfuse.com/docs)
- [Arize Phoenix Documentation](https://docs.arize.com/phoenix)
- [Helicone Documentation](https://docs.helicone.ai/)
- [OpenLLMetry / Traceloop](https://www.traceloop.com/docs)
- [OpenTelemetry Semantic Conventions for GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Hamel Husain - Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/)
- [Braintrust AI Observability Guide](https://www.braintrust.dev/docs)
