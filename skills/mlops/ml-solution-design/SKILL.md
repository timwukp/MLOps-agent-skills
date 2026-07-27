---
name: ml-solution-design
description: >
  Turn customer requirements into an ML solution architecture. Covers requirements intake
  (business goals, data reality, SLOs, budget, team skills, compliance), decision frameworks
  (batch vs real-time vs streaming inference, managed SageMaker/Bedrock vs self-hosted K8s/vLLM,
  build vs buy, classical ML vs LLM vs hybrid), reference architectures, cost estimation,
  and engagement deliverables. Use when starting a customer engagement, gathering ML requirements,
  choosing an architecture, writing a solution proposal, deciding batch vs real-time,
  SageMaker vs self-hosted, or scoping an MLOps/LLMOps delivery.
license: Apache-2.0
metadata:
  author: mlops-skills
  version: "1.0"
  category: mlops
---

# ML Solution Design

## Overview

Solution design is step zero of every ML engagement: before writing a single pipeline, translate
customer requirements into an architecture, a cost estimate, and acceptance criteria. Most failed
ML projects fail here — not in modeling. This skill provides the intake checklist, the decision
frameworks, and reference architectures that map directly onto the other skills in this repo.

## When to Use This Skill

- Kicking off a customer or internal ML/LLM engagement
- Gathering and structuring ML requirements
- Choosing between batch, real-time, and streaming inference
- Deciding managed (SageMaker/Bedrock) vs self-hosted (Kubernetes/vLLM)
- Making build-vs-buy calls for feature store, registry, monitoring
- Writing a solution proposal, architecture doc, or statement of work

## Step-by-Step Instructions

### 1. Requirements Intake Checklist

Work through every row before proposing anything. Missing answers are risks, not gaps to fill
with assumptions. Use `scripts/intake_questionnaire.py` to capture answers as `requirements.json`.

| Area | Questions to answer | Red flags |
|------|--------------------|-----------|
| Business goal | What decision does the model drive? What is the $ value of a 1% metric improvement? Who owns the KPI? | "We want AI" with no KPI owner |
| Data reality | Where does data live? Volume/velocity? Labels available? Label quality? PII present? | No labels, no data access yet, "data team will provide it later" |
| Latency SLO | p50/p99 latency budget for a prediction? Is a 24h-old prediction acceptable? | "Real-time" requested without a number |
| Throughput | Predictions per second/day? Peak vs average? Growth in 12 months? | Peak load unknown |
| Budget | Monthly infra budget? One-time build budget? Team cost included? | Budget undefined until architecture chosen |
| Team skills | Who operates this after handover? K8s experience? Python-only? On-call rotation exists? | Handover to a team that has never run the proposed stack |
| Compliance | Data residency? PII/PHI? Audit requirements? Model explainability mandated? | Compliance discovered after architecture is fixed |
| Timeline | Hard deadline? PoC-first or straight to production? | Production expectations on a PoC timeline |

### 2. Batch vs Real-Time vs Streaming Inference

Decide from the latency SLO and how fresh inputs must be — not from what sounds impressive.

| Criterion | Batch | Real-time (online) | Streaming |
|-----------|-------|--------------------|-----------|
| Prediction freshness needed | Hours to days OK | Milliseconds to seconds | Seconds; continuous events |
| Trigger | Schedule (cron, pipeline) | Request/response API | Event arrival (Kafka/Kinesis) |
| Typical latency SLO | None per-prediction | p99 < 100ms–1s | End-to-end seconds |
| Cost profile | Cheapest (spot, scale-to-zero) | Always-on endpoints | Always-on consumers |
| Complexity | Low | Medium | High |
| Examples | Churn scores, demand forecast, nightly recommendations | Fraud check at checkout, search ranking, chatbot | Anomaly detection on sensor/clickstream |
| Repo skills involved | ml-pipeline-orchestration, model-serving (batch transform) | model-serving, model-monitoring | data-ingestion (streaming), model-serving |

Decision rules:
- If a prediction computed last night still drives the same business action, choose **batch**. Roughly 70-80% of first ML use cases in an enterprise are fine as batch.
- Choose **real-time** only when the input is unknowable in advance (user query, transaction in flight).
- Choose **streaming** only when actions must fire on events without a request (alerting, live aggregation). Never as a default — it has the highest operational cost.
- Hybrid is normal: batch-precompute candidates + real-time ranking is the standard recommender pattern.

### 3. Managed (SageMaker/Bedrock) vs Self-Hosted (K8s/vLLM)

| Criterion | Managed (SageMaker, Bedrock) | Self-hosted (K8s + KServe/vLLM, MLflow, Airflow) |
|-----------|------------------------------|--------------------------------------------------|
| Time to first production model | Weeks | Months (platform build first) |
| Ops burden | AWS operates infra; you operate models | You operate everything, need platform team + on-call |
| Cost at low volume | Lower (pay-per-use, serverless endpoints) | Higher (idle cluster baseline ~$1.5-3K/mo minimum) |
| Cost at high sustained volume | Higher (managed premium ~20-40%) | Lower once cluster utilization > ~60% |
| Control / portability | AWS-coupled; SDK v3 (`ModelTrainer` — `Estimator` is removed) | Full control; cloud-portable |
| Team skills required | Python + AWS basics | Python + K8s + Helm + observability stack |
| Compliance / residency | AWS regions, HIPAA/SOC2 inherited | Anything, including on-prem/air-gapped |
| LLM serving | Bedrock: zero-ops, per-token pricing, e.g. `global.anthropic.claude-sonnet-5` | vLLM/TGI on GPU nodes: fixed cost, open-weights models |

Decision rules:
- Team has no K8s experience and no platform team → **managed**. Do not build a platform for one model.
- Data cannot leave premises / air-gapped → **self-hosted** (only real option).
- LLM with spiky or low traffic → **Bedrock per-token** beats a dedicated GPU. Break-even vs a self-hosted GPU node is typically in the tens of millions of tokens/month range — compute it, don't guess.
- Sustained high GPU utilization on open-weights models → self-hosted vLLM wins on cost.
- Start managed, revisit at scale. Migration cost is real but smaller than a premature platform build.

### 4. Build vs Buy per Component

| Component | Buy/managed option | Build/OSS option | Default recommendation |
|-----------|-------------------|------------------|------------------------|
| Feature store | SageMaker Feature Store, Tecton | Feast on Redis/DynamoDB | Skip entirely until you have >2 models sharing features (see feature-store skill) |
| Model registry | SageMaker Model Registry | MLflow Model Registry (3.x, alias-based) | MLflow if self-hosted anything; SageMaker registry if all-in AWS (see model-registry skill) |
| Experiment tracking | SageMaker MLflow (managed) | Self-hosted MLflow, W&B | Managed MLflow — tracking servers are annoying to operate (see ml-experiment-tracking) |
| Orchestration | SageMaker Pipelines, Step Functions | Airflow, Dagster, Argo | Match to team: existing Airflow → keep it (see ml-pipeline-orchestration) |
| Monitoring | SageMaker Model Monitor, CloudWatch | Evidently + Prometheus/Grafana | Evidently for drift either way (see model-monitoring, model-drift-detection) |
| LLM guardrails | Bedrock Guardrails | Custom filters + LLM-judge | Bedrock Guardrails if on Bedrock (see llm-guardrails) |

Rule of thumb: build only what differentiates the customer's business. Feature stores, registries,
and monitoring dashboards never do.

### 5. Classical ML vs LLM vs Hybrid

| Signal | Points to classical ML | Points to LLM |
|--------|------------------------|---------------|
| Data type | Tabular, time series | Unstructured text, documents, conversation |
| Labels | Thousands of labeled rows exist | Few/no labels; task describable in instructions |
| Latency/cost per call | Sub-10ms, fractions of a cent | 100ms-10s, cents per call acceptable |
| Explainability required | SHAP on gradient boosting is defensible | Hard to certify; regulator pushback |
| Output | Score, class, number | Text, extraction, reasoning, dialogue |

Decision rules:
- Tabular prediction (churn, fraud, forecast) → **classical ML** (XGBoost/LightGBM). An LLM is slower, costlier, and usually less accurate here.
- Document extraction, summarization, Q&A over a corpus, chatbots → **LLM**; retrieval-augmented before fine-tuning (see llm-rag, llm-fine-tuning).
- **Hybrid** is the common enterprise answer: LLM extracts structure from documents → classical model scores it; or classical retrieval/ranking feeds an LLM answerer.
- If an LLM is used on AWS, default to Bedrock with current models (e.g. `global.anthropic.claude-sonnet-5`); apply llm-cost-optimization before scaling.

### 6. Reference Architectures

Each stage names the repo skill that implements it.

**A. AWS-managed stack** (default for AWS-committed customers, small ops teams):

```
 sources (RDS/S3/APIs)
        |
 [data-ingestion]          Glue / Kinesis -> S3 data lake
        |
 [data-validation]         Great Expectations checks in pipeline
        |
 [feature-engineering]     SageMaker Processing / Glue jobs
        |
 [model-training]          SageMaker ModelTrainer (SDK v3)
        |
 [model-registry]          SageMaker Model Registry (approval gates)
        |
 [model-serving]           SageMaker endpoints / Batch Transform
        |
 [model-monitoring]        Model Monitor + CloudWatch -> alerts
        |
 [model-drift-detection]   scheduled drift jobs -> retrain trigger
```

**B. Open-source-on-K8s stack** (platform team exists, portability/residency required):

```
 sources
        |
 [data-ingestion]          Kafka / Airbyte -> object store (S3/MinIO)
        |
 [data-validation]         Great Expectations / pandera in DAGs
        |
 [feature-engineering]     Spark / dbt;  [feature-store] Feast (only if shared)
        |
 [ml-pipeline-orchestration]  Airflow / Argo Workflows drives:
        |
 [model-training]          K8s GPU jobs;  [ml-experiment-tracking] MLflow
        |
 [model-registry]          MLflow Registry (aliases: models:/name@champion)
        |
 [model-serving]           KServe / Seldon; vLLM for open-weights LLMs
        |
 [model-monitoring]        Prometheus + Grafana + Evidently
```

**C. Hybrid classical + LLM stack** (document/text workloads with structured scoring):

```
 documents/text                          tabular features
        |                                       |
 [llm-rag] or [llm-fine-tuning]          [feature-engineering]
        |                                       |
 [llm-evaluation]  eval harness          [model-training] -> [model-registry]
        |                                       |
 [llm-deployment]  Bedrock / vLLM        [model-serving]
        |                                       |
 [llm-guardrails]  I/O filtering                |
        |                                       |
 [llm-observability]  traces/cost  <---- joined application layer
                                                |
                                   [model-monitoring] both paths
```

LLM path ordering: llm-rag / llm-fine-tuning → llm-evaluation → llm-deployment →
llm-guardrails → llm-observability. Never deploy an LLM feature without the evaluation
and guardrails stages — they are the LLM equivalents of test suites and input validation.

### 7. Rough Cost Bands (order-of-magnitude, us-east-1, 2026)

| Setup | Monthly infra band |
|-------|--------------------|
| Batch scoring, 1 model, SageMaker processing + batch transform | $150 – $800 |
| Real-time endpoint, 1 model, 2x ml.m5.xlarge + monitoring | $500 – $1,500 |
| K8s ML platform baseline (3-node cluster, MLflow, Airflow, monitoring) | $1,500 – $4,000 before GPUs |
| Single always-on GPU serving node (g5.xlarge – g5.12xlarge) | $750 – $4,000 |
| Bedrock LLM app at moderate volume (10M in / 2M out tokens/mo, mid-tier model) | $100 – $500 (token-dependent — always compute from actual prices) |

Use `scripts/architecture_recommender.py` for a requirements-driven band; treat every number
as ±50% until validated with the pricing calculator and a 2-week PoC measurement.

### 8. Deliverables Checklist

An engagement's design phase is done when these exist (templates in references/REFERENCE.md):

1. **Architecture doc** — context, decisions with rationale (use the decision tables above), diagram, skill/stage mapping, explicitly listed rejected alternatives.
2. **Cost estimate** — monthly infra band + build effort, assumptions stated, ±50% caveat.
3. **Risk register** — every unanswered intake question becomes a risk with owner and mitigation.
4. **Acceptance criteria** — measurable: model metric threshold on holdout, latency SLO, pipeline SLA, handover checklist. "Model works well" is not a criterion.

## Best Practices

1. **Intake before architecture** — never present a stack in the first meeting
2. **Latency SLO is a number** — refuse "real-time" without milliseconds attached
3. **Default to batch** — upgrade to real-time only with evidence
4. **Default to managed** — self-host only for residency, scale economics, or an existing platform team
5. **Classical ML for tabular** — LLMs are for language, not for churn scores
6. **Cost every option** — a decision table without cost columns is advocacy, not analysis
7. **Write down rejected alternatives** — the architecture doc must say why NOT
8. **Every unknown is a risk register entry** — not a silent assumption
9. **Acceptance criteria before build** — agree on the finish line first
10. **Map stages to skills** — each pipeline stage in the diagram should name the repo skill that implements it

## Scripts

- `scripts/intake_questionnaire.py` - Capture requirements interactively or via flags; outputs `requirements.json` plus gap warnings
- `scripts/architecture_recommender.py` - Rule-based stack recommendation from `requirements.json`: architecture pattern, skill mapping, monthly cost band

## References

See [references/REFERENCE.md](references/REFERENCE.md) for decision rationale, cost model
assumptions, and the architecture doc / risk register / acceptance-criteria templates.

## Related skills

**Upstream:** none — lifecycle entry point (business requirements come in here) · **Downstream:** `data-ingestion` (approved architecture, data-source inventory); for LLM projects, `llm-data-preparation`
**See also:** `ml-cost-optimization` for budget sizing at design time · `ml-security` for compliance constraints up front · `ml-pipeline-orchestration` when the design calls for automated pipelines
