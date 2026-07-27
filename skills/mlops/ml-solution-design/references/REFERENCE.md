# ML Solution Design Reference Guide

## Deeper Decision Rationale

### Why batch is the default inference mode

The cost asymmetry is large and usually invisible in the first meeting:

- A batch job runs minutes per day on spot/preemptible capacity and scales to zero.
  A real-time endpoint runs 24/7 with at least 2 replicas for availability.
- Real-time serving forces a whole second stack into existence: autoscaling policy,
  p99 latency monitoring, on-call ownership, canary deploys, timeout/retry semantics
  in every caller. Batch needs a scheduler and a table.
- The business action attached to most first ML use cases (retention offer, restock
  order, weekly report) happens on human or daily cadence anyway. A millisecond-fresh
  churn score feeding a weekly email campaign is wasted engineering.

Test to apply in the intake meeting: "If this prediction were computed at 2am and used
at 2pm, what breaks?" If the answer is "nothing", it is batch.

Valid real-time cases: the input does not exist until the request (search query,
transaction in flight, conversation turn), or the action must happen inside the
user's session.

### Why managed is the default hosting model

Self-hosting is a platform product with its own roadmap, on-call, and upgrade
treadmill (K8s minor versions, GPU drivers, MLflow/Airflow upgrades, CVE patching).
That is justified when:

1. **Residency**: data cannot leave premises — managed cloud is simply unavailable.
2. **Scale economics**: sustained utilization is high enough that the managed premium
   (roughly 20-40% over raw compute) exceeds the fully-loaded cost of the platform
   team. This rarely happens below several models in continuous production.
3. **An existing platform team** already operates K8s for other workloads — the
   marginal cost of adding ML serving is then genuinely low.

Anti-pattern to name explicitly in proposals: building a Kubeflow/Feast/Seldon
platform for a single model. The platform outlives its only tenant.

### Managed LLM vs self-hosted GPU break-even

Rough method (put the actual arithmetic in the cost estimate, not adjectives):

1. Compute monthly Bedrock cost: `(input_tokens x input_price + output_tokens x output_price)`.
2. Compute self-hosted cost: GPU node(s) sized for peak concurrency, 24/7,
   + ops time. A single `g5.2xlarge` is ~ $870/mo on-demand; real deployments need
   headroom and usually 2+ nodes for availability.
3. Self-hosting only wins when token volume is high AND steady. Spiky traffic
   destroys self-hosted economics because the GPU idles between spikes while
   Bedrock bills nothing.

Also weigh: open-weights model quality vs frontier models (e.g. Claude on Bedrock,
model id `global.anthropic.claude-sonnet-5`), fine-tuning needs, and prompt-caching
discounts on the managed side.

### Classical vs LLM: the accuracy trap

For tabular prediction, gradient boosting is not just cheaper — it is usually more
accurate than prompting an LLM with serialized rows. LLMs lose on tabular tasks
because they cannot learn dataset-specific feature interactions from a prompt.
The correct LLM role in tabular pipelines is upstream: extracting structured
features from unstructured columns (free-text notes, documents), then handing off
to a classical model. That is the "hybrid" pattern.

## Cost Model Assumptions

Bands used by `architecture_recommender.py` (USD/month, us-east-1, mid-2026 pricing,
order-of-magnitude, treat as ±50%):

| Key | Band | Assumptions |
|-----|------|-------------|
| batch + managed | $150-800 | SageMaker Processing ~1h/day on ml.m5.2xlarge; Batch Transform; S3 storage <1TB |
| batch + self-hosted | $1,500-4,000 | 3-node K8s baseline (m5.xlarge class) + MLflow + Airflow + monitoring; batch marginal |
| realtime + managed | $500-1,500 | 2x ml.m5.xlarge endpoint 24/7 (~$560) + Model Monitor + CloudWatch |
| realtime + self-hosted | $1,800-5,000 | K8s baseline + 2-4 serving replicas + Prometheus/Grafana |
| streaming + managed | $800-2,500 | Kinesis shards + always-on consumer (ECS/Lambda) + endpoint |
| streaming + self-hosted | $2,500-6,000 | 3-broker Kafka + stream processors + K8s baseline |
| LLM adder, managed | +$100-500 | Bedrock at 10M input / 2M output tokens/mo on a mid-tier model; scales linearly with tokens |
| LLM adder, self-hosted | +$750-4,000 | 1 GPU node, g5.xlarge ($750/mo region-dependent) to g5.12xlarge (~$4,000/mo), on-demand |

Excluded from all bands: people cost, data egress, one-time build effort, dev/staging
environments (add ~30-50% for non-prod copies), and enterprise support plans.

## Architecture Doc Template

```markdown
# <Engagement> — ML Solution Architecture
Version: 0.x (draft) | Author: | Date: | Reviewers:

## 1. Context
- Business goal and KPI (owner: <name>)
- Current state / why now

## 2. Requirements Summary
- Link to requirements.json; table of key answers (latency SLO, throughput,
  budget, residency, compliance)

## 3. Key Decisions
| # | Decision | Choice | Rationale | Rejected alternatives (and why) |
|---|----------|--------|-----------|--------------------------------|
| 1 | Inference mode | batch | No per-prediction SLO; daily action cadence | real-time (cost, no need) |
| 2 | Hosting | AWS managed | No K8s team; AWS-committed | self-hosted K8s (ops burden) |
| 3 | Model family | classical (XGBoost) | tabular, labels exist | LLM (cost/latency/accuracy) |

## 4. Architecture Diagram
<ASCII or image; every stage labeled with the repo skill that implements it>

## 5. Stage-to-Skill Mapping
| Stage | Repo skill | Tooling |
|-------|-----------|---------|
| Ingest | data-ingestion | Glue |
| ...   | ...       | ...     |

## 6. Cost Estimate
Monthly band, assumptions, ±50% caveat, link to pricing-calculator worksheet.

## 7. Risks
Link to risk register.

## 8. Acceptance Criteria
Link or inline (see template below).

## 9. Out of Scope
Explicit non-goals — the most dispute-preventing section in the doc.
```

## Risk Register Template

```markdown
| ID | Risk | Source (intake gap?) | Likelihood | Impact | Mitigation | Owner | Status |
|----|------|----------------------|------------|--------|------------|-------|--------|
| R1 | Data access not yet granted | intake gap: data_access_confirmed | High | Blocks week 1 | Access request filed <date>; escalation path <name> | <customer> | Open |
| R2 | No labeled data for supervised task | intake gap: labels_available | Med | Rework of approach | Labeling sprint scoped as phase 0 | <us> | Open |
| R3 | Latency SLO undefined | intake gap: latency_slo_ms | Med | Wrong serving tier built | SLO workshop scheduled | <customer> | Open |
| R4 | Handover team lacks K8s skills | intake | High | Post-handover outages | Chose managed stack; runbook + training | <us> | Mitigated |
```

Rules: every high-severity gap from `intake_questionnaire.py` gets a row. A risk
without an owner and a next action is a complaint, not a register entry.

## Acceptance Criteria Examples

Every criterion measurable, with a dataset/window and a number:

- **Model quality**: F1 (weighted) >= 0.82 on the agreed frozen holdout set
  `s3://.../holdout-2026-06.parquet`; measured by `ml-testing` gate in CI.
- **Latency**: p99 endpoint latency <= 150 ms at 50 RPS sustained for 30 min
  (load test script in repo).
- **Pipeline SLA**: daily batch scoring completes by 06:00 local, >= 99% of days
  over a 30-day acceptance window.
- **Monitoring**: drift and data-quality alarms configured (model-monitoring skill);
  test alert fires end-to-end to the on-call channel.
- **Reproducibility**: any registered model version can be retrained from config +
  data snapshot to within 1% of the registered metric.
- **Handover**: operating team runs one full retrain-register-deploy cycle without
  vendor assistance; runbook covers the top 5 failure modes.

Anti-examples (reject these in review): "model performs well", "system is fast",
"monitoring is in place", "documentation is provided".

## Further Reading

- [AWS Well-Architected Machine Learning Lens](https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/machine-learning-lens.html)
- [Google Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml)
- [Hidden Technical Debt in Machine Learning Systems (Sculley et al.)](https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems)
- [ML Design Patterns (Lakshmanan, Robinson, Munn)](https://www.oreilly.com/library/view/machine-learning-design/9781098115777/)
- [AWS Pricing Calculator](https://calculator.aws/)
