# MLOps & LLMOps Agent Skills

A comprehensive collection of **27 Agent Skills** for MLOps and LLMOps, following the
[Agent Skills standard](https://agentskills.io). These skills provide domain-expert
guidance, executable scripts, and reference documentation for the complete ML/LLM lifecycle.

Compatible with: **Claude Code**, **Kiro CLI/IDE**, **Cursor**, **VS Code Copilot**,
**Gemini CLI**, and any agent supporting the Agent Skills format.

## Skills Overview

### MLOps Skills (17)

| Skill | Description |
|-------|-------------|
| `ml-solution-design` | Requirements intake, architecture decisions, reference architectures, proposals |
| `data-ingestion` | Batch/streaming ingestion, ETL/ELT, data lake, versioning |
| `data-validation` | Great Expectations, Pandera, data contracts, quality checks |
| `feature-engineering` | Transformations, encoding, selection, sklearn Pipelines |
| `feature-store` | Feast, online/offline stores, point-in-time joins |
| `ml-experiment-tracking` | MLflow, W&B, experiment comparison, reproducibility |
| `model-training` | HPO (Optuna), distributed training (DDP), mixed precision |
| `model-registry` | Versioning, promotion, lineage, model cards, packaging |
| `ml-cicd` | Model CI, GitHub Actions, SageMaker Pipelines, promotion, rollback |
| `model-serving` | FastAPI, BentoML, Triton, K8s, A/B testing, batching |
| `model-monitoring` | Evidently, Whylogs, performance tracking, alerting |
| `model-drift-detection` | PSI, KS test, chi-squared, retraining triggers |
| `model-observability` | SHAP, LIME, tracing, fairness, prediction logging |
| `ml-pipeline-orchestration` | Airflow, Prefect, Dagster, Kubeflow, ZenML |
| `ml-testing` | Behavioral tests, quality gates, regression, CI/CD |
| `ml-security` | Adversarial robustness, differential privacy, RBAC, PII |
| `ml-cost-optimization` | GPU selection, quantization, spot instances, FinOps |

### LLMOps Skills (10)

| Skill | Description |
|-------|-------------|
| `llm-fine-tuning` | LoRA, QLoRA, PEFT, DPO, SFT, dataset preparation |
| `llm-evaluation` | RAGAS, LLM-as-judge, benchmarks, safety evaluation |
| `llm-deployment` | vLLM, TGI, Ollama, quantization (AWQ/GPTQ/GGUF) |
| `llm-prompt-engineering` | Prompt patterns, templates, versioning, injection defense |
| `llm-rag` | Chunking, embeddings, vector stores, hybrid search, reranking |
| `llm-guardrails` | Input/output validation, PII, toxicity, jailbreak prevention |
| `llm-observability` | Token tracking, latency (TTFT/TPS), LangSmith, feedback |
| `llm-agent-orchestration` | Tool use, LangGraph, CrewAI, memory, human-in-the-loop |
| `llm-cost-optimization` | Model routing, semantic caching, prompt compression, batch API |
| `llm-data-preparation` | Synthetic data, annotation (Argilla), deduplication, quality |

## Delivery Lifecycle

The skills compose into two end-to-end delivery chains. Each SKILL.md ends with a
"Related skills" section that names its upstream and downstream neighbors.

### MLOps chain

```
ml-solution-design → data-ingestion → data-validation → feature-engineering
        │                                                      │
        │                                              (feature-store)
        │                                                      │
        └──────────────────────────────────────────► model-training ◄─────────┐
                                                           │                  │
                                          ml-experiment-tracking (parallel)   │
                                                           │                  │
                                                     model-registry           │ retraining
                                                           │                  │ trigger
                                                        ml-cicd               │
                                                           │                  │
                                                     model-serving            │
                                                           │                  │
                                                    model-monitoring          │
                                                           │                  │
                                                 model-drift-detection ───────┘
```

Cross-cutting: `ml-testing` (quality gates inside `ml-cicd`), `ml-security` (all stages),
`ml-cost-optimization` (design-time sizing + operational right-sizing),
`ml-pipeline-orchestration` (automates the chain as DAGs), `model-observability`
(explainability companion to serving + monitoring). The training → registry → serving →
monitoring handoff is specified in
[`skills/mlops/model-registry/references/ARTIFACT_CONTRACT.md`](skills/mlops/model-registry/references/ARTIFACT_CONTRACT.md).

### LLMOps chain

```
ml-solution-design → llm-data-preparation ─┬─► llm-fine-tuning ─┬─► llm-evaluation
                                           └─► llm-rag ─────────┘        │
                                                                   llm-deployment
                                                                          │
                                            llm-agent-orchestration (app layer)
                                                                          │
                                                                   llm-guardrails
                                                                          │
                                                                 llm-observability
                                                                          │
                                                              llm-cost-optimization
```

Cross-cutting: `llm-prompt-engineering` (every stage from RAG generation to agent system
prompts). Bridges: fine-tuned LLMs flow through `model-registry` / `model-serving` /
`model-monitoring` like any other model artifact.

## Installation

### Claude Code

```bash
# Copy skills to your project
cp -r skills/ .claude/skills/

# Or to global skills
cp -r skills/ ~/.claude/skills/
```

### Kiro CLI/IDE

```bash
# Copy to workspace
cp -r skills/ .kiro/skills/

# Or to global skills
cp -r skills/ ~/.kiro/skills/
```

### Other Compatible Agents

Copy the skill folders to the location specified by your agent's skills documentation.

## Skill Structure

Each skill follows the Agent Skills standard:

```
skill-name/
├── SKILL.md           # Instructions and guidance (required)
├── scripts/           # Executable Python/Bash scripts
│   ├── main_tool.py   # Primary automation script
│   └── helper.py      # Supporting script
└── references/        # Detailed reference documentation
    └── REFERENCE.md   # Tool comparisons, deep-dives
```

## Design Principles

- **Platform-agnostic**: Works with any cloud or on-prem setup
- **Framework-inclusive**: PyTorch, TensorFlow, scikit-learn, XGBoost, HuggingFace
- **Practical**: Real code examples, not just theory
- **Production-ready**: Scripts include error handling, logging, CLI interfaces
- **Progressive disclosure**: Quick guidance in SKILL.md, deep details in references

## Testing & Validation

Every skill is continuously tested — see [TEST_RESULTS.md](TEST_RESULTS.md) for the full report:

- **Schema validation**: all 27 skills validated against the Agent Skills specification (`tests/validate_skills.py`)
- **Script tests**: all 55 bundled scripts compile and pass CLI tests (`tests/test_scripts.py`), run automatically in CI on every push
- **Accuracy review (2026-07)**: all code examples and technical claims deep-reviewed against current library releases (MLflow 3.x, Evidently 0.7+, Great Expectations 1.x, Airflow 3.x, Prefect 3.x, TRL 1.x, RAGAS 0.2+, LangChain/LangGraph 1.x, SageMaker SDK v3) and current model pricing
- **Live AWS validation**: the model-training, model-registry, and model-serving patterns were executed end-to-end on a real AWS SageMaker account — registry lifecycle 7/7 checks passed, serverless endpoint deploy + invoke 5/5 checks passed. Reproducible scripts in [`tests/aws_validation/`](tests/aws_validation/)

## License

Apache-2.0
