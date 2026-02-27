# MLOps & LLMOps Agent Skills

A comprehensive collection of **25 Agent Skills** for MLOps and LLMOps, following the
[Agent Skills standard](https://agentskills.io). These skills provide domain-expert
guidance, executable scripts, and reference documentation for the complete ML/LLM lifecycle.

Compatible with: **Claude Code**, **Kiro CLI/IDE**, **Cursor**, **VS Code Copilot**,
**Gemini CLI**, and any agent supporting the Agent Skills format.

## Skills Overview

### MLOps Skills (15)

| Skill | Description |
|-------|-------------|
| `data-ingestion` | Batch/streaming ingestion, ETL/ELT, data lake, versioning |
| `data-validation` | Great Expectations, Pandera, data contracts, quality checks |
| `feature-engineering` | Transformations, encoding, selection, sklearn Pipelines |
| `feature-store` | Feast, online/offline stores, point-in-time joins |
| `ml-experiment-tracking` | MLflow, W&B, experiment comparison, reproducibility |
| `model-training` | HPO (Optuna), distributed training (DDP), mixed precision |
| `model-registry` | Versioning, promotion, lineage, model cards, packaging |
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

## License

Apache-2.0
