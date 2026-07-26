# Accuracy Review — July 2026

Full findings from the deep technical-accuracy review that produced the 2026-07 overhaul PR.
Every issue below was verified at the cited file:line before being fixed; fixes were
functionally tested (see `TEST_RESULTS.md` at the repo root and `tests/aws_validation/`).

## Contents

| File | Scope |
|---|---|
| `mlops-batch1.md` | data-ingestion, data-validation, feature-engineering, feature-store, ml-experiment-tracking |
| `mlops-batch2.md` | model-training, model-registry, model-serving, model-monitoring, model-drift-detection |
| `mlops-batch3.md` | model-observability, ml-pipeline-orchestration, ml-testing, ml-security, ml-cost-optimization |
| `llmops-batch1.md` | llm-fine-tuning, llm-evaluation, llm-deployment, llm-prompt-engineering, llm-rag |
| `llmops-batch2.md` | llm-guardrails, llm-observability, llm-agent-orchestration, llm-cost-optimization, llm-data-preparation |
| `fact-sheet-2026.md` | The verified July-2026 library/API/pricing fact sheet all fixes were checked against (sources: PyPI, GitHub releases, official docs) |

## Method

1. **Baseline** — full test suite run before any change (25/25 schema, 48 script tests).
2. **Review** — five parallel deep-review passes, each reading every SKILL.md,
   REFERENCE.md, and script in its 5-skill batch in full, citing file:line for each issue.
3. **Research** — independent fact-checking pass against primary sources for library
   versions, breaking API changes, and model pricing (see `fact-sheet-2026.md`).
4. **Fix** — five parallel fix passes applying root-cause fixes with per-change
   functional verification (e.g. the chi-squared fix was proven against data where
   a category is absent from the reference window; the PSI fix was proven to detect
   a 3σ location shift the old code scored ~0).
5. **Live AWS validation** — model-training/registry/serving patterns executed
   end-to-end on a real SageMaker account: 12/12 checks passed
   (`tests/aws_validation/`).

These findings files are kept as-is (working notes, terse by design) so every fix in
the PR can be traced back to its original finding.
