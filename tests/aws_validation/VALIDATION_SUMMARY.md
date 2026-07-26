# AWS SageMaker Live Validation Summary

**Account:** <ACCOUNT_ID> (us-east-1) · **Date:** 2026-07-26 · **SDKs:** boto3 1.43.56, sagemaker 3.17.0, Python 3.12

## Run 1: Model Registry lifecycle (validate_registry_patterns.py) — 7/7 PASS
Validates the `model-training` + `model-registry` skill patterns end-to-end on live AWS:
| Check | Result | Time |
|---|---|---|
| S3 bucket setup | PASS | 0.8s |
| Train RandomForest (acc 0.94 > 0.85 gate) + package model.tar.gz + upload S3 | PASS | 25.4s |
| Create Model Package Group | PASS | 0.9s |
| Register model version w/ metrics metadata (PendingManualApproval) | PASS | 0.6s |
| Promote Pending → Approved (registry promotion pattern) | PASS | 0.6s |
| Lineage/metadata retrieval intact | PASS | 0.6s |
| Full cleanup (packages, group, S3 object) | PASS | 1.3s |

## Run 2: Model Serving on Serverless Inference (validate_serving_pattern.py) — 5/5 PASS (after 1 fix)
Validates the `model-serving` skill deployment pattern on a real SageMaker serverless endpoint:
| Check | Result | Time |
|---|---|---|
| Train iris LogisticRegression + package + upload | PASS | 2.9s |
| Create SageMaker Model (sklearn 1.2-1 container) | PASS | 1.2s |
| Create serverless endpoint (1024MB, conc 1) → InService | PASS | 172s |
| Invoke endpoint with CSV → correct predictions | PASS | 0.9s |
| Full cleanup (endpoint, config, model, S3) | PASS | 2.0s |

### Real-world lesson learned (fed back into skill docs)
First attempt FAILED: `ModuleNotFoundError: No module named 'inference'`.
Root cause: when `SAGEMAKER_SUBMIT_DIRECTORY` points at the model.tar.gz itself,
the sklearn container pip-installs the **archive root** as the module dir — so
`inference.py` must be at the tarball root, NOT under `code/`. The common
`code/inference.py` layout only applies when the SDK repacks the model
(sagemaker SDK SKLearnModel path), not when wiring the container env vars by
hand via boto3. This nuance is now documented in the model-serving skill.

Cost of both runs: < $0.05 (serverless per-invocation billing + S3 pennies; everything deleted).
