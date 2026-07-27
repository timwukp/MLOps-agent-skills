---
name: ml-cicd
description: >
  CI/CD for machine learning models: from git push to production. Covers model CI
  (train-on-PR with data slices, quality gates), promotion across dev/staging/prod with
  GitOps and approval gates, container builds for model images, complete GitHub Actions
  workflows, SageMaker Pipelines (SDK v3) as managed CD, EventBridge triggers on model
  approval, and rollback strategy. Use when setting up CI/CD for ML models, automating
  training pipelines on git events, promoting models dev->staging->prod, building
  GitHub Actions/GitLab CI for ML, SageMaker Pipelines, or automating deployment
  on model approval.
license: Apache-2.0
metadata:
  author: mlops-skills
  version: "1.0"
  category: mlops
---

# ML CI/CD

## Overview

ML CI/CD extends software CI/CD with a third axis: alongside code and config, the *model
artifact* (a function of code + data + hyperparameters) must be built, tested, versioned,
promoted, and rolled back. The pipeline below takes a git push through lint, a fast training
run on a data slice, quality gates, registry promotion, staged deployment, and automated
rollback — cross-referencing the sibling skills that own each stage: ml-testing (gates),
model-registry (promotion), model-serving (deploy targets), model-monitoring (post-deploy
verification).

## When to Use This Skill

- Setting up CI for a model repo (train-on-PR, quality gates)
- Automating dev → staging → prod model promotion
- Writing GitHub Actions / GitLab CI pipelines for ML
- Building SageMaker Pipelines for managed CD
- Wiring "deploy on model approval" automation
- Designing rollback for bad model deployments

## Step-by-Step Instructions

### 1. What Runs Where: the ML CI/CD Stage Map

```
 PR opened                push to main              model approved
    |                          |                          |
 [CI: fast]              [CD: full]                [CD: release]
 lint + unit tests       full training run         deploy to staging
 train on data slice     evaluate vs holdout       smoke test endpoint
 quality gates           register-if-better        manual approval gate
 (ml-testing skill)      (model-registry skill)    deploy to prod
                                                   (model-serving skill)
                                                   post-deploy checks
                                                   (model-monitoring skill)
```

Principles:
- **PR CI must be fast (<15 min)**: train on a fixed small slice (e.g. 5-10% stratified
  sample committed as a versioned artifact), not the full dataset. The goal is catching
  broken code and regressions, not producing the production model.
- **Only main builds candidates**: full training runs on merge to main (or nightly), never per-PR.
- **Gates are pipeline stages, not review comments**: encode thresholds from the ml-testing
  skill (min accuracy/F1, max latency, no-worse-than-champion) as steps that fail the job.
- **The registry is the promotion boundary**: CI/CD moves *registry state* (aliases /
  approval status); deployment automation reacts to registry state, not to git.

### 2. Model CI on Pull Requests

Quality gates as an explicit, versioned config (consumed by the workflow below):

```yaml
# ci/gates.yaml
gates:
  min_f1_weighted: 0.80          # absolute floor on the CI data slice
  max_regression_vs_main: 0.02   # candidate may be at most 2pts worse than main's slice score
  max_p95_latency_ms: 100        # single-prediction latency on CI hardware
  max_model_size_mb: 500
```

```python
# ci/check_gates.py -- exit non-zero if any gate fails (fails the pipeline stage)
import json, sys, yaml

gates = yaml.safe_load(open("ci/gates.yaml"))["gates"]
metrics = json.load(open("output/metrics.json"))

failures = []
if metrics["f1_weighted"] < gates["min_f1_weighted"]:
    failures.append(f"f1 {metrics['f1_weighted']:.3f} < floor {gates['min_f1_weighted']}")
if metrics["p95_latency_ms"] > gates["max_p95_latency_ms"]:
    failures.append(f"p95 {metrics['p95_latency_ms']:.1f}ms > {gates['max_p95_latency_ms']}ms")

if failures:
    print("GATE FAILURES:\n  " + "\n  ".join(failures))
    sys.exit(1)
print("All gates passed.")
```

See the ml-testing skill for the full gate catalog (behavioral tests, invariance tests,
data contract checks) — wire each as its own pipeline step so failures are attributable.

### 3. Complete GitHub Actions Workflow

Generated variants (MLflow vs SageMaker registry, different deploy targets) via
`scripts/generate_ml_pipeline.py`. Canonical shape:

```yaml
# .github/workflows/ml-cicd.yml
name: ml-cicd
on:
  pull_request:
    paths: ["src/**", "ci/**", "configs/**"]
  push:
    branches: [main]

permissions:
  id-token: write      # OIDC to AWS -- no long-lived keys (see references)
  contents: read

env:
  AWS_REGION: us-east-1

jobs:
  lint-and-unit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12", cache: pip }
      - run: pip install -r requirements.txt -r requirements-dev.txt
      - run: ruff check src/ && ruff format --check src/
      - run: pytest tests/unit -q

  train-and-evaluate:
    needs: lint-and-unit
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12", cache: pip }
      - run: pip install -r requirements.txt
      # PR: small slice. main: full data.
      - name: Train
        run: |
          SLICE=$([ "${{ github.event_name }}" = "pull_request" ] && echo "--slice ci" || echo "")
          python src/train.py --config configs/train.yaml $SLICE --output ./output
      - name: Quality gates
        run: python ci/check_gates.py --config ci/gates.yaml --gates accuracy,latency
      - uses: actions/upload-artifact@v4
        with: { name: model-and-metrics, path: output/ }

  register-if-better:
    if: github.ref == 'refs/heads/main'
    needs: train-and-evaluate
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/download-artifact@v4
        with: { name: model-and-metrics, path: output/ }
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ vars.AWS_CICD_ROLE_ARN }}
          aws-region: ${{ env.AWS_REGION }}
      - name: Register candidate if it beats champion
        run: python ci/register_if_better.py --metrics output/metrics.json

  deploy-staging:
    needs: register-if-better
    runs-on: ubuntu-latest
    environment: staging          # env-scoped secrets + optional reviewers
    steps:
      - uses: actions/checkout@v4
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ vars.AWS_STAGING_ROLE_ARN }}
          aws-region: ${{ env.AWS_REGION }}
      - run: python ci/deploy.py --target sagemaker --env staging --alias challenger
      - name: Smoke test
        run: python ci/smoke_test.py --endpoint staging --requests 50 --max-p95-ms 200

  deploy-prod:
    needs: deploy-staging
    runs-on: ubuntu-latest
    environment: production       # REQUIRED REVIEWERS configured = manual gate
    steps:
      - uses: actions/checkout@v4
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ vars.AWS_PROD_ROLE_ARN }}
          aws-region: ${{ env.AWS_REGION }}
      - run: python ci/deploy.py --target sagemaker --env prod --alias champion
      - name: Post-deploy verification
        run: python ci/smoke_test.py --endpoint prod --requests 100 --max-p95-ms 150
```

Key mechanics:
- The **manual gate** is GitHub's `environment: production` with required reviewers —
  the job pauses until a human approves in the UI. No custom code needed.
- **OIDC** (`permissions: id-token: write` + `aws-actions/configure-aws-credentials@v4`
  with `role-to-assume`) replaces stored AWS keys entirely (details in references).
- Registration is conditional: `register_if_better.py` compares the candidate's metrics
  to the current champion (fetched from the registry) and only registers on improvement.

### 4. Promotion: dev → staging → prod

Promotion is a registry-state change, executed by `scripts/promote_model.py`:

**MLflow 3.x — aliases, not stages.** Model stages are removed; use aliases:

```python
from mlflow import MlflowClient
client = MlflowClient()

# Promote: point the environment alias at a version (atomic flip)
client.set_registered_model_alias("churn-model", alias="staging", version="12")
client.set_registered_model_alias("churn-model", alias="champion", version="12")

# Serving side loads by alias -- redeploy-free promotion if the server re-resolves:
model = mlflow.pyfunc.load_model("models:/churn-model@champion")
```

**SageMaker Model Registry — approval status:**

```python
import boto3
sm = boto3.client("sagemaker")
sm.update_model_package(
    ModelPackageArn=candidate_arn,
    ModelApprovalStatus="Approved",   # PendingManualApproval -> Approved
)
```

GitOps rule either way: the promotion command runs *only* from the pipeline (or a
reviewed script invocation), never from a laptop against prod. The audit trail is the
pipeline run + the registry event history. See the model-registry skill for
registry setup and metadata conventions.

### 5. Container Build for Model Images

Two patterns — pick one deliberately:

| Pattern | Image contains | Rebuild frequency | Best for |
|---------|---------------|-------------------|----------|
| Baked-in model | code + weights | Every model version | K8s/immutable deploys, small models, strict provenance |
| Model-at-startup | code only; weights pulled from registry/S3 by alias at boot | Only on code change | Large models, fast promotion (no rebuild to promote) |

```dockerfile
# Dockerfile.serve -- baked-in pattern
FROM python:3.12-slim
WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY src/serve.py src/
COPY output/model.joblib model/          # baked in at build time
EXPOSE 8080
CMD ["python", "src/serve.py", "--model-path", "model/model.joblib", "--port", "8080"]
```

Build step (add to the workflow after gates pass): tag with both the git SHA and the
registry model version (`myrepo/churn:git-abc1234-mv12`) so any running container is
traceable to exact code and model. Push to ECR/GHCR; deploy targets are covered in the
model-serving skill.

### 6. SageMaker Pipelines as Managed CD (SDK v3)

SageMaker Python SDK v3 removed `Estimator`/`Model`/`Predictor`; training steps are
built from `ModelTrainer` and wired with `sagemaker.workflow`:

```python
# pipeline.py -- sagemaker >= 3.0
from sagemaker.train import ModelTrainer
from sagemaker.train.configs import SourceCode, Compute
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterString

input_data = ParameterString(name="InputDataS3", default_value="s3://bucket/train/")

trainer = ModelTrainer(
    training_image="763104351884.dkr.ecr.us-east-1.amazonaws.com/sklearn-training:1.4-cpu-py311",
    source_code=SourceCode(source_dir="./src", entry_script="train.py"),
    compute=Compute(instance_type="ml.m5.xlarge", instance_count=1),
)
train_step = TrainingStep(name="TrainModel", step_args=trainer.train(wait=False))

# Register into the Model Registry pending manual approval
register_step = ModelStep(
    name="RegisterModel",
    step_args=...,  # model.register(model_package_group_name="churn",
                    #                approval_status="PendingManualApproval")
)

pipeline = Pipeline(name="churn-train-register",
                    parameters=[input_data],
                    steps=[train_step, register_step])
pipeline.upsert(role_arn=EXECUTION_ROLE)
pipeline.start()
```

Division of labor that works well: **GitHub Actions owns git events, lint/unit, and
gates; it triggers the SageMaker Pipeline for heavy training and registration** (via
`pipeline.start()` in a job step), then waits or reacts to the approval event below.

### 7. Deploy on Approval: EventBridge Trigger

Approving a model package emits an event — wire it to deployment so approval *is* the
release action:

```json
{
  "source": ["aws.sagemaker"],
  "detail-type": ["SageMaker Model Package State Change"],
  "detail": {
    "ModelPackageGroupName": ["churn"],
    "ModelApprovalStatus": ["Approved"]
  }
}
```

Target: a Lambda (or Step Functions state machine) that creates/updates the endpoint
from the newly approved package. This decouples "human approved in the registry UI or
pipeline" from "deployment executes" — the same path serves both GitHub-driven and
console-driven approvals.

### 8. Rollback Strategy

Model rollback must be faster than model deployment:

- **Registry-level**: re-point the alias / approval at the previous good version —
  MLflow: `set_registered_model_alias("churn-model", "champion", previous_version)`;
  SageMaker: previous package is still `Approved` — redeploy the endpoint from it
  (keep N-1 approved, only revoke on known-bad).
- **Serving-level**: keep the previous endpoint config/variant for instant traffic
  shift (blue/green, canary weights — see model-serving and the canary patterns in
  references/REFERENCE.md).
- **Trigger**: rollback decisions come from post-deploy monitoring (model-monitoring
  skill): error-rate or drift alarms should page a human by default; auto-rollback
  only on unambiguous signals (5xx spike, latency SLO breach), not on quality metrics
  that need investigation.
- **Drill it**: `promote_model.py --dry-run` against the previous version is the
  rollback rehearsal; run it in staging quarterly.

## Best Practices

1. **PR CI trains on a slice** — full training only on main/nightly
2. **Gates are versioned config** — thresholds in `ci/gates.yaml`, reviewed like code
3. **Registry state is the promotion source of truth** — deploys react to aliases/approval, not to git SHAs
4. **OIDC, never long-lived cloud keys** in workflows
5. **Manual gate before prod** — `environment` required reviewers; automate everything up to it
6. **Tag images with git SHA + model version** — every container traceable
7. **Register only if better** — compare to champion before writing to the registry
8. **Smoke test after every deploy** — a deploy without verification is a hope
9. **Keep N-1 deployable** — rollback is an alias flip, not a rebuild
10. **One pipeline file, generated variants** — avoid hand-maintained workflow drift across repos

## Scripts

- `scripts/generate_ml_pipeline.py` - Emit a complete GitHub Actions workflow from flags (`--registry mlflow|sagemaker`, `--deploy-target sagemaker|k8s|docker`, `--gates accuracy,latency[,size,regression]`, `--output`)
- `scripts/promote_model.py` - Registry-agnostic promotion/rollback: MLflow 3.x alias flip (`--model-name --version --alias`) or SageMaker approval update (`--model-package-arn --status`), with `--dry-run`

## References

See [references/REFERENCE.md](references/REFERENCE.md) for full pipeline anatomy,
environment strategy, OIDC secrets handling, the SageMaker Pipelines vs Step Functions
vs GitHub Actions comparison, and canary/shadow deployment patterns.

## Related skills

**Upstream:** `model-registry` (approved model version to promote) · **Downstream:** `model-serving` (deployed endpoint, staged rollout)
**See also:** `ml-testing` supplies the quality gates this pipeline enforces · `ml-pipeline-orchestration` for the training-side automation that produces candidates
