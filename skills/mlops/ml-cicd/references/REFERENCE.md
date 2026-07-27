# ML CI/CD Reference Guide

## Pipeline Anatomy Deep-Dive

The canonical pipeline (as emitted by `scripts/generate_ml_pipeline.py`) has five
jobs; each exists for a specific failure class:

| Job | Catches | Runs on | Typical duration |
|-----|---------|---------|------------------|
| lint-and-unit | Broken code, style drift, contract violations | Every PR + main | 2-5 min |
| train-and-evaluate | Training crashes, metric regressions, gate breaches | Every PR (slice) + main (full) | 5-15 min PR / longer on main |
| register-if-better | Registering a model worse than the champion | main only | 1-2 min |
| deploy-staging + smoke test | Serving-time failures (deps, serialization, latency) | main, after registration | 3-10 min |
| deploy-prod (env gate) | Everything a human should eyeball before release | After manual approval | 3-10 min |

Design decisions worth defending in review:

- **Why the PR job trains at all**: unit tests cannot catch "the model trains but is
  garbage". A fixed, versioned 5-10% stratified slice gives a reproducible signal
  in minutes. Keep the slice frozen (commit its manifest/hash); a moving slice makes
  gate history meaningless.
- **Why register-if-better is its own job**: separating "produce a candidate" from
  "admit it to the registry" means a gate change or champion comparison bug never
  silently pollutes the registry. The job fetches the champion's metrics from the
  registry (not from git) and registers only on improvement.
- **Why the smoke test is not optional**: it is the only stage that exercises the
  *serving* path (deserialization, feature schema at inference time, endpoint auth).
  Training metrics say nothing about any of this.
- **Why the manual gate lives at prod, not staging**: everything up to staging must
  be fully automated or nobody trusts the pipeline; the single human decision point
  is placed where blast radius justifies it.
- **Artifact flow**: `train-and-evaluate` uploads `output/` (model + metrics.json)
  via `actions/upload-artifact@v4`; downstream jobs download it rather than
  retraining. The artifact, not the runner filesystem, is the hand-off contract.

### Failure attribution

One assertion per step. A job that runs lint, tests, and gates in a single `run:`
block produces a red X that requires log spelunking. The generated workflow keeps
lint, format, unit tests, training, and gates as separate named steps so the
failing stage is visible from the run summary.

## Environment Strategy: dev / staging / prod Accounts

Use separate AWS *accounts* per environment, not prefixes inside one account:

```
 dev account          staging account          prod account
 (experiments,        (production-shaped,      (real traffic,
  scratch endpoints)   synthetic traffic)       real data)
      |                      |                        |
  AWS_DEV_ROLE_ARN     AWS_STAGING_ROLE_ARN     AWS_PROD_ROLE_ARN
      |                      |                        |
      +---------- GitHub environments map 1:1 --------+
```

- **Blast radius**: an experiment that deletes an endpoint in dev cannot touch prod.
  IAM inside one account is too easy to get wrong; account boundaries are hard walls.
- **GitHub `environment:` blocks map 1:1 to accounts**: each environment holds its own
  role ARN variable and (for prod) required reviewers. The workflow job declares
  `environment: production` and automatically gets prod-scoped variables plus the
  approval gate — no custom code.
- **The registry spans environments**: a central registry (MLflow server or a
  SageMaker Model Registry in a shared services account with cross-account resource
  policies) is the single source of truth; environments differ only in *which alias
  or approval status they deploy*. Staging serves `@challenger`, prod serves
  `@champion`.
- **Same artifact, different config**: the model artifact and container image that
  passed staging are byte-identical to what deploys to prod. Only endpoint config
  (instance count, autoscaling, env vars) differs. Rebuilding "for prod" invalidates
  everything staging verified.

## Secrets: OIDC, Not Long-Lived Keys

Long-lived `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` in repo secrets are the
number-one CI credential leak vector: they never expire, they work from anywhere,
and they leak via forked-PR misconfiguration and log echoes. GitHub's OIDC provider
eliminates them:

1. Workflow declares `permissions: id-token: write`.
2. `aws-actions/configure-aws-credentials@v4` requests a short-lived OIDC token
   from GitHub and exchanges it via `sts:AssumeRoleWithWebIdentity` for temporary
   credentials (minutes-to-hours lifetime).
3. The IAM role's trust policy pins exactly who may assume it:

```json
{
  "Effect": "Allow",
  "Principal": { "Federated": "arn:aws:iam::123456789012:oidc-provider/token.actions.githubusercontent.com" },
  "Action": "sts:AssumeRoleWithWebIdentity",
  "Condition": {
    "StringEquals": { "token.actions.githubusercontent.com:aud": "sts.amazonaws.com" },
    "StringLike":   { "token.actions.githubusercontent.com:sub": "repo:my-org/my-repo:environment:production" }
  }
}
```

Rules:

- **One role per environment**, each pinned to its GitHub environment in the `sub`
  condition (`repo:org/repo:environment:production`). The prod role is unusable from
  a PR branch or from staging jobs by construction.
- **Least privilege per role**: the CI role can write to the registry and read data;
  the staging role can update staging endpoints; the prod role can update prod
  endpoints. None of them can do the others' jobs.
- Store role ARNs as environment-scoped *variables* (`vars.`), not secrets — ARNs
  are not sensitive; keeping them visible aids debugging.
- Whatever must remain a secret (e.g. `MLFLOW_TRACKING_TOKEN` for a private MLflow
  server) goes in environment-scoped secrets so staging credentials never appear in
  prod jobs and vice versa.

## Orchestrator Comparison: SageMaker Pipelines vs Step Functions vs GitHub Actions

| Criterion | SageMaker Pipelines | AWS Step Functions | GitHub Actions |
|-----------|--------------------|--------------------|----------------|
| Primary role | Managed ML DAG (train, evaluate, register) | General AWS service orchestration | Git-event CI/CD |
| Trigger model | SDK/API call, EventBridge schedule | Any AWS event, API | push/PR/schedule/manual dispatch |
| ML-native steps | Yes: TrainingStep, ModelStep, quality/clarify checks, registry integration | No — call SageMaker APIs yourself | No — run SDK code in job steps |
| Compute | SageMaker jobs (GPU, spot, distributed) | Delegates to other services | GitHub runners (small CPU; self-host for more) |
| Max duration | Long-running training fine | 1 year (standard workflows) | 6 h per job |
| Human approval | Registry approval status (out-of-band) | `.waitForTaskToken` callback pattern | `environment` required reviewers (built-in UI) |
| Retries/caching | Step caching, retry policies built in | Rich retry/catch per state | `retry`-less; re-run whole job |
| Lineage/tracking | Automatic (experiments, model registry links) | None ML-specific | None ML-specific |
| Cost model | Pay for underlying jobs; pipeline free | Per state transition | Runner minutes |
| Best at | Heavy training + registration DAGs on AWS | Event-driven glue (approval -> deploy), multi-service sagas | Everything triggered by git; lint/tests/gates; calling the other two |

Working combination (what the skill's workflow assumes):

- **GitHub Actions** owns git events, lint/unit, slice training, gates, and the
  manual prod gate. For heavy training it *starts* a SageMaker Pipeline
  (`pipeline.start()`) instead of training on a runner.
- **SageMaker Pipelines** (SDK v3: `ModelTrainer` + `sagemaker.workflow`; the legacy
  `Estimator` classes are removed in v3) owns full training, evaluation, and
  registration with `PendingManualApproval`.
- **Step Functions / Lambda** reacts to the approval EventBridge event and performs
  the endpoint update — the deployment path that works for both pipeline-driven and
  console-driven approvals.

Anti-pattern: re-implementing training DAGs in raw Step Functions when SageMaker
Pipelines exists, or running multi-hour GPU training on GitHub runners.

## Canary and Shadow Deployment Patterns

### Canary (traffic-split) release

Route a small, growing fraction of live traffic to the new model; promote on healthy
metrics, roll back on breach:

```
 100% -> [champion variant]                 95% -> [champion]    0% -> [champion]
                              == step ==>    5% -> [challenger]  == ... ==> 100% -> [challenger]
```

- **SageMaker**: either two production variants with `DesiredWeight` on one endpoint,
  or (fully managed) endpoint *deployment guardrails* with a canary/linear traffic
  shifting policy plus CloudWatch alarms that auto-rollback the endpoint update.
- **K8s**: Istio/ALB weighted routing or Argo Rollouts canary steps
  (`setWeight: 5` -> `pause` -> `setWeight: 25` ...) with automated analysis.
- Watch *both* ops metrics (5xx, p99 latency) and model metrics (prediction
  distribution vs champion). Auto-rollback on ops breaches; page a human for model
  metric anomalies — quality shifts need investigation, not reflexes.
- Canary duration must cover a representative traffic cycle (at least one daily
  peak); a 10-minute canary on overnight traffic proves nothing.

### Shadow (mirror) deployment

The challenger receives a *copy* of live traffic; its responses are logged, never
returned to users:

- Zero user-facing risk — the only pattern that lets a model see true production
  traffic before any exposure.
- **SageMaker**: shadow variants are first-class (`ShadowProductionVariants` in the
  endpoint config); the shadow's outputs land in data capture for offline comparison.
- **K8s**: Istio traffic mirroring, or an app-level tee in the inference service.
- Compare champion vs shadow on: agreement rate, per-segment metric deltas (where
  delayed labels exist), latency distribution, error rate under real payloads.
- Cost: shadow doubles inference compute for the mirrored fraction. Mirror a sample
  (10-25%) when the model is expensive.
- Sequence for high-stakes models: shadow (days) -> canary (hours-days) -> full.

## EventBridge: Deploy on Model-Package Approval

Approving a SageMaker model package emits an event; wiring a rule to it makes
"approve in the registry" the *release action*, independent of which UI or pipeline
performed the approval.

Rule pattern:

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

Target options:

- **Lambda** (simple): reads the event's `ModelPackageArn`, builds a model from the
  package, and calls `create_endpoint_config` + `update_endpoint`. Fine when deploy
  is a single API call.
- **Step Functions** (robust): update endpoint -> wait -> describe endpoint status ->
  run smoke invocation -> on failure, redeploy previous approved package and notify.
  Prefer this once rollback and verification are part of the deploy.

Operational notes:

- Idempotence: approvals can be re-emitted or replayed; the target must no-op when
  the endpoint already serves the approved package version.
- Add a rule for `ModelApprovalStatus: ["Rejected"]` too — revoking a bad package
  should alert (and optionally trigger rollback) rather than silently do nothing.
- Put a DLQ on the target; a lost approval event is a release that never happened
  and nobody was told.
- The same decoupling exists for MLflow via registry webhooks (or polling the
  alias from the deploy job) — the principle is identical: deployment reacts to
  registry state, not to git.

## Further Reading

- [GitHub Actions: OIDC with AWS](https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/configuring-openid-connect-in-amazon-web-services)
- [GitHub Actions: environments and required reviewers](https://docs.github.com/en/actions/deployment/targeting-different-environments/using-environments-for-deployment)
- [SageMaker Pipelines](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines.html)
- [SageMaker deployment guardrails (canary/linear traffic shifting)](https://docs.aws.amazon.com/sagemaker/latest/dg/deployment-guardrails.html)
- [SageMaker shadow tests](https://docs.aws.amazon.com/sagemaker/latest/dg/shadow-tests.html)
- [Automate deployment on model registry approval (EventBridge)](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry-eventbridge.html)
- [MLflow model registry aliases](https://mlflow.org/docs/latest/model-registry.html)
- [Argo Rollouts canary strategy](https://argoproj.github.io/argo-rollouts/features/canary/)
