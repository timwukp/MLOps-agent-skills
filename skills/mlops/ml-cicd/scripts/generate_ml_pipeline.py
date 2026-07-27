#!/usr/bin/env python3
"""
ML CI/CD GitHub Actions Workflow Generator

Emits a complete, ready-to-commit GitHub Actions workflow for an ML repo:
lint + unit tests -> train + evaluate (slice on PR, full on main) -> quality
gates -> register-if-better -> deploy to staging -> smoke test -> manual
approval (environment gate) -> deploy to prod -> post-deploy verification.

The generated workflow uses OIDC (aws-actions/configure-aws-credentials@v4
with role-to-assume) -- never long-lived access keys. Variants are selected
by flags so every repo generates the same canonical shape instead of
hand-maintaining drifting copies.

Usage:
    # MLflow registry, deploy to SageMaker endpoints
    python generate_ml_pipeline.py --registry mlflow --deploy-target sagemaker \
        --gates accuracy,latency --output .github/workflows/ml-cicd.yml

    # SageMaker Model Registry, deploy to Kubernetes (EKS)
    python generate_ml_pipeline.py --registry sagemaker --deploy-target k8s \
        --gates accuracy,latency,size

    # Print to stdout for review
    python generate_ml_pipeline.py --registry sagemaker --deploy-target docker

Dependencies:
    - Python 3.8+ standard library only (PyYAML used for validation if present)
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

REGISTRIES = ("mlflow", "sagemaker")
DEPLOY_TARGETS = ("sagemaker", "k8s", "docker")
KNOWN_GATES = ("accuracy", "latency", "size", "regression")

PYTHON_VERSION = "3.12"


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def checkout_and_python():
    return [
        "      - uses: actions/checkout@v4",
        "      - uses: actions/setup-python@v5",
        "        with:",
        '          python-version: "%s"' % PYTHON_VERSION,
        "          cache: pip",
    ]


def aws_oidc_credentials(role_var):
    """OIDC role assumption -- requires `permissions: id-token: write` at top level."""
    return [
        "      - uses: aws-actions/configure-aws-credentials@v4",
        "        with:",
        "          role-to-assume: ${{ vars.%s }}" % role_var,
        "          aws-region: ${{ env.AWS_REGION }}",
    ]


def registry_env(registry):
    """Job-level env for talking to the model registry."""
    if registry == "mlflow":
        return [
            "    env:",
            "      MLFLOW_TRACKING_URI: ${{ vars.MLFLOW_TRACKING_URI }}",
        ]
    return []  # sagemaker registry needs only the assumed role


def deploy_steps(target, environment, alias):
    """Deploy command block per target. `alias` is the registry alias/tag deployed."""
    if target == "sagemaker":
        return [
            "      - name: Deploy to %s (SageMaker endpoint)" % environment,
            "        run: |",
            "          python ci/deploy.py --target sagemaker --env %s --alias %s" % (environment, alias),
        ]
    if target == "k8s":
        return [
            "      - name: Update kubeconfig (EKS)",
            "        run: aws eks update-kubeconfig --name ${{ vars.EKS_CLUSTER_NAME }} --region ${{ env.AWS_REGION }}",
            "      - name: Deploy to %s (Kubernetes)" % environment,
            "        run: |",
            "          kubectl -n ml-%s set image deployment/model-server \\" % environment,
            "            model-server=${{ vars.ECR_REPOSITORY }}:git-${{ github.sha }}",
            "          kubectl -n ml-%s rollout status deployment/model-server --timeout=300s" % environment,
        ]
    # docker: build once, push to ECR, deploy by tag
    return [
        "      - name: Login to Amazon ECR",
        "        run: |",
        "          aws ecr get-login-password --region ${{ env.AWS_REGION }} \\",
        "            | docker login --username AWS --password-stdin ${{ vars.ECR_REGISTRY }}",
        "      - name: Build and push image for %s" % environment,
        "        run: |",
        "          docker build -f Dockerfile.serve \\",
        "            -t ${{ vars.ECR_REGISTRY }}/${{ vars.ECR_REPOSITORY }}:git-${{ github.sha }} .",
        "          docker push ${{ vars.ECR_REGISTRY }}/${{ vars.ECR_REPOSITORY }}:git-${{ github.sha }}",
        "      - name: Deploy to %s (container service)" % environment,
        "        run: |",
        "          python ci/deploy.py --target docker --env %s \\" % environment,
        "            --image ${{ vars.ECR_REGISTRY }}/${{ vars.ECR_REPOSITORY }}:git-${{ github.sha }}",
    ]


def smoke_test_steps(environment, requests, max_p95_ms):
    name = "Smoke test" if environment == "staging" else "Post-deploy verification"
    return [
        "      - name: %s" % name,
        "        run: |",
        "          python ci/smoke_test.py --endpoint %s --requests %d --max-p95-ms %d" % (
            environment, requests, max_p95_ms),
    ]


# ---------------------------------------------------------------------------
# Workflow assembly
# ---------------------------------------------------------------------------

def generate_workflow(registry, deploy_target, gates):
    gate_list = ",".join(gates)
    lines = []
    add = lines.extend

    add([
        "# Generated by generate_ml_pipeline.py -- registry=%s deploy-target=%s gates=%s" % (
            registry, deploy_target, gate_list),
        "# Regenerate instead of hand-editing; see the ml-cicd skill.",
        "# Repo prerequisites: ci/check_gates.py, ci/register_if_better.py, ci/deploy.py,",
        "# ci/smoke_test.py, and ci/promote_model.py (copy from the skill's scripts/).",
        "name: ml-cicd",
        "",
        "on:",
        "  pull_request:",
        "    paths:",
        '      - "src/**"',
        '      - "ci/**"',
        '      - "configs/**"',
        "  push:",
        "    branches:",
        "      - main",
        "",
        "permissions:",
        "  id-token: write   # OIDC to AWS -- no long-lived access keys",
        "  contents: read",
        "",
        "env:",
        "  AWS_REGION: us-east-1",
        "",
        "jobs:",
    ])

    # --- Job 1: lint + unit -------------------------------------------------
    add([
        "  lint-and-unit:",
        "    runs-on: ubuntu-latest",
        "    steps:",
    ])
    add(checkout_and_python())
    add([
        "      - name: Install dependencies",
        "        run: pip install -r requirements.txt -r requirements-dev.txt",
        "      - name: Lint",
        "        run: |",
        "          ruff check src/",
        "          ruff format --check src/",
        "      - name: Unit tests",
        "        run: pytest tests/unit -q",
        "",
    ])

    # --- Job 2: train + evaluate + gates -------------------------------------
    add([
        "  train-and-evaluate:",
        "    needs: lint-and-unit",
        "    runs-on: ubuntu-latest",
        "    steps:",
    ])
    add(checkout_and_python())
    add([
        "      - name: Install dependencies",
        "        run: pip install -r requirements.txt",
        "      - name: Train (slice on PR, full on main)",
        "        run: |",
        '          SLICE=$([ "${{ github.event_name }}" = "pull_request" ] && echo "--slice ci" || echo "")',
        "          python src/train.py --config configs/train.yaml $SLICE --output ./output",
        "      - name: Quality gates (%s)" % gate_list,
        "        run: python ci/check_gates.py --config ci/gates.yaml --gates %s" % gate_list,
        "      - uses: actions/upload-artifact@v4",
        "        with:",
        "          name: model-and-metrics",
        "          path: output/",
        "",
    ])

    # --- Job 3: register-if-better (main only) --------------------------------
    add([
        "  register-if-better:",
        "    if: github.ref == 'refs/heads/main'",
        "    needs: train-and-evaluate",
        "    runs-on: ubuntu-latest",
    ])
    add(registry_env(registry))
    add(["    steps:"])
    add(checkout_and_python())
    add([
        "      - uses: actions/download-artifact@v4",
        "        with:",
        "          name: model-and-metrics",
        "          path: output/",
    ])
    add(aws_oidc_credentials("AWS_CICD_ROLE_ARN"))
    if registry == "mlflow":
        add([
            "      - name: Register candidate if it beats champion (MLflow 3.x aliases)",
            "        run: |",
            "          pip install 'mlflow>=3.0'",
            "          python ci/register_if_better.py --registry mlflow \\",
            "            --metrics output/metrics.json --model-name ${{ vars.MODEL_NAME }}",
        ])
    else:
        add([
            "      - name: Register candidate if it beats champion (SageMaker Model Registry)",
            "        run: |",
            "          pip install boto3 'sagemaker>=3.0'",
            "          python ci/register_if_better.py --registry sagemaker \\",
            "            --metrics output/metrics.json \\",
            "            --model-package-group ${{ vars.MODEL_PACKAGE_GROUP }}",
        ])
    add([""])

    # --- Job 4: deploy-staging + smoke test -----------------------------------
    add([
        "  deploy-staging:",
        "    needs: register-if-better",
        "    runs-on: ubuntu-latest",
        "    environment: staging   # env-scoped vars/secrets; optional reviewers",
        "    steps:",
    ])
    add(checkout_and_python())
    add(aws_oidc_credentials("AWS_STAGING_ROLE_ARN"))
    add(deploy_steps(deploy_target, "staging", "challenger"))
    add(smoke_test_steps("staging", requests=50, max_p95_ms=200))
    add([""])

    # --- Job 5: deploy-prod behind a manual approval gate ----------------------
    add([
        "  deploy-prod:",
        "    needs: deploy-staging",
        "    runs-on: ubuntu-latest",
        "    # Manual approval gate: configure REQUIRED REVIEWERS on the",
        "    # 'production' environment in repo settings; the job pauses here",
        "    # until a human approves in the GitHub UI.",
        "    environment: production",
        "    steps:",
    ])
    add(checkout_and_python())
    add(aws_oidc_credentials("AWS_PROD_ROLE_ARN"))
    if registry == "mlflow":
        add([
            "      - name: Promote in registry (alias flip to champion)",
            "        env:",
            "          MLFLOW_TRACKING_URI: ${{ vars.MLFLOW_TRACKING_URI }}",
            "        run: |",
            "          pip install 'mlflow>=3.0'",
            "          python ci/promote_model.py --registry mlflow \\",
            "            --model-name ${{ vars.MODEL_NAME }} --alias champion \\",
            "            --version $(cat output/registered_version.txt 2>/dev/null || echo latest)",
        ])
    else:
        add([
            "      - name: Promote in registry (approval status -> Approved)",
            "        run: |",
            "          pip install boto3",
            "          python ci/promote_model.py --registry sagemaker \\",
            "            --model-package-arn $(cat output/model_package_arn.txt) \\",
            "            --status Approved",
        ])
    add(deploy_steps(deploy_target, "prod", "champion"))
    add(smoke_test_steps("prod", requests=100, max_p95_ms=150))

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_gates(raw):
    gates = [g.strip() for g in raw.split(",") if g.strip()]
    unknown = [g for g in gates if g not in KNOWN_GATES]
    if unknown:
        raise argparse.ArgumentTypeError(
            "Unknown gate(s): %s (known: %s)" % (", ".join(unknown), ", ".join(KNOWN_GATES)))
    if not gates:
        raise argparse.ArgumentTypeError("At least one gate is required")
    return gates


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate a complete GitHub Actions ML CI/CD workflow (lint -> train -> "
                    "gates -> register -> staging -> smoke test -> manual approval -> prod).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--registry", required=True, choices=REGISTRIES,
                        help="Model registry: mlflow (3.x aliases) or sagemaker (approval status)")
    parser.add_argument("--deploy-target", required=True, choices=DEPLOY_TARGETS,
                        help="Where models are deployed: sagemaker endpoints, k8s (EKS), or docker (ECR image)")
    parser.add_argument("--gates", type=parse_gates, default=["accuracy", "latency"],
                        metavar="GATE[,GATE...]",
                        help="Comma-separated quality gates: %s (default: accuracy,latency)" % ",".join(KNOWN_GATES))
    parser.add_argument("--output", default=None,
                        help="Write the workflow YAML here (default: print to stdout)")
    return parser


def main():
    args = build_parser().parse_args()
    workflow = generate_workflow(args.registry, args.deploy_target, args.gates)

    try:  # validate if PyYAML is available; generation itself is stdlib-only
        import yaml
        yaml.safe_load(workflow)
        logger.info("Generated workflow parsed OK (yaml.safe_load)")
    except ImportError:
        logger.info("PyYAML not installed -- skipping parse validation")
    except Exception as exc:  # pragma: no cover -- template bug guard
        logger.error("Generated YAML failed to parse: %s", exc)
        sys.exit(1)

    if args.output:
        with open(args.output, "w") as f:
            f.write(workflow)
        logger.info("Workflow written to %s (registry=%s, deploy-target=%s, gates=%s)",
                    args.output, args.registry, args.deploy_target, ",".join(args.gates))
        logger.info("Remember: configure required reviewers on the 'production' environment, "
                    "and set the AWS_*_ROLE_ARN repository variables for OIDC.")
    else:
        print(workflow, end="")


if __name__ == "__main__":
    main()
