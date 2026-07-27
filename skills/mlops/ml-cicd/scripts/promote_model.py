#!/usr/bin/env python3
"""
Registry-Agnostic Model Promotion (and Rollback)

Executes the promotion step of an ML CI/CD pipeline as a registry-state change:

  - MLflow 3.x:   atomically point an alias (e.g. `champion`, `staging`) at a
                  model version via `set_registered_model_alias`. MLflow 3.x
                  uses aliases -- model *stages* are removed.
  - SageMaker:    flip a model package's ModelApprovalStatus (e.g.
                  PendingManualApproval -> Approved) via `update_model_package`;
                  EventBridge rules on the state change can then drive deployment.

Rollback is the same operation pointed at the previous good version -- run it
with --dry-run first as the rollback rehearsal.

Usage:
    # MLflow: promote version 12 to champion
    python promote_model.py --registry mlflow --model-name churn-model \
        --version 12 --alias champion

    # MLflow: rollback = flip the alias back (dry-run rehearsal)
    python promote_model.py --registry mlflow --model-name churn-model \
        --version 11 --alias champion --dry-run

    # SageMaker: approve a candidate package
    python promote_model.py --registry sagemaker \
        --model-package-arn arn:aws:sagemaker:us-east-1:123456789012:model-package/churn/12 \
        --status Approved

    # SageMaker: revoke a known-bad package
    python promote_model.py --registry sagemaker \
        --model-package-arn arn:...:model-package/churn/12 --status Rejected

Dependencies:
    - mlflow>=3.0 for --registry mlflow
    - boto3 for --registry sagemaker
    (imported lazily -- only the SDK for the chosen registry is required)
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

SAGEMAKER_STATUSES = ("Approved", "Rejected", "PendingManualApproval")


# ---------------------------------------------------------------------------
# MLflow (3.x aliases)
# ---------------------------------------------------------------------------

def promote_mlflow(model_name, version, alias, dry_run):
    try:
        from mlflow import MlflowClient
    except ImportError:
        logger.error("mlflow is not installed. Install it with: pip install 'mlflow>=3.0' "
                     "(MLflow 3.x required -- this script uses aliases, not the removed stages API)")
        sys.exit(2)

    client = MlflowClient()

    # Resolve and show what the alias currently points at (audit trail / rollback info).
    # A missing alias is normal (first promotion); any other registry error is fatal
    # unless --dry-run, which must be runnable as a rehearsal without registry access.
    try:
        current = client.get_model_version_by_alias(model_name, alias)
        logger.info("Alias '%s' on '%s' currently points at version %s",
                    alias, model_name, current.version)
        if str(current.version) == str(version):
            logger.info("Alias already points at version %s -- nothing to do.", version)
            return
    except Exception as exc:
        if dry_run:
            logger.warning("Cannot resolve alias '%s' on '%s' (%s) -- continuing "
                           "because --dry-run makes no changes", alias, model_name, exc)
        else:
            logger.info("Alias '%s' on '%s' is not currently set (or not resolvable: %s)",
                        alias, model_name, exc)

    # Verify the target version exists before flipping
    try:
        client.get_model_version(model_name, str(version))
    except Exception as exc:
        if dry_run:
            logger.warning("Model version %s of '%s' not found (%s) -- continuing "
                           "because --dry-run makes no changes", version, model_name, exc)
        else:
            logger.error("Model version %s of '%s' not found in the registry: %s",
                         version, model_name, exc)
            sys.exit(1)

    if dry_run:
        logger.info("[DRY-RUN] Would run: set_registered_model_alias(%r, alias=%r, version=%r)",
                    model_name, alias, str(version))
        return

    client.set_registered_model_alias(model_name, alias=alias, version=str(version))
    logger.info("Promoted: models:/%s@%s -> version %s", model_name, alias, version)
    logger.info("Serving that loads by alias (models:/%s@%s) picks this up on next resolve.",
                model_name, alias)


# ---------------------------------------------------------------------------
# SageMaker Model Registry (approval status)
# ---------------------------------------------------------------------------

def promote_sagemaker(model_package_arn, status, dry_run, region):
    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError:
        logger.error("boto3 is not installed. Install it with: pip install boto3")
        sys.exit(2)

    sm = boto3.client("sagemaker", **({"region_name": region} if region else {}))

    # Show current state first (audit trail / idempotence)
    current = None
    try:
        pkg = sm.describe_model_package(ModelPackageName=model_package_arn)
        current = pkg.get("ModelApprovalStatus")
        logger.info("Model package %s current ModelApprovalStatus: %s",
                    model_package_arn, current)
    except (BotoCoreError, ClientError) as exc:
        if dry_run:
            logger.warning("Cannot describe model package (%s) -- continuing because "
                           "--dry-run makes no changes", exc)
        else:
            logger.error("Cannot describe model package %s: %s", model_package_arn, exc)
            sys.exit(1)

    if current == status:
        logger.info("Already '%s' -- nothing to do.", status)
        return

    if dry_run:
        logger.info("[DRY-RUN] Would run: update_model_package(ModelPackageArn=%r, "
                    "ModelApprovalStatus=%r)", model_package_arn, status)
        return

    try:
        sm.update_model_package(
            ModelPackageArn=model_package_arn,
            ModelApprovalStatus=status,
        )
    except (BotoCoreError, ClientError) as exc:
        logger.error("update_model_package failed: %s", exc)
        sys.exit(1)

    logger.info("Updated ModelApprovalStatus: %s -> %s", current, status)
    if status == "Approved":
        logger.info("An EventBridge rule on 'SageMaker Model Package State Change' with "
                    "ModelApprovalStatus=Approved can now trigger deployment (see references).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="Promote (or roll back) a model by changing registry state: "
                    "MLflow 3.x alias flip, or SageMaker model-package approval status.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--registry", required=True, choices=("mlflow", "sagemaker"),
                        help="Which registry to operate on")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show current state and the exact change, without applying it")

    mlflow_group = parser.add_argument_group("mlflow mode")
    mlflow_group.add_argument("--model-name", help="Registered model name (mlflow)")
    mlflow_group.add_argument("--version", help="Model version number to promote (mlflow)")
    mlflow_group.add_argument("--alias", default="champion",
                              help="Alias to point at the version, e.g. champion, staging "
                                   "(default: champion)")

    sm_group = parser.add_argument_group("sagemaker mode")
    sm_group.add_argument("--model-package-arn", help="Model package ARN (sagemaker)")
    sm_group.add_argument("--status", choices=SAGEMAKER_STATUSES, default="Approved",
                          help="Target ModelApprovalStatus (default: Approved)")
    sm_group.add_argument("--region", default=None,
                          help="AWS region (default: environment/profile default)")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.registry == "mlflow":
        missing = [f for f, v in (("--model-name", args.model_name),
                                  ("--version", args.version)) if not v]
        if missing:
            parser.error("mlflow mode requires: " + ", ".join(missing))
        promote_mlflow(args.model_name, args.version, args.alias, args.dry_run)
    else:
        if not args.model_package_arn:
            parser.error("sagemaker mode requires: --model-package-arn")
        promote_sagemaker(args.model_package_arn, args.status, args.dry_run, args.region)


if __name__ == "__main__":
    main()
