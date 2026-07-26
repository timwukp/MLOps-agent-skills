#!/usr/bin/env python3
"""Validate MLOps-agent-skills patterns against real AWS SageMaker.

Exercises the model-training packaging + model-registry lifecycle patterns
from the skills repo on a live AWS account:
  1. Train a small sklearn model (model-training skill pattern)
  2. Package as model.tar.gz and upload to S3 (model-registry packaging)
  3. Create a SageMaker Model Package Group + register a model version
  4. Promote (update approval status Pending -> Approved) — registry promotion pattern
  5. Verify lineage/metadata retrieval
  6. Clean up all created resources

Cost: S3 storage pennies; SageMaker Model Registry API calls are free.
"""

import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import boto3

REGION = "us-east-1"
ACCOUNT = boto3.client("sts").get_caller_identity()["Account"]
BUCKET = f"sagemaker-{REGION}-{ACCOUNT}"
GROUP_NAME = "mlops-skills-validation"
RESULTS = []


def check(name, fn):
    t0 = time.time()
    try:
        detail = fn()
        RESULTS.append({"check": name, "status": "PASS", "detail": str(detail)[:200],
                        "seconds": round(time.time() - t0, 1)})
        print(f"PASS  {name} ({time.time()-t0:.1f}s)")
    except Exception as e:
        RESULTS.append({"check": name, "status": "FAIL", "detail": f"{type(e).__name__}: {e}",
                        "seconds": round(time.time() - t0, 1)})
        print(f"FAIL  {name}: {e}")


sm = boto3.client("sagemaker", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)

workdir = Path(tempfile.mkdtemp(prefix="skills-val-"))
model_s3_uri = None


def train_and_package():
    """model-training skill pattern: sklearn train + joblib persist."""
    global model_s3_uri
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score
    import joblib

    X, y = make_classification(n_samples=2000, n_features=20, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    metrics = {"accuracy": accuracy_score(y_te, preds), "f1": f1_score(y_te, preds)}
    assert metrics["accuracy"] > 0.85, f"model quality gate failed: {metrics}"

    joblib.dump(model, workdir / "model.joblib")
    (workdir / "metrics.json").write_text(json.dumps(metrics))
    with tarfile.open(workdir / "model.tar.gz", "w:gz") as tar:
        tar.add(workdir / "model.joblib", arcname="model.joblib")

    key = "mlops-skills-validation/model.tar.gz"
    s3.upload_file(str(workdir / "model.tar.gz"), BUCKET, key)
    model_s3_uri = f"s3://{BUCKET}/{key}"
    return f"trained acc={metrics['accuracy']:.3f}, uploaded {model_s3_uri}"


def ensure_bucket():
    try:
        s3.head_bucket(Bucket=BUCKET)
        return f"bucket {BUCKET} exists"
    except Exception:
        s3.create_bucket(Bucket=BUCKET)
        return f"created bucket {BUCKET}"


def create_group():
    try:
        sm.create_model_package_group(
            ModelPackageGroupName=GROUP_NAME,
            ModelPackageGroupDescription="Validation of MLOps-agent-skills model-registry patterns",
        )
        return "created group"
    except sm.exceptions.ClientError as e:
        if "already exists" in str(e) or "ValidationException" in str(e):
            return "group already exists"
        raise


pkg_arn = None


def register_version():
    """model-registry skill pattern: register version with metrics metadata."""
    global pkg_arn
    # sklearn inference image for us-east-1 (SageMaker managed)
    image = f"683313688378.dkr.ecr.{REGION}.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"
    resp = sm.create_model_package(
        ModelPackageGroupName=GROUP_NAME,
        ModelPackageDescription="rf-classifier acc>0.85 quality gate passed",
        ModelApprovalStatus="PendingManualApproval",
        InferenceSpecification={
            "Containers": [{"Image": image, "ModelDataUrl": model_s3_uri}],
            "SupportedContentTypes": ["text/csv"],
            "SupportedResponseMIMETypes": ["text/csv"],
        },
        CustomerMetadataProperties={
            "accuracy": json.loads((workdir / "metrics.json").read_text())["accuracy"].__str__()[:10],
            "source": "mlops-agent-skills-validation",
        },
    )
    pkg_arn = resp["ModelPackageArn"]
    return pkg_arn


def promote():
    """model-registry promotion pattern: Pending -> Approved."""
    sm.update_model_package(ModelPackageArn=pkg_arn, ModelApprovalStatus="Approved")
    desc = sm.describe_model_package(ModelPackageName=pkg_arn)
    assert desc["ModelApprovalStatus"] == "Approved"
    return f"approved, status={desc['ModelApprovalStatus']}"


def verify_lineage():
    """registry lineage/listing pattern."""
    versions = sm.list_model_packages(ModelPackageGroupName=GROUP_NAME)["ModelPackageSummaryList"]
    assert len(versions) >= 1
    latest = sm.describe_model_package(ModelPackageName=versions[0]["ModelPackageArn"])
    meta = latest.get("CustomerMetadataProperties", {})
    assert meta.get("source") == "mlops-agent-skills-validation"
    return f"{len(versions)} version(s), metadata intact"


def cleanup():
    for v in sm.list_model_packages(ModelPackageGroupName=GROUP_NAME)["ModelPackageSummaryList"]:
        sm.delete_model_package(ModelPackageName=v["ModelPackageArn"])
    sm.delete_model_package_group(ModelPackageGroupName=GROUP_NAME)
    s3.delete_object(Bucket=BUCKET, Key="mlops-skills-validation/model.tar.gz")
    return "deleted model packages, group, and S3 artifact"


check("s3-bucket", ensure_bucket)
check("train-and-package (model-training pattern)", train_and_package)
check("create-model-package-group", create_group)
check("register-model-version (model-registry pattern)", register_version)
check("promote-pending-to-approved (promotion pattern)", promote)
check("verify-lineage-and-metadata", verify_lineage)
check("cleanup", cleanup)

out = Path("/tmp/sagemaker-validation/results.json")
out.write_text(json.dumps(RESULTS, indent=2))
print(f"\nResults -> {out}")
failed = [r for r in RESULTS if r["status"] == "FAIL"]
sys.exit(1 if failed else 0)
