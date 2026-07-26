#!/usr/bin/env python3
"""Validate model-serving skill pattern on real SageMaker Serverless Inference.

Pattern under test (from skills/mlops/model-serving):
  package sklearn model -> SageMaker Model -> Serverless endpoint -> invoke -> cleanup

Serverless inference bills per-invocation only (no idle cost); this run
costs < $0.01. All resources are deleted at the end.
"""

import json
import tarfile
import tempfile
import time
import sys
from pathlib import Path

import boto3

REGION = "us-east-1"
ACCOUNT = boto3.client("sts").get_caller_identity()["Account"]
BUCKET = f"sagemaker-{REGION}-{ACCOUNT}"
NAME = "mlops-skills-serving-val"
RESULTS = []

sm = boto3.client("sagemaker", region_name=REGION)
smr = boto3.client("sagemaker-runtime", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)
iam = boto3.client("iam")


def check(name, fn):
    t0 = time.time()
    try:
        detail = fn()
        RESULTS.append({"check": name, "status": "PASS", "detail": str(detail)[:300],
                        "seconds": round(time.time() - t0, 1)})
        print(f"PASS  {name} ({time.time()-t0:.1f}s)", flush=True)
    except Exception as e:
        RESULTS.append({"check": name, "status": "FAIL", "detail": f"{type(e).__name__}: {e}",
                        "seconds": round(time.time() - t0, 1)})
        print(f"FAIL  {name}: {e}", flush=True)
        raise SystemExit(1)


def get_role():
    """Find an existing SageMaker execution role."""
    for r in iam.list_roles(PathPrefix="/service-role/")["Roles"]:
        if "AmazonSageMaker-ExecutionRole" in r["RoleName"]:
            return r["Arn"]
    for r in iam.list_roles()["Roles"]:
        if "SageMaker" in r["RoleName"] and "Execution" in r["RoleName"]:
            return r["Arn"]
    raise RuntimeError("no SageMaker execution role found")


ROLE = get_role()
print(f"Using role: {ROLE}", flush=True)

workdir = Path(tempfile.mkdtemp(prefix="serving-val-"))
model_uri = None


def train_package_upload():
    global model_uri
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    import joblib

    X, y = load_iris(return_X_y=True)
    clf = LogisticRegression(max_iter=500).fit(X, y)
    joblib.dump(clf, workdir / "model.joblib")

    # inference script per SageMaker sklearn container contract
    (workdir / "inference.py").write_text(
        "import os, joblib\n"
        "def model_fn(model_dir):\n"
        "    return joblib.load(os.path.join(model_dir, 'model.joblib'))\n"
    )
    code_dir = workdir / "code"
    code_dir.mkdir()
    (code_dir / "inference.py").write_text((workdir / "inference.py").read_text())

    with tarfile.open(workdir / "model.tar.gz", "w:gz") as tar:
        tar.add(workdir / "model.joblib", arcname="model.joblib")
        # SAGEMAKER_SUBMIT_DIRECTORY installs the tarball root as the module dir,
        # so inference.py must sit at the archive root (not under code/)
        tar.add(workdir / "inference.py", arcname="inference.py")
    key = f"{NAME}/model.tar.gz"
    s3.upload_file(str(workdir / "model.tar.gz"), BUCKET, key)
    model_uri = f"s3://{BUCKET}/{key}"
    return model_uri


def create_model():
    image = f"683313688378.dkr.ecr.{REGION}.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"
    sm.create_model(
        ModelName=NAME,
        PrimaryContainer={
            "Image": image,
            "ModelDataUrl": model_uri,
            "Environment": {
                "SAGEMAKER_PROGRAM": "inference.py",
                "SAGEMAKER_SUBMIT_DIRECTORY": model_uri,
            },
        },
        ExecutionRoleArn=ROLE,
    )
    return NAME


def create_endpoint():
    sm.create_endpoint_config(
        EndpointConfigName=NAME,
        ProductionVariants=[{
            "VariantName": "AllTraffic",
            "ModelName": NAME,
            "ServerlessConfig": {"MemorySizeInMB": 1024, "MaxConcurrency": 1},
        }],
    )
    sm.create_endpoint(EndpointName=NAME, EndpointConfigName=NAME)
    waiter = sm.get_waiter("endpoint_in_service")
    waiter.wait(EndpointName=NAME, WaiterConfig={"Delay": 15, "MaxAttempts": 40})
    st = sm.describe_endpoint(EndpointName=NAME)["EndpointStatus"]
    assert st == "InService", st
    return f"endpoint InService"


def invoke():
    payload = "5.1,3.5,1.4,0.2\n6.7,3.0,5.2,2.3"
    resp = smr.invoke_endpoint(
        EndpointName=NAME, ContentType="text/csv", Body=payload,
    )
    body = resp["Body"].read().decode()
    preds = [x for x in body.replace("[", "").replace("]", "").replace("\n", ",").split(",") if x.strip()]
    assert len(preds) == 2, body
    return f"predictions={body.strip()[:80]}"


def cleanup():
    sm.delete_endpoint(EndpointName=NAME)
    sm.delete_endpoint_config(EndpointConfigName=NAME)
    sm.delete_model(ModelName=NAME)
    s3.delete_object(Bucket=BUCKET, Key=f"{NAME}/model.tar.gz")
    return "endpoint, config, model, s3 artifact deleted"


check("train-package-upload", train_package_upload)
check("create-sagemaker-model", create_model)
check("create-serverless-endpoint (model-serving pattern)", create_endpoint)
check("invoke-endpoint (real inference)", invoke)
check("cleanup", cleanup)

Path("/tmp/sagemaker-validation/serving_results.json").write_text(json.dumps(RESULTS, indent=2))
print("Serving validation complete")
