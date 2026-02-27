#!/usr/bin/env python3
"""Model packaging script for deployment.

Generates deployment-ready artifacts from trained models: Docker images,
ONNX exports, MLflow serving bundles, model cards, and validation checks.

Usage examples:
    # Generate a Dockerfile for serving a model
    python model_packager.py --action docker --model-path ./model \
        --output ./deploy --framework sklearn

    # Convert a scikit-learn model to ONNX
    python model_packager.py --action onnx --model-path ./model.pkl \
        --framework sklearn --output ./model.onnx

    # Package in MLflow model serving format
    python model_packager.py --action mlflow --model-path ./model \
        --output ./serve_bundle

    # Generate a model card (Markdown)
    python model_packager.py --action model-card --model-path ./model \
        --output ./MODEL_CARD.md --framework sklearn

    # Validate a packaged model by running sample inference
    python model_packager.py --action validate --model-path ./model \
        --framework sklearn
"""

import argparse
import json
import logging
import os
import sys
import shutil
import textwrap
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SUPPORTED_FRAMEWORKS = ("sklearn", "pytorch", "xgboost")

# ---------------------------------------------------------------------------
# Docker packaging
# ---------------------------------------------------------------------------

_DOCKERFILE_TEMPLATE = textwrap.dedent("""\
    FROM python:3.11-slim

    LABEL maintainer="mlops-team"
    LABEL description="Model serving container for {framework} model"

    WORKDIR /opt/model

    RUN pip install --no-cache-dir \\
        mlflow=={mlflow_version} \\
        {framework_pip} \\
        gunicorn

    COPY {model_dir} /opt/model/model

    ENV MLFLOW_MODEL_URI=/opt/model/model
    EXPOSE 8080

    CMD ["mlflow", "models", "serve", \\
         "-m", "/opt/model/model", \\
         "--host", "0.0.0.0", \\
         "--port", "8080", \\
         "--no-conda"]
""")

_FRAMEWORK_PIP = {
    "sklearn": "scikit-learn",
    "pytorch": "torch torchvision",
    "xgboost": "xgboost",
}


def package_docker(model_path: str, output_dir: str, framework: str):
    """Generate a Dockerfile and supporting files for model serving."""
    try:
        import mlflow
        mlflow_version = mlflow.__version__
    except ImportError:
        mlflow_version = "2.12.1"
        logger.warning("mlflow not installed; defaulting to version %s in Dockerfile", mlflow_version)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_dir = Path(model_path).name
    dest_model = out / model_dir
    if not dest_model.exists():
        if Path(model_path).is_dir():
            shutil.copytree(model_path, dest_model)
        else:
            shutil.copy2(model_path, dest_model)

    dockerfile_content = _DOCKERFILE_TEMPLATE.format(
        framework=framework,
        mlflow_version=mlflow_version,
        framework_pip=_FRAMEWORK_PIP.get(framework, framework),
        model_dir=model_dir,
    )
    dockerfile_path = out / "Dockerfile"
    dockerfile_path.write_text(dockerfile_content)

    dockerignore_path = out / ".dockerignore"
    dockerignore_path.write_text("__pycache__\n*.pyc\n.git\n")

    logger.info("Dockerfile written to %s", dockerfile_path)
    logger.info("Build with: docker build -t model-serve %s", out)
    return str(dockerfile_path)


# ---------------------------------------------------------------------------
# ONNX conversion
# ---------------------------------------------------------------------------

def _convert_sklearn_to_onnx(model_path: str, output_path: str):
    """Convert a pickled sklearn model to ONNX."""
    import pickle
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        logger.error("skl2onnx is not installed. Run: pip install skl2onnx")
        sys.exit(1)

    with open(model_path, "rb") as fh:
        model = pickle.load(fh)

    n_features = getattr(model, "n_features_in_", 10)
    initial_type = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    with open(output_path, "wb") as fh:
        fh.write(onnx_model.SerializeToString())
    logger.info("ONNX model saved to %s", output_path)


def _convert_pytorch_to_onnx(model_path: str, output_path: str):
    """Export a saved PyTorch model to ONNX."""
    try:
        import torch
    except ImportError:
        logger.error("torch is not installed. Run: pip install torch")
        sys.exit(1)

    model = torch.load(model_path, map_location="cpu", weights_only=False)
    model.eval()

    # Assume a simple 1-D float input; users may need to adjust shape
    dummy = torch.randn(1, 10)
    torch.onnx.export(
        model, dummy, output_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    logger.info("ONNX model saved to %s", output_path)


def _convert_xgboost_to_onnx(model_path: str, output_path: str):
    """Convert an XGBoost model to ONNX via onnxmltools."""
    try:
        import xgboost as xgb
        from onnxmltools import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType
    except ImportError:
        logger.error("onnxmltools or xgboost missing. Run: pip install onnxmltools xgboost")
        sys.exit(1)

    booster = xgb.Booster()
    booster.load_model(model_path)

    n_features = int(booster.num_features())
    initial_type = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_xgboost(booster, initial_types=initial_type)

    with open(output_path, "wb") as fh:
        fh.write(onnx_model.SerializeToString())
    logger.info("ONNX model saved to %s", output_path)


_ONNX_CONVERTERS = {
    "sklearn": _convert_sklearn_to_onnx,
    "pytorch": _convert_pytorch_to_onnx,
    "xgboost": _convert_xgboost_to_onnx,
}


def package_onnx(model_path: str, output_path: str, framework: str):
    """Convert a model to ONNX format."""
    converter = _ONNX_CONVERTERS.get(framework)
    if converter is None:
        logger.error("ONNX conversion not supported for '%s'", framework)
        sys.exit(1)
    converter(model_path, output_path)
    return output_path


# ---------------------------------------------------------------------------
# MLflow serving bundle
# ---------------------------------------------------------------------------

def package_mlflow(model_path: str, output_dir: str):
    """Re-package a model into the MLflow model serving directory layout."""
    try:
        import mlflow.pyfunc
    except ImportError:
        logger.error("mlflow is not installed. Run: pip install mlflow")
        sys.exit(1)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = mlflow.pyfunc.load_model(model_path)
    logger.info("Loaded pyfunc model (flavour: %s)", model.metadata.flavors.keys())

    dest = out / "model"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(model_path, dest)
    logger.info("MLflow serving bundle ready at %s", dest)
    return str(dest)


# ---------------------------------------------------------------------------
# Model card generation
# ---------------------------------------------------------------------------

def generate_model_card(model_path: str, output_path: str, framework: str):
    """Create a Markdown model card with metadata, metrics, and caveats."""
    meta = {
        "model_path": str(Path(model_path).resolve()),
        "framework": framework,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }

    # Try to pull metrics from MLmodel file
    mlmodel_file = Path(model_path) / "MLmodel"
    metrics_section = "_No metrics available._"
    if mlmodel_file.exists():
        meta["mlmodel"] = mlmodel_file.read_text()

    card = textwrap.dedent(f"""\
        # Model Card

        ## Overview
        | Field | Value |
        |-------|-------|
        | Framework | {framework} |
        | Path | `{meta['model_path']}` |
        | Generated | {meta['generated_at']} |

        ## Intended Use
        Describe the primary intended use-cases for this model.

        ## Metrics
        {metrics_section}

        ## Limitations
        - Document known failure modes here.
        - Specify populations or conditions where performance degrades.

        ## Ethical Considerations
        - Note any fairness evaluations performed.
        - List potential biases in training data.
    """)

    Path(output_path).write_text(card)
    logger.info("Model card written to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_model(model_path: str, framework: str):
    """Load the model and run a trivial inference to verify packaging."""
    logger.info("Validating model at %s (framework=%s)", model_path, framework)

    try:
        import numpy as np
    except ImportError:
        logger.error("numpy is not installed. Run: pip install numpy")
        sys.exit(1)

    sample = np.random.randn(1, 10).astype(np.float32)

    if framework == "sklearn":
        import pickle
        with open(model_path, "rb") as fh:
            model = pickle.load(fh)
        pred = model.predict(sample)
    elif framework == "pytorch":
        import torch
        model = torch.load(model_path, map_location="cpu", weights_only=False)
        model.eval()
        with torch.no_grad():
            pred = model(torch.from_numpy(sample)).numpy()
    elif framework == "xgboost":
        import xgboost as xgb
        dmat = xgb.DMatrix(sample)
        booster = xgb.Booster()
        booster.load_model(model_path)
        pred = booster.predict(dmat)
    else:
        logger.error("Unsupported framework '%s'", framework)
        sys.exit(1)

    logger.info("Inference succeeded. Output shape: %s", pred.shape)
    logger.info("Sample prediction: %s", pred.ravel()[:5])
    return pred


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Package models for deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--action",
        required=True,
        choices=["docker", "onnx", "mlflow", "model-card", "validate"],
        help="Packaging action to perform",
    )
    parser.add_argument("--model-path", required=True, help="Path to the model artifact")
    parser.add_argument(
        "--framework",
        choices=SUPPORTED_FRAMEWORKS,
        default="sklearn",
        help="ML framework of the source model",
    )
    parser.add_argument("--output", default="./output", help="Output path or directory")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    try:
        if args.action == "docker":
            package_docker(args.model_path, args.output, args.framework)

        elif args.action == "onnx":
            package_onnx(args.model_path, args.output, args.framework)

        elif args.action == "mlflow":
            package_mlflow(args.model_path, args.output)

        elif args.action == "model-card":
            generate_model_card(args.model_path, args.output, args.framework)

        elif args.action == "validate":
            validate_model(args.model_path, args.framework)

    except Exception:
        logger.exception("Action '%s' failed", args.action)
        sys.exit(1)


if __name__ == "__main__":
    main()
