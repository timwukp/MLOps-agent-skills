#!/usr/bin/env python3
"""MLflow Model Registry management script.

Provides CLI operations for managing the full model lifecycle:
registering models from runs, promoting through stages, comparing
versions, inspecting lineage, and cleaning up archived versions.

Usage examples:
    # Register a model from a completed MLflow run
    python registry_manager.py --action register --model-name fraud-detector \
        --run-id abc123def456 --artifact-path model

    # Promote a model version to Staging
    python registry_manager.py --action promote --model-name fraud-detector \
        --version 3 --stage Staging

    # Archive a model version
    python registry_manager.py --action archive --model-name fraud-detector \
        --version 2

    # Compare two model versions side-by-side
    python registry_manager.py --action compare --model-name fraud-detector \
        --versions 2,3

    # List all versions of a model
    python registry_manager.py --action list --model-name fraud-detector

    # Show full lineage for a specific version
    python registry_manager.py --action lineage --model-name fraud-detector \
        --version 3

    # Clean up old archived versions (keep most recent N)
    python registry_manager.py --action cleanup --model-name fraud-detector \
        --keep 2

    # Add description and tags to a version
    python registry_manager.py --action describe --model-name fraud-detector \
        --version 3 --description "Improved recall" --tags '{"team":"fraud"}'
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _get_client():
    """Lazy-import MLflow and return the tracking MlflowClient."""
    try:
        from mlflow.tracking import MlflowClient
    except ImportError:
        logger.error("mlflow is not installed. Run: pip install mlflow")
        sys.exit(1)
    return MlflowClient()


def register_model(model_name: str, run_id: str, artifact_path: str = "model"):
    """Register a model from an existing MLflow run."""
    try:
        import mlflow
    except ImportError:
        logger.error("mlflow is not installed. Run: pip install mlflow")
        sys.exit(1)

    model_uri = f"runs:/{run_id}/{artifact_path}"
    logger.info("Registering model '%s' from URI: %s", model_name, model_uri)

    result = mlflow.register_model(model_uri, model_name)
    logger.info(
        "Registered version %s of '%s' (status: %s)",
        result.version,
        result.name,
        result.status,
    )
    return result


def promote_model(model_name: str, version: int, stage: str):
    """Transition a model version to a new stage."""
    valid_stages = {"None", "Staging", "Production", "Archived"}
    if stage not in valid_stages:
        logger.error("Invalid stage '%s'. Must be one of %s", stage, valid_stages)
        sys.exit(1)

    client = _get_client()
    logger.info(
        "Transitioning '%s' v%s -> %s", model_name, version, stage
    )
    updated = client.transition_model_version_stage(
        name=model_name,
        version=str(version),
        stage=stage,
        archive_existing_versions=(stage == "Production"),
    )
    logger.info(
        "Version %s is now in stage '%s'", updated.version, updated.current_stage
    )
    return updated


def archive_model(model_name: str, version: int):
    """Shortcut to archive a specific model version."""
    return promote_model(model_name, version, "Archived")


def compare_versions(model_name: str, versions: list[int]):
    """Print a side-by-side comparison of metrics for the given versions."""
    client = _get_client()
    rows: list[dict] = []

    for ver in versions:
        mv = client.get_model_version(model_name, str(ver))
        run = client.get_run(mv.run_id)
        rows.append(
            {
                "version": ver,
                "stage": mv.current_stage,
                "run_id": mv.run_id,
                "metrics": run.data.metrics,
                "params": run.data.params,
            }
        )

    # Gather all metric keys across versions
    all_metrics = sorted(
        {k for r in rows for k in r["metrics"]}
    )

    header = f"{'Metric':<30}" + "".join(
        f"v{r['version']:<15}" for r in rows
    )
    print("\n" + header)
    print("-" * len(header))
    for metric in all_metrics:
        vals = "".join(
            f"{r['metrics'].get(metric, 'N/A'):<16}" for r in rows
        )
        print(f"{metric:<30}{vals}")

    # Stage info
    print()
    stage_row = "".join(f"{r['stage']:<16}" for r in rows)
    print(f"{'Stage':<30}{stage_row}")
    print()


def list_versions(model_name: str):
    """List every version of a registered model with key metadata."""
    client = _get_client()
    versions = client.search_model_versions(f"name='{model_name}'")

    if not versions:
        logger.warning("No versions found for model '%s'", model_name)
        return

    print(f"\nModel: {model_name}  ({len(versions)} version(s))\n")
    print(
        f"{'Version':<10}{'Stage':<15}{'Status':<12}{'Run ID':<36}{'Created'}"
    )
    print("-" * 95)
    for mv in sorted(versions, key=lambda v: int(v.version)):
        created = datetime.fromtimestamp(
            mv.creation_timestamp / 1000, tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M UTC")
        print(
            f"{mv.version:<10}{mv.current_stage:<15}{mv.status:<12}"
            f"{mv.run_id:<36}{created}"
        )
    print()


def get_production_model(model_name: str):
    """Print details of the current Production version."""
    client = _get_client()
    versions = client.search_model_versions(f"name='{model_name}'")
    prod = [v for v in versions if v.current_stage == "Production"]

    if not prod:
        logger.warning("No Production version found for '%s'", model_name)
        return None

    mv = prod[0]
    run = client.get_run(mv.run_id)
    print(f"\nProduction model: {model_name} v{mv.version}")
    print(f"  Run ID      : {mv.run_id}")
    print(f"  Description : {mv.description or '(none)'}")
    print(f"  Tags        : {dict(mv.tags) if mv.tags else '{}'}")
    print(f"  Metrics     : {run.data.metrics}")
    print(f"  Params      : {run.data.params}")
    print()
    return mv


def describe_version(
    model_name: str, version: int, description: str = None, tags: dict = None
):
    """Add or update description and tags on a model version."""
    client = _get_client()
    if description:
        client.update_model_version(
            name=model_name, version=str(version), description=description
        )
        logger.info("Updated description for '%s' v%s", model_name, version)

    if tags:
        for key, value in tags.items():
            client.set_model_version_tag(model_name, str(version), key, value)
        logger.info("Set %d tag(s) on '%s' v%s", len(tags), model_name, version)


def cleanup_archived(model_name: str, keep: int = 1):
    """Delete old Archived versions, retaining the most recent *keep*."""
    client = _get_client()
    versions = client.search_model_versions(f"name='{model_name}'")
    archived = sorted(
        [v for v in versions if v.current_stage == "Archived"],
        key=lambda v: int(v.version),
        reverse=True,
    )

    to_delete = archived[keep:]
    if not to_delete:
        logger.info("Nothing to clean up (archived: %d, keep: %d)", len(archived), keep)
        return

    for mv in to_delete:
        logger.info("Deleting '%s' v%s (Archived)", model_name, mv.version)
        client.delete_model_version(name=model_name, version=mv.version)

    logger.info("Deleted %d archived version(s)", len(to_delete))


def show_lineage(model_name: str, version: int):
    """Display full lineage: run parameters, metrics, and artifacts."""
    client = _get_client()
    mv = client.get_model_version(model_name, str(version))
    run = client.get_run(mv.run_id)

    print(f"\nLineage for {model_name} v{version}")
    print(f"  Run ID  : {mv.run_id}")
    print(f"  Source  : {mv.source}")
    print(f"  Stage   : {mv.current_stage}")

    print("\n  Parameters:")
    for k, v in sorted(run.data.params.items()):
        print(f"    {k:<30} {v}")

    print("\n  Metrics:")
    for k, v in sorted(run.data.metrics.items()):
        print(f"    {k:<30} {v}")

    artifacts = client.list_artifacts(mv.run_id)
    print("\n  Artifacts:")
    for art in artifacts:
        print(f"    {art.path}  ({art.file_size or '?'} bytes)")
    print()


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="MLflow Model Registry manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--action",
        required=True,
        choices=[
            "register", "promote", "archive", "compare",
            "list", "production", "describe", "cleanup", "lineage",
        ],
        help="Operation to perform",
    )
    parser.add_argument("--model-name", required=True, help="Registered model name")
    parser.add_argument("--version", type=int, help="Model version number")
    parser.add_argument("--versions", help="Comma-separated versions to compare")
    parser.add_argument("--stage", help="Target stage for promotion")
    parser.add_argument("--run-id", help="MLflow run ID (for register)")
    parser.add_argument("--artifact-path", default="model", help="Artifact sub-path")
    parser.add_argument("--description", help="Version description text")
    parser.add_argument("--tags", help="JSON dict of tags to set")
    parser.add_argument(
        "--keep", type=int, default=1,
        help="Number of archived versions to keep during cleanup",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    try:
        if args.action == "register":
            if not args.run_id:
                logger.error("--run-id is required for register")
                sys.exit(1)
            register_model(args.model_name, args.run_id, args.artifact_path)

        elif args.action == "promote":
            if args.version is None or not args.stage:
                logger.error("--version and --stage are required for promote")
                sys.exit(1)
            promote_model(args.model_name, args.version, args.stage)

        elif args.action == "archive":
            if args.version is None:
                logger.error("--version is required for archive")
                sys.exit(1)
            archive_model(args.model_name, args.version)

        elif args.action == "compare":
            if not args.versions:
                logger.error("--versions (e.g. '2,3') is required for compare")
                sys.exit(1)
            ver_list = [int(v.strip()) for v in args.versions.split(",")]
            compare_versions(args.model_name, ver_list)

        elif args.action == "list":
            list_versions(args.model_name)

        elif args.action == "production":
            get_production_model(args.model_name)

        elif args.action == "describe":
            if args.version is None:
                logger.error("--version is required for describe")
                sys.exit(1)
            tags = json.loads(args.tags) if args.tags else None
            describe_version(args.model_name, args.version, args.description, tags)

        elif args.action == "cleanup":
            cleanup_archived(args.model_name, keep=args.keep)

        elif args.action == "lineage":
            if args.version is None:
                logger.error("--version is required for lineage")
                sys.exit(1)
            show_lineage(args.model_name, args.version)

    except Exception:
        logger.exception("Operation '%s' failed", args.action)
        sys.exit(1)


if __name__ == "__main__":
    main()
