#!/usr/bin/env python3
"""Lightweight feature registry / catalog for ML teams.

Provides a JSON-file-backed registry for feature definitions, lineage
tracking, search, export, and consistency validation -- useful when a
full Feast deployment is not required.

Usage:
    python feature_registry.py --action register --name daily_spend --dtype float64 \
        --description "Average daily spend" --owner growth-team --tag finance --tag user
    python feature_registry.py --action list
    python feature_registry.py --action list --owner growth-team
    python feature_registry.py --action search --name spend
    python feature_registry.py --action search --tag finance
    python feature_registry.py --action export --format markdown
    python feature_registry.py --action export --format json
    python feature_registry.py --action validate
"""
import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

VALID_DTYPES = {"int32", "int64", "float32", "float64", "string", "bool", "datetime", "category"}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class FeatureDefinition:
    name: str
    dtype: str
    description: str = ""
    source: str = ""
    owner: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    version: int = 1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "FeatureDefinition":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class FeatureLineage:
    output_feature: str
    input_features: List[str]
    transformation: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class FeatureRegistry:
    """JSON-file-backed feature catalogue."""

    def __init__(self, registry_path: str = "feature_registry.json"):
        self.path = Path(registry_path)
        self._features: Dict[str, dict] = {}
        self._lineage: List[dict] = []
        if self.path.exists():
            self._load()

    # -- persistence ---------------------------------------------------------

    def _load(self):
        data = json.loads(self.path.read_text())
        self._features = data.get("features", {})
        self._lineage = data.get("lineage", [])

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"features": self._features, "lineage": self._lineage}
        self.path.write_text(json.dumps(payload, indent=2, default=str))

    # -- CRUD ----------------------------------------------------------------

    def register(self, feat: FeatureDefinition) -> None:
        if feat.name in self._features:
            logger.warning("Feature '%s' already exists -- updating", feat.name)
            feat.version = self._features[feat.name].get("version", 0) + 1
        self._features[feat.name] = feat.to_dict()
        self._save()
        logger.info("Registered feature '%s' (v%d)", feat.name, feat.version)

    def update(self, name: str, **kwargs) -> None:
        if name not in self._features:
            raise KeyError(f"Feature '{name}' not found in registry")
        self._features[name].update(kwargs)
        self._features[name]["version"] = self._features[name].get("version", 0) + 1
        self._save()
        logger.info("Updated feature '%s'", name)

    def delete(self, name: str) -> None:
        if name not in self._features:
            raise KeyError(f"Feature '{name}' not found in registry")
        del self._features[name]
        self._lineage = [l for l in self._lineage if l.get("output_feature") != name]
        self._save()
        logger.info("Deleted feature '%s'", name)

    def get(self, name: str) -> Optional[FeatureDefinition]:
        data = self._features.get(name)
        return FeatureDefinition.from_dict(data) if data else None

    # -- query ---------------------------------------------------------------

    def list_all(self) -> List[FeatureDefinition]:
        return [FeatureDefinition.from_dict(v) for v in self._features.values()]

    def list_by_owner(self, owner: str) -> List[FeatureDefinition]:
        return [FeatureDefinition.from_dict(v) for v in self._features.values()
                if v.get("owner") == owner]

    def list_by_tag(self, tag: str) -> List[FeatureDefinition]:
        return [FeatureDefinition.from_dict(v) for v in self._features.values()
                if tag in v.get("tags", [])]

    def list_by_source(self, source: str) -> List[FeatureDefinition]:
        return [FeatureDefinition.from_dict(v) for v in self._features.values()
                if source in v.get("source", "")]

    def search(self, query: str) -> List[FeatureDefinition]:
        q = query.lower()
        results = []
        for v in self._features.values():
            searchable = f"{v['name']} {v.get('description','')} {' '.join(v.get('tags',[]))}".lower()
            if q in searchable:
                results.append(FeatureDefinition.from_dict(v))
        return results

    # -- lineage -------------------------------------------------------------

    def add_lineage(self, lineage: FeatureLineage) -> None:
        self._lineage.append(lineage.to_dict())
        self._save()
        logger.info("Lineage recorded: %s -> %s", lineage.input_features, lineage.output_feature)

    def get_lineage(self, feature_name: str) -> List[FeatureLineage]:
        return [FeatureLineage(**l) for l in self._lineage
                if l.get("output_feature") == feature_name]

    # -- export --------------------------------------------------------------

    def export_json(self) -> str:
        return json.dumps(
            [v for v in self._features.values()],
            indent=2,
            default=str,
        )

    def export_markdown(self) -> str:
        lines = ["| Name | Dtype | Owner | Tags | Version | Description |",
                 "|------|-------|-------|------|---------|-------------|"]
        for v in self._features.values():
            tags = ", ".join(v.get("tags", []))
            lines.append(
                f"| {v['name']} | {v['dtype']} | {v.get('owner','')} "
                f"| {tags} | {v.get('version',1)} | {v.get('description','')} |"
            )
        return "\n".join(lines)

    # -- validation ----------------------------------------------------------

    def validate(self) -> Dict[str, list]:
        errors: List[str] = []
        warnings: List[str] = []

        seen_names: Dict[str, int] = {}
        for name, v in self._features.items():
            # Duplicate check (should not happen with dict keys, but validates payload)
            seen_names[name] = seen_names.get(name, 0) + 1
            if seen_names[name] > 1:
                errors.append(f"Duplicate feature name: '{name}'")

            # Dtype check
            if v.get("dtype") not in VALID_DTYPES:
                errors.append(f"Feature '{name}': invalid dtype '{v.get('dtype')}' "
                              f"(expected one of {sorted(VALID_DTYPES)})")

            # Missing metadata
            if not v.get("description"):
                warnings.append(f"Feature '{name}': missing description")
            if not v.get("owner"):
                warnings.append(f"Feature '{name}': missing owner")

        # Lineage references
        for entry in self._lineage:
            out = entry.get("output_feature", "")
            if out and out not in self._features:
                warnings.append(f"Lineage output '{out}' not found in registry")
            for inp in entry.get("input_features", []):
                if inp not in self._features:
                    warnings.append(f"Lineage input '{inp}' not found in registry")

        return {"errors": errors, "warnings": warnings}


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _print_features(features: List[FeatureDefinition]) -> None:
    for f in features:
        tags = ", ".join(f.tags) if f.tags else ""
        print(f"  {f.name:30s}  {f.dtype:10s}  owner={f.owner or '-':15s}  "
              f"v{f.version}  tags=[{tags}]  {f.description}")


def main():
    parser = argparse.ArgumentParser(description="Lightweight ML feature registry")
    parser.add_argument(
        "--action",
        required=True,
        choices=["register", "update", "delete", "list", "search", "export", "validate"],
        help="Action to perform",
    )
    parser.add_argument("--name", default=None, help="Feature name")
    parser.add_argument("--dtype", default=None, help="Feature data type")
    parser.add_argument("--description", default="", help="Feature description")
    parser.add_argument("--source", default="", help="Data source identifier")
    parser.add_argument("--owner", default="", help="Feature owner (team or person)")
    parser.add_argument("--tag", action="append", default=[], dest="tags", help="Tag (repeatable)")
    parser.add_argument("--registry-path", default="feature_registry.json", help="Registry file")
    parser.add_argument("--format", default="markdown", choices=["markdown", "json"],
                        help="Export format")

    args = parser.parse_args()
    registry = FeatureRegistry(args.registry_path)

    try:
        if args.action == "register":
            if not args.name or not args.dtype:
                parser.error("--name and --dtype are required for register")
            feat = FeatureDefinition(
                name=args.name,
                dtype=args.dtype,
                description=args.description,
                source=args.source,
                owner=args.owner,
                tags=args.tags,
            )
            registry.register(feat)

        elif args.action == "update":
            if not args.name:
                parser.error("--name is required for update")
            updates = {}
            if args.dtype:
                updates["dtype"] = args.dtype
            if args.description:
                updates["description"] = args.description
            if args.owner:
                updates["owner"] = args.owner
            if args.tags:
                updates["tags"] = args.tags
            registry.update(args.name, **updates)

        elif args.action == "delete":
            if not args.name:
                parser.error("--name is required for delete")
            registry.delete(args.name)

        elif args.action == "list":
            if args.owner:
                features = registry.list_by_owner(args.owner)
            elif args.tags:
                features = registry.list_by_tag(args.tags[0])
            elif args.source:
                features = registry.list_by_source(args.source)
            else:
                features = registry.list_all()
            logger.info("Found %d features", len(features))
            _print_features(features)

        elif args.action == "search":
            query = args.name or (args.tags[0] if args.tags else "")
            if not query:
                parser.error("--name or --tag required for search")
            features = registry.search(query)
            logger.info("Search '%s' returned %d features", query, len(features))
            _print_features(features)

        elif args.action == "export":
            if args.format == "json":
                print(registry.export_json())
            else:
                print(registry.export_markdown())

        elif args.action == "validate":
            result = registry.validate()
            for err in result["errors"]:
                logger.error("ERROR: %s", err)
            for warn in result["warnings"]:
                logger.warning("WARN:  %s", warn)
            logger.info(
                "Validation complete: %d errors, %d warnings",
                len(result["errors"]),
                len(result["warnings"]),
            )
            if result["errors"]:
                sys.exit(1)

    except Exception as exc:
        logger.error("Action '%s' failed: %s", args.action, exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
