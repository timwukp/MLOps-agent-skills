#!/usr/bin/env python3
"""Prompt template management and versioning with YAML-backed registry.

Usage:
    python prompt_manager.py --action register --name summarizer \
        --template "Summarize the following {document} in {style} style." \
        --metadata '{"author": "team-a", "task": "summarization"}'
    python prompt_manager.py --action get --name summarizer
    python prompt_manager.py --action list
    python prompt_manager.py --action render --name summarizer \
        --variables '{"document": "Long article text...", "style": "concise"}'
    python prompt_manager.py --action update --name summarizer \
        --template "Provide a {style} summary of: {document}"
    python prompt_manager.py --action delete --name summarizer
    python prompt_manager.py --action export --registry-path prompts.yaml
    python prompt_manager.py --action import --registry-path prompts_backup.yaml
"""
import argparse
import json
import logging
import re
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_REGISTRY = "prompt_registry.yaml"


class PromptTemplate:
    """A versioned prompt template with variable placeholders."""

    def __init__(self, name, template, version=1, metadata=None, created_at=None):
        self.name = name
        self.template = template
        self.version = version
        self.metadata = metadata or {}
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()

    @property
    def variables(self):
        """Extract {variable} placeholders from the template string."""
        return sorted(set(re.findall(r"\{(\w+)\}", self.template)))

    def render(self, **kwargs):
        """Fill template variables. Raises ValueError if any variable is missing."""
        missing = [v for v in self.variables if v not in kwargs]
        if missing:
            raise ValueError(f"Missing variables for template '{self.name}': {missing}")
        return self.template.format(**kwargs)

    def to_dict(self):
        return {
            "name": self.name,
            "template": self.template,
            "version": self.version,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            name=d["name"],
            template=d["template"],
            version=d.get("version", 1),
            metadata=d.get("metadata", {}),
            created_at=d.get("created_at"),
        )


class PromptRegistry:
    """YAML-file backed storage for prompt templates with version history."""

    def __init__(self, path=None):
        self.path = Path(path or DEFAULT_REGISTRY)
        self._data = self._load()

    def _load(self):
        if not self.path.exists():
            return {"templates": {}, "history": {}, "ab_tests": {}}
        try:
            import yaml
        except ImportError:
            logger.error("Install: pip install pyyaml")
            sys.exit(1)
        with open(self.path, "r") as f:
            return yaml.safe_load(f) or {"templates": {}, "history": {}, "ab_tests": {}}

    def _save(self):
        try:
            import yaml
        except ImportError:
            logger.error("Install: pip install pyyaml")
            sys.exit(1)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            yaml.dump(self._data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    def register(self, name, template, metadata=None):
        """Register a new prompt template. Fails if name already exists."""
        if name in self._data["templates"]:
            raise ValueError(f"Template '{name}' already exists. Use update() instead.")
        pt = PromptTemplate(name=name, template=template, version=1, metadata=metadata)
        self._data["templates"][name] = pt.to_dict()
        self._data["history"].setdefault(name, [])
        self._data["history"][name].append(pt.to_dict())
        self._save()
        logger.info(f"Registered template '{name}' v1")
        return pt

    def update(self, name, template=None, metadata=None):
        """Update a template, auto-incrementing its version and keeping history."""
        if name not in self._data["templates"]:
            raise KeyError(f"Template '{name}' not found.")
        current = self._data["templates"][name]
        new_version = current["version"] + 1
        pt = PromptTemplate(
            name=name,
            template=template if template is not None else current["template"],
            version=new_version,
            metadata=metadata if metadata is not None else current.get("metadata", {}),
        )
        self._data["templates"][name] = pt.to_dict()
        self._data["history"].setdefault(name, [])
        self._data["history"][name].append(pt.to_dict())
        self._save()
        logger.info(f"Updated template '{name}' to v{new_version}")
        return pt

    def get(self, name, version=None):
        """Retrieve a template by name. Optionally specify a historical version."""
        if version is not None:
            history = self._data.get("history", {}).get(name, [])
            for entry in history:
                if entry["version"] == version:
                    return PromptTemplate.from_dict(entry)
            raise KeyError(f"Template '{name}' v{version} not found in history.")
        if name not in self._data["templates"]:
            raise KeyError(f"Template '{name}' not found.")
        return PromptTemplate.from_dict(self._data["templates"][name])

    def list_templates(self):
        """Return a list of all current templates."""
        return [PromptTemplate.from_dict(v) for v in self._data["templates"].values()]

    def delete(self, name):
        """Delete a template and its history."""
        if name not in self._data["templates"]:
            raise KeyError(f"Template '{name}' not found.")
        del self._data["templates"][name]
        self._data["history"].pop(name, None)
        self._data["ab_tests"].pop(name, None)
        self._save()
        logger.info(f"Deleted template '{name}'")

    def get_history(self, name):
        """Return version history for a template."""
        entries = self._data.get("history", {}).get(name, [])
        return [PromptTemplate.from_dict(e) for e in entries]

    # -- A/B testing support --------------------------------------------------

    def setup_ab_test(self, name, variants):
        """Configure A/B test variants for a template name.

        Args:
            name: logical test name
            variants: dict mapping variant label to template name in registry
        """
        self._data["ab_tests"][name] = {
            "variants": variants,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save()
        logger.info(f"A/B test '{name}' created with variants: {list(variants.keys())}")

    def assign_variant(self, test_name, session_id):
        """Deterministically assign a variant based on session_id hash."""
        import hashlib
        test = self._data["ab_tests"].get(test_name)
        if not test:
            raise KeyError(f"A/B test '{test_name}' not found.")
        variant_keys = sorted(test["variants"].keys())
        idx = int(hashlib.sha256(session_id.encode()).hexdigest(), 16) % len(variant_keys)
        chosen = variant_keys[idx]
        template_name = test["variants"][chosen]
        pt = self.get(template_name)
        return {"variant": chosen, "template_name": template_name, "template": pt.to_dict()}

    # -- Export / Import ------------------------------------------------------

    def export_library(self, dest_path):
        """Export full registry to a YAML file."""
        try:
            import yaml
        except ImportError:
            logger.error("Install: pip install pyyaml")
            sys.exit(1)
        data = deepcopy(self._data)
        with open(dest_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        n = len(data.get("templates", {}))
        logger.info(f"Exported {n} templates to {dest_path}")

    def import_library(self, src_path, overwrite=False):
        """Import templates from a YAML file into this registry."""
        try:
            import yaml
        except ImportError:
            logger.error("Install: pip install pyyaml")
            sys.exit(1)
        with open(src_path, "r") as f:
            incoming = yaml.safe_load(f) or {}
        added, skipped = 0, 0
        for name, tpl in incoming.get("templates", {}).items():
            if name in self._data["templates"] and not overwrite:
                logger.warning(f"Skipping '{name}' (already exists). Use --overwrite to replace.")
                skipped += 1
                continue
            self._data["templates"][name] = tpl
            self._data["history"].setdefault(name, [])
            self._data["history"][name].append(tpl)
            added += 1
        self._save()
        logger.info(f"Imported {added} templates, skipped {skipped}")


def main():
    parser = argparse.ArgumentParser(description="Prompt template manager")
    parser.add_argument("--action", required=True,
                        choices=["register", "get", "list", "update", "delete",
                                 "render", "export", "import", "history"],
                        help="Action to perform")
    parser.add_argument("--name", help="Template name")
    parser.add_argument("--template", help="Template string with {variables}")
    parser.add_argument("--variables", help="JSON dict of variable values for render")
    parser.add_argument("--metadata", help="JSON dict of metadata")
    parser.add_argument("--version", type=int, help="Specific version to retrieve")
    parser.add_argument("--registry-path", default=DEFAULT_REGISTRY, help="Path to registry YAML")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite on import")

    args = parser.parse_args()
    registry = PromptRegistry(args.registry_path)

    if args.action == "register":
        if not args.name or not args.template:
            parser.error("--name and --template are required for register")
        meta = json.loads(args.metadata) if args.metadata else None
        pt = registry.register(args.name, args.template, metadata=meta)
        print(json.dumps(pt.to_dict(), indent=2))

    elif args.action == "get":
        if not args.name:
            parser.error("--name is required for get")
        pt = registry.get(args.name, version=args.version)
        print(json.dumps(pt.to_dict(), indent=2))

    elif args.action == "list":
        for pt in registry.list_templates():
            print(f"  {pt.name:30s} v{pt.version}  vars={pt.variables}")

    elif args.action == "update":
        if not args.name:
            parser.error("--name is required for update")
        meta = json.loads(args.metadata) if args.metadata else None
        pt = registry.update(args.name, template=args.template, metadata=meta)
        print(json.dumps(pt.to_dict(), indent=2))

    elif args.action == "delete":
        if not args.name:
            parser.error("--name is required for delete")
        registry.delete(args.name)

    elif args.action == "render":
        if not args.name:
            parser.error("--name is required for render")
        if not args.variables:
            parser.error("--variables (JSON) is required for render")
        pt = registry.get(args.name)
        variables = json.loads(args.variables)
        rendered = pt.render(**variables)
        print(rendered)

    elif args.action == "history":
        if not args.name:
            parser.error("--name is required for history")
        for pt in registry.get_history(args.name):
            print(f"  v{pt.version}  created={pt.created_at}  vars={pt.variables}")

    elif args.action == "export":
        dest = args.registry_path if args.registry_path != DEFAULT_REGISTRY else "prompts_export.yaml"
        registry.export_library(dest)

    elif args.action == "import":
        if not args.registry_path:
            parser.error("--registry-path is required for import")
        registry.import_library(args.registry_path, overwrite=args.overwrite)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
