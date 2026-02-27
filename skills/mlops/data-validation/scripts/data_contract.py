#!/usr/bin/env python3
"""Data contract definition and enforcement for MLOps pipelines.

Defines, loads, validates, and compares data contracts expressed as YAML
specifications.  A contract captures schema expectations (column names, types,
nullability, constraints), freshness requirements, volume expectations, and
custom quality rules.  The script can also infer a draft contract from an
existing CSV dataset and produce a detailed validation report.

Typical usage:
    python data_contract.py --action validate --data sales.csv --contract sales_contract.yaml
    python data_contract.py --action generate --data sales.csv --output draft_contract.yaml
    python data_contract.py --action compare --contract v1.yaml --contract-v2 v2.yaml
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:
    yaml = None  # graceful degradation; error raised when YAML is actually needed

logger = logging.getLogger("data_contract")

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ColumnSpec:
    """Specification for a single column inside a data contract."""
    name: str
    dtype: str = "string"
    nullable: bool = True
    constraints: Dict[str, Any] = field(default_factory=dict)
    # constraints may include: min, max, pattern, allowed_values, unique


@dataclass
class DataContract:
    """Top-level data contract definition."""
    name: str
    version: str
    owner: str
    columns: List[ColumnSpec] = field(default_factory=list)
    freshness_max_hours: Optional[float] = None
    volume_min_rows: Optional[int] = None
    volume_max_rows: Optional[int] = None
    quality_rules: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def _require_yaml():
    if yaml is None:
        logger.error("PyYAML is required. Install with: pip install pyyaml")
        sys.exit(1)


def load_contract(path: str) -> DataContract:
    """Load a DataContract from a YAML file."""
    _require_yaml()
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh)
    cols = [ColumnSpec(**c) for c in raw.get("columns", [])]
    return DataContract(
        name=raw.get("name", "unnamed"),
        version=raw.get("version", "0.1.0"),
        owner=raw.get("owner", "unknown"),
        columns=cols,
        freshness_max_hours=raw.get("freshness_max_hours"),
        volume_min_rows=raw.get("volume_min_rows"),
        volume_max_rows=raw.get("volume_max_rows"),
        quality_rules=raw.get("quality_rules", []),
    )


def save_contract(contract: DataContract, path: str) -> None:
    """Serialize a DataContract to a YAML file."""
    _require_yaml()
    data: Dict[str, Any] = {
        "name": contract.name,
        "version": contract.version,
        "owner": contract.owner,
        "columns": [
            {"name": c.name, "dtype": c.dtype, "nullable": c.nullable, "constraints": c.constraints}
            for c in contract.columns
        ],
        "freshness_max_hours": contract.freshness_max_hours,
        "volume_min_rows": contract.volume_min_rows,
        "volume_max_rows": contract.volume_max_rows,
        "quality_rules": contract.quality_rules,
    }
    with open(path, "w") as fh:
        yaml.dump(data, fh, default_flow_style=False, sort_keys=False)
    logger.info("Contract written to %s", path)


# ---------------------------------------------------------------------------
# Validation engine
# ---------------------------------------------------------------------------

def _read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader)


def validate(contract: DataContract, data_path: str) -> List[Dict[str, Any]]:
    """Validate *data_path* (CSV) against *contract*.  Returns a list of
    result dicts, each with keys ``rule``, ``passed``, and ``detail``."""
    rows = _read_csv(data_path)
    results: List[Dict[str, Any]] = []
    headers = list(rows[0].keys()) if rows else []

    # Volume check
    if contract.volume_min_rows is not None:
        ok = len(rows) >= contract.volume_min_rows
        results.append({"rule": "volume_min_rows", "passed": ok,
                        "detail": f"rows={len(rows)}, min={contract.volume_min_rows}"})
    if contract.volume_max_rows is not None:
        ok = len(rows) <= contract.volume_max_rows
        results.append({"rule": "volume_max_rows", "passed": ok,
                        "detail": f"rows={len(rows)}, max={contract.volume_max_rows}"})

    # Freshness check (file mtime)
    if contract.freshness_max_hours is not None:
        mtime = datetime.fromtimestamp(os.path.getmtime(data_path), tz=timezone.utc)
        age_hours = (datetime.now(timezone.utc) - mtime).total_seconds() / 3600
        ok = age_hours <= contract.freshness_max_hours
        results.append({"rule": "freshness", "passed": ok,
                        "detail": f"age_hours={age_hours:.2f}, max={contract.freshness_max_hours}"})

    # Per-column checks
    for col_spec in contract.columns:
        cname = col_spec.name
        if cname not in headers:
            results.append({"rule": f"schema_presence:{cname}", "passed": False,
                            "detail": "Column missing from data"})
            continue
        results.append({"rule": f"schema_presence:{cname}", "passed": True, "detail": "Column present"})

        values = [r[cname] for r in rows]

        # Nullable
        if not col_spec.nullable:
            blanks = sum(1 for v in values if v.strip() == "")
            ok = blanks == 0
            results.append({"rule": f"not_null:{cname}", "passed": ok,
                            "detail": f"blanks={blanks}"})

        cons = col_spec.constraints
        # Min / max (numeric)
        if "min" in cons or "max" in cons:
            try:
                nums = [float(v) for v in values if v.strip() != ""]
            except ValueError:
                nums = []
            if nums:
                if "min" in cons:
                    ok = all(n >= cons["min"] for n in nums)
                    results.append({"rule": f"min:{cname}", "passed": ok,
                                    "detail": f"actual_min={min(nums)}, expected_min={cons['min']}"})
                if "max" in cons:
                    ok = all(n <= cons["max"] for n in nums)
                    results.append({"rule": f"max:{cname}", "passed": ok,
                                    "detail": f"actual_max={max(nums)}, expected_max={cons['max']}"})

        # Regex pattern
        if "pattern" in cons:
            pat = re.compile(cons["pattern"])
            non_empty = [v for v in values if v.strip() != ""]
            mismatches = [v for v in non_empty if not pat.fullmatch(v)]
            ok = len(mismatches) == 0
            results.append({"rule": f"pattern:{cname}", "passed": ok,
                            "detail": f"mismatches={len(mismatches)}"})

        # Allowed values
        if "allowed_values" in cons:
            allowed = set(cons["allowed_values"])
            bad = [v for v in values if v.strip() != "" and v not in allowed]
            ok = len(bad) == 0
            results.append({"rule": f"allowed_values:{cname}", "passed": ok,
                            "detail": f"invalid_count={len(bad)}"})

        # Uniqueness
        if cons.get("unique"):
            non_empty = [v for v in values if v.strip() != ""]
            ok = len(non_empty) == len(set(non_empty))
            results.append({"rule": f"unique:{cname}", "passed": ok,
                            "detail": f"total={len(non_empty)}, distinct={len(set(non_empty))}"})

    return results


# ---------------------------------------------------------------------------
# Contract generation (infer from data)
# ---------------------------------------------------------------------------

def generate_contract(data_path: str, name: str = "auto_generated") -> DataContract:
    """Infer a draft contract from an existing CSV file."""
    rows = _read_csv(data_path)
    if not rows:
        return DataContract(name=name, version="0.1.0", owner="auto")
    columns: List[ColumnSpec] = []
    for col_name in rows[0].keys():
        values = [r[col_name] for r in rows]
        nullable = any(v.strip() == "" for v in values)
        dtype = _infer_dtype(values)
        columns.append(ColumnSpec(name=col_name, dtype=dtype, nullable=nullable))
    return DataContract(
        name=name, version="0.1.0", owner="auto",
        columns=columns, volume_min_rows=len(rows),
    )


def _infer_dtype(values: List[str]) -> str:
    non_empty = [v for v in values if v.strip() != ""]
    if not non_empty:
        return "string"
    try:
        for v in non_empty:
            int(v)
        return "integer"
    except ValueError:
        pass
    try:
        for v in non_empty:
            float(v)
        return "float"
    except ValueError:
        pass
    return "string"


# ---------------------------------------------------------------------------
# Contract comparison / versioning
# ---------------------------------------------------------------------------

def compare_contracts(old: DataContract, new: DataContract) -> List[Dict[str, str]]:
    """Compare two contract versions.  Returns a list of change dicts."""
    changes: List[Dict[str, str]] = []
    old_cols = {c.name: c for c in old.columns}
    new_cols = {c.name: c for c in new.columns}
    for name in old_cols:
        if name not in new_cols:
            changes.append({"type": "breaking", "detail": f"Column '{name}' removed"})
        else:
            oc, nc = old_cols[name], new_cols[name]
            if oc.dtype != nc.dtype:
                changes.append({"type": "breaking",
                                "detail": f"Column '{name}' dtype changed {oc.dtype}->{nc.dtype}"})
            if oc.nullable and not nc.nullable:
                changes.append({"type": "breaking",
                                "detail": f"Column '{name}' changed from nullable to non-nullable"})
            if not oc.nullable and nc.nullable:
                changes.append({"type": "non-breaking",
                                "detail": f"Column '{name}' changed from non-nullable to nullable"})
    for name in new_cols:
        if name not in old_cols:
            changes.append({"type": "non-breaking", "detail": f"Column '{name}' added"})
    if old.version != new.version:
        changes.append({"type": "info", "detail": f"Version changed {old.version}->{new.version}"})
    return changes


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def print_report(results: List[Dict[str, Any]]) -> None:
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed
    print(f"\n{'='*60}")
    print(f"  Validation Report  |  Passed: {passed}  Failed: {failed}  Total: {total}")
    print(f"{'='*60}")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['rule']:40s} {r['detail']}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Data contract enforcement tool")
    parser.add_argument("--action", choices=["validate", "generate", "compare"], required=True)
    parser.add_argument("--data", help="Path to CSV data file")
    parser.add_argument("--contract", help="Path to contract YAML file")
    parser.add_argument("--contract-v2", help="Second contract for comparison")
    parser.add_argument("--output", help="Output path for generated contract")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.action == "validate":
        if not args.data or not args.contract:
            parser.error("--data and --contract are required for validate")
        contract = load_contract(args.contract)
        results = validate(contract, args.data)
        print_report(results)
        if args.output:
            with open(args.output, "w") as fh:
                json.dump(results, fh, indent=2)
            logger.info("Report written to %s", args.output)
        if any(not r["passed"] for r in results):
            sys.exit(1)

    elif args.action == "generate":
        if not args.data:
            parser.error("--data is required for generate")
        contract = generate_contract(args.data)
        out = args.output or "contract_draft.yaml"
        save_contract(contract, out)

    elif args.action == "compare":
        if not args.contract or not args.contract_v2:
            parser.error("--contract and --contract-v2 are required for compare")
        old = load_contract(args.contract)
        new = load_contract(args.contract_v2)
        changes = compare_contracts(old, new)
        breaking = [c for c in changes if c["type"] == "breaking"]
        print(f"\nChanges detected: {len(changes)}  (breaking: {len(breaking)})")
        for c in changes:
            print(f"  [{c['type'].upper():12s}] {c['detail']}")
        if breaking:
            logger.warning("Breaking changes detected between contract versions")
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
