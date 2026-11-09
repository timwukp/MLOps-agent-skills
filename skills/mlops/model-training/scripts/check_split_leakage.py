#!/usr/bin/env python3
"""
Check train/validation/test splits for temporal and data leakage.

Inspects the frames that were actually used, independent of the framework or code that
built them. Splits are given in chronological order (earliest first). Checks:

  time_order              every row of an earlier split precedes the next split (with --gap)
  label_maturity          earlier splits' labels were known by the next split's first time
  entity_overlap          entities shared between splits (fails with --disjoint-entities)
  duplicate_keys          repeated entity/time keys within a split
  cross_split_duplicates  identical rows (ignoring the time column) in two splits
  target_proxy            single numeric features that almost determine the target

A check without the settings it needs is reported as skipped, never as passed. The two
heuristic checks (cross_split_duplicates, target_proxy) warn instead of failing.
Run once per fold for a multi-fold backtest. Background and rationale:
references/TEMPORAL_VALIDATION.md in this skill.

Usage:
    # Separate files in chronological order
    python check_split_leakage.py --split train=train.parquet --split valid=valid.parquet \\
        --split test=test.parquet --time-column timestamp --entity-column device_id \\
        --target fault --label-window 24h --reporting-delay 2h

    # One file with a split column
    python check_split_leakage.py --input dataset.csv --split-column split \\
        --order train,valid,test --time-column timestamp --target label

    # Tabular split without time: duplicates, entity overlap and target proxies only
    python check_split_leakage.py --split train=train.csv --split test=test.csv \\
        --target churned --entity-column customer_id --disjoint-entities

Exit code: 0 pass/warn, 1 any failure (or any warning with --strict), 2 invalid usage.
Dependencies: pandas (plus pyarrow or fastparquet for Parquet input).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

TZ_SUFFIX = r"(?:Z|[+-]\d{2}:?\d{2})$"


def _result(check, status, detail, **extra):
    return {"check": check, "status": status, "detail": detail, **extra}


def _parse_times(splits, column, allow_missing=False):
    """Parse a time column in every split to UTC; reject mixed naive/aware input."""
    import pandas as pd
    from pandas.api import types

    aware, parsed = set(), {}
    for name, frame in splits.items():
        if column not in frame:
            raise ValueError(f"column {column!r} missing from split {name!r}")
        raw = frame[column]
        if types.is_numeric_dtype(raw):
            raise ValueError(f"{column!r} is numeric; convert it to timestamps first")
        if isinstance(raw.dtype, pd.DatetimeTZDtype):
            aware.add(True)
        elif types.is_datetime64_dtype(raw):
            aware.add(False)
        else:
            text = raw.dropna().astype(str).str.strip()
            aware.update(text.str.contains(TZ_SUFFIX).unique().tolist())
        if len(aware) > 1:
            raise ValueError(
                f"{column!r} mixes timezone-aware and naive timestamps; normalize first"
            )
        values = pd.to_datetime(raw, utc=True)
        if not allow_missing and values.isna().any():
            raise ValueError(f"missing or unparseable {column!r} values in split {name!r}")
        parsed[name] = values
    return parsed


def _pairs(names):
    return list(zip(names, names[1:]))


def _time_order(splits, times, gap):
    violations = []
    for a, b in _pairs(list(splits)):
        if times[a].empty or times[b].empty:
            continue
        cutoff = times[b].min()
        bad = int((times[a] >= cutoff - gap).sum())
        if bad:
            violations.append({
                "earlier": a, "later": b, "rows": bad,
                "earlier_max": str(times[a].max()), "later_min": str(cutoff),
            })
    if violations:
        return _result(
            "time_order", "fail",
            "rows of an earlier split are at or after the next split's start minus the gap; "
            "per-entity cutoffs or random splits put future data in training",
            violations=violations,
        )
    return _result("time_order", "pass", f"splits are ordered in time (gap {gap})")


def _label_maturity(splits, times, known):
    violations = []
    for a, b in _pairs(list(splits)):
        if times[b].empty:
            continue
        cutoff = times[b].min()
        bad = int((known[a].isna() | (known[a] > cutoff)).sum())
        if bad:
            violations.append({
                "split": a, "cutoff": str(cutoff), "rows": bad,
                "latest_known": str(known[a].max()),
            })
    if violations:
        return _result(
            "label_maturity", "fail",
            "labels were not yet known when the next split starts; drop or purge these rows",
            violations=violations,
        )
    return _result("label_maturity", "pass", "all labels were known before each next split")


def _entity_overlap(splits, entities, disjoint):
    names = list(splits)
    shared = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            left = splits[a][entities].drop_duplicates()
            right = splits[b][entities].drop_duplicates()
            count = len(left.merge(right, on=entities))
            if count:
                shared.append({"splits": [a, b], "shared_entities": count})
    if shared and disjoint:
        return _result(
            "entity_overlap", "fail",
            "entities appear in more than one split but --disjoint-entities was required",
            overlaps=shared,
        )
    detail = "no shared entities" if not shared else "entities shared (allowed)"
    return _result("entity_overlap", "pass", detail, overlaps=shared)


def _duplicate_keys(splits, keys):
    counts = {name: int(frame.duplicated(keys).sum()) for name, frame in splits.items()}
    found = {name: count for name, count in counts.items() if count}
    if found:
        return _result(
            "duplicate_keys", "warn",
            f"repeated {keys} keys; check for join fan-out or unaggregated records",
            duplicates=found,
        )
    return _result("duplicate_keys", "pass", f"{keys} keys are unique within each split")


def _cross_split_duplicates(splits, ignore):
    import pandas as pd

    names = list(splits)
    common = set(splits[names[0]].columns)
    for name in names[1:]:
        common &= set(splits[name].columns)
    columns = sorted(common - set(ignore))
    if not columns:
        return _result("cross_split_duplicates", "skipped", "no shared columns to compare")
    rows = {
        name: pd.util.hash_pandas_object(frame[columns], index=False)
        for name, frame in splits.items()
    }
    found = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            count = int(rows[b].isin(rows[a]).sum())
            if count:
                found.append({"earlier": a, "later": b, "rows": count,
                              "fraction_of_later": round(count / len(rows[b]), 4)})
    if found:
        return _result(
            "cross_split_duplicates", "warn",
            "identical rows (all compared columns, time ignored) occur in two splits; "
            "expected for repeated constant readings, a leak for copied records",
            columns=columns, duplicates=found,
        )
    return _result("cross_split_duplicates", "pass", "no identical rows across splits",
                   columns=columns)


def _target_proxy(frame, target, ignore, threshold, split_name):
    from pandas.api import types

    y = frame[target]
    values = y.dropna().unique()
    binary = len(values) == 2
    if not binary and not types.is_numeric_dtype(y):
        return _result("target_proxy", "skipped",
                       "target is neither binary nor numeric (multiclass labels)")
    positive = sorted(values, key=str)[1] if binary else None
    flagged, evaluated = [], 0
    for column in frame.columns:
        if column in ignore or not types.is_numeric_dtype(frame[column]):
            continue
        pair = frame[[column, target]].dropna()
        x = pair[column].astype(float)
        if len(pair) < 10 or x.nunique() < 2:
            continue
        evaluated += 1
        if binary:
            is_pos = pair[target] == positive
            n1, n0 = int(is_pos.sum()), int((~is_pos).sum())
            if not n1 or not n0:
                continue
            auc = (x.rank()[is_pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
            score, metric = max(auc, 1 - auc), "auc"
        else:
            score, metric = abs(x.rank().corr(pair[target].rank())), "abs_spearman"
        if score == score and score >= threshold:
            flagged.append({"column": column, metric: round(float(score), 4)})
    if flagged:
        return _result(
            "target_proxy", "warn",
            f"single features in {split_name!r} nearly determine the target; confirm they are "
            "available at prediction time and not written by the outcome process",
            threshold=threshold, features=flagged, evaluated=evaluated,
        )
    return _result("target_proxy", "pass",
                   f"no single numeric feature reaches {threshold}", evaluated=evaluated)


def check_splits(
    splits,
    time_column=None,
    entity_columns=(),
    target=None,
    label_known_column=None,
    label_window=None,
    reporting_delay=None,
    gap=None,
    disjoint_entities=False,
    exclude_columns=(),
    proxy_threshold=0.98,
):
    """Check ordered {name: DataFrame} splits; return a JSON-serializable report."""
    import pandas as pd

    entity_columns = list(entity_columns)
    if len(splits) < 2:
        raise ValueError("provide at least two splits, earliest first")
    if label_known_column and label_window:
        raise ValueError("use either label_known_column or label_window, not both")
    if reporting_delay and not label_window:
        raise ValueError("reporting_delay requires label_window")
    if not time_column and (label_known_column or label_window or gap):
        raise ValueError("label timing and gap require time_column")
    if disjoint_entities and not entity_columns:
        raise ValueError("disjoint_entities requires entity_columns")
    for name, frame in splits.items():
        missing = [c for c in [*entity_columns, target] if c and c not in frame]
        if missing:
            raise ValueError(f"split {name!r} lacks columns {missing}")

    checks = []
    times = None
    if time_column:
        try:
            times = _parse_times(splits, time_column)
        except (ValueError, TypeError) as exc:
            checks.append(_result("time_parse", "fail", str(exc)))
    if times is not None:
        checks.append(_time_order(splits, times, pd.Timedelta(gap or 0)))
        known = None
        try:
            if label_known_column:
                known = _parse_times(splits, label_known_column, allow_missing=True)
            elif label_window:
                delay = pd.Timedelta(label_window) + pd.Timedelta(reporting_delay or 0)
                known = {name: values + delay for name, values in times.items()}
        except (ValueError, TypeError) as exc:
            checks.append(_result("label_maturity", "fail", str(exc)))
        else:
            if known is None:
                checks.append(_result(
                    "label_maturity", "skipped",
                    "label timing unknown; pass --label-window or --label-known-column",
                ))
            else:
                checks.append(_label_maturity(splits, times, known))
        checks.append(_duplicate_keys(splits, [*entity_columns, time_column]))
    elif not time_column:
        for name in ["time_order", "label_maturity"]:
            checks.append(_result(name, "skipped", "no --time-column given"))

    if entity_columns:
        checks.append(_entity_overlap(splits, entity_columns, disjoint_entities))
    else:
        checks.append(_result("entity_overlap", "skipped", "no --entity-column given"))

    ignore = {time_column, label_known_column, *exclude_columns} - {None}
    checks.append(_cross_split_duplicates(splits, ignore))

    if target:
        first = next(iter(splits))
        proxy_ignore = ignore | set(entity_columns) | {target}
        checks.append(
            _target_proxy(splits[first], target, proxy_ignore, proxy_threshold, first)
        )
    else:
        checks.append(_result("target_proxy", "skipped", "no --target given"))

    statuses = {c["status"] for c in checks}
    status = "fail" if "fail" in statuses else "warn" if "warn" in statuses else "pass"
    return {
        "status": status,
        "skipped": [c["check"] for c in checks if c["status"] == "skipped"],
        "splits": {name: {"rows": len(frame)} for name, frame in splits.items()},
        "settings": {
            "time_column": time_column, "entity_columns": entity_columns,
            "target": target, "label_known_column": label_known_column,
            "label_window": label_window, "reporting_delay": reporting_delay,
            "gap": gap, "disjoint_entities": disjoint_entities,
            "exclude_columns": list(exclude_columns), "proxy_threshold": proxy_threshold,
            "timestamps": "compared in UTC; naive timestamps are read as UTC",
        },
        "checks": checks,
    }


def _read(path):
    import pandas as pd

    suffix = Path(path).suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported file type: {path} (use .csv or .parquet)")


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_splits(args):
    """Return ordered splits and {path: sha256} for the files read."""
    if args.input:
        if args.split or not args.split_column or not args.order:
            raise ValueError("--input needs --split-column and --order, and no --split")
        frame = _read(args.input)
        order = [name.strip() for name in args.order.split(",") if name.strip()]
        values = set(frame[args.split_column].astype(str))
        unknown = values - set(order)
        if unknown:
            raise ValueError(f"split values not listed in --order: {sorted(unknown)}")
        labels = frame[args.split_column].astype(str)
        splits = {
            name: frame[labels == name].drop(columns=args.split_column) for name in order
        }
        return splits, {args.input: _sha256(args.input)}
    splits, files = {}, {}
    for item in args.split:
        name, sep, path = item.partition("=")
        if not sep or not name or not path:
            raise ValueError(f"--split expects NAME=PATH, got {item!r}")
        if name in splits:
            raise ValueError(f"duplicate split name {name!r}")
        splits[name] = _read(path)
        files[path] = _sha256(path)
    return splits, files


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_argument_group("input")
    source.add_argument("--split", action="append", default=[], metavar="NAME=PATH",
                        help="split file, repeated in chronological order")
    source.add_argument("--input", help="single CSV/Parquet file with a split column")
    source.add_argument("--split-column", help="column naming each row's split")
    source.add_argument("--order", help="comma-separated split names, earliest first")
    parser.add_argument("--time-column", help="prediction time of each row")
    parser.add_argument("--entity-column", action="append", default=[],
                        help="entity key column; repeat for composite keys")
    parser.add_argument("--target", help="target/label column")
    parser.add_argument("--label-known-column",
                        help="column with the time each row's label became known")
    parser.add_argument("--label-window",
                        help="label window after the prediction time, e.g. 24h")
    parser.add_argument("--reporting-delay",
                        help="delay before a label is recorded, e.g. 2h")
    parser.add_argument("--gap", help="required gap between consecutive splits, e.g. 6h")
    parser.add_argument("--disjoint-entities", action="store_true",
                        help="fail if any entity appears in two splits")
    parser.add_argument("--exclude-column", action="append", default=[],
                        help="column ignored by duplicate and proxy checks; repeatable")
    parser.add_argument("--proxy-threshold", type=float, default=0.98,
                        help="single-feature AUC or |Spearman| that triggers a warning")
    parser.add_argument("--strict", action="store_true", help="exit 1 on warnings too")
    parser.add_argument("--output", help="write the JSON report here instead of stdout")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.input and not args.split:
        parser.error("give --split NAME=PATH (at least twice) or --input")
    try:
        splits, files = load_splits(args)
        report = check_splits(
            splits,
            time_column=args.time_column,
            entity_columns=args.entity_column,
            target=args.target,
            label_known_column=args.label_known_column,
            label_window=args.label_window,
            reporting_delay=args.reporting_delay,
            gap=args.gap,
            disjoint_entities=args.disjoint_entities,
            exclude_columns=args.exclude_column,
            proxy_threshold=args.proxy_threshold,
        )
    except (OSError, KeyError, ValueError) as exc:
        parser.error(str(exc))
    report["files"] = files
    text = json.dumps(report, indent=2, default=str)
    if args.output:
        Path(args.output).write_text(text + "\n")
    else:
        print(text)
    failed = report["status"] == "fail" or (args.strict and report["status"] == "warn")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
