#!/usr/bin/env python3
"""Continuous drift monitoring daemon for MLOps pipelines.

Periodically compares a reference dataset against incoming data windows,
computes Population Stability Index (PSI) and Kolmogorov-Smirnov (KS)
statistics per feature, records drift scores to a JSON-lines history
file, analyses drift trends, and optionally triggers retraining when
thresholds are breached.

Typical usage:
    python drift_monitor.py --reference ref.csv --current-dir ./incoming \
        --interval 5 --threshold 0.25 --history-file drift_history.jsonl

    python drift_monitor.py --reference ref.csv --current-dir ./incoming \
        --interval 10 --threshold 0.20 --webhook-url https://hooks.example.com/drift \
        --retrain-signal retrain.flag
"""

import argparse
import csv
import glob
import json
import logging
import math
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("drift_monitor")

EPSILON = 1e-10  # smoothing constant for PSI

# ---------------------------------------------------------------------------
# Statistical helpers (self-contained, no numpy/scipy dependency)
# ---------------------------------------------------------------------------

def _histogram(values: List[float], bins: int = 10):
    """Return bin edges and normalised frequencies."""
    if not values:
        return [], []
    lo, hi = min(values), max(values)
    if lo == hi:
        hi = lo + 1.0
    width = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        idx = min(int((v - lo) / width), bins - 1)
        counts[idx] += 1
    total = len(values)
    freqs = [(c / total) if total else 0 for c in counts]
    edges = [lo + i * width for i in range(bins + 1)]
    return edges, freqs


def compute_psi(reference: List[float], current: List[float], bins: int = 10) -> float:
    """Population Stability Index between two distributions."""
    _, ref_freq = _histogram(reference, bins)
    _, cur_freq = _histogram(current, bins)
    psi = 0.0
    for r, c in zip(ref_freq, cur_freq):
        r = max(r, EPSILON)
        c = max(c, EPSILON)
        psi += (c - r) * math.log(c / r)
    return psi


def compute_ks(reference: List[float], current: List[float]) -> float:
    """Two-sample Kolmogorov-Smirnov statistic."""
    all_vals = sorted(set(reference + current))
    if not all_vals:
        return 0.0
    n_ref, n_cur = len(reference), len(current)
    ref_sorted = sorted(reference)
    cur_sorted = sorted(current)

    def ecdf_val(sorted_data, n, x):
        count = 0
        for v in sorted_data:
            if v <= x:
                count += 1
            else:
                break
        return count / n if n else 0

    max_diff = 0.0
    for x in all_vals:
        diff = abs(ecdf_val(ref_sorted, n_ref, x) - ecdf_val(cur_sorted, n_cur, x))
        if diff > max_diff:
            max_diff = diff
    return max_diff


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _read_csv_numeric(path: str) -> Dict[str, List[float]]:
    """Read a CSV file and return numeric columns as lists of floats."""
    columns: Dict[str, List[float]] = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            for k, v in row.items():
                try:
                    columns.setdefault(k, []).append(float(v))
                except (ValueError, TypeError):
                    pass  # skip non-numeric
    return columns


def _latest_files(directory: str, window_size: int) -> List[str]:
    """Return the most recent *window_size* CSV files from *directory*."""
    pattern = os.path.join(directory, "*.csv")
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return files[:window_size]


def _merge_csvs(paths: List[str]) -> Dict[str, List[float]]:
    merged: Dict[str, List[float]] = {}
    for p in paths:
        for col, vals in _read_csv_numeric(p).items():
            merged.setdefault(col, []).extend(vals)
    return merged


# ---------------------------------------------------------------------------
# Drift monitor
# ---------------------------------------------------------------------------

@dataclass
class DriftResult:
    timestamp: str
    column: str
    psi: float
    ks: float
    drifted: bool


class DriftMonitor:
    """Periodically compare reference data against a sliding window of current data."""

    def __init__(self, reference_path: str, current_dir: str, threshold: float = 0.25,
                 window_size: int = 5, history_file: str = "drift_history.jsonl",
                 webhook_url: Optional[str] = None, retrain_signal: Optional[str] = None):
        self.reference = _read_csv_numeric(reference_path)
        self.current_dir = current_dir
        self.threshold = threshold
        self.window_size = window_size
        self.history_file = history_file
        self.webhook_url = webhook_url
        self.retrain_signal = retrain_signal

    def check_once(self) -> List[DriftResult]:
        """Run one round of drift detection against the current window."""
        files = _latest_files(self.current_dir, self.window_size)
        if not files:
            logger.warning("No CSV files found in %s", self.current_dir)
            return []
        current = _merge_csvs(files)
        ts = datetime.now(timezone.utc).isoformat()
        results: List[DriftResult] = []
        any_drift = False
        for col, ref_vals in self.reference.items():
            cur_vals = current.get(col)
            if not cur_vals:
                logger.debug("Column '%s' missing from current window, skipping", col)
                continue
            psi = compute_psi(ref_vals, cur_vals)
            ks = compute_ks(ref_vals, cur_vals)
            drifted = psi > self.threshold or ks > self.threshold
            results.append(DriftResult(ts, col, round(psi, 6), round(ks, 6), drifted))
            if drifted:
                any_drift = True
                logger.warning("Drift detected on '%s': PSI=%.4f KS=%.4f", col, psi, ks)
        self._append_history(results)
        if any_drift:
            self._fire_alert(results)
            self._write_retrain_signal(results)
        return results

    def run_loop(self, interval_minutes: float) -> None:
        """Blocking loop that calls *check_once* every *interval_minutes*."""
        logger.info("Starting drift monitor loop (interval=%s min, threshold=%s)",
                     interval_minutes, self.threshold)
        try:
            while True:
                self.check_once()
                time.sleep(interval_minutes * 60)
        except KeyboardInterrupt:
            logger.info("Drift monitor stopped by user")

    # -- persistence --

    def _append_history(self, results: List[DriftResult]) -> None:
        with open(self.history_file, "a") as fh:
            for r in results:
                fh.write(json.dumps(r.__dict__) + "\n")

    # -- alerting --

    def _fire_alert(self, results: List[DriftResult]) -> None:
        drifted = [r for r in results if r.drifted]
        msg = f"Drift alert: {len(drifted)} column(s) exceeded threshold {self.threshold}"
        for r in drifted:
            msg += f"\n  {r.column}: PSI={r.psi} KS={r.ks}"
        logger.warning(msg)
        if self.webhook_url:
            payload = json.dumps({"text": msg, "results": [r.__dict__ for r in drifted]}).encode()
            req = urllib.request.Request(self.webhook_url, data=payload,
                                        headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    logger.info("Webhook responded %s", resp.status)
            except urllib.error.URLError as exc:
                logger.error("Webhook notification failed: %s", exc)

    def _write_retrain_signal(self, results: List[DriftResult]) -> None:
        if not self.retrain_signal:
            return
        drifted = [r for r in results if r.drifted]
        signal = {"timestamp": datetime.now(timezone.utc).isoformat(),
                  "reason": "drift_threshold_exceeded",
                  "columns": [r.column for r in drifted]}
        with open(self.retrain_signal, "w") as fh:
            json.dump(signal, fh, indent=2)
        logger.info("Retrain signal written to %s", self.retrain_signal)


# ---------------------------------------------------------------------------
# Trend analysis
# ---------------------------------------------------------------------------

def analyse_trends(history_file: str, last_n: int = 10) -> Dict[str, str]:
    """Read recent history and report whether drift is increasing or decreasing per column."""
    if not os.path.exists(history_file):
        return {}
    records: List[Dict[str, Any]] = []
    with open(history_file) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    col_scores: Dict[str, List[float]] = {}
    for rec in records:
        col_scores.setdefault(rec["column"], []).append(rec["psi"])
    trends: Dict[str, str] = {}
    for col, scores in col_scores.items():
        recent = scores[-last_n:]
        if len(recent) < 2:
            trends[col] = "insufficient_data"
            continue
        mid = len(recent) // 2
        first_half = sum(recent[:mid]) / mid
        second_half = sum(recent[mid:]) / (len(recent) - mid)
        if second_half > first_half * 1.1:
            trends[col] = "increasing"
        elif second_half < first_half * 0.9:
            trends[col] = "decreasing"
        else:
            trends[col] = "stable"
    return trends


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Continuous drift monitoring daemon")
    p.add_argument("--reference", required=True, help="Path to reference CSV")
    p.add_argument("--current-dir", required=True, help="Directory with timestamped CSV files")
    p.add_argument("--interval", type=float, default=5, help="Check interval in minutes")
    p.add_argument("--window-size", type=int, default=5, help="Number of recent files to merge")
    p.add_argument("--threshold", type=float, default=0.25, help="Drift threshold (PSI/KS)")
    p.add_argument("--webhook-url", help="Webhook URL for drift alerts")
    p.add_argument("--history-file", default="drift_history.jsonl", help="JSONL history path")
    p.add_argument("--retrain-signal", help="Path to write retrain signal JSON file")
    p.add_argument("--once", action="store_true", help="Run a single check then exit")
    p.add_argument("--trends", action="store_true", help="Analyse and print drift trends then exit")
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.trends:
        trends = analyse_trends(args.history_file)
        if not trends:
            print("No history data available for trend analysis.")
        for col, trend in trends.items():
            print(f"  {col:30s} -> {trend}")
        return

    monitor = DriftMonitor(
        reference_path=args.reference,
        current_dir=args.current_dir,
        threshold=args.threshold,
        window_size=args.window_size,
        history_file=args.history_file,
        webhook_url=args.webhook_url,
        retrain_signal=args.retrain_signal,
    )

    if args.once:
        results = monitor.check_once()
        for r in results:
            status = "DRIFT" if r.drifted else "OK"
            print(f"  [{status:5s}] {r.column:30s} PSI={r.psi:.4f} KS={r.ks:.4f}")
        if any(r.drifted for r in results):
            sys.exit(1)
    else:
        monitor.run_loop(args.interval)


if __name__ == "__main__":
    main()
