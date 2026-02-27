#!/usr/bin/env python3
"""LLM observability and monitoring - log calls, track costs, detect anomalies.

Usage:
    python llm_monitor.py --action log --model gpt-4o --prompt "Hello" --response "Hi" --latency 320
    python llm_monitor.py --action stats --period 1d --model-filter gpt-4o
    python llm_monitor.py --action alerts --period 1h
    python llm_monitor.py --action export --period 7d --output dashboard.json
    python llm_monitor.py --action cleanup --period 30d
"""
import argparse
import json
import logging
import sqlite3
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Default cost per 1M tokens (input, output) in USD
DEFAULT_PRICING = {
    "gpt-4o":            (2.50,  10.00),
    "gpt-4o-mini":       (0.15,   0.60),
    "gpt-4-turbo":      (10.00,  30.00),
    "gpt-3.5-turbo":     (0.50,   1.50),
    "claude-3-opus":    (15.00,  75.00),
    "claude-3-sonnet":   (3.00,  15.00),
    "claude-3-haiku":    (0.25,   1.25),
    "claude-3.5-sonnet": (3.00,  15.00),
    "llama-3-70b":       (0.00,   0.00),
    "llama-3-8b":        (0.00,   0.00),
}
PERIOD_MAP = {"1h": 1, "1d": 24, "7d": 168, "30d": 720}


def estimate_tokens(text):
    """Estimate token count via tiktoken or word-based fallback."""
    try:
        import tiktoken
        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except ImportError:
        return max(1, int(len(text.split()) * 1.3))


def estimate_cost(model, input_tokens, output_tokens, pricing=None):
    """Estimate USD cost for a single LLM call."""
    in_r, out_r = (pricing or DEFAULT_PRICING).get(model, (0.0, 0.0))
    return (input_tokens * in_r + output_tokens * out_r) / 1_000_000


class LLMCallLogger:
    """Log and query LLM calls backed by SQLite."""

    def __init__(self, db_path="llm_calls.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS llm_calls ("
            "id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, model TEXT NOT NULL, "
            "prompt TEXT, response TEXT, input_tokens INTEGER DEFAULT 0, "
            "output_tokens INTEGER DEFAULT 0, total_tokens INTEGER DEFAULT 0, "
            "latency_ms REAL DEFAULT 0, cost_estimate REAL DEFAULT 0, "
            "status TEXT DEFAULT 'success')")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_ts ON llm_calls(timestamp)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_model ON llm_calls(model)")
        self.conn.commit()

    def _cutoff(self, period):
        hours = PERIOD_MAP.get(period, 24)
        return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()

    def log_call(self, model, prompt="", response="", input_tokens=None,
                 output_tokens=None, latency_ms=0.0, status="success"):
        """Record a single LLM call."""
        if input_tokens is None:
            input_tokens = estimate_tokens(prompt)
        if output_tokens is None:
            output_tokens = estimate_tokens(response)
        total = input_tokens + output_tokens
        cost = estimate_cost(model, input_tokens, output_tokens)
        cid = uuid.uuid4().hex[:16]
        ts = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "INSERT INTO llm_calls VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (cid, ts, model, prompt[:500], response[:500],
             input_tokens, output_tokens, total, latency_ms, cost, status))
        self.conn.commit()
        logger.info(f"Logged {cid}: model={model} tokens={total} cost=${cost:.6f}")
        return cid

    def get_stats(self, period="1d", model_filter=None):
        """Aggregate metrics for a time period."""
        where, params = "WHERE timestamp >= ?", [self._cutoff(period)]
        if model_filter:
            where += " AND model = ?"; params.append(model_filter)
        row = self.conn.execute(
            f"SELECT COUNT(*) as total_calls, COALESCE(SUM(total_tokens),0) as total_tokens,"
            f" COALESCE(SUM(cost_estimate),0) as total_cost, COALESCE(AVG(latency_ms),0) as avg_latency,"
            f" COALESCE(SUM(CASE WHEN status!='success' THEN 1 ELSE 0 END),0) as errors"
            f" FROM llm_calls {where}", params).fetchone()
        total = row["total_calls"] or 0
        err_rate = (row["errors"] / total * 100) if total > 0 else 0.0
        by_model = self.conn.execute(
            f"SELECT model, COUNT(*) as calls, SUM(total_tokens) as tokens,"
            f" SUM(cost_estimate) as cost, AVG(latency_ms) as avg_lat"
            f" FROM llm_calls {where} GROUP BY model ORDER BY calls DESC", params).fetchall()
        return {
            "period": period, "total_calls": total,
            "total_tokens": row["total_tokens"],
            "total_cost": round(row["total_cost"], 6),
            "avg_latency_ms": round(row["avg_latency"], 2),
            "error_rate_pct": round(err_rate, 2),
            "by_model": [dict(r) for r in by_model],
        }

    def check_alerts(self, period="1h"):
        """Detect latency spikes, error rate increases, cost anomalies."""
        cutoff = self._cutoff(period)
        alerts = []
        rows = self.conn.execute(
            "SELECT latency_ms FROM llm_calls WHERE timestamp>=? AND status='success'",
            (cutoff,)).fetchall()
        latencies = sorted(r["latency_ms"] for r in rows)
        if len(latencies) >= 5:
            p95 = latencies[int(len(latencies) * 0.95)]
            if any(v > p95 * 1.5 for v in latencies[-5:]):
                alerts.append({"type": "latency_spike",
                               "message": f"Recent calls exceed 1.5x p95 ({p95:.0f}ms)",
                               "p95_ms": round(p95, 1)})
        stats = self.get_stats(period)
        if stats["total_calls"] >= 10 and stats["error_rate_pct"] > 5:
            alerts.append({"type": "high_error_rate",
                           "message": f"Error rate {stats['error_rate_pct']:.1f}% exceeds 5%",
                           "error_rate_pct": stats["error_rate_pct"]})
        hours = PERIOD_MAP.get(period, 1)
        prev_cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours * 2)).isoformat()
        prev_cost = self.conn.execute(
            "SELECT COALESCE(SUM(cost_estimate),0) as c FROM llm_calls "
            "WHERE timestamp>=? AND timestamp<?", (prev_cutoff, cutoff)).fetchone()["c"]
        if prev_cost > 0 and stats["total_cost"] > prev_cost * 2:
            alerts.append({"type": "cost_anomaly",
                           "message": f"Cost ${stats['total_cost']:.4f} is >2x previous ${prev_cost:.4f}",
                           "current_cost": stats["total_cost"],
                           "previous_cost": round(prev_cost, 6)})
        return alerts

    def export_dashboard(self, period="7d", model_filter=None):
        """Export time-series data suitable for dashboard plotting."""
        where, params = "WHERE timestamp >= ?", [self._cutoff(period)]
        if model_filter:
            where += " AND model = ?"; params.append(model_filter)
        rows = self.conn.execute(
            f"SELECT strftime('%Y-%m-%dT%H:00:00', timestamp) as hour,"
            f" COUNT(*) as calls, SUM(total_tokens) as tokens,"
            f" SUM(cost_estimate) as cost, AVG(latency_ms) as avg_latency"
            f" FROM llm_calls {where} GROUP BY hour ORDER BY hour", params).fetchall()
        return {"generated_at": datetime.now(timezone.utc).isoformat(), "period": period,
                "summary": self.get_stats(period, model_filter),
                "time_series": [dict(r) for r in rows]}

    def cleanup(self, period="30d"):
        """Delete records older than the given period."""
        cur = self.conn.execute("DELETE FROM llm_calls WHERE timestamp<?", (self._cutoff(period),))
        self.conn.commit()
        logger.info(f"Deleted {cur.rowcount} records older than {period}")
        return cur.rowcount

    def close(self):
        self.conn.close()


def main():
    p = argparse.ArgumentParser(description="LLM observability and monitoring")
    p.add_argument("--action", required=True, choices=["log", "stats", "alerts", "export", "cleanup"])
    p.add_argument("--db-path", default="llm_calls.db", help="SQLite database path")
    p.add_argument("--period", default="1d", choices=PERIOD_MAP.keys(), help="Time period")
    p.add_argument("--model-filter", default=None, help="Filter by model name")
    p.add_argument("--output", default=None, help="Output file path (JSON)")
    p.add_argument("--model", default="gpt-4o", help="Model name for logging")
    p.add_argument("--prompt", default="", help="Prompt text")
    p.add_argument("--response", default="", help="Response text")
    p.add_argument("--input-tokens", type=int, default=None)
    p.add_argument("--output-tokens", type=int, default=None)
    p.add_argument("--latency", type=float, default=0.0, help="Latency in ms")
    p.add_argument("--status", default="success", help="Call status")
    args = p.parse_args()
    mon = LLMCallLogger(args.db_path)
    try:
        if args.action == "log":
            cid = mon.log_call(model=args.model, prompt=args.prompt, response=args.response,
                               input_tokens=args.input_tokens, output_tokens=args.output_tokens,
                               latency_ms=args.latency, status=args.status)
            print(json.dumps({"call_id": cid}))
        elif args.action == "stats":
            print(json.dumps(mon.get_stats(args.period, args.model_filter), indent=2))
        elif args.action == "alerts":
            alerts = mon.check_alerts(args.period)
            for a in alerts:
                logger.warning(f"[{a['type']}] {a['message']}")
            if not alerts:
                logger.info("No alerts detected.")
            print(json.dumps(alerts, indent=2))
        elif args.action == "export":
            out = json.dumps(mon.export_dashboard(args.period, args.model_filter), indent=2)
            if args.output:
                Path(args.output).write_text(out)
                logger.info(f"Dashboard exported to {args.output}")
            else:
                print(out)
        elif args.action == "cleanup":
            print(json.dumps({"deleted_records": mon.cleanup(args.period)}))
    except Exception as exc:
        logger.error(f"Action '{args.action}' failed: {exc}")
        sys.exit(1)
    finally:
        mon.close()


if __name__ == "__main__":
    main()
