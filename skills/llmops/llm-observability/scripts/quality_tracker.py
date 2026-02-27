#!/usr/bin/env python3
"""LLM response quality tracking - collect feedback, detect degradation, run A/B analysis.

Usage:
    python quality_tracker.py --action log-feedback --request-id abc123 --rating 5 --feedback "Great"
    python quality_tracker.py --action stats --period 7d --model-filter gpt-4o
    python quality_tracker.py --action compare --variant-a v1-prompt --variant-b v2-prompt
    python quality_tracker.py --action export --period 30d --output report.json
"""
import argparse
import csv
import io
import json
import logging
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PERIOD_MAP = {"1h": 1, "1d": 24, "7d": 168, "30d": 720}


class QualityTracker:
    """Track and analyse LLM response quality via user feedback."""

    def __init__(self, db_path="quality.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS feedback ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, "
            "request_id TEXT NOT NULL, model TEXT DEFAULT '', variant TEXT DEFAULT '', "
            "use_case TEXT DEFAULT '', rating INTEGER CHECK(rating BETWEEN 1 AND 5), "
            "thumbs_up INTEGER DEFAULT NULL, comment TEXT DEFAULT '')")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_fb_ts ON feedback(timestamp)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_fb_req ON feedback(request_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_fb_var ON feedback(variant)")
        self.conn.commit()

    def _cutoff(self, period):
        return (datetime.now(timezone.utc) - timedelta(hours=PERIOD_MAP.get(period, 24))).isoformat()

    def log_feedback(self, request_id, rating=None, thumbs_up=None,
                     comment="", model="", variant="", use_case=""):
        """Record user feedback for a given request."""
        if rating is not None and not 1 <= rating <= 5:
            raise ValueError("Rating must be between 1 and 5")
        ts = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "INSERT INTO feedback (timestamp,request_id,model,variant,use_case,rating,thumbs_up,comment)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (ts, request_id, model, variant, use_case, rating, thumbs_up, comment))
        self.conn.commit()
        logger.info(f"Feedback logged for {request_id}: rating={rating} thumbs_up={thumbs_up}")

    def get_stats(self, period="7d", model_filter=None):
        """Compute satisfaction rate and average rating."""
        where, params = "WHERE timestamp >= ?", [self._cutoff(period)]
        if model_filter:
            where += " AND model = ?"; params.append(model_filter)
        row = self.conn.execute(
            f"SELECT COUNT(*) as total, COALESCE(AVG(rating),0) as avg_rating,"
            f" COALESCE(SUM(CASE WHEN thumbs_up=1 THEN 1 ELSE 0 END),0) as pos,"
            f" COALESCE(SUM(CASE WHEN thumbs_up=0 THEN 1 ELSE 0 END),0) as neg,"
            f" COALESCE(SUM(CASE WHEN rating>=4 THEN 1 ELSE 0 END),0) as high"
            f" FROM feedback {where}", params).fetchone()
        total, thumb_total = row["total"], row["pos"] + row["neg"]
        sat = (row["pos"] / thumb_total * 100) if thumb_total > 0 else None
        by_model = self.conn.execute(
            f"SELECT model, COUNT(*) as count, AVG(rating) as avg_rating"
            f" FROM feedback {where} AND model!='' GROUP BY model ORDER BY count DESC",
            params).fetchall()
        by_variant = self.conn.execute(
            f"SELECT variant, COUNT(*) as count, AVG(rating) as avg_rating"
            f" FROM feedback {where} AND variant!='' GROUP BY variant ORDER BY count DESC",
            params).fetchall()
        return {
            "period": period, "total_feedback": total,
            "avg_rating": round(row["avg_rating"], 2),
            "satisfaction_pct": round(sat, 2) if sat is not None else None,
            "high_rating_pct": round(row["high"] / total * 100, 2) if total else 0.0,
            "by_model": [dict(r) for r in by_model],
            "by_variant": [dict(r) for r in by_variant],
        }

    def detect_degradation(self, period="1d", threshold=0.5):
        """Compare recent quality against the historical baseline."""
        hours = PERIOD_MAP.get(period, 24)
        recent_cutoff = self._cutoff(period)
        baseline_cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours * 4)).isoformat()
        recent = self.conn.execute(
            "SELECT AVG(rating) as avg, COUNT(*) as n FROM feedback "
            "WHERE timestamp>=? AND rating IS NOT NULL", (recent_cutoff,)).fetchone()
        baseline = self.conn.execute(
            "SELECT AVG(rating) as avg, COUNT(*) as n FROM feedback "
            "WHERE timestamp>=? AND timestamp<? AND rating IS NOT NULL",
            (baseline_cutoff, recent_cutoff)).fetchone()
        if not recent["n"] or not baseline["n"] or not baseline["avg"]:
            return {"degradation_detected": False, "reason": "insufficient data"}
        drop = baseline["avg"] - recent["avg"]
        return {"degradation_detected": drop > threshold,
                "baseline_avg": round(baseline["avg"], 2),
                "recent_avg": round(recent["avg"], 2), "drop": round(drop, 2),
                "threshold": threshold, "baseline_n": baseline["n"], "recent_n": recent["n"]}

    def compare_variants(self, variant_a, variant_b, period="30d"):
        """Compare quality metrics between two prompt/model variants."""
        cutoff = self._cutoff(period)

        def _stats(v):
            r = self.conn.execute(
                "SELECT COUNT(*) as n, COALESCE(AVG(rating),0) as avg,"
                " COALESCE(SUM(CASE WHEN thumbs_up=1 THEN 1 ELSE 0 END),0) as pos,"
                " COALESCE(SUM(CASE WHEN thumbs_up=0 THEN 1 ELSE 0 END),0) as neg"
                " FROM feedback WHERE variant=? AND timestamp>=?", (v, cutoff)).fetchone()
            tt = r["pos"] + r["neg"]
            return {"variant": v, "count": r["n"], "avg_rating": round(r["avg"], 2),
                    "satisfaction_pct": round(r["pos"] / tt * 100, 2) if tt else None}

        a, b = _stats(variant_a), _stats(variant_b)
        winner = None
        if a["count"] >= 5 and b["count"] >= 5:
            winner = variant_a if a["avg_rating"] > b["avg_rating"] else variant_b
        return {"variant_a": a, "variant_b": b, "winner": winner, "period": period}

    def export_report(self, period="30d", fmt="json"):
        """Export quality report as JSON or CSV."""
        rows = self.conn.execute(
            "SELECT * FROM feedback WHERE timestamp>=? ORDER BY timestamp",
            (self._cutoff(period),)).fetchall()
        records = [dict(r) for r in rows]
        if fmt == "csv":
            if not records:
                return ""
            buf = io.StringIO()
            w = csv.DictWriter(buf, fieldnames=records[0].keys())
            w.writeheader(); w.writerows(records)
            return buf.getvalue()
        return json.dumps({"generated_at": datetime.now(timezone.utc).isoformat(),
                           "period": period, "stats": self.get_stats(period),
                           "degradation": self.detect_degradation(period),
                           "records": records}, indent=2)

    def close(self):
        self.conn.close()


def main():
    p = argparse.ArgumentParser(description="LLM response quality tracking")
    p.add_argument("--action", required=True, choices=["log-feedback", "stats", "compare", "export"])
    p.add_argument("--db-path", default="quality.db", help="SQLite database path")
    p.add_argument("--period", default="7d", choices=PERIOD_MAP.keys())
    p.add_argument("--output", default=None, help="Output file path")
    p.add_argument("--format", default="json", choices=["json", "csv"], help="Export format")
    p.add_argument("--request-id", default=None, help="Request ID to attach feedback to")
    p.add_argument("--rating", type=int, default=None, help="Rating 1-5")
    p.add_argument("--thumbs-up", type=int, default=None, choices=[0, 1])
    p.add_argument("--feedback", default="", help="Text feedback comment")
    p.add_argument("--model-filter", default=None, help="Filter by model")
    p.add_argument("--model", default="", help="Model name for feedback")
    p.add_argument("--variant", default="", help="Variant label for feedback")
    p.add_argument("--use-case", default="", help="Use case label")
    p.add_argument("--variant-a", default=None, help="First variant for comparison")
    p.add_argument("--variant-b", default=None, help="Second variant for comparison")
    args = p.parse_args()
    tracker = QualityTracker(args.db_path)
    try:
        if args.action == "log-feedback":
            if not args.request_id:
                logger.error("--request-id is required"); sys.exit(1)
            if args.rating is None and args.thumbs_up is None:
                logger.error("Provide --rating or --thumbs-up"); sys.exit(1)
            tracker.log_feedback(request_id=args.request_id, rating=args.rating,
                                 thumbs_up=args.thumbs_up, comment=args.feedback,
                                 model=args.model, variant=args.variant, use_case=args.use_case)
            print(json.dumps({"status": "ok", "request_id": args.request_id}))
        elif args.action == "stats":
            stats = tracker.get_stats(args.period, args.model_filter)
            deg = tracker.detect_degradation(args.period)
            print(json.dumps({"stats": stats, "degradation": deg}, indent=2))
        elif args.action == "compare":
            if not args.variant_a or not args.variant_b:
                logger.error("--variant-a and --variant-b required"); sys.exit(1)
            print(json.dumps(tracker.compare_variants(args.variant_a, args.variant_b, args.period), indent=2))
        elif args.action == "export":
            report = tracker.export_report(args.period, args.format)
            if args.output:
                Path(args.output).write_text(report)
                logger.info(f"Report exported to {args.output}")
            else:
                print(report)
    except Exception as exc:
        logger.error(f"Action '{args.action}' failed: {exc}")
        sys.exit(1)
    finally:
        tracker.close()


if __name__ == "__main__":
    main()
