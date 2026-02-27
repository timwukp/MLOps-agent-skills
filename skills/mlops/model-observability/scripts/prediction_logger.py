#!/usr/bin/env python3
"""Prediction logging and audit trail for ML models.

Usage:
    python prediction_logger.py --action log --db-path preds.db \\
        --input '{"age":35,"income":50000}' --prediction 1 --probability 0.87 \\
        --model-version v2.1 --latency 12.5
    python prediction_logger.py --action query --db-path preds.db \\
        --start-date 2025-01-01 --end-date 2025-01-31 --model-version v2.1
    python prediction_logger.py --action stats --db-path preds.db
    python prediction_logger.py --action anomalies --db-path preds.db --output anomalies.json
    python prediction_logger.py --action export --db-path preds.db --output predictions.parquet
"""
import argparse
import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class PredictionLogger:
    """Log, query, and analyse model predictions with SQLite + optional JSONL."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS predictions (
        request_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL,
        input_features TEXT NOT NULL, prediction TEXT NOT NULL,
        probability REAL, model_version TEXT, latency_ms REAL
    );
    CREATE INDEX IF NOT EXISTS idx_ts ON predictions(timestamp);
    CREATE INDEX IF NOT EXISTS idx_mv ON predictions(model_version);
    """

    def __init__(self, db_path="predictions.db"):
        self.db_path = db_path
        with self._conn() as conn:
            conn.executescript(self.SCHEMA)
        logger.info(f"Database ready at {self.db_path}")

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def log(self, input_features, prediction, probability=None,
            model_version=None, latency_ms=None, request_id=None):
        """Record a single prediction to SQLite."""
        request_id = request_id or str(uuid.uuid4())
        ts = datetime.now(timezone.utc).isoformat()
        feat_json = json.dumps(input_features) if isinstance(input_features, dict) else str(input_features)
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO predictions (request_id,timestamp,input_features,prediction,"
                "probability,model_version,latency_ms) VALUES (?,?,?,?,?,?,?)",
                (request_id, ts, feat_json, str(prediction), probability, model_version, latency_ms),
            )
        logger.info(f"Logged prediction {request_id} (model={model_version})")
        return request_id

    def log_jsonl(self, input_features, prediction, probability=None,
                  model_version=None, latency_ms=None, request_id=None, jsonl_path=None):
        """Append a prediction record to a JSON-lines file."""
        jsonl_path = jsonl_path or self.db_path.replace(".db", ".jsonl")
        request_id = request_id or str(uuid.uuid4())
        record = {
            "request_id": request_id, "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_features": input_features, "prediction": prediction,
            "probability": probability, "model_version": model_version, "latency_ms": latency_ms,
        }
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
        logger.info(f"Appended prediction to {jsonl_path}")
        return request_id

    def query(self, start_date=None, end_date=None, model_version=None,
              min_confidence=None, max_confidence=None, limit=1000):
        """Retrieve predictions matching the given filters."""
        clauses, params = [], []
        if start_date:
            clauses.append("timestamp >= ?"); params.append(start_date)
        if end_date:
            clauses.append("timestamp <= ?"); params.append(end_date)
        if model_version:
            clauses.append("model_version = ?"); params.append(model_version)
        if min_confidence is not None:
            clauses.append("probability >= ?"); params.append(min_confidence)
        if max_confidence is not None:
            clauses.append("probability <= ?"); params.append(max_confidence)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM predictions{where} ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
        records = [dict(r) for r in rows]
        logger.info(f"Query returned {len(records)} prediction(s)")
        return records

    def stats(self, start_date=None, end_date=None):
        """Compute summary statistics: volume, confidence distribution, class counts."""
        records = self.query(start_date=start_date, end_date=end_date, limit=10_000_000)
        if not records:
            logger.warning("No predictions found for stats"); return {}
        df = pd.DataFrame(records)
        df["probability"] = pd.to_numeric(df["probability"], errors="coerce")
        df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        summary = {
            "total_predictions": len(df),
            "date_range": {"min": str(df["timestamp"].min()), "max": str(df["timestamp"].max())},
            "model_versions": df["model_version"].value_counts().to_dict(),
            "prediction_class_distribution": df["prediction"].value_counts().to_dict(),
        }
        if df["probability"].notna().any():
            prob = df["probability"].dropna()
            summary["confidence"] = {
                "mean": float(prob.mean()), "std": float(prob.std()),
                "p5": float(prob.quantile(0.05)), "p50": float(prob.quantile(0.50)),
                "p95": float(prob.quantile(0.95)),
            }
        if df["latency_ms"].notna().any():
            lat = df["latency_ms"].dropna()
            summary["latency_ms"] = {
                "mean": float(lat.mean()), "p50": float(lat.quantile(0.50)),
                "p95": float(lat.quantile(0.95)), "p99": float(lat.quantile(0.99)),
            }
        df.set_index("timestamp", inplace=True)
        hourly = df.resample("h").size()
        summary["volume_hourly"] = {str(k): int(v) for k, v in hourly.items() if v > 0}
        logger.info(f"Stats: {summary['total_predictions']} total predictions")
        return summary

    def detect_anomalies(self, contamination=0.05, min_records=20):
        """Flag anomalous predictions using Isolation Forest on numeric features."""
        records = self.query(limit=10_000_000)
        if len(records) < min_records:
            logger.warning(f"Only {len(records)} records; need >= {min_records} for anomaly detection")
            return []
        df = pd.DataFrame(records)
        df["probability"] = pd.to_numeric(df["probability"], errors="coerce")
        df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce")
        feat_dicts = []
        for raw in df["input_features"]:
            try:
                feat_dicts.append(json.loads(raw))
            except (json.JSONDecodeError, TypeError):
                feat_dicts.append({})
        feat_df = pd.DataFrame(feat_dicts).apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
        numeric = pd.concat([df[["probability", "latency_ms"]], feat_df], axis=1).fillna(0)

        from sklearn.ensemble import IsolationForest
        iso = IsolationForest(contamination=contamination, random_state=42)
        labels = iso.fit_predict(numeric)
        scores = iso.decision_function(numeric)
        anomalies = []
        for idx in np.where(labels == -1)[0]:
            rec = records[idx].copy(); rec["anomaly_score"] = float(scores[idx])
            anomalies.append(rec)
        anomalies.sort(key=lambda r: r["anomaly_score"])
        logger.info(f"Detected {len(anomalies)} anomalous prediction(s) out of {len(records)}")
        return anomalies

    def export(self, output_path, start_date=None, end_date=None, model_version=None):
        """Export predictions to CSV or Parquet."""
        records = self.query(start_date=start_date, end_date=end_date,
                             model_version=model_version, limit=10_000_000)
        if not records:
            logger.warning("No records to export"); return
        df = pd.DataFrame(records)
        if output_path.endswith(".parquet"):
            df.to_parquet(output_path, index=False)
        else:
            df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(df)} records to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prediction logging and audit trail")
    parser.add_argument("--action", required=True, choices=["log", "query", "stats", "anomalies", "export"])
    parser.add_argument("--db-path", default="predictions.db", help="SQLite database path")
    parser.add_argument("--start-date", default=None, help="Start date filter (ISO format)")
    parser.add_argument("--end-date", default=None, help="End date filter (ISO format)")
    parser.add_argument("--model-version", default=None, help="Model version filter")
    parser.add_argument("--output", default=None, help="Output file path")
    parser.add_argument("--input", default=None, help="Input features as JSON string")
    parser.add_argument("--prediction", default=None, help="Prediction value")
    parser.add_argument("--probability", type=float, default=None, help="Prediction probability")
    parser.add_argument("--latency", type=float, default=None, help="Prediction latency in ms")
    parser.add_argument("--jsonl", action="store_true", help="Also write to JSON-lines file")
    parser.add_argument("--contamination", type=float, default=0.05, help="Anomaly fraction (default: 0.05)")
    args = parser.parse_args()
    pl = PredictionLogger(db_path=args.db_path)

    if args.action == "log":
        if not args.input or args.prediction is None:
            parser.error("--input and --prediction are required for action=log")
        try:
            features = json.loads(args.input)
        except json.JSONDecodeError:
            features = args.input
        rid = pl.log(features, args.prediction, probability=args.probability,
                     model_version=args.model_version, latency_ms=args.latency)
        if args.jsonl:
            pl.log_jsonl(features, args.prediction, probability=args.probability,
                         model_version=args.model_version, latency_ms=args.latency, request_id=rid)
        print(json.dumps({"request_id": rid}))
    elif args.action in ("query", "stats", "anomalies"):
        if args.action == "query":
            data = pl.query(start_date=args.start_date, end_date=args.end_date,
                            model_version=args.model_version)
        elif args.action == "stats":
            data = pl.stats(start_date=args.start_date, end_date=args.end_date)
        else:
            data = pl.detect_anomalies(contamination=args.contamination)
        out = json.dumps(data, indent=2, default=str)
        if args.output:
            with open(args.output, "w") as f: f.write(out)
        else:
            print(out)
    elif args.action == "export":
        if not args.output:
            parser.error("--output is required for action=export")
        pl.export(args.output, start_date=args.start_date, end_date=args.end_date,
                  model_version=args.model_version)


if __name__ == "__main__":
    main()
