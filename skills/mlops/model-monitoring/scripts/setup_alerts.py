#!/usr/bin/env python3
"""Model monitoring alerting configuration and evaluation.

Loads alert rules from a YAML configuration, evaluates incoming metrics
against those rules, fires notifications through configurable channels
(log file, webhook, email via SMTP, Slack), and persists alert history
in a lightweight SQLite database for later querying.

Typical usage:
    python setup_alerts.py --action check --config alerts.yaml --metrics '{"accuracy": 0.72}'
    python setup_alerts.py --action history --db-path alerts.db --severity critical
    python setup_alerts.py --action test-alert --channel slack --config alerts.yaml
"""

import argparse
import json
import logging
import os
import smtplib
import sqlite3
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger("setup_alerts")

SEVERITY_LEVELS = ("info", "warning", "critical")

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AlertRule:
    """Single alerting rule definition."""
    name: str
    metric: str
    condition: str  # gt, lt, eq, gte, lte, neq
    threshold: float
    severity: str = "warning"
    cooldown_minutes: int = 15
    channels: List[str] = field(default_factory=lambda: ["log"])
    message: str = ""


@dataclass
class FiredAlert:
    """Record of a fired alert."""
    rule_name: str
    metric: str
    value: float
    threshold: float
    severity: str
    timestamp: str
    message: str


# ---------------------------------------------------------------------------
# Alert manager
# ---------------------------------------------------------------------------

class AlertManager:
    """Evaluate rules against live metrics and manage cooldowns."""

    _OPS = {
        "gt": lambda v, t: v > t,
        "lt": lambda v, t: v < t,
        "eq": lambda v, t: v == t,
        "gte": lambda v, t: v >= t,
        "lte": lambda v, t: v <= t,
        "neq": lambda v, t: v != t,
    }

    def __init__(self, rules: List[AlertRule], db_path: str = "alerts.db"):
        self.rules = rules
        self.db_path = db_path
        self._last_fired: Dict[str, float] = {}
        self._init_db()

    def _init_db(self) -> None:
        con = sqlite3.connect(self.db_path)
        con.execute(
            "CREATE TABLE IF NOT EXISTS alert_history ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, rule_name TEXT, metric TEXT, "
            "value REAL, threshold REAL, severity TEXT, timestamp TEXT, message TEXT)"
        )
        con.commit()
        con.close()

    def evaluate(self, metrics: Dict[str, float]) -> List[FiredAlert]:
        """Check all rules against *metrics* and return fired alerts."""
        fired: List[FiredAlert] = []
        now = time.time()
        for rule in self.rules:
            value = metrics.get(rule.metric)
            if value is None:
                logger.debug("Metric '%s' not present, skipping rule '%s'", rule.metric, rule.name)
                continue
            op = self._OPS.get(rule.condition)
            if op is None:
                logger.warning("Unknown condition '%s' in rule '%s'", rule.condition, rule.name)
                continue
            if op(value, rule.threshold):
                last = self._last_fired.get(rule.name, 0)
                if (now - last) < rule.cooldown_minutes * 60:
                    logger.info("Rule '%s' in cooldown, skipping", rule.name)
                    continue
                self._last_fired[rule.name] = now
                ts = datetime.now(timezone.utc).isoformat()
                msg = rule.message or f"[{rule.severity.upper()}] {rule.name}: {rule.metric}={value} {rule.condition} {rule.threshold}"
                alert = FiredAlert(rule.name, rule.metric, value, rule.threshold, rule.severity, ts, msg)
                fired.append(alert)
                self._persist(alert)
        return fired

    def _persist(self, alert: FiredAlert) -> None:
        con = sqlite3.connect(self.db_path)
        con.execute(
            "INSERT INTO alert_history (rule_name, metric, value, threshold, severity, timestamp, message) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (alert.rule_name, alert.metric, alert.value, alert.threshold,
             alert.severity, alert.timestamp, alert.message),
        )
        con.commit()
        con.close()

    def query_history(self, severity: Optional[str] = None,
                      since: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve past alerts from SQLite."""
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        sql = "SELECT * FROM alert_history WHERE 1=1"
        params: List[Any] = []
        if severity:
            sql += " AND severity = ?"
            params.append(severity)
        if since:
            sql += " AND timestamp >= ?"
            params.append(since)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        rows = [dict(r) for r in con.execute(sql, params).fetchall()]
        con.close()
        return rows


# ---------------------------------------------------------------------------
# Notification channels
# ---------------------------------------------------------------------------

def notify_log(alert: FiredAlert, log_path: str = "alerts.log") -> None:
    """Append alert to a plain-text log file."""
    line = f"{alert.timestamp} | {alert.severity.upper():8s} | {alert.message}\n"
    with open(log_path, "a") as fh:
        fh.write(line)
    logger.info("Logged alert to %s", log_path)


def notify_webhook(alert: FiredAlert, url: str) -> None:
    """Send alert as JSON via HTTP POST."""
    payload = json.dumps({"text": alert.message, "severity": alert.severity,
                          "metric": alert.metric, "value": alert.value}).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            logger.info("Webhook responded %s", resp.status)
    except urllib.error.URLError as exc:
        logger.error("Webhook failed: %s", exc)


def notify_slack(alert: FiredAlert, webhook_url: str) -> None:
    """Post alert to a Slack incoming webhook."""
    payload = json.dumps({"text": alert.message}).encode()
    req = urllib.request.Request(webhook_url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            logger.info("Slack webhook responded %s", resp.status)
    except urllib.error.URLError as exc:
        logger.error("Slack notification failed: %s", exc)


def notify_email(alert: FiredAlert, smtp_host: str = "localhost", smtp_port: int = 25,
                 sender: str = "alerts@mlops.local", recipients: Optional[List[str]] = None) -> None:
    """Send alert via SMTP email."""
    recipients = recipients or ["team@mlops.local"]
    msg = MIMEText(alert.message)
    msg["Subject"] = f"[{alert.severity.upper()}] {alert.rule_name}"
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as srv:
            srv.sendmail(sender, recipients, msg.as_string())
        logger.info("Email sent to %s", recipients)
    except Exception as exc:
        logger.error("Email notification failed: %s", exc)


CHANNEL_DISPATCH = {
    "log": notify_log,
    "webhook": notify_webhook,
    "slack": notify_slack,
    "email": notify_email,
}

# ---------------------------------------------------------------------------
# YAML config loader
# ---------------------------------------------------------------------------

def _require_yaml():
    if yaml is None:
        logger.error("PyYAML is required. Install with: pip install pyyaml")
        sys.exit(1)


def load_rules(path: str) -> Dict[str, Any]:
    """Load alert rules and channel config from YAML."""
    _require_yaml()
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh)
    rules = [AlertRule(**r) for r in raw.get("rules", [])]
    return {"rules": rules, "channels": raw.get("channels", {})}


# ---------------------------------------------------------------------------
# Dispatching
# ---------------------------------------------------------------------------

def dispatch_alert(alert: FiredAlert, channels: List[str], channel_config: Dict[str, Any]) -> None:
    """Route a fired alert to the requested notification channels."""
    for ch in channels:
        cfg = channel_config.get(ch, {})
        if ch == "log":
            notify_log(alert, cfg.get("path", "alerts.log"))
        elif ch == "webhook":
            url = cfg.get("url")
            if url:
                notify_webhook(alert, url)
        elif ch == "slack":
            url = cfg.get("webhook_url")
            if url:
                notify_slack(alert, url)
        elif ch == "email":
            notify_email(alert, cfg.get("smtp_host", "localhost"), cfg.get("smtp_port", 25),
                         cfg.get("sender", "alerts@mlops.local"), cfg.get("recipients"))
        else:
            logger.warning("Unknown channel: %s", ch)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Model monitoring alert manager")
    p.add_argument("--action", choices=["check", "history", "test-alert"], required=True)
    p.add_argument("--config", help="Path to alert rules YAML")
    p.add_argument("--metrics", help="JSON string or file path of current metric values")
    p.add_argument("--db-path", default="alerts.db", help="SQLite database path")
    p.add_argument("--severity", choices=SEVERITY_LEVELS, help="Filter history by severity")
    p.add_argument("--since", help="ISO timestamp lower bound for history query")
    p.add_argument("--channel", default="log", help="Channel for test-alert (log/webhook/slack/email)")
    p.add_argument("--verbose", action="store_true")
    return p


def _load_metrics(raw: str) -> Dict[str, float]:
    if os.path.isfile(raw):
        with open(raw) as fh:
            return json.load(fh)
    return json.loads(raw)


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.action == "check":
        if not args.config or not args.metrics:
            parser.error("--config and --metrics required for check")
        cfg = load_rules(args.config)
        metrics = _load_metrics(args.metrics)
        manager = AlertManager(cfg["rules"], db_path=args.db_path)
        fired = manager.evaluate(metrics)
        if not fired:
            print("No alerts fired.")
        for alert in fired:
            print(f"  FIRED: {alert.message}")
            rule = next((r for r in cfg["rules"] if r.name == alert.rule_name), None)
            chs = rule.channels if rule else ["log"]
            dispatch_alert(alert, chs, cfg.get("channels", {}))

    elif args.action == "history":
        manager = AlertManager([], db_path=args.db_path)
        rows = manager.query_history(severity=args.severity, since=args.since)
        if not rows:
            print("No alert history found.")
        for r in rows:
            print(f"  {r['timestamp']} [{r['severity']:8s}] {r['message']}")

    elif args.action == "test-alert":
        ts = datetime.now(timezone.utc).isoformat()
        test = FiredAlert("test_rule", "test_metric", 0.0, 0.0, "info", ts,
                          "This is a test alert from setup_alerts.py")
        print(f"Sending test alert via channel '{args.channel}'...")
        cfg = load_rules(args.config) if args.config else {"channels": {}}
        dispatch_alert(test, [args.channel], cfg.get("channels", {}))
        print("Test alert dispatched.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
