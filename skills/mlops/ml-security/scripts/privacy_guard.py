#!/usr/bin/env python3
"""
Privacy Guard — Privacy-preserving ML utilities.

Features:
    - Differential privacy training wrapper (Opacus for PyTorch)
    - Data anonymization utilities
    - PII masking in datasets (CSV, JSON, Parquet, plain text)
    - Privacy budget tracking and accounting
    - Privacy audit report generation
    - CLI interface

Usage:
    python privacy_guard.py --help
    python privacy_guard.py mask-pii --input data.csv --output data_masked.csv
    python privacy_guard.py anonymize --input data.csv --output data_anon.csv --method hash
    python privacy_guard.py audit --epsilon 3.0 --delta 1e-5 --n-samples 50000
    python privacy_guard.py dp-train --model model.pt --data train.csv --epsilon 8.0
"""

import argparse
import hashlib
import json
import logging
import math
import os
import re
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("privacy_guard")


# ===========================================================================
# PII Detection and Masking
# ===========================================================================

@dataclass
class PIIMatch:
    """A single PII detection result."""
    text: str
    category: str
    start: int
    end: int
    confidence: float


# Regex-based PII patterns with confidence scores
PII_PATTERNS: Dict[str, Tuple[str, float, str]] = {
    "email": (
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        0.95,
        "critical",
    ),
    "phone_us": (
        r'\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        0.85,
        "high",
    ),
    "phone_intl": (
        r'\b\+\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{2,4}[-.\s]?\d{2,4}[-.\s]?\d{0,4}\b',
        0.75,
        "high",
    ),
    "ssn": (
        r'\b\d{3}-\d{2}-\d{4}\b',
        0.92,
        "critical",
    ),
    "credit_card": (
        r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        0.88,
        "critical",
    ),
    "ip_address": (
        r'\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b',
        0.70,
        "medium",
    ),
    "date_of_birth": (
        r'\b(?:0[1-9]|1[0-2])[/\-](?:0[1-9]|[12]\d|3[01])[/\-](?:19|20)\d{2}\b',
        0.75,
        "high",
    ),
    "us_passport": (
        r'\b[A-Z]\d{8}\b',
        0.60,
        "critical",
    ),
    "aws_access_key": (
        r'AKIA[0-9A-Z]{16}',
        0.98,
        "critical",
    ),
    "generic_api_key": (
        r'(?i)(?:api[_-]?key|secret|token|password)\s*[:=]\s*["\']?[A-Za-z0-9+/=_\-]{16,}',
        0.80,
        "critical",
    ),
    "iban": (
        r'\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}(?:[A-Z0-9]?\d{0,16})\b',
        0.85,
        "critical",
    ),
}


class PIIDetector:
    """Detect PII in text using configurable regex patterns."""

    def __init__(self, patterns: Dict[str, Tuple[str, float, str]] = None,
                 min_confidence: float = 0.0):
        self.patterns = patterns or PII_PATTERNS
        self.min_confidence = min_confidence

    def detect(self, text: str) -> List[PIIMatch]:
        """Scan text for all configured PII patterns."""
        matches = []
        for category, (pattern, confidence, _severity) in self.patterns.items():
            if confidence < self.min_confidence:
                continue
            for match in re.finditer(pattern, text):
                matches.append(PIIMatch(
                    text=match.group(),
                    category=category,
                    start=match.start(),
                    end=match.end(),
                    confidence=confidence,
                ))
        return sorted(matches, key=lambda m: m.start)

    def detect_in_columns(self, df, columns: List[str] = None,
                          sample_size: int = 1000) -> Dict[str, List[Dict]]:
        """Detect PII across DataFrame columns. Returns findings per column."""
        import pandas as pd
        findings: Dict[str, List[Dict]] = {}
        columns = columns or df.select_dtypes(include=["object"]).columns.tolist()
        for col in columns:
            col_findings = []
            sample = df[col].dropna().head(sample_size).astype(str)
            combined_text = "\n".join(sample)
            matches = self.detect(combined_text)
            # Aggregate by category
            category_counts: Dict[str, int] = {}
            for m in matches:
                category_counts[m.category] = category_counts.get(m.category, 0) + 1
            for cat, count in category_counts.items():
                _, confidence, severity = self.patterns[cat]
                col_findings.append({
                    "column": col,
                    "pii_type": cat,
                    "match_count": count,
                    "sample_size": len(sample),
                    "confidence": confidence,
                    "severity": severity,
                })
            if col_findings:
                findings[col] = col_findings
        return findings


class PIIMasker:
    """Mask PII in text with configurable replacement strategies."""

    def __init__(self, detector: PIIDetector = None,
                 default_replacement: str = "[REDACTED]"):
        self.detector = detector or PIIDetector()
        self.default_replacement = default_replacement
        self.category_replacements = {
            "email": "[EMAIL]",
            "phone_us": "[PHONE]",
            "phone_intl": "[PHONE]",
            "ssn": "[SSN]",
            "credit_card": "[CREDIT_CARD]",
            "ip_address": "[IP_ADDR]",
            "date_of_birth": "[DOB]",
            "us_passport": "[PASSPORT]",
            "aws_access_key": "[AWS_KEY]",
            "generic_api_key": "[API_KEY]",
            "iban": "[IBAN]",
        }

    def mask_text(self, text: str, typed_replacement: bool = True) -> Tuple[str, List[PIIMatch]]:
        """Mask all detected PII in text. Returns (masked_text, matches)."""
        matches = self.detector.detect(text)
        # Process in reverse order to preserve indices
        for m in sorted(matches, key=lambda x: x.start, reverse=True):
            if typed_replacement:
                replacement = self.category_replacements.get(m.category, self.default_replacement)
            else:
                replacement = self.default_replacement
            text = text[:m.start] + replacement + text[m.end:]
        return text, matches

    def mask_dataframe(self, df, columns: List[str] = None) -> "Any":
        """Mask PII in all string columns of a DataFrame."""
        import pandas as pd
        df = df.copy()
        columns = columns or df.select_dtypes(include=["object"]).columns.tolist()
        for col in columns:
            df[col] = df[col].apply(
                lambda x: self.mask_text(str(x))[0] if pd.notna(x) else x
            )
        return df


# ===========================================================================
# Data Anonymization
# ===========================================================================

class DataAnonymizer:
    """Anonymize data using various strategies."""

    def __init__(self, salt: str = ""):
        self.salt = salt

    def hash_value(self, value: str, algorithm: str = "sha256",
                   truncate: int = 16) -> str:
        """Hash a value with optional salt and truncation."""
        salted = f"{self.salt}{value}".encode("utf-8")
        h = hashlib.new(algorithm, salted)
        return h.hexdigest()[:truncate]

    def k_anonymize_age(self, age: int, k: int = 5) -> str:
        """Generalize age into k-width buckets for k-anonymity."""
        lower = (age // k) * k
        upper = lower + k - 1
        return f"{lower}-{upper}"

    def k_anonymize_zip(self, zipcode: str, mask_digits: int = 2) -> str:
        """Mask last N digits of a zip code."""
        if len(zipcode) <= mask_digits:
            return "*" * len(zipcode)
        return zipcode[:-mask_digits] + "*" * mask_digits

    def suppress(self, value: Any, threshold: int = None,
                 group_counts: Dict = None) -> Any:
        """Suppress (remove) values that appear fewer than threshold times."""
        if threshold and group_counts:
            if group_counts.get(value, 0) < threshold:
                return None
        return value

    def anonymize_dataframe(self, df, config: Dict[str, Dict]) -> "Any":
        """
        Anonymize a DataFrame according to a configuration dict.

        Config format:
            {
                "column_name": {
                    "method": "hash" | "mask_pii" | "k_anonymize_age" | "k_anonymize_zip" |
                              "suppress" | "drop",
                    "params": { ... }  # method-specific parameters
                }
            }
        """
        import pandas as pd
        df = df.copy()
        for col, col_config in config.items():
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found in DataFrame, skipping.")
                continue
            method = col_config.get("method", "hash")
            params = col_config.get("params", {})

            if method == "hash":
                df[col] = df[col].apply(
                    lambda x: self.hash_value(str(x), **params) if pd.notna(x) else x
                )
            elif method == "mask_pii":
                masker = PIIMasker()
                df[col] = df[col].apply(
                    lambda x: masker.mask_text(str(x))[0] if pd.notna(x) else x
                )
            elif method == "k_anonymize_age":
                k = params.get("k", 5)
                df[col] = df[col].apply(
                    lambda x: self.k_anonymize_age(int(x), k)
                    if pd.notna(x) else x
                )
            elif method == "k_anonymize_zip":
                mask_digits = params.get("mask_digits", 2)
                df[col] = df[col].apply(
                    lambda x: self.k_anonymize_zip(str(x), mask_digits)
                    if pd.notna(x) else x
                )
            elif method == "suppress":
                threshold = params.get("threshold", 5)
                counts = df[col].value_counts().to_dict()
                df[col] = df[col].apply(
                    lambda x: x if counts.get(x, 0) >= threshold else None
                )
            elif method == "drop":
                df = df.drop(columns=[col])
            else:
                logger.warning(f"Unknown anonymization method '{method}' for column '{col}'.")
        return df


# ===========================================================================
# Privacy Budget Tracking
# ===========================================================================

@dataclass
class PrivacyExpenditure:
    """Record of a single privacy expenditure."""
    timestamp: str
    operation: str
    epsilon: float
    delta: float
    mechanism: str       # "gaussian", "laplace", "dp-sgd", etc.
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PrivacyBudget:
    """Track cumulative privacy budget across operations."""
    total_epsilon_budget: float
    total_delta_budget: float
    composition_method: str = "basic"  # "basic", "advanced", "rdp"
    expenditures: List[PrivacyExpenditure] = field(default_factory=list)

    @property
    def epsilon_spent(self) -> float:
        """Compute total epsilon spent based on composition method."""
        epsilons = [e.epsilon for e in self.expenditures]
        if not epsilons:
            return 0.0
        if self.composition_method == "basic":
            # Basic composition: sum of epsilons
            return sum(epsilons)
        elif self.composition_method == "advanced":
            # Advanced composition theorem
            k = len(epsilons)
            max_eps = max(epsilons)
            sum_eps_sq = sum(e ** 2 for e in epsilons)
            return min(
                sum(epsilons),
                math.sqrt(2 * k * math.log(1 / self.total_delta_budget)) * max_eps
                + k * max_eps * (math.exp(max_eps) - 1),
            )
        elif self.composition_method == "rdp":
            # Simplified RDP: tighter than basic, looser than full RDP accounting
            # Full RDP requires per-mechanism alpha parameters
            return sum(epsilons) * 0.7  # Rough approximation
        return sum(epsilons)

    @property
    def delta_spent(self) -> float:
        """Compute total delta spent (basic: sum of deltas)."""
        return sum(e.delta for e in self.expenditures)

    @property
    def epsilon_remaining(self) -> float:
        return max(0.0, self.total_epsilon_budget - self.epsilon_spent)

    @property
    def delta_remaining(self) -> float:
        return max(0.0, self.total_delta_budget - self.delta_spent)

    @property
    def budget_exhausted(self) -> bool:
        return self.epsilon_remaining <= 0 or self.delta_remaining <= 0

    def record_expenditure(self, operation: str, epsilon: float, delta: float,
                           mechanism: str = "gaussian", description: str = "",
                           metadata: Dict = None) -> PrivacyExpenditure:
        """Record a privacy expenditure. Raises if budget would be exceeded."""
        # Check if this would exceed budget
        test_budget = PrivacyBudget(
            total_epsilon_budget=self.total_epsilon_budget,
            total_delta_budget=self.total_delta_budget,
            composition_method=self.composition_method,
            expenditures=self.expenditures + [PrivacyExpenditure(
                timestamp="", operation=operation, epsilon=epsilon,
                delta=delta, mechanism=mechanism,
            )],
        )
        if test_budget.epsilon_spent > self.total_epsilon_budget:
            raise ValueError(
                f"Epsilon budget exceeded: spending {epsilon} would bring total to "
                f"{test_budget.epsilon_spent:.4f}, exceeding budget {self.total_epsilon_budget}"
            )
        if test_budget.delta_spent > self.total_delta_budget:
            raise ValueError(
                f"Delta budget exceeded: spending {delta} would bring total to "
                f"{test_budget.delta_spent}, exceeding budget {self.total_delta_budget}"
            )

        expenditure = PrivacyExpenditure(
            timestamp=datetime.now(timezone.utc).isoformat(),
            operation=operation,
            epsilon=epsilon,
            delta=delta,
            mechanism=mechanism,
            description=description,
            metadata=metadata or {},
        )
        self.expenditures.append(expenditure)
        logger.info(
            f"Privacy expenditure recorded: eps={epsilon:.4f}, delta={delta:.2e}, "
            f"remaining: eps={self.epsilon_remaining:.4f}, delta={self.delta_remaining:.2e}"
        )
        return expenditure

    def to_dict(self) -> dict:
        return {
            "total_epsilon_budget": self.total_epsilon_budget,
            "total_delta_budget": self.total_delta_budget,
            "composition_method": self.composition_method,
            "epsilon_spent": self.epsilon_spent,
            "delta_spent": self.delta_spent,
            "epsilon_remaining": self.epsilon_remaining,
            "delta_remaining": self.delta_remaining,
            "budget_exhausted": self.budget_exhausted,
            "num_operations": len(self.expenditures),
            "expenditures": [asdict(e) for e in self.expenditures],
        }

    def save(self, path: str):
        """Save privacy budget state to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PrivacyBudget":
        """Load privacy budget state from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        budget = cls(
            total_epsilon_budget=data["total_epsilon_budget"],
            total_delta_budget=data["total_delta_budget"],
            composition_method=data.get("composition_method", "basic"),
        )
        for exp_data in data.get("expenditures", []):
            budget.expenditures.append(PrivacyExpenditure(**exp_data))
        return budget


# ===========================================================================
# Differential Privacy Training Wrapper
# ===========================================================================

class DPTrainer:
    """Wrapper for differential privacy training with Opacus (PyTorch)."""

    def __init__(self, privacy_budget: PrivacyBudget = None):
        self.privacy_budget = privacy_budget

    def train_pytorch(self, model, train_dataset, epochs: int = 10,
                      target_epsilon: float = 8.0, target_delta: float = 1e-5,
                      max_grad_norm: float = 1.0, batch_size: int = 64,
                      lr: float = 0.001, device: str = "cpu"):
        """
        Train a PyTorch model with differential privacy using Opacus.

        Returns:
            (model, privacy_engine, final_epsilon)
        """
        try:
            import torch
            from torch.utils.data import DataLoader
            from opacus import PrivacyEngine
            from opacus.validators import ModuleValidator
        except ImportError as e:
            raise ImportError(
                "Opacus and PyTorch are required for DP training. "
                "Install with: pip install torch opacus"
            ) from e

        # Validate model compatibility with Opacus
        if not ModuleValidator.is_valid(model):
            logger.info("Model is not compatible with Opacus. Attempting automatic fix...")
            model = ModuleValidator.fix(model)
            logger.info("Model fixed for Opacus compatibility.")

        model = model.to(device)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=epochs,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            max_grad_norm=max_grad_norm,
        )

        loss_fn = torch.nn.CrossEntropyLoss()
        logger.info(
            f"Starting DP training: epochs={epochs}, target_eps={target_epsilon}, "
            f"delta={target_delta}, max_grad_norm={max_grad_norm}"
        )

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            num_batches = 0
            for batch in train_loader:
                if isinstance(batch, (list, tuple)):
                    x, y = batch[0].to(device), batch[1].to(device)
                else:
                    x, y = batch.to(device), None

                optimizer.zero_grad()
                outputs = model(x)
                loss = loss_fn(outputs, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

            current_epsilon = privacy_engine.get_epsilon(delta=target_delta)
            avg_loss = total_loss / max(num_batches, 1)
            logger.info(
                f"Epoch {epoch+1}/{epochs} -- Loss: {avg_loss:.4f} -- "
                f"epsilon: {current_epsilon:.2f}, delta: {target_delta}"
            )

        final_epsilon = privacy_engine.get_epsilon(delta=target_delta)

        # Record in privacy budget if available
        if self.privacy_budget:
            self.privacy_budget.record_expenditure(
                operation="dp_sgd_training",
                epsilon=final_epsilon,
                delta=target_delta,
                mechanism="dp-sgd",
                description=f"Training for {epochs} epochs with Opacus",
                metadata={
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "max_grad_norm": max_grad_norm,
                    "lr": lr,
                },
            )

        logger.info(f"DP Training complete. Final epsilon = {final_epsilon:.4f}")
        return model, privacy_engine, final_epsilon

    @staticmethod
    def compute_noise_multiplier(target_epsilon: float, target_delta: float,
                                 n_samples: int, batch_size: int,
                                 epochs: int) -> float:
        """
        Estimate the noise multiplier needed to achieve a target epsilon.

        This uses a simple binary search approach. For production use,
        prefer Opacus or TF Privacy built-in calibration.
        """
        try:
            from opacus.accountants.utils import get_noise_multiplier
            return get_noise_multiplier(
                target_epsilon=target_epsilon,
                target_delta=target_delta,
                sample_rate=batch_size / n_samples,
                epochs=epochs,
            )
        except ImportError:
            # Fallback rough estimate
            steps = epochs * (n_samples // batch_size)
            sampling_rate = batch_size / n_samples
            # Simplified Gaussian mechanism estimate
            return math.sqrt(2 * math.log(1.25 / target_delta)) / target_epsilon * math.sqrt(steps * sampling_rate)


# ===========================================================================
# Privacy Audit Report
# ===========================================================================

@dataclass
class PrivacyAuditReport:
    """Comprehensive privacy audit report."""
    timestamp: str
    data_path: Optional[str]
    pii_findings: Dict[str, List[Dict]] = field(default_factory=dict)
    privacy_budget: Optional[Dict] = None
    recommendations: List[str] = field(default_factory=list)
    risk_level: str = "unknown"  # "low", "medium", "high", "critical"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def print_report(self):
        print("\n" + "=" * 70)
        print("  PRIVACY AUDIT REPORT")
        print("=" * 70)
        print(f"  Timestamp  : {self.timestamp}")
        print(f"  Data Path  : {self.data_path or 'N/A'}")
        print(f"  Risk Level : {self.risk_level.upper()}")
        print("-" * 70)

        if self.pii_findings:
            print("\n  PII Findings:")
            for col, findings in self.pii_findings.items():
                for f in findings:
                    print(f"    [{f.get('severity', '?').upper()}] Column '{col}': "
                          f"{f['pii_type']} -- {f['match_count']} matches "
                          f"(confidence: {f.get('confidence', 'N/A')})")
        else:
            print("\n  No PII detected in scanned data.")

        if self.privacy_budget:
            print(f"\n  Privacy Budget:")
            print(f"    Epsilon spent / budget: "
                  f"{self.privacy_budget.get('epsilon_spent', 0):.4f} / "
                  f"{self.privacy_budget.get('total_epsilon_budget', 'N/A')}")
            print(f"    Delta spent / budget:   "
                  f"{self.privacy_budget.get('delta_spent', 0):.2e} / "
                  f"{self.privacy_budget.get('total_delta_budget', 'N/A')}")
            print(f"    Budget exhausted:       "
                  f"{self.privacy_budget.get('budget_exhausted', False)}")

        if self.recommendations:
            print(f"\n  Recommendations:")
            for i, rec in enumerate(self.recommendations, 1):
                print(f"    {i}. {rec}")

        print("=" * 70 + "\n")


def run_privacy_audit(data_path: str = None, budget_path: str = None,
                      epsilon: float = None, delta: float = None,
                      n_samples: int = None) -> PrivacyAuditReport:
    """Run a comprehensive privacy audit."""
    report = PrivacyAuditReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        data_path=data_path,
    )
    recommendations = []

    # 1. PII Detection
    if data_path and Path(data_path).exists():
        try:
            ext = Path(data_path).suffix.lower()
            import pandas as pd
            if ext == ".csv":
                df = pd.read_csv(data_path, nrows=10000)
            elif ext == ".parquet":
                df = pd.read_parquet(data_path)
                if len(df) > 10000:
                    df = df.head(10000)
            elif ext in {".json", ".jsonl"}:
                df = pd.read_json(data_path, lines=(ext == ".jsonl"), nrows=10000)
            else:
                df = None

            if df is not None:
                detector = PIIDetector()
                report.pii_findings = detector.detect_in_columns(df)
                if report.pii_findings:
                    critical_count = sum(
                        1 for col_findings in report.pii_findings.values()
                        for f in col_findings if f.get("severity") == "critical"
                    )
                    if critical_count > 0:
                        recommendations.append(
                            f"CRITICAL: {critical_count} critical PII types detected. "
                            "Mask or remove before training."
                        )
                    recommendations.append(
                        "Run: python privacy_guard.py mask-pii "
                        f"--input {data_path} --output {data_path}.masked"
                    )
            else:
                # Scan raw text
                with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read(500000)
                detector = PIIDetector()
                matches = detector.detect(text)
                if matches:
                    from collections import Counter
                    counts = Counter(m.category for m in matches)
                    report.pii_findings["_raw_text"] = [
                        {"pii_type": cat, "match_count": cnt, "column": "_raw_text",
                         "sample_size": len(text), "severity": PII_PATTERNS[cat][2],
                         "confidence": PII_PATTERNS[cat][1]}
                        for cat, cnt in counts.items()
                    ]

        except ImportError:
            recommendations.append(
                "Install pandas for structured data PII scanning: pip install pandas"
            )
        except Exception as e:
            logger.warning(f"PII scan error: {e}")

    # 2. Privacy Budget Assessment
    if budget_path and Path(budget_path).exists():
        budget = PrivacyBudget.load(budget_path)
        report.privacy_budget = budget.to_dict()
        if budget.budget_exhausted:
            recommendations.append(
                "CRITICAL: Privacy budget is exhausted. No further operations allowed."
            )
        elif budget.epsilon_remaining < budget.total_epsilon_budget * 0.2:
            recommendations.append(
                f"WARNING: Only {budget.epsilon_remaining:.4f} epsilon remaining "
                f"({budget.epsilon_remaining/budget.total_epsilon_budget:.0%} of budget)."
            )

    # 3. Epsilon/Delta Assessment (if provided directly)
    if epsilon is not None and delta is not None:
        if epsilon > 10:
            recommendations.append(
                f"Epsilon={epsilon} is high. Consider epsilon <= 10 for meaningful privacy."
            )
        elif epsilon > 1:
            recommendations.append(
                f"Epsilon={epsilon} provides moderate privacy. "
                "Epsilon <= 1 is recommended for strong guarantees."
            )
        else:
            recommendations.append(
                f"Epsilon={epsilon} provides strong privacy protection."
            )

        if n_samples and delta > 1.0 / n_samples:
            recommendations.append(
                f"Delta={delta} exceeds 1/n={1.0/n_samples:.2e}. "
                "Set delta < 1/n for meaningful guarantees."
            )

    # 4. General recommendations
    if not report.pii_findings:
        recommendations.append("No PII detected, but consider manual review for domain-specific PII.")
    recommendations.append("Ensure all data is encrypted at rest and in transit.")
    recommendations.append("Implement access logging for all data access operations.")

    # Determine risk level
    has_critical = any(
        f.get("severity") == "critical"
        for col_findings in report.pii_findings.values()
        for f in col_findings
    )
    has_high = any(
        f.get("severity") == "high"
        for col_findings in report.pii_findings.values()
        for f in col_findings
    )

    if has_critical or (report.privacy_budget and report.privacy_budget.get("budget_exhausted")):
        report.risk_level = "critical"
    elif has_high:
        report.risk_level = "high"
    elif report.pii_findings:
        report.risk_level = "medium"
    else:
        report.risk_level = "low"

    report.recommendations = recommendations
    return report


# ===========================================================================
# CLI
# ===========================================================================

def cmd_mask_pii(args):
    """Mask PII in a data file."""
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    output_path = args.output or str(input_path.with_suffix(".masked" + input_path.suffix))
    ext = input_path.suffix.lower()

    masker = PIIMasker()
    total_matches = 0

    if ext == ".csv":
        try:
            import pandas as pd
            df = pd.read_csv(args.input)
            columns = args.columns.split(",") if args.columns else None
            masked_df = masker.mask_dataframe(df, columns=columns)
            masked_df.to_csv(output_path, index=False)
            logger.info(f"Masked CSV written to {output_path}")
        except ImportError:
            logger.error("pandas is required for CSV processing: pip install pandas")
            sys.exit(1)
    elif ext == ".parquet":
        try:
            import pandas as pd
            df = pd.read_parquet(args.input)
            columns = args.columns.split(",") if args.columns else None
            masked_df = masker.mask_dataframe(df, columns=columns)
            masked_df.to_parquet(output_path)
            logger.info(f"Masked Parquet written to {output_path}")
        except ImportError:
            logger.error("pandas and pyarrow are required: pip install pandas pyarrow")
            sys.exit(1)
    else:
        # Plain text or JSON
        with open(args.input, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        masked_text, matches = masker.mask_text(text)
        total_matches = len(matches)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(masked_text)
        logger.info(f"Masked {total_matches} PII instances. Output: {output_path}")


def cmd_anonymize(args):
    """Anonymize a data file."""
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    output_path = args.output or str(input_path.with_suffix(".anon" + input_path.suffix))
    method = args.method

    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas is required: pip install pandas")
        sys.exit(1)

    ext = input_path.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(args.input)
    elif ext == ".parquet":
        df = pd.read_parquet(args.input)
    elif ext in {".json", ".jsonl"}:
        df = pd.read_json(args.input, lines=(ext == ".jsonl"))
    else:
        logger.error(f"Unsupported file format: {ext}")
        sys.exit(1)

    columns = args.columns.split(",") if args.columns else \
        df.select_dtypes(include=["object"]).columns.tolist()

    anonymizer = DataAnonymizer(salt=args.salt)

    if method == "hash":
        for col in columns:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: anonymizer.hash_value(str(x)) if pd.notna(x) else x
                )
    elif method == "mask_pii":
        masker = PIIMasker()
        df = masker.mask_dataframe(df, columns=columns)
    elif method == "drop":
        df = df.drop(columns=[c for c in columns if c in df.columns])
    else:
        logger.error(f"Unknown method: {method}. Use: hash, mask_pii, drop")
        sys.exit(1)

    if ext == ".csv":
        df.to_csv(output_path, index=False)
    elif ext == ".parquet":
        df.to_parquet(output_path)
    else:
        df.to_json(output_path, orient="records", indent=2)

    logger.info(f"Anonymized data written to {output_path}")


def cmd_audit(args):
    """Run a privacy audit."""
    report = run_privacy_audit(
        data_path=args.data,
        budget_path=args.budget,
        epsilon=args.epsilon,
        delta=args.delta,
        n_samples=args.n_samples,
    )
    report.print_report()

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(report.to_json())
        logger.info(f"Audit report written to {args.output}")


def cmd_dp_train(args):
    """Run differential privacy training (PyTorch/Opacus)."""
    logger.info("DP training requires a PyTorch model and dataset.")
    logger.info("This command provides parameter guidance and budget tracking.")

    budget = None
    if args.budget:
        if Path(args.budget).exists():
            budget = PrivacyBudget.load(args.budget)
            logger.info(f"Loaded privacy budget from {args.budget}")
        else:
            budget = PrivacyBudget(
                total_epsilon_budget=args.total_epsilon or args.epsilon * 2,
                total_delta_budget=args.delta * 10,
                composition_method="advanced",
            )
            logger.info("Created new privacy budget.")

    # Compute recommended noise multiplier
    if args.n_samples and args.batch_size:
        try:
            noise_mult = DPTrainer.compute_noise_multiplier(
                target_epsilon=args.epsilon,
                target_delta=args.delta,
                n_samples=args.n_samples,
                batch_size=args.batch_size,
                epochs=args.epochs,
            )
            logger.info(f"Recommended noise multiplier: {noise_mult:.4f}")
        except Exception as e:
            logger.warning(f"Could not compute noise multiplier: {e}")

    print("\n--- DP Training Configuration ---")
    print(f"  Target epsilon  : {args.epsilon}")
    print(f"  Target delta    : {args.delta}")
    print(f"  Epochs          : {args.epochs}")
    print(f"  Max grad norm   : {args.max_grad_norm}")
    print(f"  Batch size      : {args.batch_size}")
    print(f"  Learning rate   : {args.lr}")
    if budget:
        print(f"  Budget remaining: eps={budget.epsilon_remaining:.4f}")
    print("-" * 35)

    if budget and args.budget:
        budget.save(args.budget)
        logger.info(f"Budget saved to {args.budget}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Privacy Guard -- Privacy-preserving ML utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # mask-pii
    p_mask = subparsers.add_parser("mask-pii", help="Mask PII in a data file")
    p_mask.add_argument("--input", "-i", required=True, help="Input file path")
    p_mask.add_argument("--output", "-o", help="Output file path (default: input.masked.ext)")
    p_mask.add_argument("--columns", help="Comma-separated column names to scan (CSV/Parquet)")
    p_mask.set_defaults(func=cmd_mask_pii)

    # anonymize
    p_anon = subparsers.add_parser("anonymize", help="Anonymize a data file")
    p_anon.add_argument("--input", "-i", required=True, help="Input file path")
    p_anon.add_argument("--output", "-o", help="Output file path")
    p_anon.add_argument("--method", "-m", default="hash",
                        choices=["hash", "mask_pii", "drop"],
                        help="Anonymization method (default: hash)")
    p_anon.add_argument("--columns", help="Comma-separated column names")
    p_anon.add_argument("--salt", default="", help="Salt for hashing (default: empty)")
    p_anon.set_defaults(func=cmd_anonymize)

    # audit
    p_audit = subparsers.add_parser("audit", help="Run a privacy audit")
    p_audit.add_argument("--data", "-d", help="Data file to scan for PII")
    p_audit.add_argument("--budget", "-b", help="Privacy budget JSON file")
    p_audit.add_argument("--epsilon", "-e", type=float, help="Epsilon value to assess")
    p_audit.add_argument("--delta", type=float, default=1e-5, help="Delta value (default: 1e-5)")
    p_audit.add_argument("--n-samples", "-n", type=int, help="Number of training samples")
    p_audit.add_argument("--output", "-o", help="Output report JSON path")
    p_audit.set_defaults(func=cmd_audit)

    # dp-train
    p_dp = subparsers.add_parser("dp-train", help="DP training parameter guidance")
    p_dp.add_argument("--model", help="Path to PyTorch model")
    p_dp.add_argument("--data", help="Path to training data")
    p_dp.add_argument("--epsilon", "-e", type=float, default=8.0,
                      help="Target epsilon (default: 8.0)")
    p_dp.add_argument("--delta", type=float, default=1e-5, help="Target delta (default: 1e-5)")
    p_dp.add_argument("--total-epsilon", type=float, help="Total epsilon budget")
    p_dp.add_argument("--epochs", type=int, default=10, help="Number of epochs (default: 10)")
    p_dp.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    p_dp.add_argument("--max-grad-norm", type=float, default=1.0,
                      help="Max gradient norm (default: 1.0)")
    p_dp.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    p_dp.add_argument("--n-samples", type=int, help="Number of training samples")
    p_dp.add_argument("--budget", help="Privacy budget JSON file to load/save")
    p_dp.set_defaults(func=cmd_dp_train)

    args = parser.parse_args(argv)
    return args


def main(argv=None):
    args = parse_args(argv)
    if not args.command:
        parse_args(["--help"])
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
