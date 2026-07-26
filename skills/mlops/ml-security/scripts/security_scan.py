#!/usr/bin/env python3
"""
ML Security Scanner — Comprehensive security scanning for ML systems.

Features:
    - Model load smoke check (framework load + stochastic-layer / pickle exposure
      report; does NOT run adversarial attacks — see ModelLoadSmokeCheckScanner)
    - Input validation checks
    - Model artifact integrity verification
    - Dependency vulnerability scanning (pip-audit, safety)
    - PII detection in training data (regex + Luhn; format-only, see PIIScanner)
    - Pickle safety analysis
    - Security report generation
    - CLI interface

Usage:
    python security_scan.py --help
    python security_scan.py --scan all --model model.pt --data training_data.csv
    python security_scan.py --scan robustness --model model.pt --framework pytorch
    python security_scan.py --scan dependencies --requirements requirements.txt
    python security_scan.py --scan pii --data training_data.csv
    python security_scan.py --scan artifact --model model.pt --manifest manifest.json
"""

import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
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
logger = logging.getLogger("ml_security_scan")


# ---------------------------------------------------------------------------
# Data classes for scan results
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    """A single security finding."""
    category: str
    severity: str  # "critical", "high", "medium", "low", "info"
    title: str
    description: str
    remediation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}

    def severity_rank(self) -> int:
        return self.SEVERITY_ORDER.get(self.severity, 5)


@dataclass
class ScanResult:
    """Results from a single scan module."""
    scanner_name: str
    status: str  # "passed", "warnings", "failed", "error", "skipped"
    findings: List[Finding] = field(default_factory=list)
    duration_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityReport:
    """Complete security report aggregating all scans."""
    timestamp: str
    model_path: Optional[str]
    data_path: Optional[str]
    scan_results: List[ScanResult] = field(default_factory=list)

    def overall_status(self) -> str:
        statuses = [r.status for r in self.scan_results]
        if "failed" in statuses:
            return "FAILED"
        if "error" in statuses:
            return "ERROR"
        if "warnings" in statuses:
            return "WARNINGS"
        return "PASSED"

    def total_findings(self) -> Dict[str, int]:
        counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
        for result in self.scan_results:
            for finding in result.findings:
                counts[finding.severity] = counts.get(finding.severity, 0) + 1
        return counts

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "model_path": self.model_path,
            "data_path": self.data_path,
            "overall_status": self.overall_status(),
            "finding_counts": self.total_findings(),
            "scan_results": [
                {
                    "scanner_name": r.scanner_name,
                    "status": r.status,
                    "duration_seconds": round(r.duration_seconds, 3),
                    "findings": [asdict(f) for f in r.findings],
                    "metadata": r.metadata,
                }
                for r in self.scan_results
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def print_summary(self):
        """Print a human-readable summary to stdout."""
        counts = self.total_findings()
        status = self.overall_status()
        print("\n" + "=" * 70)
        print("  ML SECURITY SCAN REPORT")
        print("=" * 70)
        print(f"  Timestamp : {self.timestamp}")
        print(f"  Model     : {self.model_path or 'N/A'}")
        print(f"  Data      : {self.data_path or 'N/A'}")
        print(f"  Status    : {status}")
        print(f"  Findings  : "
              f"{counts['critical']} critical, {counts['high']} high, "
              f"{counts['medium']} medium, {counts['low']} low, {counts['info']} info")
        print("-" * 70)
        for result in self.scan_results:
            icon = {
                "passed": "[PASS]", "warnings": "[WARN]",
                "failed": "[FAIL]", "error": "[ERR ]", "skipped": "[SKIP]",
            }.get(result.status, "[????]")
            print(f"  {icon} {result.scanner_name} "
                  f"({result.duration_seconds:.2f}s) — {len(result.findings)} finding(s)")
            for f in sorted(result.findings, key=lambda x: x.severity_rank()):
                sev = f.severity.upper()
                print(f"         [{sev}] {f.title}")
                if f.description:
                    for line in f.description.split("\n"):
                        print(f"                {line}")
        print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Scanner base class
# ---------------------------------------------------------------------------

class BaseScanner(ABC):
    """Abstract base class for all security scanners."""

    name: str = "base"

    @abstractmethod
    def scan(self, **kwargs) -> ScanResult:
        ...


# ---------------------------------------------------------------------------
# 1. Adversarial Robustness Scanner
# ---------------------------------------------------------------------------

class ModelLoadSmokeCheckScanner(BaseScanner):
    """
    Load-time smoke check for model artifacts.

    SCOPE: this scanner does NOT run adversarial attacks. It verifies that the
    artifact loads in the declared framework, reports what it found (layer
    types, stochastic layers, pickle exposure), and tells you whether the ART
    toolchain needed for a real robustness evaluation is installed.

    A real robustness evaluation needs a labelled evaluation set and a wrapped
    estimator, neither of which a file-path-only scan has. Do it explicitly:

        from art.estimators.classification import PyTorchClassifier
        from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
        clf = PyTorchClassifier(model=model, loss=loss_fn, optimizer=opt,
                                input_shape=(3, 224, 224), nb_classes=10)
        x_adv = ProjectedGradientDescent(estimator=clf, eps=0.03,
                                         max_iter=40).generate(x=x_test)
        # then compare accuracy on x_test vs x_adv

    See references/REFERENCE.md sections 3 and 17 for full examples.
    """

    name = "model_load_smoke_check"

    def scan(self, model_path: str = None, framework: str = "pytorch",
             epsilon: float = 0.03, attacks: List[str] = None,
             allow_unsafe_load: bool = False, **kwargs) -> ScanResult:
        start = time.time()
        findings = []

        if model_path is None:
            return ScanResult(
                scanner_name=self.name, status="skipped",
                findings=[Finding(
                    category="robustness", severity="info",
                    title="No model path provided",
                    description="Skipping model load smoke check — no model specified.",
                )],
                duration_seconds=time.time() - start,
            )

        attacks = attacks or ["fgsm", "pgd"]

        # Report whether the toolchain for a real robustness evaluation exists.
        try:
            import art  # noqa: F401
            art_available = True
        except ImportError:
            art_available = False
            findings.append(Finding(
                category="robustness",
                severity="medium",
                title="ART library not installed — no robustness evaluation possible",
                description=(
                    "The Adversarial Robustness Toolbox (ART) is required to run FGSM/PGD "
                    "evaluations. This scanner only smoke-checks that the artifact loads; "
                    "attacks must be run separately against a labelled evaluation set."
                ),
                remediation="pip install adversarial-robustness-toolbox",
            ))

        if not Path(model_path).exists():
            findings.append(Finding(
                category="robustness", severity="high",
                title="Model file not found",
                description=f"The model file '{model_path}' does not exist.",
            ))
            return ScanResult(
                scanner_name=self.name, status="error",
                findings=findings, duration_seconds=time.time() - start,
            )

        findings.extend(
            self._run_load_check(model_path, framework, allow_unsafe_load)
        )
        findings.append(Finding(
            category="robustness", severity="info",
            title="No adversarial attacks were executed",
            description=(
                f"Requested attacks {attacks} at epsilon={epsilon} were NOT run. "
                "A robustness evaluation needs labelled evaluation data and a wrapped "
                "ART estimator; see the class docstring and REFERENCE.md section 3."
            ),
            remediation=(
                "Run ART FastGradientMethod / ProjectedGradientDescent against your "
                "held-out test set and compare clean vs adversarial accuracy."
            ),
        ))

        # Determine overall status
        severities = [f.severity for f in findings]
        if "critical" in severities or "high" in severities:
            status = "failed"
        elif "medium" in severities:
            status = "warnings"
        else:
            status = "passed"

        return ScanResult(
            scanner_name=self.name, status=status,
            findings=findings, duration_seconds=time.time() - start,
            metadata={
                "epsilon": epsilon, "attacks_requested": attacks,
                "attacks_executed": [], "framework": framework,
                "art_available": art_available,
            },
        )

    def _run_load_check(self, model_path: str, framework: str,
                        allow_unsafe_load: bool = False) -> List[Finding]:
        """Load the artifact in the declared framework and describe it."""
        findings = []
        try:
            if framework == "pytorch":
                findings.extend(self._scan_pytorch(model_path, allow_unsafe_load))
            elif framework == "tensorflow":
                findings.extend(self._scan_tensorflow(model_path))
            elif framework == "sklearn":
                findings.extend(self._scan_sklearn(model_path, allow_unsafe_load))
            else:
                findings.append(Finding(
                    category="robustness", severity="info",
                    title=f"Framework '{framework}' not directly supported",
                    description="Supported frameworks: pytorch, tensorflow, sklearn. "
                                "Export to ONNX for cross-framework testing.",
                ))
        except Exception as e:
            findings.append(Finding(
                category="robustness", severity="medium",
                title="Model load check encountered an error",
                description=str(e),
                remediation="Check model format compatibility and framework version.",
            ))
        return findings

    def _scan_pytorch(self, model_path: str,
                      allow_unsafe_load: bool = False) -> List[Finding]:
        findings = []
        try:
            import torch
        except ImportError:
            return [Finding(
                category="robustness", severity="medium",
                title="PyTorch not installed",
                description="Cannot load the artifact without torch.",
                remediation="pip install torch",
            )]

        # Load weights-only by default: unpickling a whole nn.Module executes
        # arbitrary code, which is exactly the risk this scanner reports on.
        try:
            obj = torch.load(model_path, map_location="cpu", weights_only=True)
            loaded_unsafely = False
        except Exception as safe_err:
            if not allow_unsafe_load:
                findings.append(Finding(
                    category="robustness", severity="medium",
                    title="Artifact requires unsafe (pickle) deserialization",
                    description=(
                        f"torch.load(weights_only=True) failed: {safe_err}\n"
                        "The file therefore contains pickled Python objects, not just "
                        "tensors. The scanner did NOT unpickle it, because doing so would "
                        "execute code from the artifact under audit."
                    ),
                    remediation=(
                        "Re-save as a state_dict or safetensors. To inspect anyway, "
                        "re-run in a sandbox with --allow-unsafe-load, and check the file "
                        "first with `fickling <file> --check-safety`."
                    ),
                ))
                return findings
            logger.warning(
                "Loading %s with weights_only=False as explicitly requested — "
                "this executes pickle payloads in the artifact.", model_path,
            )
            obj = torch.load(model_path, map_location="cpu", weights_only=False)
            loaded_unsafely = True

        if isinstance(obj, dict):
            findings.append(Finding(
                category="robustness", severity="info",
                title="PyTorch state_dict loaded successfully",
                description=(
                    f"Loaded {len(obj)} tensor entries from {model_path} with "
                    "weights_only=True. Instantiate the architecture separately to "
                    "evaluate the model."
                ),
            ))
            return findings

        model = obj
        try:
            model.eval()
        except Exception:
            pass
        findings.append(Finding(
            category="robustness", severity="info",
            title="PyTorch module loaded successfully",
            description=(
                f"Model loaded from {model_path}"
                + (" (unsafe pickle load, explicitly allowed)." if loaded_unsafely else ".")
            ),
        ))

        # Detect stochastic layers by CLASS name; named_modules() yields the
        # attribute path (e.g. "features.3"), which usually contains no type info.
        module_types = {type(m).__name__.lower() for _, m in model.named_modules()}
        has_dropout = any("dropout" in t for t in module_types)
        has_batchnorm = any("batchnorm" in t or "instancenorm" in t for t in module_types)
        if has_dropout or has_batchnorm:
            findings.append(Finding(
                category="robustness", severity="low",
                title="Model uses stochastic / running-statistics layers",
                description=(
                    "Model contains Dropout or BatchNorm layers. Ensure model.eval() "
                    "is called before inference to disable stochastic behavior."
                ),
                remediation="Always call model.eval() before adversarial testing.",
                metadata={"dropout": has_dropout, "batchnorm": has_batchnorm},
            ))
        return findings

    def _scan_tensorflow(self, model_path: str) -> List[Finding]:
        findings = []
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(model_path)
            layer_types = [type(layer).__name__ for layer in model.layers]
            findings.append(Finding(
                category="robustness", severity="info",
                title="TensorFlow model loaded successfully",
                description=f"Model loaded from {model_path}. Layers: {len(model.layers)}",
                metadata={"layer_types": sorted(set(layer_types))},
            ))
            if "Lambda" in layer_types:
                findings.append(Finding(
                    category="robustness", severity="high",
                    title="Keras Lambda layer detected",
                    description=(
                        "Lambda layers store Python bytecode that is executed on load. "
                        "This artifact can run arbitrary code in any process that loads it."
                    ),
                    remediation=(
                        "Replace Lambda layers with a registered custom layer class, and "
                        "load untrusted models with safe_mode=True."
                    ),
                ))
        except Exception as e:
            findings.append(Finding(
                category="robustness", severity="medium",
                title="Failed to load TensorFlow model",
                description=str(e),
            ))
        return findings

    def _scan_sklearn(self, model_path: str,
                      allow_unsafe_load: bool = False) -> List[Finding]:
        findings = []
        # joblib.load unpickles; do not do it unless explicitly allowed.
        findings.append(Finding(
            category="robustness", severity="medium",
            title="Pickle deserialization risk",
            description=(
                "scikit-learn models saved with joblib/pickle execute arbitrary code "
                "on load."
            ),
            remediation=(
                "Use skops.io for safer serialization, or verify artifact integrity "
                "with signed manifests before loading."
            ),
        ))
        if not allow_unsafe_load:
            findings.append(Finding(
                category="robustness", severity="info",
                title="sklearn artifact not deserialized",
                description=(
                    "The scanner did not unpickle the artifact. Re-run with "
                    "--allow-unsafe-load in a sandbox to inspect the estimator, or use "
                    "`fickling <file> --check-safety` for a static check."
                ),
            ))
            return findings
        try:
            import joblib
            model = joblib.load(model_path)
            findings.append(Finding(
                category="robustness", severity="info",
                title="scikit-learn model loaded",
                description=f"Model type: {type(model).__name__}.",
            ))
        except Exception as e:
            findings.append(Finding(
                category="robustness", severity="medium",
                title="Failed to load sklearn model",
                description=str(e),
            ))
        return findings


# ---------------------------------------------------------------------------
# 2. Input Validation Scanner
# ---------------------------------------------------------------------------

class InputValidationScanner(BaseScanner):
    """Check model serving code for input validation best practices."""

    name = "input_validation"

    # Patterns that indicate missing input validation
    RISKY_PATTERNS = [
        (r'pickle\.loads?\(', "high",
         "Unsafe pickle deserialization",
         "Using pickle.load/loads on untrusted input enables arbitrary code execution.",
         "Use safer alternatives (JSON, protobuf) or validate with fickling."),
        (r'torch\.load\([^)]*weights_only\s*=\s*False', "high",
         "Unsafe torch.load with weights_only=False",
         "torch.load with weights_only=False uses pickle and is vulnerable to code execution.",
         "Use torch.load(..., weights_only=True) or safetensors format."),
        (r'torch\.load\([^)]*\)', "medium",
         "torch.load usage detected",
         "torch.load defaults to pickle-based deserialization in older PyTorch versions.",
         "Use weights_only=True (PyTorch >= 2.0) or safetensors."),
        (r'yaml\.load\([^)]*\)', "medium",
         "Unsafe YAML loading",
         "yaml.load without Loader=SafeLoader can execute arbitrary Python.",
         "Use yaml.safe_load() instead."),
        (r'eval\(', "high",
         "Use of eval()",
         "eval() on user input enables arbitrary code execution.",
         "Remove eval() calls; use safe parsers for expressions."),
        (r'exec\(', "high",
         "Use of exec()",
         "exec() on user input enables arbitrary code execution.",
         "Remove exec() calls."),
        (r'subprocess\.(?:call|run|Popen)\([^)]*shell\s*=\s*True', "high",
         "Shell injection risk",
         "subprocess with shell=True is vulnerable to command injection.",
         "Use shell=False and pass arguments as a list."),
        (r'__import__\(', "medium",
         "Dynamic import detected",
         "Dynamic imports can be exploited to load malicious modules.",
         "Use static imports or a strict allowlist."),
    ]

    def scan(self, project_dir: str = ".", **kwargs) -> ScanResult:
        start = time.time()
        findings = []
        project_path = Path(project_dir)

        if not project_path.exists():
            return ScanResult(
                scanner_name=self.name, status="error",
                findings=[Finding(
                    category="input_validation", severity="medium",
                    title="Project directory not found",
                    description=f"Directory '{project_dir}' does not exist.",
                )],
                duration_seconds=time.time() - start,
            )

        python_files = list(project_path.rglob("*.py"))
        if not python_files:
            findings.append(Finding(
                category="input_validation", severity="info",
                title="No Python files found",
                description=f"No .py files found in '{project_dir}'.",
            ))
        else:
            for py_file in python_files:
                try:
                    content = py_file.read_text(encoding="utf-8", errors="ignore")
                    for pattern, severity, title, desc, remediation in self.RISKY_PATTERNS:
                        matches = list(re.finditer(pattern, content))
                        if matches:
                            lines = [
                                content[:m.start()].count("\n") + 1 for m in matches
                            ]
                            findings.append(Finding(
                                category="input_validation",
                                severity=severity,
                                title=f"{title} in {py_file.name}",
                                description=f"{desc}\nFound at line(s): {lines} in {py_file}",
                                remediation=remediation,
                            ))
                except Exception as e:
                    logger.warning(f"Could not scan {py_file}: {e}")

        # Check for input validation patterns (positive signals)
        validation_patterns_found = 0
        for py_file in python_files:
            try:
                content = py_file.read_text(encoding="utf-8", errors="ignore")
                if re.search(r'(validate|sanitize|InputSpec|InputValidator)', content):
                    validation_patterns_found += 1
            except Exception:
                pass

        if python_files and validation_patterns_found == 0:
            findings.append(Finding(
                category="input_validation", severity="medium",
                title="No input validation patterns detected",
                description=(
                    "No input validation or sanitization code was detected in the project. "
                    "ML APIs should validate input shape, dtype, range, and size."
                ),
                remediation="Implement input validation. See SKILL.md section 10.2.",
            ))

        severities = [f.severity for f in findings]
        if "critical" in severities or "high" in severities:
            status = "failed"
        elif "medium" in severities:
            status = "warnings"
        elif findings:
            status = "passed"
        else:
            status = "passed"

        return ScanResult(
            scanner_name=self.name, status=status,
            findings=findings, duration_seconds=time.time() - start,
            metadata={"files_scanned": len(python_files)},
        )


# ---------------------------------------------------------------------------
# 3. Model Artifact Integrity Scanner
# ---------------------------------------------------------------------------

class ArtifactIntegrityScanner(BaseScanner):
    """Verify model artifact integrity against a signed manifest."""

    name = "artifact_integrity"

    def scan(self, model_path: str = None, manifest_path: str = None,
             signing_key: str = None, pickle_severity: str = "high",
             **kwargs) -> ScanResult:
        """
        Args:
            pickle_severity: severity assigned to pickle-based formats
                (.pkl/.pickle/.joblib). Defaults to "high", which makes the
                overall scan FAIL (exit 2). Teams whose registry is legitimately
                joblib-based can lower this to "medium" (warning) or "low" via
                --pickle-severity so the gate stays useful.
        """
        start = time.time()
        findings = []

        if model_path is None:
            return ScanResult(
                scanner_name=self.name, status="skipped",
                findings=[Finding(
                    category="artifact_integrity", severity="info",
                    title="No model path provided",
                    description="Skipping artifact integrity scan.",
                )],
                duration_seconds=time.time() - start,
            )

        model_file = Path(model_path)
        if not model_file.exists():
            findings.append(Finding(
                category="artifact_integrity", severity="high",
                title="Model file not found",
                description=f"'{model_path}' does not exist.",
            ))
            return ScanResult(
                scanner_name=self.name, status="error",
                findings=findings, duration_seconds=time.time() - start,
            )

        # Compute model hash
        model_hash = self._compute_hash(model_path)
        findings.append(Finding(
            category="artifact_integrity", severity="info",
            title="Model hash computed",
            description=f"SHA-256: {model_hash}",
            metadata={"hash": model_hash},
        ))

        # Check file extension safety
        ext = model_file.suffix.lower()
        unsafe_extensions = {".pkl", ".pickle", ".joblib"}
        safe_extensions = {".onnx", ".safetensors", ".tflite"}
        # Formats that carry executable graph/config content but not raw pickle.
        code_bearing_extensions = {
            ".h5": (
                "Keras HDF5 models store layer configuration, including Lambda layers whose "
                "Python code is deserialized (marshalled bytecode) on load. Loading an "
                "untrusted .h5 can therefore execute attacker-controlled code."
            ),
            ".keras": (
                "The .keras v3 archive also serializes Lambda layers and custom objects; "
                "Keras 3 refuses them unless safe_mode=False, but older loaders do not."
            ),
            ".pb": (
                "TensorFlow SavedModel graphs can contain ops that read/write the filesystem "
                "or shell out (e.g. py_function, ReadFile/WriteFile). Treat untrusted "
                "SavedModels as untrusted code."
            ),
        }

        if ext in unsafe_extensions:
            findings.append(Finding(
                category="artifact_integrity", severity=pickle_severity,
                title=f"Unsafe serialization format: {ext}",
                description=(
                    f"The model uses {ext} format which is vulnerable to arbitrary code "
                    f"execution during deserialization."
                ),
                remediation=(
                    "Convert to a safer format: ONNX (.onnx), SafeTensors (.safetensors), "
                    "or TFLite (.tflite)."
                ),
            ))
        elif ext in code_bearing_extensions:
            findings.append(Finding(
                category="artifact_integrity", severity="medium",
                title=f"Format can carry executable content: {ext}",
                description=code_bearing_extensions[ext],
                remediation=(
                    "Load only from trusted, hash-verified sources. Prefer weights-only "
                    "formats (.safetensors) or export to ONNX. For Keras, load with "
                    "safe_mode=True (default in Keras 3) and reject Lambda layers."
                ),
            ))
        elif ext in safe_extensions:
            findings.append(Finding(
                category="artifact_integrity", severity="info",
                title=f"Safe serialization format: {ext}",
                description=(
                    "Model uses a weights/graph-only format that does not deserialize "
                    "Python objects."
                ),
            ))
        elif ext == ".pt" or ext == ".pth":
            findings.append(Finding(
                category="artifact_integrity", severity="medium",
                title="PyTorch native format detected",
                description=(
                    "PyTorch .pt/.pth files use pickle internally. Use weights_only=True "
                    "when loading, or convert to safetensors."
                ),
                remediation="pip install safetensors && convert to .safetensors format.",
            ))

        # Check model file size (unusually small or large)
        file_size = model_file.stat().st_size
        if file_size < 1024:  # Less than 1KB
            findings.append(Finding(
                category="artifact_integrity", severity="medium",
                title="Suspiciously small model file",
                description=f"Model file is only {file_size} bytes. This may indicate corruption.",
            ))
        if file_size > 10 * 1024 * 1024 * 1024:  # Greater than 10GB
            findings.append(Finding(
                category="artifact_integrity", severity="low",
                title="Very large model file",
                description=f"Model file is {file_size / (1024**3):.1f} GB.",
            ))

        # Verify against manifest if provided
        if manifest_path:
            findings.extend(
                self._verify_manifest(model_path, model_hash, manifest_path, signing_key)
            )
        else:
            findings.append(Finding(
                category="artifact_integrity", severity="medium",
                title="No manifest provided for verification",
                description="Model artifact has no signed manifest for integrity verification.",
                remediation="Generate a signed manifest using the model signing utilities.",
            ))

        severities = [f.severity for f in findings]
        if "critical" in severities or "high" in severities:
            status = "failed"
        elif "medium" in severities:
            status = "warnings"
        else:
            status = "passed"

        return ScanResult(
            scanner_name=self.name, status=status,
            findings=findings, duration_seconds=time.time() - start,
            metadata={"model_hash": model_hash, "file_size": file_size},
        )

    def _compute_hash(self, file_path: str, algorithm: str = "sha256") -> str:
        h = hashlib.new(algorithm)
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _verify_manifest(self, model_path: str, model_hash: str,
                         manifest_path: str, signing_key: str = None) -> List[Finding]:
        findings = []
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            findings.append(Finding(
                category="artifact_integrity", severity="high",
                title="Invalid manifest file",
                description=f"Could not parse manifest: {e}",
            ))
            return findings

        # Check hash match
        manifest_hash = manifest.get("model_hash", "")
        if manifest_hash != model_hash:
            findings.append(Finding(
                category="artifact_integrity", severity="critical",
                title="Model hash mismatch!",
                description=(
                    f"Manifest hash: {manifest_hash}\n"
                    f"Computed hash: {model_hash}\n"
                    "The model artifact may have been tampered with."
                ),
                remediation="Do NOT use this model. Investigate provenance and re-download.",
            ))
        else:
            findings.append(Finding(
                category="artifact_integrity", severity="info",
                title="Model hash matches manifest",
                description="Artifact integrity verified via hash comparison.",
            ))

        # Verify signature if signing key is provided
        if signing_key and "signature" in manifest:
            import hmac as hmac_mod
            signature = manifest.pop("signature")
            manifest_bytes = json.dumps(manifest, sort_keys=True).encode()
            expected = hmac_mod.new(
                signing_key.encode(), manifest_bytes, hashlib.sha256
            ).hexdigest()
            manifest["signature"] = signature
            if not hmac_mod.compare_digest(signature, expected):
                findings.append(Finding(
                    category="artifact_integrity", severity="critical",
                    title="Manifest signature verification FAILED",
                    description="The manifest signature does not match. Possible tampering.",
                    remediation="Do NOT use this model. Re-sign from a trusted source.",
                ))
            else:
                findings.append(Finding(
                    category="artifact_integrity", severity="info",
                    title="Manifest signature verified",
                    description="HMAC signature matches.",
                ))
        elif signing_key and "signature" not in manifest:
            findings.append(Finding(
                category="artifact_integrity", severity="medium",
                title="Manifest is not signed",
                description="Signing key provided but manifest has no signature field.",
                remediation="Re-generate the manifest with signing enabled.",
            ))

        return findings


# ---------------------------------------------------------------------------
# 4. Dependency Vulnerability Scanner
# ---------------------------------------------------------------------------

class DependencyScanner(BaseScanner):
    """Scan Python dependencies for known vulnerabilities."""

    name = "dependency_vulnerabilities"

    def scan(self, requirements_path: str = None, project_dir: str = ".", **kwargs) -> ScanResult:
        start = time.time()
        findings = []

        # Find requirements file
        if requirements_path is None:
            for candidate in ["requirements.txt", "requirements-lock.txt",
                              "requirements.in", "setup.cfg", "pyproject.toml"]:
                check_path = Path(project_dir) / candidate
                if check_path.exists():
                    requirements_path = str(check_path)
                    break

        if requirements_path is None:
            findings.append(Finding(
                category="dependencies", severity="medium",
                title="No requirements file found",
                description="Could not find a Python requirements/dependency file.",
                remediation="Create a requirements.txt or pyproject.toml.",
            ))

        # Try pip-audit
        pip_audit_findings = self._run_pip_audit(requirements_path)
        findings.extend(pip_audit_findings)

        # Try safety check
        safety_findings = self._run_safety(requirements_path)
        findings.extend(safety_findings)

        # Check for known risky packages
        if requirements_path and Path(requirements_path).exists():
            findings.extend(self._check_risky_packages(requirements_path))

        severities = [f.severity for f in findings]
        if "critical" in severities or "high" in severities:
            status = "failed"
        elif "medium" in severities:
            status = "warnings"
        elif findings:
            status = "passed"
        else:
            status = "passed"

        return ScanResult(
            scanner_name=self.name, status=status,
            findings=findings, duration_seconds=time.time() - start,
            metadata={"requirements_path": requirements_path},
        )

    def _run_pip_audit(self, requirements_path: str = None) -> List[Finding]:
        findings = []
        try:
            cmd = ["pip-audit", "--format", "json"]
            if requirements_path and Path(requirements_path).exists():
                cmd.extend(["--requirement", requirements_path])
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120,
            )
            # pip-audit exit codes: 0 = no vulnerabilities, 1 = vulnerabilities
            # found (JSON still on stdout), >1 = tool error. Parsing only on 0
            # would silently drop every real finding.
            if result.returncode in (0, 1):
                try:
                    data = json.loads(result.stdout)
                except json.JSONDecodeError as e:
                    findings.append(Finding(
                        category="dependencies", severity="medium",
                        title="pip-audit output could not be parsed",
                        description=f"{e}. stderr: {result.stderr[:300]}",
                        remediation="Run pip-audit manually to inspect the failure.",
                    ))
                    return findings
                # --format json emits {"dependencies": [...]}; older versions
                # emit a bare list of dependency objects.
                deps = data.get("dependencies", []) if isinstance(data, dict) else data
                for dep in deps:
                    for vuln in dep.get("vulns", []):
                        severity = self._cvss_to_severity(vuln.get("fix_versions", []))
                        findings.append(Finding(
                            category="dependencies",
                            severity=severity,
                            title=f"Vulnerability in {dep.get('name')} {dep.get('version')}",
                            description=f"{vuln.get('id', 'Unknown')}: {vuln.get('description', '')}",
                            remediation=f"Upgrade to: {', '.join(vuln.get('fix_versions', ['latest']))}",
                            metadata={"vuln_id": vuln.get("id"), "package": dep.get("name")},
                        ))
                if not any(dep.get("vulns") for dep in deps):
                    findings.append(Finding(
                        category="dependencies", severity="info",
                        title="pip-audit: No vulnerabilities found",
                        description="All scanned packages are free of known vulnerabilities.",
                    ))
            else:
                # Tool error: report it instead of letting the scan look clean.
                findings.append(Finding(
                    category="dependencies", severity="medium",
                    title=f"pip-audit failed (exit {result.returncode})",
                    description=(result.stderr or result.stdout)[:500]
                                or "pip-audit produced no output.",
                    remediation="Fix the pip-audit invocation; the dependency scan did not run.",
                ))
        except FileNotFoundError:
            findings.append(Finding(
                category="dependencies", severity="low",
                title="pip-audit not installed",
                description="pip-audit is not available for dependency scanning.",
                remediation="pip install pip-audit",
            ))
        except subprocess.TimeoutExpired:
            findings.append(Finding(
                category="dependencies", severity="low",
                title="pip-audit timed out",
                description="Dependency scan timed out after 120 seconds.",
            ))
        except Exception as e:
            logger.warning(f"pip-audit error: {e}")
        return findings

    def _run_safety(self, requirements_path: str = None) -> List[Finding]:
        """
        Run Safety and parse its JSON.

        Output shape differs by major version:
          - safety 1.x  : bare list of 5+ element lists
          - safety 2.x  : {"vulnerabilities": [{package_name, analyzed_version, ...}]}
          - safety 3.x  : `safety scan --output json` (the `check` subcommand is
                          deprecated) with vulnerabilities nested under
                          scan_results/dependencies
        A non-zero exit code means "vulnerabilities found", so the JSON must be
        parsed regardless of the return code.
        """
        findings = []
        # Prefer the modern `scan` subcommand, fall back to legacy `check`.
        commands = [["safety", "scan", "--output", "json"]]
        legacy = ["safety", "check", "--json"]
        if requirements_path and Path(requirements_path).exists():
            commands[0] += ["--target", str(Path(requirements_path).parent or ".")]
            legacy += ["--file", requirements_path]
        commands.append(legacy)

        for cmd in commands:
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=120,
                )
            except FileNotFoundError:
                return findings  # safety not installed; nothing to report
            except subprocess.TimeoutExpired:
                findings.append(Finding(
                    category="dependencies", severity="low",
                    title="safety timed out",
                    description="Safety scan timed out after 120 seconds.",
                ))
                return findings
            except Exception as e:
                logger.warning(f"safety error: {e}")
                return findings

            if not result.stdout.strip():
                continue
            try:
                data = json.loads(result.stdout)
            except json.JSONDecodeError:
                continue
            parsed = self._parse_safety_json(data)
            if parsed:
                return parsed
            # Valid JSON with zero vulnerabilities: report the clean result.
            return [Finding(
                category="dependencies", severity="info",
                title="safety: No vulnerabilities found",
                description=f"Safety ({cmd[1]}) reported no known vulnerabilities.",
            )]
        return findings

    @staticmethod
    def _parse_safety_json(data: Any) -> List[Finding]:
        """Normalize Safety output (1.x/2.x/3.x) into Findings."""
        findings: List[Finding] = []

        def add(pkg, version, advisory, vuln_id, fixed):
            findings.append(Finding(
                category="dependencies", severity="high",
                title=f"Safety: {pkg} {version}",
                description=f"{vuln_id or 'advisory'}: {advisory or ''}",
                remediation=(f"Upgrade to: {fixed}" if fixed
                             else "Upgrade to a patched version."),
                metadata={"package": pkg, "vuln_id": vuln_id},
            ))

        if isinstance(data, list):
            # safety 1.x: [[package, affected_spec, installed, advisory, id], ...]
            for vuln in data:
                if isinstance(vuln, list) and len(vuln) >= 5:
                    add(vuln[0], vuln[2], vuln[3], vuln[4], None)
                elif isinstance(vuln, dict):
                    add(vuln.get("package_name"), vuln.get("analyzed_version"),
                        vuln.get("advisory"), vuln.get("vulnerability_id"),
                        vuln.get("fixed_versions"))
            return findings

        if not isinstance(data, dict):
            return findings

        # safety 2.x
        for vuln in data.get("vulnerabilities", []) or []:
            if isinstance(vuln, dict):
                add(vuln.get("package_name"), vuln.get("analyzed_version"),
                    vuln.get("advisory"), vuln.get("vulnerability_id"),
                    vuln.get("fixed_versions"))

        # safety 3.x: scan_results -> projects/files -> dependencies -> specifications
        scan_results = data.get("scan_results") or {}
        containers = []
        if isinstance(scan_results, dict):
            containers = (scan_results.get("projects") or []) + \
                         (scan_results.get("files") or [])
        for container in containers:
            for dep in (container.get("dependencies") or []):
                for spec in (dep.get("specifications") or []):
                    known = ((spec.get("vulnerabilities") or {})
                             .get("known_vulnerabilities") or [])
                    for vuln in known:
                        add(dep.get("name"), spec.get("raw"),
                            vuln.get("advisory") or vuln.get("id"),
                            vuln.get("id"),
                            ", ".join(vuln.get("fixed_versions", []) or []) or None)
        return findings

    def _check_risky_packages(self, requirements_path: str) -> List[Finding]:
        """Check for packages with known security concerns."""
        findings = []
        risky = {
            "pickle5": "Extends pickle with known deserialization risks.",
            "dill": "Extends pickle; can serialize arbitrary objects including code.",
            "cloudpickle": "Powerful pickle extension with code serialization.",
        }
        try:
            content = Path(requirements_path).read_text()
            for pkg, reason in risky.items():
                if re.search(rf'^{pkg}\b', content, re.MULTILINE | re.IGNORECASE):
                    findings.append(Finding(
                        category="dependencies", severity="medium",
                        title=f"Risky package: {pkg}",
                        description=reason,
                        remediation="Evaluate if this package is necessary; prefer safer alternatives.",
                    ))
        except Exception:
            pass
        return findings

    @staticmethod
    def _cvss_to_severity(fix_versions: list) -> str:
        # Heuristic: if no fix exists, it is higher severity
        if not fix_versions:
            return "high"
        return "medium"


# ---------------------------------------------------------------------------
# 5. PII Detection Scanner
# ---------------------------------------------------------------------------

def _luhn_valid(digits: str) -> bool:
    """Luhn (mod-10) checksum used by payment card numbers."""
    nums = [int(c) for c in digits if c.isdigit()]
    if len(nums) < 12:
        return False
    total = 0
    for i, d in enumerate(reversed(nums)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


class PIIScanner(BaseScanner):
    """
    Detect PII (Personally Identifiable Information) in training data.

    SCOPE: format-based detection only (regex + a Luhn checksum for card
    numbers). It cannot find names, addresses, or free-text identifiers, so it
    is a pre-filter, not evidence of HIPAA Safe Harbor de-identification. For
    compliance-grade discovery use Microsoft Presidio, AWS Comprehend
    (DetectPiiEntities / Comprehend Medical), or GCP DLP.
    """

    name = "pii_detection"

    # Extra validators applied after a regex hit; a False result drops the match.
    VALIDATORS = {"credit_card": _luhn_valid}

    PII_PATTERNS = {
        "email": (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "high"),
        "phone_us": (r'\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', "medium"),
        "ssn": (r'\b\d{3}-\d{2}-\d{4}\b', "critical"),
        "credit_card": (r'\b(?:\d{4}[-\s]?){3}\d{4}\b', "critical"),
        "ip_address": (r'\b(?:\d{1,3}\.){3}\d{1,3}\b', "low"),
        "date_of_birth": (
            r'\b(?:0[1-9]|1[0-2])/(?:0[1-9]|[12]\d|3[01])/(?:19|20)\d{2}\b', "medium"
        ),
        "aws_key": (r'AKIA[0-9A-Z]{16}', "critical"),
        "generic_secret": (
            r'(?i)(?:password|secret|token|api_key|apikey)\s*[:=]\s*["\']?[A-Za-z0-9+/=]{8,}',
            "critical",
        ),
    }

    def scan(self, data_path: str = None, sample_size: int = 10000, **kwargs) -> ScanResult:
        start = time.time()
        findings = []

        if data_path is None:
            return ScanResult(
                scanner_name=self.name, status="skipped",
                findings=[Finding(
                    category="pii", severity="info",
                    title="No data path provided",
                    description="Skipping PII detection scan.",
                )],
                duration_seconds=time.time() - start,
            )

        data_file = Path(data_path)
        if not data_file.exists():
            findings.append(Finding(
                category="pii", severity="medium",
                title="Data file not found",
                description=f"'{data_path}' does not exist.",
            ))
            return ScanResult(
                scanner_name=self.name, status="error",
                findings=findings, duration_seconds=time.time() - start,
            )

        # Read and scan data
        try:
            text_content = self._read_data(data_path, sample_size)
            pii_counts = {}
            for pii_type, (pattern, severity) in self.PII_PATTERNS.items():
                matches = [m.group() for m in re.finditer(pattern, text_content)]
                validator = self.VALIDATORS.get(pii_type)
                if validator:
                    matches = [m for m in matches if validator(m)]
                if matches:
                    pii_counts[pii_type] = len(matches)
                    # Redact examples for the report
                    examples = [self._redact(m) for m in matches[:3]]
                    findings.append(Finding(
                        category="pii",
                        severity=severity,
                        title=f"PII detected: {pii_type} ({len(matches)} instances)",
                        description=(
                            f"Found {len(matches)} potential {pii_type} pattern(s). "
                            f"Examples (redacted): {examples}"
                        ),
                        remediation=(
                            f"Remove or mask {pii_type} data before training. "
                            f"See scripts/privacy_guard.py for anonymization utilities."
                        ),
                        metadata={"count": len(matches), "pii_type": pii_type},
                    ))

            if not pii_counts:
                findings.append(Finding(
                    category="pii", severity="info",
                    title="No PII patterns detected",
                    description=(
                        f"Scanned {len(text_content)} characters. "
                        "No common PII patterns found."
                    ),
                ))

        except Exception as e:
            findings.append(Finding(
                category="pii", severity="medium",
                title="Error scanning data for PII",
                description=str(e),
            ))

        severities = [f.severity for f in findings]
        if "critical" in severities:
            status = "failed"
        elif "high" in severities:
            status = "failed"
        elif "medium" in severities:
            status = "warnings"
        else:
            status = "passed"

        return ScanResult(
            scanner_name=self.name, status=status,
            findings=findings, duration_seconds=time.time() - start,
        )

    def _read_data(self, data_path: str, sample_size: int) -> str:
        """Read data file and return text content for PII scanning."""
        ext = Path(data_path).suffix.lower()

        if ext == ".csv":
            try:
                import pandas as pd
                df = pd.read_csv(data_path, nrows=sample_size)
                return df.to_string()
            except ImportError:
                pass

        if ext == ".parquet":
            try:
                import pandas as pd
                df = pd.read_parquet(data_path)
                if len(df) > sample_size:
                    df = df.head(sample_size)
                return df.to_string()
            except ImportError:
                pass

        if ext in {".json", ".jsonl"}:
            lines = []
            with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f):
                    if i >= sample_size:
                        break
                    lines.append(line)
            return "\n".join(lines)

        # Default: read as text
        with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read(sample_size * 500)  # ~500 chars per record estimate
        return content

    @staticmethod
    def _redact(value: str) -> str:
        """Partially redact a PII value for safe display."""
        if len(value) <= 4:
            return "****"
        return value[:2] + "*" * (len(value) - 4) + value[-2:]


# ---------------------------------------------------------------------------
# Main scanner orchestrator
# ---------------------------------------------------------------------------

SCANNERS = {
    # "robustness" kept as the CLI name for backwards compatibility; the
    # scanner is a load smoke check, not an attack runner (see its docstring).
    "robustness": ModelLoadSmokeCheckScanner,
    "input_validation": InputValidationScanner,
    "artifact": ArtifactIntegrityScanner,
    "dependencies": DependencyScanner,
    "pii": PIIScanner,
}


def run_security_scan(
    scans: List[str],
    model_path: str = None,
    data_path: str = None,
    manifest_path: str = None,
    signing_key: str = None,
    framework: str = "pytorch",
    requirements_path: str = None,
    project_dir: str = ".",
    epsilon: float = 0.03,
    attacks: List[str] = None,
    output_path: str = None,
    pickle_severity: str = "high",
    allow_unsafe_load: bool = False,
) -> SecurityReport:
    """Run specified security scans and generate a report."""

    report = SecurityReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        model_path=model_path,
        data_path=data_path,
    )

    if "all" in scans:
        scans = list(SCANNERS.keys())

    for scan_name in scans:
        scanner_cls = SCANNERS.get(scan_name)
        if scanner_cls is None:
            logger.warning(f"Unknown scan type: {scan_name}. Available: {list(SCANNERS.keys())}")
            continue

        logger.info(f"Running scanner: {scan_name}")
        scanner = scanner_cls()
        # Every scanner accepts **kwargs, so a single call site is enough; a
        # TypeError here is a real bug and must not be swallowed.
        result = scanner.scan(
            model_path=model_path,
            data_path=data_path,
            manifest_path=manifest_path,
            signing_key=signing_key,
            framework=framework,
            requirements_path=requirements_path,
            project_dir=project_dir,
            epsilon=epsilon,
            attacks=attacks,
            pickle_severity=pickle_severity,
            allow_unsafe_load=allow_unsafe_load,
        )
        report.scan_results.append(result)

    # Output
    report.print_summary()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report.to_json())
        logger.info(f"Report written to {output_path}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="ML Security Scanner — scan models, data, and code for security issues.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --scan all --model model.pt --data train.csv
  %(prog)s --scan robustness --model model.pt --framework pytorch --epsilon 0.05
  %(prog)s --scan dependencies --requirements requirements.txt
  %(prog)s --scan pii --data dataset.csv
  %(prog)s --scan artifact --model model.onnx --manifest manifest.json
  %(prog)s --scan robustness,input_validation --model model.pt --project-dir ./src
        """,
    )
    parser.add_argument(
        "--scan", type=str, required=True,
        help=f"Comma-separated list of scans to run. Options: {', '.join(SCANNERS.keys())}, all",
    )
    parser.add_argument("--model", type=str, default=None, help="Path to model artifact")
    parser.add_argument("--data", type=str, default=None, help="Path to training data file")
    parser.add_argument("--manifest", type=str, default=None, help="Path to model manifest JSON")
    parser.add_argument("--signing-key", type=str, default=None, help="HMAC signing key for manifest")
    parser.add_argument("--framework", type=str, default="pytorch",
                        choices=["pytorch", "tensorflow", "sklearn", "onnx"],
                        help="ML framework (default: pytorch)")
    parser.add_argument("--requirements", type=str, default=None, help="Path to requirements.txt")
    parser.add_argument("--project-dir", type=str, default=".", help="Project directory to scan")
    parser.add_argument("--epsilon", type=float, default=0.03, help="Adversarial epsilon (default: 0.03)")
    parser.add_argument("--attacks", type=str, default=None, help="Comma-separated attack types (fgsm,pgd)")
    parser.add_argument("--pickle-severity", type=str, default="high",
                        choices=["critical", "high", "medium", "low", "info"],
                        help="Severity for pickle-based model formats (.pkl/.joblib). "
                             "Default 'high' fails the scan; lower it if your registry "
                             "is legitimately joblib-based (default: high)")
    parser.add_argument("--allow-unsafe-load", action="store_true",
                        help="Permit pickle deserialization of the model under audit "
                             "(torch.load weights_only=False, joblib.load). Off by default; "
                             "only use in a sandbox")
    parser.add_argument("--output", type=str, default=None, help="Path to write JSON report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    scans = [s.strip() for s in args.scan.split(",")]
    attacks = [a.strip() for a in args.attacks.split(",")] if args.attacks else None

    report = run_security_scan(
        scans=scans,
        model_path=args.model,
        data_path=args.data,
        manifest_path=args.manifest,
        signing_key=args.signing_key,
        framework=args.framework,
        requirements_path=args.requirements,
        project_dir=args.project_dir,
        epsilon=args.epsilon,
        attacks=attacks,
        output_path=args.output,
        pickle_severity=args.pickle_severity,
        allow_unsafe_load=args.allow_unsafe_load,
    )

    # Exit code based on overall status
    status = report.overall_status()
    if status == "FAILED":
        sys.exit(2)
    elif status == "ERROR":
        sys.exit(1)
    elif status == "WARNINGS":
        sys.exit(0)  # Warnings are non-fatal
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
