#!/usr/bin/env python3
"""Input/output guardrails pipeline for LLMs.

Usage:
    python guardrails_pipeline.py check-input --input "Tell me John's SSN 123-45-6789"
    python guardrails_pipeline.py check-output --input "The answer is ..." --context "source doc"
    python guardrails_pipeline.py run-pipeline --input "Summarize this" --model gpt-4o-mini
    python guardrails_pipeline.py check-input --input "some text" --config guardrails.yaml
"""
import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}
TOXICITY_KEYWORDS = [
    "kill", "murder", "hate", "slur", "terrorist", "bomb", "attack",
    "assault", "abuse", "torture", "suicide", "self-harm", "genocide",
]
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts|rules)",
    r"disregard\s+(your|all|any)\s+(instructions|guidelines|rules)",
    r"you\s+are\s+now\s+(DAN|evil|unrestricted|jailbroken)",
    r"pretend\s+you\s+(are|have)\s+no\s+(restrictions|filters|rules)",
    r"system\s*prompt\s*[:=]",
    r"<\|?(system|im_start|endoftext)\|?>",
    r"\[INST\]|\[/INST\]|<<SYS>>",
    r"act\s+as\s+(an?\s+)?unrestricted",
    r"bypass\s+(your\s+)?(safety|content)\s+(filters?|rules)",
]
SENSITIVE_TOPICS = [
    "medical advice", "legal advice", "financial advice", "diagnosis",
    "prescription", "lawsuit", "investment recommendation", "classified",
    "top secret", "confidential government",
]
PII_PATTERNS = {
    "email": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    "phone": r"(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
}


@dataclass
class CheckResult:
    passed: bool
    check_name: str
    details: str
    severity: str = "low"


@dataclass
class GuardrailConfig:
    max_input_length: int = 8000
    blocked_severity: str = "high"
    failure_action: str = "block"  # block, warn, redact, fallback
    fallback_response: str = "I cannot process this request due to safety constraints."
    custom_blocked_patterns: List[str] = field(default_factory=list)
    custom_allowed_topics: List[str] = field(default_factory=list)
    enabled_input_checks: List[str] = field(default_factory=lambda: [
        "injection", "pii", "toxicity", "length", "language"])
    enabled_output_checks: List[str] = field(default_factory=lambda: [
        "pii_leakage", "hallucination", "toxicity", "format", "sensitive_topic"])

    @classmethod
    def from_yaml(cls, path: str) -> "GuardrailConfig":
        try:
            import yaml
        except ImportError:
            logger.error("Install: pip install pyyaml"); sys.exit(1)
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def _detect_pii(text: str) -> Dict[str, int]:
    found = {}
    for pii_type, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, text)
        if matches:
            found[pii_type] = len(matches)
    return found


def _find_toxic(text: str) -> List[str]:
    lower = text.lower()
    return [kw for kw in TOXICITY_KEYWORDS if re.search(rf"\b{re.escape(kw)}\b", lower)]


class InputGuardrail:
    """Checks applied to user input before sending to the LLM."""
    def __init__(self, config: GuardrailConfig):
        self.config = config

    def check_all(self, text: str) -> List[CheckResult]:
        results = []
        check_map = {"injection": self._injection, "pii": self._pii,
                      "toxicity": self._toxicity, "length": self._length,
                      "language": self._language}
        for name in self.config.enabled_input_checks:
            if name in check_map:
                results.append(check_map[name](text))
        for pattern in self.config.custom_blocked_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                results.append(CheckResult(False, "custom_pattern",
                               f"Matched blocked pattern: {pattern}", "high"))
        return results

    def _injection(self, text: str) -> CheckResult:
        for pattern in INJECTION_PATTERNS:
            m = re.search(pattern, text.lower())
            if m:
                return CheckResult(False, "prompt_injection",
                                   f"Injection pattern: '{m.group()}'", "critical")
        return CheckResult(True, "prompt_injection", "No injection detected")

    def _pii(self, text: str) -> CheckResult:
        found = _detect_pii(text)
        if found:
            return CheckResult(False, "pii_detection", f"PII detected: {found}", "high")
        return CheckResult(True, "pii_detection", "No PII detected")

    def _toxicity(self, text: str) -> CheckResult:
        found = _find_toxic(text)
        if found:
            return CheckResult(False, "toxicity", f"Toxic keywords: {found}", "high")
        return CheckResult(True, "toxicity", "No toxicity detected")

    def _length(self, text: str) -> CheckResult:
        if len(text) > self.config.max_input_length:
            return CheckResult(False, "input_length",
                               f"Length {len(text)} exceeds max {self.config.max_input_length}", "medium")
        return CheckResult(True, "input_length", f"Length OK ({len(text)} chars)")

    def _language(self, text: str) -> CheckResult:
        try:
            from langdetect import detect
            lang = detect(text)
        except ImportError:
            return CheckResult(True, "language", "langdetect not installed, skipping")
        except Exception:
            return CheckResult(True, "language", "Detection failed, skipping")
        if lang == "en":
            return CheckResult(True, "language", f"Language: {lang}")
        return CheckResult(False, "language", f"Non-English detected: {lang}", "low")


class OutputGuardrail:
    """Checks applied to LLM output before returning to the user."""
    def __init__(self, config: GuardrailConfig):
        self.config = config

    def check_all(self, response: str, context: Optional[str] = None) -> List[CheckResult]:
        results = []
        check_map = {"pii_leakage": lambda: self._pii(response),
                      "hallucination": lambda: self._hallucination(response, context),
                      "toxicity": lambda: self._toxicity(response),
                      "format": lambda: self._format(response),
                      "sensitive_topic": lambda: self._sensitive(response)}
        for name in self.config.enabled_output_checks:
            if name in check_map:
                results.append(check_map[name]())
        return results

    def _pii(self, response: str) -> CheckResult:
        found = _detect_pii(response)
        if found:
            return CheckResult(False, "pii_leakage", f"PII in output: {found}", "critical")
        return CheckResult(True, "pii_leakage", "No PII leakage detected")

    def _hallucination(self, response: str, context: Optional[str]) -> CheckResult:
        if not context:
            return CheckResult(True, "hallucination", "No context provided, skipping")
        sentences = [s.strip() for s in re.split(r'[.!?]+', response) if len(s.strip()) > 20]
        if not sentences:
            return CheckResult(True, "hallucination", "Response too short to evaluate")
        ctx_words = set(context.lower().split())
        ungrounded = sum(1 for s in sentences
                         if len(set(s.lower().split()) & ctx_words) / max(len(s.split()), 1) < 0.3)
        if ungrounded > len(sentences) * 0.5:
            return CheckResult(False, "hallucination",
                               f"{ungrounded}/{len(sentences)} sentences ungrounded", "high")
        return CheckResult(True, "hallucination", "Response appears grounded in context")

    def _toxicity(self, response: str) -> CheckResult:
        found = _find_toxic(response)
        if found:
            return CheckResult(False, "output_toxicity", f"Toxic output: {found}", "high")
        return CheckResult(True, "output_toxicity", "No toxicity in output")

    def _format(self, response: str) -> CheckResult:
        issues = []
        stripped = response.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                json.loads(response)
            except json.JSONDecodeError:
                issues.append("Malformed JSON")
        if response.count("`") % 2 != 0:
            issues.append("Unclosed backtick")
        if issues:
            return CheckResult(False, "format_validation", f"Issues: {issues}", "low")
        return CheckResult(True, "format_validation", "Format OK")

    def _sensitive(self, response: str) -> CheckResult:
        lower = response.lower()
        allowed = {t.lower() for t in self.config.custom_allowed_topics}
        flagged = [t for t in SENSITIVE_TOPICS if t in lower and t not in allowed]
        if flagged:
            return CheckResult(False, "sensitive_topic", f"Sensitive topics: {flagged}", "medium")
        return CheckResult(True, "sensitive_topic", "No sensitive topics detected")


class GuardrailsPipeline:
    """Chain: input guardrails -> LLM call -> output guardrails."""
    def __init__(self, config: GuardrailConfig):
        self.config = config
        self.input_guard = InputGuardrail(config)
        self.output_guard = OutputGuardrail(config)

    def run(self, user_input: str, model: str = "gpt-4o-mini",
            context: Optional[str] = None) -> Dict[str, Any]:
        result = {"input": user_input, "model": model,
                  "input_checks": [], "output_checks": [], "blocked": False}
        # Input checks
        in_results = self.input_guard.check_all(user_input)
        result["input_checks"] = [asdict(r) for r in in_results]
        failures = [r for r in in_results if not r.passed]
        if failures and self._should_block(failures):
            result["blocked"] = True
            result["action"] = self.config.failure_action
            if self.config.failure_action == "redact":
                result["response"] = self._call_llm(self._redact_pii(user_input), model)
            else:
                result["response"] = self.config.fallback_response
            logger.warning(f"Input blocked: {[f.check_name for f in failures]}")
            return result
        # LLM call
        llm_response = self._call_llm(user_input, model)
        result["raw_response"] = llm_response
        # Output checks
        out_results = self.output_guard.check_all(llm_response, context)
        result["output_checks"] = [asdict(r) for r in out_results]
        out_failures = [r for r in out_results if not r.passed]
        if out_failures and self._should_block(out_failures):
            result["blocked"] = True
            result["action"] = self.config.failure_action
            if self.config.failure_action == "redact":
                result["response"] = self._redact_pii(llm_response)
            else:
                result["response"] = self.config.fallback_response
            logger.warning(f"Output blocked: {[f.check_name for f in out_failures]}")
        else:
            result["response"] = llm_response
        return result

    def _should_block(self, failures: List[CheckResult]) -> bool:
        threshold = SEVERITY_ORDER.get(self.config.blocked_severity, 2)
        return any(SEVERITY_ORDER.get(f.severity, 0) >= threshold for f in failures)

    @staticmethod
    def _redact_pii(text: str) -> str:
        for pii_type, pattern in PII_PATTERNS.items():
            text = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", text)
        return text

    @staticmethod
    def _call_llm(prompt: str, model: str) -> str:
        try:
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=1024)
            return resp.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"[LLM error: {e}]"


def print_results(results: List[CheckResult], label: str):
    print(f"\n{'=' * 60}\n  {label}\n{'=' * 60}")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        sev = "" if r.passed else f" [{r.severity.upper()}]"
        print(f"  [{status}]{sev} {r.check_name}: {r.details}")
    print(f"\n  Summary: {sum(1 for r in results if r.passed)}/{len(results)} checks passed")


def main():
    parser = argparse.ArgumentParser(description="LLM guardrails pipeline")
    parser.add_argument("action", choices=["check-input", "check-output", "run-pipeline"])
    parser.add_argument("--input", required=True, help="Input text to check")
    parser.add_argument("--context", default=None, help="Context for hallucination checking")
    parser.add_argument("--config", default=None, help="YAML config file path")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model for pipeline mode")
    parser.add_argument("--failure-action", choices=["block", "warn", "redact", "fallback"],
                        default=None, help="Override failure action")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    config = GuardrailConfig.from_yaml(args.config) if args.config else GuardrailConfig()
    if args.failure_action:
        config.failure_action = args.failure_action

    if args.action == "check-input":
        guard = InputGuardrail(config)
        results = guard.check_all(args.input)
        if args.json:
            print(json.dumps([asdict(r) for r in results], indent=2))
        else:
            print_results(results, "Input Guardrail Results")
    elif args.action == "check-output":
        guard = OutputGuardrail(config)
        results = guard.check_all(args.input, args.context)
        if args.json:
            print(json.dumps([asdict(r) for r in results], indent=2))
        else:
            print_results(results, "Output Guardrail Results")
    elif args.action == "run-pipeline":
        pipeline = GuardrailsPipeline(config)
        result = pipeline.run(args.input, model=args.model, context=args.context)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print_results([CheckResult(**r) for r in result["input_checks"]],
                          "Input Guardrail Results")
            if not result["blocked"] or result.get("raw_response"):
                print_results([CheckResult(**r) for r in result["output_checks"]],
                              "Output Guardrail Results")
            print(f"\n{'=' * 60}\n  Pipeline Result\n{'=' * 60}")
            print(f"  Blocked: {result['blocked']}")
            if result.get("action"):
                print(f"  Action:  {result['action']}")
            print(f"  Response: {result.get('response', 'N/A')[:200]}")


if __name__ == "__main__":
    main()
