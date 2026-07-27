---
name: llm-guardrails
description: >
  Implement safety guardrails for LLM applications. Covers input validation and content filtering, output validation
  and safety checks, NeMo Guardrails, Guardrails AI, LLM Guard, toxicity detection, PII detection and redaction,
  hallucination detection and mitigation, jailbreak prevention, topic restriction, output format enforcement,
  rate limiting, content moderation pipelines, red teaming, safety benchmarks, and compliance requirements.
  Use when adding safety measures to LLM apps, preventing harmful outputs, detecting PII, or building
  content moderation systems.
license: Apache-2.0
metadata:
  author: llmops-skills
  version: "1.0"
  category: llmops
---

# LLM Guardrails

## Overview

Guardrails protect LLM applications from producing harmful, inaccurate, or non-compliant
outputs. They act as safety layers between users and the LLM.

Tested with: `guardrails-ai>=0.10`, `nemoguardrails>=0.23`, `llm-guard>=0.3`, `openai>=1.0`.

## Guardrail Architecture

```
User Input → [Input Guardrails] → LLM → [Output Guardrails] → User Response
                │                              │
                ├─ PII Detection               ├─ Toxicity Check
                ├─ Jailbreak Detection         ├─ Hallucination Check
                ├─ Topic Filtering             ├─ PII Redaction
                └─ Input Validation            ├─ Format Validation
                                               └─ Factual Grounding
```

## Step-by-Step Instructions

### 1. Input Guardrails

```python
import json
import re
from typing import Tuple

class InputGuardrails:
    def __init__(self):
        # Order matters for redaction: longest/most specific first, otherwise the
        # phone pattern consumes 10 digits of a card number and leaks the rest.
        self.pii_patterns = {
            "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        }
        # Substring blocklist: a defence-in-depth supplement only. It is trivially
        # bypassed by paraphrase, translation, or encoding. Always pair it with a
        # trained detector (Prompt Guard 2, LLM Guard PromptInjection).
        self.jailbreak_patterns = [
            "ignore previous instructions",
            "ignore all instructions",
            "disregard your training",
            "pretend you are",
            "you are now",
            "dan mode",
            "developer mode",
        ]

    def check_pii(self, text: str) -> dict:
        """Detect PII in user input."""
        findings = {}
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                findings[pii_type] = len(matches)
        return findings

    def redact_pii(self, text: str) -> str:
        """Redact PII from text."""
        redacted = text
        for pii_type, pattern in self.pii_patterns.items():
            redacted = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", redacted)
        return redacted

    def check_jailbreak(self, text: str) -> Tuple[bool, str]:
        """Detect potential jailbreak attempts."""
        text_lower = text.lower()
        for pattern in self.jailbreak_patterns:
            if pattern in text_lower:
                return True, f"Jailbreak pattern detected: '{pattern}'"
        return False, ""

    def validate(self, text: str) -> dict:
        """Run all input validations."""
        pii = self.check_pii(text)
        jailbreak, reason = self.check_jailbreak(text)
        return {
            "passed": not jailbreak and not pii,
            "pii_detected": pii,
            "jailbreak_detected": jailbreak,
            "jailbreak_reason": reason,
            "sanitized_text": self.redact_pii(text) if pii else text,
        }
```

### 2. Output Guardrails

```python
class OutputGuardrails:
    def __init__(self, llm_judge=None):
        self.llm_judge = llm_judge
        # Load the classifier once. Instantiating a transformers pipeline per call
        # costs seconds and blows the <200ms guardrail latency budget.
        from transformers import pipeline
        self.toxicity_clf = pipeline("text-classification",
                                     model="unitary/toxic-bert", truncation=True)

    def check_toxicity(self, text: str, threshold: float = 0.5) -> dict:
        """Check output for toxic content."""
        # truncation=True already caps at the model's token limit; the [:2000]
        # char guard just avoids tokenizing an unbounded string.
        result = self.toxicity_clf(text[:2000])[0]
        is_toxic = result["label"] == "toxic" and result["score"] > threshold
        return {"toxic": is_toxic, "score": result["score"]}

    def check_hallucination(self, response: str, context: str) -> dict:
        """Check if response is grounded in provided context."""
        if not self.llm_judge:
            return {"checked": False}

        prompt = f"""Determine if the response is fully supported by the context.

Context: {context}

Response: {response}

Is every claim in the response supported by the context?
Respond with JSON: {{"grounded": true/false, "unsupported_claims": ["..."]}}"""

        result = self.llm_judge(prompt)
        return result

    def check_pii_leakage(self, text: str) -> dict:
        """Check if output leaks PII."""
        guardrail = InputGuardrails()
        pii = guardrail.check_pii(text)
        return {
            "pii_leaked": bool(pii),
            "pii_types": list(pii.keys()),
            "redacted": guardrail.redact_pii(text) if pii else text,
        }

    def validate_format(self, text: str, expected_format: str = "json") -> dict:
        """Validate output format."""
        if expected_format == "json":
            try:
                json.loads(text)
                return {"valid": True}
            except json.JSONDecodeError as e:
                return {"valid": False, "error": str(e)}
        return {"valid": True}
```

### 3. Guardrails AI Integration

```python
from guardrails import Guard
from guardrails.hub import ToxicLanguage, DetectPII, RestrictToTopic

# Create guard with validators
guard = Guard().use_many(
    ToxicLanguage(threshold=0.5, on_fail="exception"),
    DetectPII(pii_entities=["EMAIL_ADDRESS", "PHONE_NUMBER"], on_fail="fix"),
    RestrictToTopic(
        valid_topics=["customer support", "product information"],
        invalid_topics=["politics", "religion"],
        on_fail="exception",
    ),
)

# Use with LLM. Guardrails >=0.5 takes a LiteLLM-style model string plus
# messages; the old llm_api=<callable> + prompt= form is removed.
result = guard(
    model="gpt-5-mini",
    messages=[{"role": "user",
               "content": f"Answer the customer question: {user_question}"}],
)

if result.validation_passed:
    print(result.validated_output)
else:
    print(f"Guardrail violated: {result.error}")
```

### 4. NeMo Guardrails

NeMo Guardrails loads every `.yml`/`.co` file in the config directory; the main
settings file must be named `config.yml` (not `rails.yaml`), and Colang flows
live in sibling `.co` files.

```yaml
# config/config.yml
models:
  - type: main
    engine: openai
    model: gpt-5-mini

rails:
  input:
    flows:
      - check jailbreak
      - check pii
  output:
    flows:
      - check toxicity
      - check factuality

instructions:
  - type: general
    content: |
      You are a helpful customer support assistant.
      Only answer questions about our products and services.
      Never discuss politics, religion, or competitors.
```

### 5. Complete Guardrail Pipeline

```python
class GuardrailPipeline:
    def __init__(self, input_guardrails, output_guardrails, llm_fn):
        self.input_guard = input_guardrails
        self.output_guard = output_guardrails
        self.llm = llm_fn

    def __call__(self, user_input, context=None):
        # Step 1: Validate input
        input_result = self.input_guard.validate(user_input)
        if not input_result["passed"]:
            if input_result["jailbreak_detected"]:
                return {"response": "I cannot process this request.", "blocked": True}
            # Use sanitized input
            user_input = input_result["sanitized_text"]

        # Step 2: Generate response
        response = self.llm(user_input)

        # Step 3: Validate output
        toxicity = self.output_guard.check_toxicity(response)
        if toxicity["toxic"]:
            return {"response": "I cannot provide this response.", "blocked": True}

        pii_check = self.output_guard.check_pii_leakage(response)
        if pii_check["pii_leaked"]:
            response = pii_check["redacted"]

        if context:
            grounding = self.output_guard.check_hallucination(response, context)

        return {
            "response": response,
            "blocked": False,
            "pii_redacted": pii_check["pii_leaked"],
        }
```

## Best Practices

1. **Separate trusted from untrusted text** - Put policy in the `system` role and
   never concatenate user input or retrieved documents into it. Wrap untrusted
   content in delimiters and tell the model that text inside them is data, not
   instructions. This structural defense holds up better than any pattern list.
2. **Defense in depth** - Apply guardrails at both input and output
3. **Fail safe** - When guardrails are uncertain, be conservative
4. **Log all violations** for analysis and improvement
5. **Red team regularly** - Test guardrails with adversarial inputs
6. **Layer multiple checks** - No single guardrail catches everything
7. **Keep guardrails fast** - Don't add > 200ms latency (load classifiers once at
   startup, not per request)
8. **Update patterns** - New jailbreak techniques emerge constantly
9. **User-facing explanations** - Tell users why their request was blocked
10. **Monitor false positive rate** - Too aggressive = bad UX

## Scripts

- `scripts/guardrails_pipeline.py` - Complete guardrail pipeline
- `scripts/red_team.py` - Red teaming test suite for LLM safety

## References

See [references/REFERENCE.md](references/REFERENCE.md) for tool comparisons and safety benchmarks.

## Related skills

**Upstream:** `llm-deployment` (live endpoint to protect) · **Downstream:** `llm-observability` (violation events to log and alert on)
**See also:** `llm-prompt-engineering` for injection-resistant prompt design · `ml-security` for the broader ML threat model
