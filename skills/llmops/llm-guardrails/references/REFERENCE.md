# LLM Guardrails Reference

## Guardrails Frameworks Comparison

| Feature | Guardrails AI | NeMo Guardrails | LLM Guard | Rebuff |
|---|---|---|---|---|
| **License** | Apache 2.0 | Apache 2.0 | Apache 2.0 | Apache 2.0 |
| **Primary Focus** | Output validation and structuring | Conversational flow control | Input/output scanning | Prompt injection detection |
| **Input Validation** | Limited (via validators) | Colang-based dialog rails | Comprehensive scanners | Specialized for injection |
| **Output Validation** | Strong (JSON, regex, semantic) | Dialog-level guardrails | Comprehensive scanners | Limited |
| **Streaming Support** | Yes | Partial | No | No |
| **Custom Rules** | Python validators | Colang scripting language | Custom scanners | Heuristic + LLM |
| **Integration Effort** | Low (decorator-based) | Medium (requires Colang) | Low (scanner pipeline) | Low (API-based) |
| **LLM Provider Agnostic** | Yes | Yes | Yes | Yes |
| **Community Size** | Large | Large (NVIDIA-backed) | Growing | Small |
| **Production Readiness** | High | High | Medium | Low |

## Input Validation Patterns

### Prompt Injection Detection

```python
# Using LLM Guard for prompt injection scanning
from llm_guard.input_scanners import PromptInjection
from llm_guard.input_scanners.prompt_injection import MatchType

scanner = PromptInjection(threshold=0.9, match_type=MatchType.FULL)
sanitized_prompt, is_valid, risk_score = scanner.scan(prompt)
```

Common prompt injection vectors to defend against:
- Direct instruction override ("Ignore previous instructions...")
- Indirect injection via retrieved context (data poisoning in RAG sources)
- Encoding-based attacks (base64, ROT13, Unicode tricks)
- Multi-turn manipulation (gradual context shifting)
- Payload splitting across multiple messages

### PII Detection

```python
# Using Presidio for PII detection in prompts
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

results = analyzer.analyze(text=user_input, language="en",
                           entities=["PHONE_NUMBER", "EMAIL_ADDRESS",
                                     "CREDIT_CARD", "US_SSN"])

anonymized = anonymizer.anonymize(text=user_input, analyzer_results=results)
```

### Topic Restriction

```python
# NeMo Guardrails Colang example for topic restriction
# config/rails.co
define user ask about competitors
  "What do you think about CompetitorX?"
  "How does CompetitorX compare?"
  "Is CompetitorX better?"

define flow
  user ask about competitors
  bot refuse to discuss competitors
  bot offer to discuss own products

define bot refuse to discuss competitors
  "I'm not able to provide opinions on competitor products.
   I can help you with questions about our own offerings."
```

## Output Validation Patterns

### Hallucination Detection

```python
# Faithfulness check against source documents
from guardrails import Guard
from guardrails.hub import DetectPII, NSFWText, ProvenanceV1

guard = Guard().use_many(
    ProvenanceV1(
        threshold=0.7,
        llm_callable="gpt-4o-mini",
        on_fail="fix"
    )
)

result = guard.validate(
    llm_output,
    metadata={"sources": retrieved_documents}
)
```

### Format Validation

```python
# Guardrails AI structured output enforcement
from guardrails import Guard
from pydantic import BaseModel, Field
from typing import List

class ExtractedEntity(BaseModel):
    name: str = Field(description="Entity name")
    category: str = Field(description="Entity category")
    confidence: float = Field(ge=0.0, le=1.0)

class ExtractionResult(BaseModel):
    entities: List[ExtractedEntity]
    summary: str = Field(max_length=500)

guard = Guard.from_pydantic(ExtractionResult)
result = guard(
    llm_api=openai.chat.completions.create,
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}]
)
```

### Toxicity Filtering

```python
# LLM Guard output toxicity scanning
from llm_guard.output_scanners import Toxicity, Bias, NoRefusal

toxicity_scanner = Toxicity(threshold=0.7)
bias_scanner = Bias(threshold=0.75)

sanitized, is_valid, score = toxicity_scanner.scan(prompt, model_output)
if not is_valid:
    model_output = "I cannot provide that response. Let me rephrase."
```

### PII Leakage Prevention

```python
# Scan LLM output for inadvertent PII leakage
from llm_guard.output_scanners import Sensitive

scanner = Sensitive(
    entity_types=["CREDIT_CARD", "US_SSN", "EMAIL_ADDRESS"],
    redact=True,
    threshold=0.5
)
sanitized_output, is_valid, score = scanner.scan(prompt, model_output)
```

## Content Moderation APIs

| API | Provider | Categories | Latency | Free Tier | Customizable |
|---|---|---|---|---|---|
| **Moderation API** | OpenAI | 11 categories (hate, violence, sexual, self-harm, etc.) | ~100ms | Free with API key | No |
| **Perspective API** | Google/Jigsaw | Toxicity, threat, insult, profanity, identity attack | ~200ms | Free (QPS limits) | Limited |
| **Content Safety** | Azure | Hate, violence, sexual, self-harm + severity levels | ~150ms | Pay-per-use | Yes (blocklists) |
| **Comprehend** | AWS | Sentiment, PII, toxicity, targeted sentiment | ~300ms | Free tier available | Custom classifiers |

```python
# OpenAI Moderation API usage
from openai import OpenAI
client = OpenAI()

response = client.moderations.create(
    model="omni-moderation-latest",
    input=user_message
)
result = response.results[0]
if result.flagged:
    blocked_categories = [
        cat for cat, flagged in result.categories.model_dump().items()
        if flagged
    ]
    print(f"Content blocked. Categories: {blocked_categories}")
```

## Guardrails Architecture Patterns

### Pre-Processing Pipeline (Input Rails)

```
User Input --> PII Scanner --> Injection Detector --> Topic Filter --> LLM
```

- Validate and sanitize all user inputs before they reach the LLM.
- Apply PII anonymization to prevent sensitive data from entering the model context.
- Run prompt injection classifiers to detect and block adversarial inputs.
- Enforce topic boundaries to keep conversations within scope.

### Post-Processing Pipeline (Output Rails)

```
LLM Output --> Format Validator --> Hallucination Check --> Toxicity Filter --> PII Redactor --> User
```

- Validate output structure (JSON schema, required fields).
- Cross-reference claims against source documents for faithfulness.
- Scan for toxic, biased, or inappropriate content.
- Redact any PII that the model may have inadvertently generated.

### Streaming Guardrails

```python
# Guardrails with streaming - buffer and validate chunks
async def guarded_stream(prompt):
    buffer = ""
    async for chunk in llm.stream(prompt):
        buffer += chunk.content
        # Check at sentence boundaries
        if buffer.endswith((".", "!", "?")):
            is_safe = await quick_safety_check(buffer)
            if not is_safe:
                yield "[Content filtered]"
                return
            yield buffer
            buffer = ""
```

Streaming considerations:
- Validate at natural boundaries (sentence-level, not token-level).
- Maintain a rolling buffer for context-aware checks.
- Have a fallback termination if cumulative output becomes unsafe.
- Accept higher latency at boundaries in exchange for safety.

## Red-Teaming Methodologies and Tools

### Automated Red-Teaming Tools

| Tool | Approach | Best For |
|---|---|---|
| **Garak** | Probe-based vulnerability scanning | Comprehensive automated testing |
| **PyRIT (Microsoft)** | Multi-turn adversarial attack generation | Enterprise red-team exercises |
| **Promptfoo** | Configuration-driven eval with adversarial plugins | CI/CD integration |
| **ART (Adversarial Robustness Toolbox)** | ML-focused adversarial attacks | Research-grade testing |

### Red-Team Attack Categories

1. **Jailbreaking**: Bypassing safety alignment through creative prompting.
2. **Prompt Leaking**: Extracting system prompts or hidden instructions.
3. **Data Extraction**: Attempting to recover training data or PII.
4. **Hallucination Exploitation**: Inducing confidently wrong outputs.
5. **Bias Amplification**: Triggering discriminatory or stereotyped outputs.
6. **Denial of Service**: Crafting inputs that cause excessive token usage or infinite loops.

```bash
# Running Garak for automated LLM vulnerability scanning
pip install garak
garak --model_type openai --model_name gpt-4o \
      --probes encoding,dan,knowledgegraph \
      --report_prefix my_redteam_report
```

## Compliance Requirements

### Content Safety Policy Checklist

- Define acceptable use policies and prohibited content categories.
- Implement age-gating for age-restricted content.
- Maintain audit logs of all guardrail triggers with timestamps and categories.
- Establish an escalation path for edge cases that automated systems cannot resolve.
- Conduct regular red-team assessments (quarterly minimum).
- Document guardrail configurations and version them alongside model deployments.
- Comply with regional regulations (EU AI Act, CCPA, GDPR) for data handling.
- Implement rate limiting per user to prevent abuse.

### Audit Logging Schema

```json
{
  "timestamp": "2026-02-27T14:30:00Z",
  "request_id": "req_abc123",
  "user_id": "user_456",
  "guardrail_type": "input_injection_detection",
  "action_taken": "blocked",
  "risk_score": 0.95,
  "category": "prompt_injection",
  "input_hash": "sha256:...",
  "model": "gpt-4o",
  "metadata": {
    "detector_version": "1.2.0",
    "threshold": 0.9
  }
}
```

## Best Practices

1. **Layer your defenses**: Use multiple guardrail types; no single check is sufficient.
2. **Fail closed**: When a guardrail is uncertain, block rather than allow.
3. **Monitor guardrail metrics**: Track false positive and false negative rates continuously.
4. **Version guardrail configs**: Treat guardrail rules as code; review and test changes.
5. **Test with adversarial inputs**: Regularly run red-team exercises against your guardrails.
6. **Keep latency budgets**: Guardrails add latency; profile and optimize the pipeline.
7. **Handle graceful degradation**: If a guardrail service is down, have a safe fallback.

## Common Pitfalls

- Over-relying on a single guardrail layer (e.g., only checking output, ignoring input).
- Setting thresholds too aggressively, causing excessive false positives.
- Not testing guardrails against multi-turn or multi-modal attacks.
- Ignoring indirect prompt injection through RAG-retrieved documents.
- Failing to update guardrails when the underlying LLM is upgraded or swapped.
- Not logging guardrail decisions, making it impossible to audit or improve.

## Further Reading

- [Guardrails AI Documentation](https://docs.guardrailsai.com/)
- [NVIDIA NeMo Guardrails](https://docs.nvidia.com/nemo/guardrails/)
- [LLM Guard Documentation](https://llm-guard.com/)
- [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [Garak LLM Vulnerability Scanner](https://docs.garak.ai/)
- [Microsoft PyRIT Red-Teaming Framework](https://github.com/Azure/PyRIT)
- [Promptfoo LLM Testing](https://www.promptfoo.dev/)
- [NIST AI Risk Management Framework](https://www.nist.gov/artificial-intelligence/ai-risk-management-framework)
