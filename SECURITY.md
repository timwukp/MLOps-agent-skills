# Security Policy

## Reporting a Vulnerability

If you find a security issue in any skill, script, or documentation in this repository,
please report it privately via [GitHub Security Advisories](https://github.com/timwukp/MLOps-agent-skills/security/advisories/new)
rather than opening a public issue. Reports are typically acknowledged within 72 hours.

Please include: the affected file(s), a description of the issue, and reproduction steps
or a proof-of-concept where applicable.

## Scope

These skills ship **executable example scripts**. They are reviewed for the classes of
issues below, but you should always review code before running it in your environment:

- **Injection**: SQL built with bound parameters, not string interpolation; shell
  commands avoid untrusted interpolation.
- **Unsafe deserialization**: `torch.load` defaults to `weights_only=True`; pickle-based
  formats (`.pkl`, `.joblib`, `.h5`, `.pt`) are treated as code-bearing, never "safe".
- **Sandboxing honesty**: nothing in this repo claims to sandbox untrusted code unless it
  actually does. The `llm-agent-orchestration` calculator uses an AST-whitelist evaluator
  (not `eval`); the Python-executor example is explicitly labeled unsafe for untrusted
  input and gated behind an opt-in flag — use OS-level isolation (containers, gVisor,
  hosted sandboxes) for real workloads.
- **Fail-closed guardrails**: security/privacy checks in the skills are designed to fail
  closed or report "unknown" — never to silently pass on error.

## Secrets & Cloud Metadata Policy

No credentials, API keys, tokens, account IDs, ARNs, or bucket names may be committed to
this repository — **including in test results, logs, and validation artifacts**. Cloud
account metadata (AWS account IDs, role ARNs, endpoint URLs) is treated as sensitive even
though it is not a credential: it enables targeted phishing and resource enumeration.

Required practice for contributors:

1. Redact before committing: account IDs → `<ACCOUNT_ID>`, ARNs/bucket names/console URLs
   → placeholders. The AWS validation artifacts in `tests/aws_validation/` follow this
   convention.
2. Run a pre-push scan covering both credential formats **and** cloud metadata:

   ```bash
   grep -rIE "AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|github_pat_|sk-ant-|-----BEGIN|[0-9]{12}" \
     --include='*.py' --include='*.md' --include='*.json' skills tests docs
   ```

   (Review 12-digit matches manually — most are AWS account IDs.)
3. If sensitive data reaches a public branch: rewrite the history immediately (clean
   commit off the last good parent, force-update the ref, delete stale branches), then
   request removal of the dangling commits via
   [GitHub Support → sensitive data removal](https://support.github.com/request), since
   force-pushed commits remain fetchable by SHA until garbage-collected. Rotate any
   credential that was exposed; for non-credential metadata, assess and monitor
   (e.g. GuardDuty for AWS accounts).

## Running the Example Scripts Safely

- Scripts that call cloud APIs (`tests/aws_validation/`) create real, billable resources.
  They clean up after themselves, but run them in a **non-production account** with
  least-privilege credentials and verify cleanup completed.
- Scripts that call LLM APIs read keys from environment variables only — never pass keys
  as CLI arguments (they leak into shell history and process lists).
- The `ml-security` skill's scanners are aids, not guarantees: the model-artifact scanner
  is a load smoke-check (it does not execute adversarial attacks), and regex-based PII
  detection is not compliance-grade — pair with ML-based detectors (e.g. Presidio,
  Amazon Comprehend PII) for regulated data.

## Supported Versions

Only the latest commit on `main` is maintained. Skills are reviewed against current
library releases (see `docs/accuracy-review-2026-07/` for the most recent full audit).
