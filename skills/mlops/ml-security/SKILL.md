---
name: ml-security
description: >
  ML security skill covering model security, adversarial attacks, adversarial robustness testing,
  data poisoning detection, model extraction defense, model inversion prevention, differential
  privacy (Opacus, TensorFlow Privacy), federated learning security, secure ML pipelines, model
  encryption, access control and RBAC for models, model signing and verification, supply chain
  security for ML artifacts, input validation for ML APIs, audit logging, PII detection, data
  anonymization, OWASP ML Top 10, MITRE ATLAS threat framework, secret management, secure model
  serving with TLS and authentication, rate limiting, dependency vulnerability scanning, model
  provenance tracking, fairness and bias auditing as security concerns, and compliance-driven
  secure data handling for production machine learning systems.
license: Apache-2.0
metadata:
  author: mlops-skills
  version: "1.0"
  category: mlops
---

# ML Security — Agent Skill

## Overview

ML systems face unique security threats beyond traditional software — adversarial inputs, data
poisoning, model theft, and privacy leakage. This skill provides comprehensive guidance and
executable tooling for securing machine learning systems across the entire lifecycle, from data
collection and training through deployment and monitoring. It is **platform-agnostic** and
supports all major ML frameworks (PyTorch, TensorFlow, scikit-learn, JAX, ONNX, and others).

## When to Use This Skill

- Security review of ML systems
- Hardening model serving endpoints
- Implementing privacy-preserving ML
- Compliance with GDPR/CCPA/HIPAA for ML
- Setting up adversarial robustness testing
- Auditing model artifacts and supply chains
- Implementing access control for model registries
- Detecting and remediating PII in training data

---

## 1. ML Threat Landscape

### 1.1 OWASP ML Top 10

| # | Risk | Description |
|---|------|-------------|
| ML01 | Input Manipulation | Adversarial examples that cause misclassification |
| ML02 | Data Poisoning | Corrupted training data leading to compromised models |
| ML03 | Model Inversion | Extracting private training data from model outputs |
| ML04 | Membership Inference | Determining whether a sample was in the training set |
| ML05 | Model Theft | Extracting model architecture or parameters |
| ML06 | AI Supply Chain | Compromised dependencies, pretrained models, or datasets |
| ML07 | Transfer Learning Attack | Exploiting vulnerabilities inherited from base models |
| ML08 | Model Skewing | Train/serve skew exploited to degrade production accuracy |
| ML09 | Output Integrity | Tampering with predictions post-inference |
| ML10 | Model Poisoning | Injecting backdoors during fine-tuning or training |

### 1.2 MITRE ATLAS (Adversarial Threat Landscape for AI Systems)

Key tactic categories:

- **Reconnaissance**: Discover model architecture, training data sources, API endpoints.
- **Resource Development**: Build adversarial tooling, acquire surrogate models.
- **Initial Access**: Exploit ML API input validation, poison public datasets.
- **Execution**: Trigger malicious model behavior via crafted inputs.
- **Persistence**: Implant backdoors that survive retraining cycles.
- **Exfiltration**: Extract model weights, training data, or membership information.
- **Impact**: Degrade model performance, cause biased predictions, deny service.

Always map your threat model to both OWASP ML Top 10 and MITRE ATLAS to ensure full coverage.
See [REFERENCE.md](references/REFERENCE.md) for the full ATLAS technique catalog and OWASP deep dives.

---

## 2. Adversarial Attacks and Defenses

### 2.1 Key Attacks

**FGSM — Fast Gradient Sign Method** (white-box):

```python
import torch

def fgsm_attack(model, images, labels, epsilon, loss_fn):
    """Generate FGSM adversarial examples."""
    images.requires_grad = True
    outputs = model(images)
    loss = loss_fn(outputs, labels)
    model.zero_grad()
    loss.backward()
    perturbed = images + epsilon * images.grad.sign()
    return torch.clamp(perturbed, 0, 1)
```

**PGD — Projected Gradient Descent** (stronger iterative attack):

```python
def pgd_attack(model, images, labels, epsilon, alpha, num_steps, loss_fn):
    """Generate PGD adversarial examples."""
    perturbed = images.clone().detach()
    for _ in range(num_steps):
        perturbed.requires_grad = True
        outputs = model(perturbed)
        loss = loss_fn(outputs, labels)
        model.zero_grad()
        loss.backward()
        adv_images = perturbed + alpha * perturbed.grad.sign()
        eta = torch.clamp(adv_images - images, min=-epsilon, max=epsilon)
        perturbed = torch.clamp(images + eta, 0, 1).detach()
    return perturbed
```

**Black-Box Attack** (using Foolbox):

```python
import foolbox as fb
fmodel = fb.PyTorchModel(model, bounds=(0, 1))
attack = fb.attacks.BoundaryAttack()
_, advs, success = attack(fmodel, images, labels, epsilons=[0.01, 0.05, 0.1])
```

### 2.2 Adversarial Training

The strongest known general defense: train on a mixture of clean and adversarial examples.

```python
def adversarial_training_step(model, images, labels, epsilon, alpha, pgd_steps, loss_fn, optimizer):
    """One step of adversarial training using PGD inner maximization."""
    model.eval()
    adv_images = pgd_attack(model, images, labels, epsilon, alpha, pgd_steps, loss_fn)
    model.train()
    outputs = model(adv_images)
    loss = loss_fn(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```

### 2.3 Additional Defenses

- **Input Preprocessing**: JPEG compression, spatial smoothing, bit-depth reduction.
- **Randomized Smoothing**: Certifiable robustness via noise injection.
- **Ensemble Adversarial Training**: Train against adversarial examples from multiple models.
- **Certified Defenses**: IBP, CROWN, randomized smoothing with provable guarantees.

See [REFERENCE.md](references/REFERENCE.md) for tool comparisons (ART, Foolbox, CleverHans), C&W attack examples, scikit-learn ART integration, and randomized smoothing code.

---

## 3-5. Data Poisoning, Model Extraction, and Model Inversion

These deep-dive attack and defense patterns are documented in [REFERENCE.md](references/REFERENCE.md):

- **Data Poisoning Detection** — Isolation Forest, spectral signature detection, prevention strategies
- **Model Extraction Attacks** — Surrogate model training, query rate limiting, extraction pattern detection
- **Membership Inference** — Loss-threshold attacks, shadow models, defenses
- **Model Inversion** — Gradient-based reconstruction, output perturbation defenses

---

## 6. Differential Privacy in ML

### 6.1 Core Concepts

- **Epsilon (privacy budget)**: Lower = stronger privacy. Typical range: 1-10.
- **Delta**: Probability of privacy breach. Usually set to 1/n for n training samples.
- **Sensitivity**: Maximum change in output caused by a single sample.

### 6.2 PyTorch with Opacus

```python
from opacus import PrivacyEngine

def train_with_dp(model, train_dataset, epochs=10, target_epsilon=8.0,
                  target_delta=1e-5, max_grad_norm=1.0, batch_size=64, lr=0.001):
    """Train a PyTorch model with differential privacy using Opacus."""
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model, optimizer=optimizer, data_loader=train_loader,
        epochs=epochs, target_epsilon=target_epsilon,
        target_delta=target_delta, max_grad_norm=max_grad_norm,
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        epsilon = privacy_engine.get_epsilon(delta=target_delta)
        print(f"Epoch {epoch+1}/{epochs} -- (epsilon={epsilon:.2f}, delta={target_delta})")
    return model, privacy_engine
```

### 6.3 TensorFlow Privacy

```python
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer

def build_dp_model(input_shape, num_classes, l2_norm_clip=1.0,
                   noise_multiplier=1.1, learning_rate=0.01):
    """Build a TF model trained with DP-SGD."""
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes),
    ])
    optimizer = DPKerasSGDOptimizer(
        l2_norm_clip=l2_norm_clip, noise_multiplier=noise_multiplier,
        num_microbatches=1, learning_rate=learning_rate,
    )
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model
```

See [REFERENCE.md](references/REFERENCE.md) for epsilon/delta selection guide, noise multiplier tables, gradient norm tuning, and privacy budget composition methods.

---

## 7. Federated Learning Security

Detailed federated learning security patterns including secure aggregation (FedAvg, Krum, trimmed mean), Byzantine-resilient protocols, and FL-specific defenses are in [REFERENCE.md](references/REFERENCE.md).

---

## 8. Model Access Control and RBAC

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Set

class Permission(Enum):
    MODEL_READ = "model:read"
    MODEL_WRITE = "model:write"
    MODEL_DEPLOY = "model:deploy"
    MODEL_DELETE = "model:delete"
    MODEL_PREDICT = "model:predict"
    DATA_READ = "data:read"
    DATA_WRITE = "data:write"
    EXPERIMENT_READ = "experiment:read"
    EXPERIMENT_WRITE = "experiment:write"
    AUDIT_READ = "audit:read"
    ADMIN = "admin"

ROLES = {
    "viewer": {Permission.MODEL_READ, Permission.EXPERIMENT_READ},
    "data_scientist": {
        Permission.MODEL_READ, Permission.MODEL_WRITE, Permission.DATA_READ,
        Permission.EXPERIMENT_READ, Permission.EXPERIMENT_WRITE, Permission.MODEL_PREDICT,
    },
    "ml_engineer": {
        Permission.MODEL_READ, Permission.MODEL_WRITE, Permission.MODEL_DEPLOY,
        Permission.DATA_READ, Permission.DATA_WRITE,
        Permission.EXPERIMENT_READ, Permission.EXPERIMENT_WRITE, Permission.MODEL_PREDICT,
    },
    "auditor": {
        Permission.MODEL_READ, Permission.DATA_READ,
        Permission.EXPERIMENT_READ, Permission.AUDIT_READ,
    },
    "admin": {Permission.ADMIN},
}

@dataclass
class AccessPolicy:
    user_id: str
    roles: Set[str] = field(default_factory=set)
    extra_permissions: Set[Permission] = field(default_factory=set)
    denied_permissions: Set[Permission] = field(default_factory=set)

    def has_permission(self, permission: Permission) -> bool:
        if Permission.ADMIN in self._all_granted():
            return True
        if permission in self.denied_permissions:
            return False
        return permission in self._all_granted()

    def _all_granted(self) -> Set[Permission]:
        perms = set(self.extra_permissions)
        for role in self.roles:
            perms.update(ROLES.get(role, set()))
        return perms

def enforce_access(policy: AccessPolicy, required: Permission, resource_id: str):
    """Raise if the user lacks the required permission."""
    if not policy.has_permission(required):
        raise PermissionError(
            f"User '{policy.user_id}' lacks '{required.value}' on '{resource_id}'"
        )
```

---

## 9-13. Artifact Signing, Supply Chain, Secrets, and Audit Logging

These operational security patterns are documented in [REFERENCE.md](references/REFERENCE.md):

- **Model Artifact Signing** — SHA-256 hashing, HMAC signing/verification, Sigstore/cosign integration
- **Supply Chain Security** — Dependency scanning (pip-audit, safety, trivy, fickling), model provenance tracking
- **Secret Management** — SecretProvider abstraction, Vault integration, best practices
- **Audit Logging** — Structured AuditEvent logging, prediction logging, data access tracking

---

## 10. Secure Model Serving

### 10.1 TLS Configuration

```yaml
server:
  tls:
    enabled: true
    cert_file: /certs/server.crt
    key_file: /certs/server.key
    ca_file: /certs/ca.crt
    min_version: TLS1.3
    client_auth: require  # Mutual TLS
  listen: 0.0.0.0:8443
```

### 10.2 Input Validation for ML APIs

```python
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List

@dataclass
class InputSpec:
    shape: Tuple[int, ...]
    dtype: str = "float32"
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    max_payload_bytes: int = 10 * 1024 * 1024

class InputValidator:
    def __init__(self, spec: InputSpec):
        self.spec = spec

    def validate(self, data: np.ndarray) -> List[str]:
        errors = []
        if data.shape[1:] != self.spec.shape:
            errors.append(f"Expected shape (batch, {self.spec.shape}), got {data.shape}")
        if data.dtype != np.dtype(self.spec.dtype):
            errors.append(f"Expected dtype {self.spec.dtype}, got {data.dtype}")
        if self.spec.min_value is not None and np.any(data < self.spec.min_value):
            errors.append(f"Values below minimum {self.spec.min_value}")
        if self.spec.max_value is not None and np.any(data > self.spec.max_value):
            errors.append(f"Values above maximum {self.spec.max_value}")
        if data.nbytes > self.spec.max_payload_bytes:
            errors.append(f"Payload too large: {data.nbytes} bytes")
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            errors.append("Input contains NaN or Inf values")
        return errors
```

### 10.3 FastAPI Secure Serving Example

```python
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from slowapi import Limiter
from slowapi.util import get_remote_address

app = FastAPI()
security = HTTPBearer()
limiter = Limiter(key_func=get_remote_address)

class MLInput(BaseModel):
    features: List[float] = Field(..., min_length=10, max_length=10)

    @validator("features", each_item=True)
    def check_feature_range(cls, v):
        if not -1000 <= v <= 1000:
            raise ValueError(f"Feature value {v} out of range [-1000, 1000]")
        return v

@app.post("/predict")
@limiter.limit("100/minute")
async def predict(request: MLInput, credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not verify_token(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid token")
    return model.predict(request.features)
```

See [REFERENCE.md](references/REFERENCE.md) for TokenAuthenticator class, query rate limiter with extraction detection, and authentication details.

---

## 14. Data Anonymization and PII Detection

```python
import re
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class PIIMatch:
    text: str
    category: str
    start: int
    end: int
    confidence: float

PII_PATTERNS = {
    "email": (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 0.95),
    "phone_us": (r'\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', 0.85),
    "ssn": (r'\b\d{3}-\d{2}-\d{4}\b', 0.90),
    "credit_card": (r'\b(?:\d{4}[-\s]?){3}\d{4}\b', 0.85),
    "ip_address": (r'\b(?:\d{1,3}\.){3}\d{1,3}\b', 0.70),
}

def detect_pii(text: str, patterns: dict = None) -> List[PIIMatch]:
    """Scan text for PII patterns."""
    patterns = patterns or PII_PATTERNS
    matches = []
    for category, (pattern, confidence) in patterns.items():
        for match in re.finditer(pattern, text):
            matches.append(PIIMatch(
                text=match.group(), category=category,
                start=match.start(), end=match.end(), confidence=confidence,
            ))
    return matches

def mask_pii(text: str, replacement: str = "[REDACTED]") -> Tuple[str, List[PIIMatch]]:
    """Detect and mask all PII in text."""
    matches = detect_pii(text)
    for m in sorted(matches, key=lambda x: x.start, reverse=True):
        text = text[:m.start] + replacement + text[m.end:]
    return text, matches
```

See [REFERENCE.md](references/REFERENCE.md) for DataFrame anonymization, fairness/bias metrics, secure data handling patterns, and compliance requirements (GDPR, CCPA, HIPAA).

---

## 15-17. Fairness, Secure Data Handling, and Vulnerability Scanning

These topics are documented in [REFERENCE.md](references/REFERENCE.md):

- **Fairness and Bias Auditing** — Group fairness metrics, disparate impact, demographic parity
- **Secure Data Handling** — Encryption at rest/transit config, data access logging decorator
- **ML Vulnerability Scanning** — pip-audit, trivy, fickling, gitleaks, checkov scan commands

---

## 18. Quick Reference: ML Security Checklist

```
[ ] Threat model documented (OWASP ML Top 10, MITRE ATLAS)
[ ] Adversarial robustness tested (FGSM, PGD at minimum)
[ ] Data poisoning detection in place
[ ] Differential privacy evaluated for sensitive data
[ ] Model artifacts signed and verified
[ ] RBAC configured for model registry and serving
[ ] Input validation on all model API endpoints
[ ] Rate limiting enabled on prediction endpoints
[ ] TLS 1.3 enforced for model serving
[ ] Mutual TLS for internal services
[ ] Dependencies scanned for vulnerabilities
[ ] Pickle / serialization risks mitigated
[ ] PII detection run on training data
[ ] Audit logging for all model access and predictions
[ ] Secrets managed via vault / secrets manager
[ ] Model provenance tracked end-to-end
[ ] Container images scanned before deployment
[ ] Fairness metrics computed and monitored
[ ] Incident response plan for model compromise
[ ] Regular security reviews scheduled
```

## Scripts

- `scripts/security_scan.py` -- ML security scanning tool (adversarial robustness, input validation, artifact integrity, dependency scanning, PII detection)
- `scripts/privacy_guard.py` -- Differential privacy training wrapper, data anonymization, PII masking, privacy budget tracking

## References

See [references/REFERENCE.md](references/REFERENCE.md) for OWASP ML Top 10 detailed guide, MITRE ATLAS attack catalog, adversarial robustness tools comparison, differential privacy parameter selection, ML security checklist, compliance requirements (GDPR, CCPA, HIPAA), and deep-dive attack/defense patterns for data poisoning, model extraction, model inversion, federated learning, artifact signing, supply chain security, secret management, audit logging, fairness/bias, secure data handling, and vulnerability scanning.
