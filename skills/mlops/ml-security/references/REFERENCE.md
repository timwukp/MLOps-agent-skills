# ML Security Reference Guide

Comprehensive reference for securing machine learning systems. Covers threat frameworks,
tooling comparisons, parameter selection guides, compliance requirements, actionable
checklists, and deep-dive attack/defense patterns.

---

## Table of Contents

1. [OWASP ML Top 10 Detailed Guide](#1-owasp-ml-top-10-detailed-guide)
2. [MITRE ATLAS Attack Catalog](#2-mitre-atlas-attack-catalog)
3. [Adversarial Robustness Tools Comparison](#3-adversarial-robustness-tools-comparison)
4. [Differential Privacy Parameter Selection Guide](#4-differential-privacy-parameter-selection-guide)
5. [ML Security Checklist](#5-ml-security-checklist)
6. [Compliance Requirements for ML (GDPR, CCPA, HIPAA)](#6-compliance-requirements-for-ml)
7. [Data Poisoning Detection and Prevention](#7-data-poisoning-detection-and-prevention)
8. [Model Extraction and Membership Inference](#8-model-extraction-and-membership-inference)
9. [Model Inversion Attacks and Defenses](#9-model-inversion-attacks-and-defenses)
10. [Federated Learning Security Patterns](#10-federated-learning-security-patterns)
11. [Model Artifact Signing and Verification](#11-model-artifact-signing-and-verification)
12. [Model Supply Chain Security](#12-model-supply-chain-security)
13. [Secret Management for ML Pipelines](#13-secret-management-for-ml-pipelines)
14. [Audit Logging for Model Access and Predictions](#14-audit-logging-for-model-access-and-predictions)
15. [Model Fairness and Bias as Security Concerns](#15-model-fairness-and-bias-as-security-concerns)
16. [Secure Data Handling in Training Pipelines](#16-secure-data-handling-in-training-pipelines)
17. [ML-Specific Vulnerability Scanning](#17-ml-specific-vulnerability-scanning)

---

## 1. OWASP ML Top 10 Detailed Guide

### ML01: Input Manipulation (Adversarial Examples)

**Description**: Adversarial examples are inputs intentionally crafted to cause a model to make
incorrect predictions. Small, often imperceptible perturbations can flip a classifier's output
with high confidence.

**Attack Vectors**:
- White-box: FGSM, PGD, C&W, AutoAttack (attacker has model access)
- Black-box: Boundary Attack, HopSkipJump, transfer attacks (attacker only sees outputs)
- Physical-world: Adversarial patches, stickers on stop signs, adversarial clothing

**Impact**: Misclassification in safety-critical systems (autonomous driving, medical imaging),
bypassing content filters, evading malware detection.

**Mitigations**:
- Adversarial training (PGD-AT is the gold standard)
- Certified defenses (randomized smoothing, IBP)
- Input preprocessing (JPEG compression, spatial smoothing)
- Ensemble methods with diverse architectures
- Runtime adversarial input detection

**Testing**:
```bash
# Using scripts/security_scan.py
python scripts/security_scan.py --scan robustness --model model.pt --framework pytorch --epsilon 0.03
```

---

### ML02: Data Poisoning

**Description**: An attacker corrupts the training data to cause the model to learn incorrect
associations or embed backdoors (triggers).

**Attack Vectors**:
- Label flipping: Swap labels to degrade accuracy on a target class
- Backdoor insertion: Inject samples with a trigger pattern (e.g., a pixel patch)
- Clean-label attacks: Poison data without changing labels (feature collisions)
- Data supply chain: Corrupt public datasets, web-scraped data, crowdsourced labels

**Impact**: Targeted misclassification, backdoor activation at deployment, model degradation.

**Mitigations**:
- Data provenance tracking (who contributed what, when)
- Spectral signature detection (Tran et al., 2018)
- Isolation Forest / DBSCAN outlier analysis on feature representations
- Robust training (trimmed loss, certified defenses)
- Regular retraining with clean, verified data

**Detection Code**:
```python
from sklearn.ensemble import IsolationForest
clf = IsolationForest(contamination=0.05, random_state=42)
outlier_labels = clf.fit_predict(feature_representations)
suspicious_indices = np.where(outlier_labels == -1)[0]
```

---

### ML03: Model Inversion

**Description**: An attacker uses model outputs (predictions, confidence scores) to reconstruct
representations of training data, violating the privacy of individuals whose data was used.

**Attack Vectors**:
- Gradient-based inversion: Optimize inputs to maximize a target class score
- Generative model inversion: Use GANs to generate training-like data from model responses
- Attribute inference: Infer sensitive attributes from model behavior

**Impact**: Exposure of PII, medical records, facial images from training data.

**Mitigations**:
- Limit output information (top-k labels only, no confidence scores)
- Add calibrated noise to outputs (output perturbation)
- Differential privacy during training
- Model distillation (serve distilled model, not original)

---

### ML04: Membership Inference

**Description**: Determine whether a specific data point was part of the model's training set.
Models typically have lower loss on training data, enabling this attack.

**Attack Vectors**:
- Loss threshold: Training members have lower loss
- Shadow model training: Train shadow models to learn the membership signal
- Metric-based: Entropy, confidence calibration differences

**Impact**: Violates data privacy, can reveal participation in sensitive datasets (medical studies,
financial data), regulatory violations.

**Mitigations**:
- Differential privacy (formal guarantee against membership inference)
- Strong regularization (dropout, weight decay, early stopping)
- Knowledge distillation
- Quantize / round output probabilities

---

### ML05: Model Theft / Extraction

**Description**: An attacker queries the model API systematically to train a functionally
equivalent copy (surrogate model), stealing intellectual property.

**Attack Vectors**:
- Random query synthesis
- Active learning-based extraction (fewer queries needed)
- Partial extraction for transfer attacks

**Impact**: IP theft, creation of surrogate models for white-box attacks, competitive harm.

**Mitigations**:
- Rate limiting and query budgets
- Prediction perturbation (add noise to outputs)
- Watermarking (embed detectable patterns in model behavior)
- Query pattern anomaly detection
- Legal protections (ToS, model licensing)

---

### ML06: AI Supply Chain Attacks

**Description**: Compromised model dependencies, pretrained models, datasets, or training
infrastructure.

**Attack Vectors**:
- Malicious pretrained model weights (trojan in HuggingFace model)
- Compromised pip/conda packages
- Poisoned public datasets
- Compromised training infrastructure (CI/CD, GPU clusters)

**Impact**: Backdoored models, arbitrary code execution via pickle, data exfiltration.

**Mitigations**:
- Pin exact dependency versions with hash verification
- Use `pip-audit` and `safety check` for vulnerability scanning
- Verify model artifact hashes and signatures
- Use safe serialization formats (SafeTensors, ONNX) instead of pickle
- Scan models with `fickling` for pickle exploits
- Software Bill of Materials (SBOM) for ML artifacts

---

### ML07: Transfer Learning Attack

**Description**: Vulnerabilities or backdoors in a base/pretrained model propagate to fine-tuned
downstream models.

**Impact**: Inherited backdoors activate in production; adversarial transferability between models.

**Mitigations**:
- Audit base models for backdoors before fine-tuning
- Fine-tune with adversarial training
- Use multiple diverse pretrained models
- Monitor for unexpected behavior changes after fine-tuning

---

### ML08: Model Skewing

**Description**: Exploit differences between training and serving environments (train/serve skew)
to degrade model accuracy in production.

**Attack Vectors**:
- Feature distribution shift exploitation
- Preprocessing pipeline inconsistencies
- Data schema changes

**Mitigations**:
- Feature validation in serving pipeline
- Statistical tests comparing training vs. serving distributions
- Model monitoring and drift detection (see model-drift-detection skill)

---

### ML09: Output Integrity

**Description**: Tampering with model predictions after inference (man-in-the-middle on
prediction responses).

**Mitigations**:
- Sign prediction responses
- End-to-end TLS with mutual authentication
- Integrity checks on prediction payloads

---

### ML10: Model Poisoning (Fine-tuning Backdoors)

**Description**: Injecting backdoors during fine-tuning or continuous learning by corrupting
update data.

**Mitigations**:
- Validate fine-tuning data with the same rigor as training data
- Monitor model behavior before and after fine-tuning
- Use differential privacy during fine-tuning
- Implement approval workflows for model updates

---

## 2. MITRE ATLAS Attack Catalog

MITRE ATLAS (Adversarial Threat Landscape for Artificial Intelligence Systems) extends the
MITRE ATT&CK framework to cover AI/ML-specific adversarial techniques.

### Tactics and Techniques

| Tactic | Technique ID | Technique Name | Description |
|--------|-------------|----------------|-------------|
| Reconnaissance | AML.T0000 | Active Scanning of ML API | Probe API to discover model type, input format, output structure |
| Reconnaissance | AML.T0001 | Discover ML Model Family | Identify framework, architecture, version |
| Reconnaissance | AML.T0002 | Discover Training Data | Identify datasets used for training |
| Resource Development | AML.T0003 | Acquire Public ML Model | Download surrogate/shadow models |
| Resource Development | AML.T0004 | Develop Adversarial ML Attacks | Build attack tooling |
| Initial Access | AML.T0005 | ML Supply Chain Compromise | Compromise pretrained models, packages |
| Initial Access | AML.T0006 | Data Poisoning | Inject malicious training data |
| Execution | AML.T0007 | Adversarial Input (Evasion) | Craft inputs causing misclassification |
| Execution | AML.T0008 | Backdoor Trigger | Activate embedded backdoor via trigger |
| Persistence | AML.T0009 | Backdoor ML Model | Embed persistent backdoor in model weights |
| Exfiltration | AML.T0010 | Model Extraction | Steal model via query-based extraction |
| Exfiltration | AML.T0011 | Model Inversion | Reconstruct training data from model |
| Exfiltration | AML.T0012 | Membership Inference | Determine training set membership |
| Impact | AML.T0013 | Denial of ML Service | Overwhelm model API or degrade performance |
| Impact | AML.T0014 | Evade ML Model | Bypass ML-based security controls |

### Using ATLAS for Threat Modeling

1. **Identify assets**: Models, training data, serving infrastructure, feature stores.
2. **Map threats**: For each asset, identify relevant ATLAS techniques.
3. **Assess risk**: Likelihood x Impact for each threat.
4. **Prioritize mitigations**: Apply defenses in order of risk.
5. **Monitor**: Set up detection for each high-risk technique.

---

## 3. Adversarial Robustness Tools Comparison

### Overview

| Feature | ART (IBM) | Foolbox | CleverHans | Advertorch |
|---------|-----------|---------|------------|------------|
| **Frameworks** | PyTorch, TF, Keras, sklearn, JAX | PyTorch, TF, JAX | TF, PyTorch | PyTorch |
| **Attack coverage** | 40+ attacks | 30+ attacks | 15+ attacks | 20+ attacks |
| **Certified defenses** | Yes | No | No | No |
| **Poisoning attacks** | Yes | No | No | No |
| **Model extraction** | Yes | No | No | No |
| **Evasion attacks** | Yes | Yes | Yes | Yes |
| **Detection methods** | Yes | No | No | No |
| **Preprocessing defenses** | Yes | No | No | No |
| **Active maintenance** | Yes (IBM) | Yes | Limited | Limited |
| **License** | MIT | MIT | MIT | MIT |

### ART (Adversarial Robustness Toolbox)

**Best for**: Comprehensive security evaluation, enterprise use, research.

```python
# Install
pip install adversarial-robustness-toolbox

# Quick start
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.attacks.evasion import CarliniLInfMethod, AutoAttack
from art.attacks.poisoning import PoisoningAttackSVM, PoisoningAttackBackdoor
from art.defences.preprocessor import SpatialSmoothing, JpegCompression
from art.defences.postprocessor import ReverseSigmoid
from art.defences.detector.evasion import BinaryInputDetector

# Key strengths:
# - Most comprehensive attack/defense library
# - Supports poisoning, extraction, inference attacks (not just evasion)
# - Certified robustness tools included
# - Enterprise-grade with IBM backing
```

### Foolbox

**Best for**: Quick robustness benchmarks, clean API, gradient-free attacks.

```python
# Install
pip install foolbox

# Quick start
import foolbox as fb

fmodel = fb.PyTorchModel(model, bounds=(0, 1))

# Run multiple attacks
attacks = [
    fb.attacks.FGSM(),
    fb.attacks.LinfPGD(),
    fb.attacks.BoundaryAttack(),
    fb.attacks.HopSkipJumpAttack(),
]
epsilons = [0.01, 0.03, 0.05, 0.1]

for attack in attacks:
    _, advs, success = attack(fmodel, images, labels, epsilons=epsilons)
    print(f"{attack.__class__.__name__}: {success.float().mean():.2%} success")

# Key strengths:
# - Clean, Pythonic API
# - Excellent gradient-free attacks
# - Supports multiple frameworks seamlessly
# - Good for quick benchmarking
```

### CleverHans

**Best for**: Research, TensorFlow-focused workflows, reference implementations.

```python
# Install
pip install cleverhans

# Quick start (TF 2.x)
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.carlini_wagner_l2 import carlini_wagner_l2

adv_x = fast_gradient_method(model, x, eps=0.03, norm=np.inf)
adv_x = projected_gradient_descent(model, x, eps=0.03, eps_iter=0.01, nb_iter=40, norm=np.inf)

# Key strengths:
# - Reference implementations from leading researchers
# - Well-documented theoretical foundations
# - Good TensorFlow integration
```

### Recommendation Matrix

| Use Case | Recommended Tool |
|----------|-----------------|
| Comprehensive security audit | ART |
| Quick robustness benchmark | Foolbox |
| Research / academic | CleverHans or ART |
| Poisoning / extraction testing | ART (only option) |
| Production pipeline integration | ART |
| Certified robustness evaluation | ART |
| Black-box only testing | Foolbox |

---

## 4. Differential Privacy Parameter Selection Guide

### Epsilon Selection

| Epsilon Range | Privacy Level | Use Case |
|---------------|---------------|----------|
| 0.01 - 0.1 | Very strong | Highly sensitive data (medical, financial PII) |
| 0.1 - 1.0 | Strong | Government data, census, health records |
| 1.0 - 5.0 | Moderate | General personal data, recommendation systems |
| 5.0 - 10.0 | Weak | Low-sensitivity data where utility is critical |
| > 10.0 | Minimal | Not recommended; privacy guarantees are weak |

### Delta Selection

- **Rule of thumb**: Set delta < 1/n, where n = number of training samples.
- For n = 10,000: delta < 1e-4
- For n = 100,000: delta < 1e-5
- For n = 1,000,000: delta < 1e-6

### Noise Multiplier vs. Epsilon

The noise multiplier (sigma) determines how much Gaussian noise is added to clipped gradients.
Higher sigma = lower epsilon = more privacy = more utility loss.

| Training Config | Noise Mult. | Approx. Epsilon | Notes |
|----------------|-------------|-----------------|-------|
| 10 epochs, batch 64, n=50K | 0.5 | ~30 | Minimal privacy |
| 10 epochs, batch 64, n=50K | 1.0 | ~8 | Moderate privacy |
| 10 epochs, batch 64, n=50K | 1.5 | ~3 | Good privacy |
| 10 epochs, batch 64, n=50K | 2.0 | ~1.5 | Strong privacy |
| 10 epochs, batch 64, n=50K | 3.0 | ~0.8 | Very strong |

### Max Gradient Norm Selection

The max gradient norm (`max_grad_norm` or `l2_norm_clip`) determines the clipping threshold
for per-sample gradients:

- **Too low**: Clips too aggressively, slowing convergence.
- **Too high**: Requires more noise for the same epsilon, degrading utility.
- **Recommendation**: Start with 1.0, then tune based on unclipped gradient norms.

```python
# Profile gradient norms to choose clipping threshold
norms = []
for batch in train_loader:
    outputs = model(batch[0])
    loss = loss_fn(outputs, batch[1])
    loss.backward()
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
    norms.append(total_norm.item())
    model.zero_grad()

import numpy as np
print(f"Gradient norm - Mean: {np.mean(norms):.2f}, "
      f"Median: {np.median(norms):.2f}, P90: {np.percentile(norms, 90):.2f}")
# Set max_grad_norm to roughly the median or P75
```

### Privacy Budget Composition

When running multiple private operations, the total privacy cost compounds:

| Method | Formula | Tightness |
|--------|---------|-----------|
| Basic Composition | epsilon_total = sum(epsilon_i) | Loose |
| Advanced Composition | epsilon_total = sqrt(2k * ln(1/delta)) * eps + k * eps * (e^eps - 1) | Moderate |
| RDP (Renyi DP) | Convert RDP orders to (eps, delta)-DP | Tight |
| zCDP (zero-concentrated DP) | Additive in rho, then convert | Tight |
| PLD (Privacy Loss Distributions) | Numerical composition | Tightest |

**Recommendation**: Use RDP accounting (available in Opacus and TF Privacy) for the tightest
practical bounds.

### Practical Decision Flowchart

```
Is the data highly sensitive (medical, financial, legal)?
  -> YES: Target epsilon <= 1.0, use strong DP
  -> NO: Does regulation require privacy guarantees?
    -> YES: Target epsilon <= 5.0, consult legal
    -> NO: Is membership inference a concern?
      -> YES: Target epsilon <= 8.0
      -> NO: Consider if DP is needed; regularization may suffice
```

---

## 5. ML Security Checklist

### Pre-Training

- [ ] **Threat model documented** (map to OWASP ML Top 10 and MITRE ATLAS)
- [ ] **Data provenance verified** (origin, lineage, contributors tracked)
- [ ] **PII detection run** on all training data (`python scripts/privacy_guard.py audit --data train.csv`)
- [ ] **Data anonymization applied** where PII is detected
- [ ] **Dataset checksums computed** and stored in manifest
- [ ] **Dependencies pinned** with exact versions and hashes
- [ ] **Dependency vulnerability scan** passed (`pip-audit`, `safety check`)
- [ ] **Secrets scanning** passed (no hardcoded credentials in code/config)
- [ ] **Access control** configured for data storage (least privilege)

### During Training

- [ ] **Differential privacy** applied if data is sensitive (Opacus / TF Privacy)
- [ ] **Privacy budget tracked** and not exceeded
- [ ] **Training environment secured** (encrypted storage, network isolation)
- [ ] **Experiment tracking** captures all hyperparameters and data versions
- [ ] **Federated learning security** applied if training is distributed (secure aggregation, client auth)

### Post-Training

- [ ] **Adversarial robustness tested** (FGSM, PGD at minimum; C&W for high-risk)
- [ ] **Data poisoning detection** run (Isolation Forest, spectral signatures)
- [ ] **Model artifact signed** with cryptographic manifest
- [ ] **Serialization format safe** (prefer ONNX / SafeTensors over pickle)
- [ ] **Model provenance recorded** (code commit, data version, hyperparameters, metrics)
- [ ] **Fairness metrics computed** across sensitive attributes
- [ ] **Membership inference resistance** evaluated

### Deployment

- [ ] **TLS 1.3 enforced** for all model serving endpoints
- [ ] **Mutual TLS** for internal service-to-service communication
- [ ] **Authentication** required for all model API access (token, OAuth, mTLS)
- [ ] **RBAC configured** (viewer, data_scientist, ml_engineer, auditor, admin)
- [ ] **Rate limiting** enabled on prediction endpoints
- [ ] **Input validation** on all model API endpoints (shape, dtype, range, NaN/Inf)
- [ ] **Output perturbation** considered to prevent model extraction
- [ ] **Audit logging** enabled for all model access and predictions
- [ ] **Container images scanned** before deployment (Trivy, Grype)

### Ongoing Operations

- [ ] **Model monitoring** for drift, degradation, and anomalous predictions
- [ ] **Dependency updates** and rescans on a regular schedule
- [ ] **Secret rotation** on schedule and after suspected compromise
- [ ] **Security reviews** conducted quarterly
- [ ] **Incident response plan** for model compromise documented and tested
- [ ] **Privacy budget** reviewed and not exhausted
- [ ] **Access logs** reviewed for anomalies

---

## 6. Compliance Requirements for ML

### GDPR (General Data Protection Regulation)

**Applies to**: Processing personal data of EU/EEA residents.

| Requirement | ML Implication | Implementation |
|-------------|---------------|----------------|
| **Lawful basis** (Art. 6) | Must have legal basis for using personal data in training | Document lawful basis; obtain consent if needed |
| **Purpose limitation** (Art. 5) | Data collected for one purpose cannot be used for another | Track data lineage; enforce purpose-bound access |
| **Data minimization** (Art. 5) | Only use data necessary for the purpose | Feature selection review; remove unnecessary PII |
| **Right to erasure** (Art. 17) | Individuals can request data deletion | Implement machine unlearning or retrain without data |
| **Right to explanation** (Art. 22) | Automated decisions must be explainable | Use interpretable models or generate explanations (SHAP, LIME) |
| **Data Protection Impact Assessment** (Art. 35) | Required for high-risk processing | Conduct DPIA before training on personal data |
| **Privacy by design** (Art. 25) | Build privacy into the system from the start | Differential privacy, anonymization, encryption |
| **Data breach notification** (Art. 33-34) | Report breaches within 72 hours | Monitoring, incident response plan, audit logging |
| **Transfer restrictions** (Art. 44-49) | Restrictions on data transfer outside EU | Use EU-region infrastructure; assess adequacy |

**ML-Specific GDPR Actions**:
1. Run PII detection on all training data before training.
2. Apply differential privacy for any model trained on personal data.
3. Implement machine unlearning capability or retrain-from-scratch procedure.
4. Generate model explanations for any automated decision-making.
5. Maintain complete data processing records including ML pipeline details.
6. Conduct a DPIA for all ML systems processing personal data.

### CCPA (California Consumer Privacy Act) / CPRA

**Applies to**: Businesses processing personal information of California residents.

| Requirement | ML Implication | Implementation |
|-------------|---------------|----------------|
| **Right to know** | Disclose what data is collected and how it is used | Data catalog with ML usage documentation |
| **Right to delete** | Consumers can request deletion of their data | Machine unlearning or retraining capability |
| **Right to opt-out** | Consumers can opt out of data sales/sharing | Exclude opted-out data from training |
| **Data minimization** (CPRA) | Only collect data necessary for the purpose | Review training data necessity |
| **Sensitive personal information** (CPRA) | Additional protections for sensitive data | Enhanced privacy measures for sensitive features |
| **Automated decision-making** (CPRA) | Right to opt out of automated profiling | Provide opt-out mechanism and human review |

### HIPAA (Health Insurance Portability and Accountability Act)

**Applies to**: Covered entities and business associates handling Protected Health Information (PHI).

| Requirement | ML Implication | Implementation |
|-------------|---------------|----------------|
| **Privacy Rule** | PHI must be protected and used only as permitted | De-identify data (Safe Harbor or Expert Determination) |
| **Security Rule** | Technical safeguards for electronic PHI (ePHI) | Encryption, access controls, audit logging |
| **Minimum Necessary** | Use/disclose only the minimum PHI needed | Feature selection; anonymize before training |
| **Business Associate Agreement** | Third-party ML services must sign BAA | Ensure cloud/ML providers sign BAAs |
| **De-identification** (Safe Harbor) | Remove 18 specified identifiers | Automated PII detection and removal |
| **Audit controls** | Track access to PHI | Comprehensive audit logging |

**HIPAA De-identification Safe Harbor Identifiers** (all must be removed):
1. Names
2. Geographic data (smaller than state)
3. Dates (except year) related to individual
4. Phone numbers
5. Fax numbers
6. Email addresses
7. Social Security numbers
8. Medical record numbers
9. Health plan beneficiary numbers
10. Account numbers
11. Certificate/license numbers
12. Vehicle identifiers
13. Device identifiers
14. Web URLs
15. IP addresses
16. Biometric identifiers
17. Full-face photographs
18. Any other unique identifying number

### Cross-Regulation Compliance Matrix

| Action | GDPR | CCPA | HIPAA |
|--------|------|------|-------|
| PII detection in training data | Required | Required | Required (PHI) |
| Data anonymization / de-identification | Recommended | Recommended | Required (Safe Harbor) |
| Differential privacy | Recommended | Recommended | Recommended |
| Right to deletion / unlearning | Required | Required | Not explicit |
| Audit logging | Required | Required | Required |
| Data encryption (at rest + transit) | Required | Required | Required |
| Access controls / RBAC | Required | Required | Required |
| Breach notification | 72 hours | 72 hours (AG) | 60 days |
| Impact assessment | DPIA required | Not required | Risk analysis |
| Model explainability | Required (Art. 22) | Right to opt out | Not explicit |
| Data processing records | Required | Required | Required |
| Consent management | Often required | Opt-out model | Authorization |

### Compliance Implementation Checklist

```
[ ] Identify applicable regulations (GDPR, CCPA, HIPAA, industry-specific)
[ ] Conduct data inventory: what personal data is used in ML?
[ ] Run PII detection: python scripts/privacy_guard.py audit --data train.csv
[ ] Apply de-identification / anonymization: python scripts/privacy_guard.py anonymize --input data.csv
[ ] Implement differential privacy where required
[ ] Set up audit logging for all data access and model predictions
[ ] Configure RBAC with least-privilege access
[ ] Enable encryption (TLS 1.3 in transit, AES-256 at rest)
[ ] Document data processing activities and ML pipeline details
[ ] Implement deletion / unlearning capability
[ ] Conduct DPIA / risk analysis
[ ] Establish breach notification procedure
[ ] Train team on compliance requirements
[ ] Schedule periodic compliance reviews
```

---

## Additional Resources

### Official References

- **OWASP ML Top 10**: https://owasp.org/www-project-machine-learning-security-top-10/
- **MITRE ATLAS**: https://atlas.mitre.org/
- **NIST AI Risk Management Framework**: https://www.nist.gov/artificial-intelligence/ai-risk-management-framework
- **EU AI Act**: https://artificialintelligenceact.eu/

### Tools

- **ART**: https://github.com/Trusted-AI/adversarial-robustness-toolbox
- **Foolbox**: https://github.com/bethgelab/foolbox
- **CleverHans**: https://github.com/cleverhans-lab/cleverhans
- **Opacus**: https://opacus.ai/
- **TensorFlow Privacy**: https://github.com/tensorflow/privacy
- **Fickling** (pickle scanner): https://github.com/trailofbits/fickling
- **pip-audit**: https://github.com/pypa/pip-audit
- **Trivy**: https://github.com/aquasecurity/trivy
- **cosign** (Sigstore): https://github.com/sigstore/cosign
- **SafeTensors**: https://github.com/huggingface/safetensors

### Research Papers

- Goodfellow et al. (2015) "Explaining and Harnessing Adversarial Examples" (FGSM)
- Madry et al. (2018) "Towards Deep Learning Models Resistant to Adversarial Attacks" (PGD-AT)
- Carlini & Wagner (2017) "Towards Evaluating the Robustness of Neural Networks" (C&W attack)
- Abadi et al. (2016) "Deep Learning with Differential Privacy" (DP-SGD)
- Tran et al. (2018) "Spectral Signatures in Backdoor Attacks"
- Shokri et al. (2017) "Membership Inference Attacks Against Machine Learning Models"
- Fredrikson et al. (2015) "Model Inversion Attacks that Exploit Confidence Information"
- Blanchard et al. (2017) "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent" (Krum)

---

## 7. Data Poisoning Detection and Prevention

### 7.1 Types of Poisoning

- **Label Flipping**: Changing labels to degrade accuracy on specific classes.
- **Backdoor / Trojan Attacks**: Injecting a trigger pattern that activates at inference time.
- **Clean-Label Attacks**: Poisoning without changing labels (harder to detect).

### 7.2 Detection Strategies

```python
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

def detect_poisoned_samples(features, labels, contamination=0.05):
    """Detect potential poisoning using per-class outlier analysis."""
    suspicious_indices = []
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        class_features = features[mask]
        if len(class_features) < 10:
            continue
        # Dimensionality reduction for stability
        n_components = min(50, class_features.shape[1], class_features.shape[0] - 1)
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(class_features)
        # Isolation Forest for outlier detection
        clf = IsolationForest(contamination=contamination, random_state=42)
        preds = clf.fit_predict(reduced)
        class_indices = np.where(mask)[0]
        suspicious_indices.extend(class_indices[preds == -1].tolist())
    return suspicious_indices

def spectral_signature_detection(features, labels, top_k=5):
    """Detect backdoor poisoning using spectral signatures (Tran et al., 2018)."""
    suspicious = {}
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        class_features = features[mask]
        centered = class_features - class_features.mean(axis=0)
        # Compute top singular vector
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        top_vector = Vt[0]
        # Score each sample by projection onto top singular vector
        scores = np.abs(centered @ top_vector)
        # Samples with high scores are suspicious
        threshold = np.percentile(scores, 95)
        class_indices = np.where(mask)[0]
        suspicious[label] = class_indices[scores > threshold].tolist()
    return suspicious
```

### 7.3 Prevention

- **Data Provenance**: Track every sample's origin, transformations, and contributors.
- **Robust Aggregation**: Use trimmed means, median-based aggregation, or Krum in federated settings.
- **Certified Data Removal** (machine unlearning): Remove the influence of specific samples.
- **Hash Verification**: Checksum datasets at every pipeline stage.

---

## 8. Model Extraction and Membership Inference

### 8.1 Model Extraction Attacks

An attacker queries the model API to reconstruct a functionally equivalent copy:

```python
import numpy as np
from sklearn.neural_network import MLPClassifier

def model_extraction_attack(target_api, input_shape, num_queries=10000):
    """Simulate a model extraction attack using random queries."""
    # Generate random queries
    queries = np.random.uniform(0, 1, size=(num_queries, *input_shape))
    # Query the target model
    labels = np.array([target_api(q) for q in queries])
    # Train a surrogate model
    surrogate = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500)
    surrogate.fit(queries.reshape(num_queries, -1), labels)
    return surrogate
```

#### Defenses Against Model Extraction

- **Rate Limiting**: Cap queries per user/IP per time window.
- **Prediction Perturbation**: Add calibrated noise to output probabilities.
- **Watermarking**: Embed verifiable watermarks in model behavior.
- **Query Pattern Detection**: Monitor for synthetic / grid-like query distributions.

```python
import hashlib
import time
from collections import defaultdict

class QueryRateLimiter:
    """Rate limiter with anomaly detection for model APIs."""

    def __init__(self, max_queries_per_minute=60, max_queries_per_hour=1000):
        self.max_per_minute = max_queries_per_minute
        self.max_per_hour = max_queries_per_hour
        self.query_log = defaultdict(list)

    def check_rate(self, client_id: str) -> bool:
        now = time.time()
        self.query_log[client_id] = [
            t for t in self.query_log[client_id] if now - t < 3600
        ]
        recent = self.query_log[client_id]
        last_minute = sum(1 for t in recent if now - t < 60)
        if last_minute >= self.max_per_minute or len(recent) >= self.max_per_hour:
            return False
        self.query_log[client_id].append(now)
        return True

    def detect_extraction_pattern(self, client_id: str, inputs: list) -> float:
        """Return a suspicion score (0-1) based on query patterns."""
        if len(inputs) < 20:
            return 0.0
        score = 0.0
        arr = np.array(inputs)
        # Check for uniformly distributed inputs (synthetic queries)
        from scipy.stats import kstest
        for col in range(min(arr.shape[1], 10)):
            col_data = arr[:, col]
            _, p_value = kstest(col_data, 'uniform',
                                args=(col_data.min(), col_data.max() - col_data.min()))
            if p_value > 0.05:
                score += 0.1
        # Check for high query frequency
        times = self.query_log.get(client_id, [])
        if len(times) > 100:
            intervals = np.diff(sorted(times))
            if np.std(intervals) < 0.1:
                score += 0.3  # Automated querying pattern
        return min(score, 1.0)
```

### 8.2 Membership Inference Attacks

Determine whether a specific sample was in the training set:

```python
def membership_inference_threshold(model, sample, label, loss_fn, threshold=0.5):
    """Simple loss-threshold membership inference."""
    import torch
    with torch.no_grad():
        output = model(sample.unsqueeze(0))
        loss = loss_fn(output, label.unsqueeze(0)).item()
    # Low loss => likely a member
    return loss < threshold, loss
```

#### Defenses

- **Regularization**: Dropout, weight decay, and early stopping reduce overfitting.
- **Differential Privacy**: Formal guarantee bounding membership leakage.
- **Knowledge Distillation**: Serve a distilled model instead of the original.
- **Output Quantization**: Round or bin prediction probabilities.

---

## 9. Model Inversion Attacks and Defenses

Model inversion reconstructs training data representations from model access:

```python
def model_inversion_attack(model, target_class, input_shape, lr=0.01, steps=1000):
    """Gradient-based model inversion to reconstruct class representative."""
    import torch
    x = torch.randn(1, *input_shape, requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=lr)
    target = torch.tensor([target_class])
    for step in range(steps):
        optimizer.zero_grad()
        output = model(x)
        loss = -output[0, target_class] + 0.01 * torch.norm(x)
        loss.backward()
        optimizer.step()
        x.data.clamp_(0, 1)
    return x.detach()
```

#### Defenses

- **Output Perturbation**: Add noise to logits or probabilities.
- **Confidence Score Masking**: Return only top-k labels, not full distributions.
- **Differential Privacy Training**: Bounds information leakage mathematically.
- **Input Transformation Ensembles**: Make gradient-based inversion unreliable.

---

## 10. Federated Learning Security Patterns

### 10.1 Threat Model

- **Honest-but-Curious Server**: Server follows protocol but tries to learn from gradients.
- **Malicious Clients**: Clients send poisoned updates.
- **Eavesdroppers**: Intercept communication between clients and server.

### 10.2 Secure Aggregation

```python
import numpy as np
from typing import List, Dict

def federated_averaging(client_updates: List[Dict[str, np.ndarray]],
                        client_weights: List[float]) -> Dict[str, np.ndarray]:
    """Standard FedAvg aggregation."""
    total_weight = sum(client_weights)
    aggregated = {}
    for key in client_updates[0]:
        aggregated[key] = sum(
            w * u[key] for w, u in zip(client_weights, client_updates)
        ) / total_weight
    return aggregated

def krum_aggregation(client_updates: List[Dict[str, np.ndarray]],
                     num_byzantine: int) -> Dict[str, np.ndarray]:
    """Krum aggregation: Byzantine-resilient by selecting the update closest to others."""
    n = len(client_updates)
    flat = []
    for update in client_updates:
        flat.append(np.concatenate([v.flatten() for v in update.values()]))
    flat = np.array(flat)
    scores = np.zeros(n)
    for i in range(n):
        dists = np.sort(np.linalg.norm(flat - flat[i], axis=1))
        scores[i] = np.sum(dists[1:n - num_byzantine - 1])
    best_idx = np.argmin(scores)
    return client_updates[best_idx]

def trimmed_mean_aggregation(client_updates: List[Dict[str, np.ndarray]],
                              trim_ratio: float = 0.1) -> Dict[str, np.ndarray]:
    """Trimmed mean: remove top/bottom trim_ratio of values per parameter."""
    aggregated = {}
    for key in client_updates[0]:
        stacked = np.stack([u[key] for u in client_updates], axis=0)
        n = stacked.shape[0]
        k = int(n * trim_ratio)
        if k > 0:
            sorted_vals = np.sort(stacked, axis=0)
            trimmed = sorted_vals[k:n-k]
        else:
            trimmed = stacked
        aggregated[key] = trimmed.mean(axis=0)
    return aggregated
```

### 10.3 Defenses

- **Secure Aggregation Protocols** (e.g., Bonawitz et al.): Cryptographic masking so server only sees the aggregated result.
- **Differential Privacy for FL**: Add noise to client updates before sending.
- **Gradient Compression and Sparsification**: Reduces information leaked per round.
- **Client Authentication**: Mutual TLS, signed updates.

---

## 11. Model Artifact Signing and Verification

### 11.1 Signing Workflow

```python
import hashlib
import json
import hmac
from pathlib import Path
from datetime import datetime, timezone

def compute_model_hash(model_path: str, algorithm: str = "sha256") -> str:
    """Compute cryptographic hash of a model artifact."""
    h = hashlib.new(algorithm)
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def sign_model_artifact(model_path: str, signing_key: str,
                        metadata: dict = None) -> dict:
    """Create a signed manifest for a model artifact."""
    model_hash = compute_model_hash(model_path)
    manifest = {
        "model_path": str(Path(model_path).name),
        "hash_algorithm": "sha256",
        "model_hash": model_hash,
        "signed_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode()
    signature = hmac.new(signing_key.encode(), manifest_bytes, hashlib.sha256).hexdigest()
    manifest["signature"] = signature
    return manifest

def verify_model_artifact(model_path: str, manifest: dict, signing_key: str) -> bool:
    """Verify a model artifact against its signed manifest."""
    expected_hash = compute_model_hash(model_path)
    if expected_hash != manifest.get("model_hash"):
        return False
    signature = manifest.pop("signature", None)
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode()
    expected_sig = hmac.new(signing_key.encode(), manifest_bytes, hashlib.sha256).hexdigest()
    manifest["signature"] = signature  # restore
    return hmac.compare_digest(signature, expected_sig)
```

### 11.2 Sigstore / cosign for ML

For production use, sign artifacts with Sigstore:

```bash
# Sign a model artifact with cosign
cosign sign-blob --key cosign.key --output-signature model.sig model.onnx

# Verify
cosign verify-blob --key cosign.pub --signature model.sig model.onnx
```

---

## 12. Model Supply Chain Security

### 12.1 Dependency Scanning

```bash
# Scan Python dependencies for known vulnerabilities
pip-audit --requirement requirements.txt --output json > audit_results.json

# Scan with Safety
safety check --json --output audit_safety.json

# Scan container images
trivy image my-ml-service:latest --format json > trivy_results.json

# Scan ONNX / pickle files for malicious code
fickling model.pkl --check-safety
```

### 12.2 Model Provenance Tracking

```python
import json
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import List

@dataclass
class ModelProvenance:
    """Track full provenance of a model artifact."""
    model_name: str
    model_version: str
    model_hash: str
    framework: str
    training_data_hash: str
    training_data_uri: str
    code_commit: str
    code_repo: str
    training_started: str
    training_completed: str
    hyperparameters: dict
    metrics: dict
    dependencies: List[str]
    environment: dict
    author: str
    approvals: List[dict] = field(default_factory=list)
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def compute_manifest_hash(self) -> str:
        content = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(content).hexdigest()
```

---

## 13. Secret Management for ML Pipelines

### 13.1 Best Practices

- **Never hardcode** credentials, API keys, or tokens in code or config files.
- Use a secrets manager: AWS Secrets Manager, HashiCorp Vault, GCP Secret Manager, Azure Key Vault.
- Inject secrets as environment variables at runtime, not build time.
- Rotate secrets on a schedule and after any suspected compromise.
- Use short-lived tokens and service accounts with minimal permissions.

```python
import os
from abc import ABC, abstractmethod

class SecretProvider(ABC):
    @abstractmethod
    def get_secret(self, key: str) -> str:
        ...

class EnvSecretProvider(SecretProvider):
    """Read secrets from environment variables."""
    def get_secret(self, key: str) -> str:
        value = os.environ.get(key)
        if value is None:
            raise KeyError(f"Secret '{key}' not found in environment")
        return value

class VaultSecretProvider(SecretProvider):
    """Read secrets from HashiCorp Vault."""
    def __init__(self, vault_addr: str, token: str):
        import hvac
        self.client = hvac.Client(url=vault_addr, token=token)

    def get_secret(self, key: str) -> str:
        path, field_name = key.rsplit("/", 1)
        secret = self.client.secrets.kv.v2.read_secret_version(path=path)
        return secret["data"]["data"][field_name]

def get_ml_secrets(provider: SecretProvider) -> dict:
    """Retrieve standard ML pipeline secrets."""
    return {
        "model_registry_token": provider.get_secret("ML_REGISTRY_TOKEN"),
        "data_store_key": provider.get_secret("DATA_STORE_KEY"),
        "experiment_tracker_key": provider.get_secret("EXPERIMENT_TRACKER_KEY"),
    }
```

---

## 14. Audit Logging for Model Access and Predictions

```python
import json
import time
import hashlib
import logging
from dataclasses import dataclass, asdict
from typing import Any, Optional

logger = logging.getLogger("ml.audit")

@dataclass
class AuditEvent:
    timestamp: float
    event_type: str          # "prediction", "model_load", "data_access", "config_change"
    user_id: str
    model_name: str
    model_version: str
    resource_id: str
    action: str
    input_hash: Optional[str] = None   # Hash of input (never log raw PII)
    output_summary: Optional[str] = None
    latency_ms: Optional[float] = None
    status: str = "success"
    ip_address: Optional[str] = None
    metadata: Optional[dict] = None

class AuditLogger:
    """Structured audit logging for ML systems."""

    def __init__(self, sink="stdout"):
        self.sink = sink

    def log(self, event: AuditEvent):
        record = asdict(event)
        record["timestamp_iso"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(event.timestamp)
        )
        msg = json.dumps(record)
        logger.info(msg)
        return record

    def log_prediction(self, user_id: str, model_name: str, model_version: str,
                       input_data: Any, output_data: Any, latency_ms: float,
                       ip_address: str = None):
        input_hash = hashlib.sha256(str(input_data).encode()).hexdigest()[:16]
        event = AuditEvent(
            timestamp=time.time(),
            event_type="prediction",
            user_id=user_id,
            model_name=model_name,
            model_version=model_version,
            resource_id=f"{model_name}:{model_version}",
            action="predict",
            input_hash=input_hash,
            output_summary=str(output_data)[:200],
            latency_ms=latency_ms,
            ip_address=ip_address,
        )
        return self.log(event)
```

---

## 15. Model Fairness and Bias as Security Concerns

Bias in ML models is a security and compliance concern -- biased models can violate
anti-discrimination laws and expose organizations to legal liability.

```python
import numpy as np
from typing import Dict

def compute_fairness_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                              sensitive_attr: np.ndarray) -> Dict[str, Dict]:
    """Compute group fairness metrics across a sensitive attribute."""
    groups = np.unique(sensitive_attr)
    metrics = {}
    for group in groups:
        mask = sensitive_attr == group
        tp = np.sum((y_pred[mask] == 1) & (y_true[mask] == 1))
        fp = np.sum((y_pred[mask] == 1) & (y_true[mask] == 0))
        fn = np.sum((y_pred[mask] == 0) & (y_true[mask] == 1))
        tn = np.sum((y_pred[mask] == 0) & (y_true[mask] == 0))
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        selection_rate = np.mean(y_pred[mask])
        metrics[str(group)] = {
            "true_positive_rate": tpr,
            "false_positive_rate": fpr,
            "selection_rate": selection_rate,
            "count": int(mask.sum()),
        }
    rates = [m["selection_rate"] for m in metrics.values() if m["selection_rate"] > 0]
    if len(rates) >= 2:
        metrics["_disparate_impact"] = min(rates) / max(rates)
        metrics["_demographic_parity_diff"] = max(rates) - min(rates)
    return metrics
```

---

## 16. Secure Data Handling in Training Pipelines

### 16.1 Encryption at Rest and in Transit

```yaml
# Example: Encrypted data pipeline configuration
data_pipeline:
  source:
    type: s3
    bucket: ml-training-data
    encryption: AES-256-SSE
    kms_key_id: arn:aws:kms:us-east-1:123456789012:key/abcd-1234
  processing:
    temp_storage:
      encryption: true
      type: tmpfs       # In-memory filesystem, never written to disk
    data_masking:
      enabled: true
      fields: [ssn, email, phone]
  output:
    encryption: AES-256-SSE
    access_logging: true
```

### 16.2 Data Access Logging

```python
import functools
import time

def log_data_access(audit_logger):
    """Decorator to log data access in training pipelines."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                audit_logger.log(AuditEvent(
                    timestamp=time.time(),
                    event_type="data_access",
                    user_id=kwargs.get("user_id", "system"),
                    model_name=kwargs.get("model_name", "unknown"),
                    model_version=kwargs.get("model_version", "unknown"),
                    resource_id=kwargs.get("data_uri", str(args[0]) if args else "unknown"),
                    action=func.__name__,
                    latency_ms=(time.time() - start) * 1000,
                    status="success",
                ))
                return result
            except Exception as e:
                audit_logger.log(AuditEvent(
                    timestamp=time.time(),
                    event_type="data_access",
                    user_id=kwargs.get("user_id", "system"),
                    model_name="", model_version="",
                    resource_id=kwargs.get("data_uri", "unknown"),
                    action=func.__name__,
                    latency_ms=(time.time() - start) * 1000,
                    status=f"error: {e}",
                ))
                raise
        return wrapper
    return decorator
```

---

## 17. ML-Specific Vulnerability Scanning

```bash
# Comprehensive ML security scan checklist (automate via scripts/security_scan.py)

# 1. Dependency vulnerabilities
pip-audit --requirement requirements.txt

# 2. Container vulnerabilities
trivy image ml-service:latest

# 3. Pickle deserialization attacks
fickling model.pkl --check-safety

# 4. ONNX model inspection
python -c "import onnx; model = onnx.load('model.onnx'); onnx.checker.check_model(model)"

# 5. Secret scanning
gitleaks detect --source . --report-format json

# 6. Infrastructure as Code scanning
checkov -d ./infra --framework terraform

# 7. Adversarial robustness (see scripts/security_scan.py)
python scripts/security_scan.py --scan all --model model.pt --data train.csv
```

### Token Authenticator for Model Endpoints

```python
import time
import hashlib
from functools import wraps
from collections import defaultdict

class TokenAuthenticator:
    """Simple token-based authentication for model endpoints."""

    def __init__(self):
        self._tokens = {}  # token_hash -> {"user_id", "scopes", "expires_at"}

    def register_token(self, token: str, user_id: str, scopes: list, ttl_seconds: int = 3600):
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        self._tokens[token_hash] = {
            "user_id": user_id,
            "scopes": set(scopes),
            "expires_at": time.time() + ttl_seconds,
        }

    def authenticate(self, token: str) -> dict:
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        record = self._tokens.get(token_hash)
        if not record:
            raise PermissionError("Invalid token")
        if time.time() > record["expires_at"]:
            raise PermissionError("Token expired")
        return record

    def authorize(self, token: str, required_scope: str) -> dict:
        record = self.authenticate(token)
        if required_scope not in record["scopes"] and "admin" not in record["scopes"]:
            raise PermissionError(f"Scope '{required_scope}' not granted")
        return record
```

### DataFrame Anonymization

```python
def anonymize_dataframe(df, columns=None, method="mask"):
    """Anonymize PII in a pandas DataFrame."""
    import pandas as pd
    import hashlib as hl
    df = df.copy()
    columns = columns or df.select_dtypes(include=["object"]).columns.tolist()
    for col in columns:
        if method == "mask":
            df[col] = df[col].apply(lambda x: mask_pii(str(x))[0] if pd.notna(x) else x)
        elif method == "hash":
            df[col] = df[col].apply(
                lambda x: hl.sha256(str(x).encode()).hexdigest()[:16]
                if pd.notna(x) else x
            )
    return df
```

### Randomized Smoothing for Certified Robustness

```python
import torch
import numpy as np

def smoothed_predict(model, x, sigma, n_samples=1000, batch_size=100):
    """Predict with randomized smoothing for certifiable robustness."""
    counts = torch.zeros(model.num_classes)
    for i in range(0, n_samples, batch_size):
        bs = min(batch_size, n_samples - i)
        noise = torch.randn(bs, *x.shape) * sigma
        noisy_inputs = (x.unsqueeze(0) + noise).clamp(0, 1)
        with torch.no_grad():
            preds = model(noisy_inputs).argmax(dim=1)
        for p in preds:
            counts[p] += 1
    return counts.argmax().item(), counts
```

### Scikit-learn Adversarial Testing with ART

```python
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.estimators.classification import SklearnClassifier
from sklearn.metrics import accuracy_score

# Wrap model
classifier = SklearnClassifier(model=sklearn_model, clip_values=(0, 1))

# FGSM attack
fgsm = FastGradientMethod(estimator=classifier, eps=0.1)
adversarial_samples = fgsm.generate(X_test)

# Evaluate robustness
clean_accuracy = accuracy_score(y_test, classifier.predict(X_test).argmax(1))
adv_accuracy = accuracy_score(y_test, classifier.predict(adversarial_samples).argmax(1))
print(f"Clean: {clean_accuracy:.2%}, Adversarial: {adv_accuracy:.2%}")
print(f"Robustness drop: {clean_accuracy - adv_accuracy:.2%}")

# PGD attack (stronger)
pgd = ProjectedGradientDescent(estimator=classifier, eps=0.1, max_iter=40)
pgd_samples = pgd.generate(X_test)
```

### C&W Attack with ART

```python
from art.attacks.evasion import CarliniLInfMethod
from art.estimators.classification import PyTorchClassifier

classifier = PyTorchClassifier(
    model=model, loss=loss_fn, optimizer=optimizer,
    input_shape=(3, 224, 224), nb_classes=10,
)
attack = CarliniLInfMethod(classifier=classifier, confidence=0.1, max_iter=100)
x_adv = attack.generate(x=x_test)
```
