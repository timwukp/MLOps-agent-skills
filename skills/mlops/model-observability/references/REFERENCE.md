# Model Observability Reference Guide

Comprehensive reference for ML model explainability, fairness assessment, prediction logging,
debugging, and regulatory compliance. Covers tooling comparisons, architecture patterns, and
actionable guidance.

---

## Table of Contents

1. [Explainability Tools Comparison](#1-explainability-tools-comparison)
2. [SHAP Explainer Types](#2-shap-explainer-types)
3. [Fairness Metrics](#3-fairness-metrics)
4. [Fairness Tools Comparison](#4-fairness-tools-comparison)
5. [Prediction Logging Architecture Patterns](#5-prediction-logging-architecture-patterns)
6. [Model Debugging Workflow](#6-model-debugging-workflow)
7. [Regulatory Requirements for Explainability](#7-regulatory-requirements-for-explainability)
8. [Further Reading](#8-further-reading)

---

## 1. Explainability Tools Comparison

| Feature | SHAP | LIME | Captum | ELI5 | InterpretML |
|---------|------|------|--------|------|-------------|
| **Type** | Global + Local | Local only | Local (primarily) | Local + some global | Global + Local |
| **Theory** | Shapley values (game theory) | Local surrogate (linear model) | Gradient-based attributions | Multiple methods wrapped | Glassbox + blackbox |
| **Model-agnostic** | Yes (KernelSHAP) | Yes | No (PyTorch only) | Partial | Yes |
| **Tree models** | Yes (TreeSHAP, exact, fast) | Yes (generic) | No | Yes (via sklearn) | Yes (EBM native) |
| **Deep learning** | Yes (DeepSHAP, GradientSHAP) | Yes (generic) | Yes (native, comprehensive) | Limited | Limited |
| **Tabular data** | Excellent | Good | Good | Good | Excellent (EBM) |
| **Text** | Yes (partition explainer) | Yes (text perturbation) | Yes (NLP attributions) | Yes (text pipeline) | Limited |
| **Images** | Yes (partition explainer) | Yes (superpixel) | Yes (pixel attributions) | No | No |
| **Speed** | Fast for trees; slow for KernelSHAP | Moderate (sampling-based) | Fast (gradient-based) | Fast | Fast for EBM; moderate for blackbox |
| **Consistency** | Mathematically guaranteed (Shapley axioms) | Approximate; varies across runs | Deterministic for gradient methods | Method-dependent | EBM is inherently interpretable |
| **Visualization** | Rich (force, summary, dependence, waterfall) | Basic (feature weights) | Basic (attribution heatmaps) | Basic (feature weights) | Good (interactive dashboard) |
| **Maintenance** | Active | Active | Active (Meta) | Low / minimal | Active (Microsoft) |
| **License** | MIT | BSD 2-Clause | BSD 3-Clause | MIT | MIT |

### Recommendation Matrix

| Use Case | Recommended Tool | Rationale |
|----------|-----------------|-----------|
| Tree model explanation (XGBoost, LightGBM, RF) | SHAP (TreeExplainer) | Exact Shapley values in polynomial time |
| Quick local explanation for any model | LIME | Fast, intuitive, model-agnostic |
| PyTorch deep learning attribution | Captum | Native integration, comprehensive gradient methods |
| Inherently interpretable model | InterpretML (EBM) | Explainable Boosting Machine matches gradient boosting accuracy with glass-box interpretability |
| Regulatory audit requiring consistency | SHAP | Shapley axioms provide mathematical guarantees |
| Non-technical stakeholder reports | SHAP (waterfall/force plots) or InterpretML dashboard | Best visualizations for communication |

---

## 2. SHAP Explainer Types

| Explainer | Model Type | Algorithm | Speed | Exact | When to Use |
|-----------|-----------|-----------|-------|-------|-------------|
| **TreeExplainer** | Tree ensembles (XGBoost, LightGBM, CatBoost, sklearn RF/GBT) | Tree path-based Shapley computation | Very fast (O(TLD)) | Yes | Always use for tree models; no reason to use KernelSHAP |
| **KernelExplainer** | Any model (model-agnostic) | Weighted linear regression on sampled coalitions | Slow (O(2^M) approx.) | No (approximate) | Black-box models; non-standard architectures; small feature sets |
| **DeepExplainer** | Deep neural networks (PyTorch, TF/Keras) | DeepLIFT-based backpropagation | Fast | No (approximate) | Deep learning classification/regression; faster than KernelSHAP |
| **GradientExplainer** | Differentiable models (PyTorch, TF/Keras) | Expected gradients (integrated gradients + SHAP) | Fast | No (approximate) | When DeepExplainer is not compatible; combines integrated gradients with Shapley |
| **LinearExplainer** | Linear models (sklearn linear, logistic) | Exact computation for linear models | Very fast | Yes | Linear/logistic regression; fast exact explanations |
| **PermutationExplainer** | Any model (model-agnostic) | Antithetic permutation sampling | Slow | No (approximate) | Benchmarking; when KernelSHAP variance is too high |
| **Partition Explainer** | Any model (model-agnostic) | Hierarchical clustering of features | Moderate | No (approximate) | Text and image data; exploits feature hierarchy |

### Usage Examples

```python
import shap

# TreeExplainer -- always use for tree models
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)

# KernelExplainer -- model-agnostic fallback
explainer = shap.KernelExplainer(model.predict, shap.sample(X_train, 100))
shap_values = explainer.shap_values(X_test[:50], nsamples=500)

# DeepExplainer -- deep learning
explainer = shap.DeepExplainer(torch_model, background_data)
shap_values = explainer.shap_values(test_batch)

# GradientExplainer -- alternative for deep learning
explainer = shap.GradientExplainer(torch_model, background_data)
shap_values = explainer.shap_values(test_batch, nsamples=200)
```

### Common Pitfalls with SHAP

- **KernelSHAP with many features**: Computation grows exponentially. Use `nsamples` parameter and limit to top features.
- **Background data selection**: Use a representative sample (100-500 points), not the full training set.
- **Correlated features**: Shapley values can distribute importance among correlated features unpredictably. Consider grouping.
- **TreeExplainer interventional vs. path-dependent**: Default is `feature_perturbation="tree_path_dependent"` which is faster but can be misleading with correlated features. Use `"interventional"` for causal interpretation.

---

## 3. Fairness Metrics

### Metric Definitions and Interpretation

| Metric | Definition | Formula | Threshold | Use When |
|--------|-----------|---------|-----------|----------|
| **Demographic Parity** | Equal positive prediction rates across groups | P(Y_hat=1 \| A=0) = P(Y_hat=1 \| A=1) | Ratio between 0.8-1.25 (4/5ths rule) | No ground truth labels; hiring, lending decisions |
| **Equalized Odds** | Equal TPR and FPR across groups | TPR_A=0 = TPR_A=1 AND FPR_A=0 = FPR_A=1 | Difference < 0.05-0.10 | Classification with ground truth; criminal justice, medical diagnosis |
| **Equal Opportunity** | Equal TPR across groups (relaxed equalized odds) | TPR_A=0 = TPR_A=1 | Difference < 0.05-0.10 | When false negatives are the primary concern |
| **Calibration** | Equal positive predictive value across groups | P(Y=1 \| Y_hat=p, A=a) = p for all a | Calibration curves overlap | Risk scoring; insurance; credit |
| **Disparate Impact** | Ratio of positive outcome rates | P(Y_hat=1 \| A=0) / P(Y_hat=1 \| A=1) | > 0.8 (80% rule, legal standard) | Legal compliance; ECOA, employment law |
| **Predictive Parity** | Equal precision across groups | PPV_A=0 = PPV_A=1 | Difference < 0.05-0.10 | When false positives are costly |
| **Treatment Equality** | Equal ratio of FN to FP across groups | FN/FP for A=0 = FN/FP for A=1 | Ratio close to 1.0 | Balancing error types across groups |

### Impossibility Results

It is mathematically impossible to simultaneously satisfy all fairness metrics (except in trivial cases):

- **Calibration + Equal FPR + Equal FNR**: Cannot hold simultaneously when base rates differ across groups (Chouldechova, 2017).
- **Demographic parity + Calibration**: Cannot hold when groups have different base rates (Kleinberg et al., 2016).
- **Implication**: Choose the fairness metric most aligned with your domain, legal requirements, and ethical priorities.

---

## 4. Fairness Tools Comparison

| Feature | AIF360 (IBM) | Fairlearn (Microsoft) | What-If Tool (Google) |
|---------|-------------|----------------------|----------------------|
| **Type** | Library | Library | Interactive visualization |
| **Language** | Python | Python | Web UI (TensorBoard plugin) |
| **Pre-processing mitigations** | Reweighing, Disparate Impact Remover, Learning Fair Representations, Optimized Preprocessing | Correlation Remover | None |
| **In-processing mitigations** | Adversarial Debiasing, Prejudice Remover, Meta-Fair Classifier, GerryFair | Exponentiated Gradient, Grid Search (constraint-based) | None |
| **Post-processing mitigations** | Calibrated Equalized Odds, Reject Option Classification | ThresholdOptimizer | Manual threshold adjustment |
| **Metrics** | 70+ fairness metrics | Core metrics (demographic parity, equalized odds, etc.) | Visual metric comparison |
| **Visualization** | Basic plots | Matplotlib/widget dashboards | Rich interactive UI |
| **Integration** | Standalone | sklearn-compatible API | TensorBoard, Jupyter |
| **Best For** | Research, comprehensive audits | Production ML pipelines, sklearn workflows | Exploration, non-technical stakeholders |
| **License** | Apache 2.0 | MIT | Apache 2.0 |

### Quick Start Examples

**AIF360**:
```python
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing

dataset = BinaryLabelDataset(df=df, label_names=['outcome'],
                              protected_attribute_names=['gender'])
metric = BinaryLabelDatasetMetric(dataset, unprivileged_groups=[{'gender': 0}],
                                   privileged_groups=[{'gender': 1}])
print(f"Disparate impact: {metric.disparate_impact():.3f}")
```

**Fairlearn**:
```python
from fairlearn.metrics import MetricFrame, demographic_parity_difference
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

metric_frame = MetricFrame(metrics={'accuracy': accuracy_score,
                                     'dp_diff': demographic_parity_difference},
                           y_true=y_test, y_pred=y_pred,
                           sensitive_features=sensitive_test)
print(metric_frame.by_group)

# Mitigation
mitigator = ExponentiatedGradient(estimator=base_model,
                                   constraints=DemographicParity())
mitigator.fit(X_train, y_train, sensitive_features=sensitive_train)
```

---

## 5. Prediction Logging Architecture Patterns

### Pattern 1: Synchronous Logging (Simple)

```
Client -> API Gateway -> Model Service -> [Log to DB] -> Response
```
- **Pros**: Simple; log is written before response is returned.
- **Cons**: Adds latency to prediction path; DB failure can block predictions.
- **Use when**: Low traffic; latency is not critical; strong audit requirements.

### Pattern 2: Asynchronous Queue-Based Logging (Recommended)

```
Client -> API Gateway -> Model Service -> Response
                              |
                              v (async)
                         Message Queue (Kafka/SQS/Pub/Sub)
                              |
                              v
                       Log Consumer -> Storage (S3/BigQuery/Data Lake)
```
- **Pros**: Does not add latency; decoupled; fault-tolerant; supports replay.
- **Cons**: Eventual consistency; requires queue infrastructure.
- **Use when**: Production systems; high traffic; latency-sensitive.

### Pattern 3: Sidecar / Agent Logging

```
Client -> API Gateway -> [Model Service + Sidecar Logger] -> Response
                                    |
                                    v
                              Log Aggregator (Fluentd/Filebeat)
                                    |
                                    v
                              Storage / SIEM
```
- **Pros**: No code changes to model service; language-agnostic.
- **Cons**: More infrastructure; potential resource contention.
- **Use when**: Kubernetes environments; multi-language services.

### What to Log

| Field | Description | Required |
|-------|------------|----------|
| `request_id` | Unique prediction identifier | Yes |
| `timestamp` | ISO 8601 timestamp | Yes |
| `model_id` | Model name + version | Yes |
| `input_features` | Raw or hashed input features | Yes (for drift detection) |
| `prediction` | Model output (class, score, probability) | Yes |
| `latency_ms` | Inference latency | Yes (for performance monitoring) |
| `ground_truth` | Actual outcome (when available, often delayed) | When available |
| `explanation` | SHAP values or feature attributions | For auditable models |
| `metadata` | User agent, region, A/B test variant | Recommended |

---

## 6. Model Debugging Workflow

### Step-by-Step Debugging Process

```
1. DETECT: Alert triggered (performance drop, drift, anomalous predictions)
     |
2. SCOPE: Determine impact -- which segments, features, time range?
     |
3. DIAGNOSE: Root cause analysis
     |-- Data issue? (schema change, missing values, upstream pipeline failure)
     |-- Drift? (feature distributions shifted)
     |-- Model issue? (overfitting, underfitting, stale model)
     |-- Infrastructure? (wrong model version, resource contention, serialization bug)
     |
4. EXPLAIN: Generate explanations for problematic predictions
     |-- SHAP values for individual failures
     |-- Feature importance shifts between reference and production
     |-- Cohort-level analysis (are specific subgroups failing?)
     |
5. FIX: Apply appropriate remedy
     |-- Data fix: Repair pipeline, add validation
     |-- Retrain: Trigger retraining with fresh data
     |-- Rollback: Revert to previous model version
     |-- Feature fix: Update feature engineering logic
     |
6. VERIFY: Confirm fix resolves the issue
     |-- Metrics recovered
     |-- Explanations look reasonable
     |-- No regressions on other segments
     |
7. PREVENT: Add monitoring, tests, or guardrails to catch this in the future
```

### Common Debugging Scenarios

| Symptom | Likely Cause | Investigation Steps |
|---------|-------------|-------------------|
| Accuracy drop across all segments | Concept drift or data pipeline failure | Check feature distributions; compare training vs. serving data |
| Accuracy drop for one segment | Label drift or subgroup-specific data issue | Slice metrics by segment; check for missing data patterns |
| Prediction distribution shift | Feature drift or upstream data change | Compare input feature distributions; check ETL logs |
| Latency spike | Model size increase, resource contention, batching issue | Check model version, infrastructure metrics, request patterns |
| Confidence scores all near 0.5 | Feature pipeline returning defaults/nulls | Inspect raw prediction inputs; check feature store health |

---

## 7. Regulatory Requirements for Explainability

### EU AI Act

**Status**: Entered into force August 2024; obligations phased in through 2027.

| Risk Level | Requirements | ML Explainability Implications |
|------------|-------------|-------------------------------|
| **Unacceptable** | Prohibited (social scoring, real-time biometric identification in public) | N/A -- these systems are banned |
| **High-risk** | Conformity assessment, risk management, data governance, transparency, human oversight, accuracy/robustness | Must provide explanations of model behavior; document training data; enable human review of decisions; maintain technical documentation |
| **Limited** | Transparency obligations | Disclose that users are interacting with an AI system |
| **Minimal** | No specific obligations | Best practice to provide explanations |

**High-risk categories include**: credit scoring, employment screening, insurance pricing, law enforcement, border control, education admission, critical infrastructure.

**Technical requirements for high-risk systems**:
- Provide clear documentation of model inputs, logic, and outputs.
- Enable human oversight and the ability to override AI decisions.
- Maintain logs of AI system operation for traceability.
- Conduct bias testing and ongoing monitoring.
- Ensure robustness against errors and adversarial attacks.

### ECOA (Equal Credit Opportunity Act) and Regulation B

**Applies to**: All creditors in the United States.

| Requirement | Implication |
|-------------|------------|
| **Adverse action notices** | When credit is denied or terms are unfavorable, must provide specific reasons (up to 4 principal reasons) |
| **Prohibited bases** | Cannot discriminate based on race, color, religion, national origin, sex, marital status, age, public assistance status |
| **Reason codes** | Must map model predictions to human-readable reason codes (e.g., "high debt-to-income ratio") |
| **Explainability requirement** | Model must be able to generate specific, actionable reasons for adverse decisions |

**Implementation approach**:
- Use SHAP values to identify top contributing features for each prediction.
- Map SHAP-derived feature importances to pre-approved reason code library.
- Ensure reason codes are specific and actionable (not "model score was low").

### SR 11-7 (Federal Reserve Supervisory Guidance on Model Risk Management)

**Applies to**: Banks and financial institutions supervised by the Federal Reserve.

| Principle | ML Implication |
|-----------|---------------|
| **Model validation** | Independent validation of all models; challenger model comparison |
| **Outcome analysis** | Back-testing predictions against actuals; ongoing performance monitoring |
| **Sensitivity analysis** | Test model behavior under stressed inputs; understand feature sensitivities |
| **Documentation** | Complete documentation of model methodology, assumptions, limitations |
| **Effective challenge** | Qualified independent reviewers must be able to understand and question the model |
| **Ongoing monitoring** | Continuous tracking of model performance, stability, and limitations |

**Key implication**: "Effective challenge" means the model must be explainable enough for a qualified reviewer (not necessarily ML expert) to understand its behavior, question its assumptions, and identify weaknesses.

### Compliance Implementation Checklist

```
[ ] Identify applicable regulations (EU AI Act, ECOA, SR 11-7, GDPR Art. 22)
[ ] Classify AI system risk level (for EU AI Act)
[ ] Select appropriate explainability method (SHAP for tabular, Captum for deep learning)
[ ] Generate global explanations (feature importance, partial dependence plots)
[ ] Generate local explanations for individual predictions
[ ] Map explanations to human-readable reason codes (for ECOA)
[ ] Compute fairness metrics across protected attributes
[ ] Document model methodology, training data, and limitations
[ ] Enable human override capability for high-risk decisions
[ ] Implement prediction logging with explanation storage
[ ] Set up ongoing performance and fairness monitoring
[ ] Establish model validation and independent review process (SR 11-7)
[ ] Conduct bias testing before deployment and on a recurring schedule
[ ] Maintain audit trail for all model decisions
```

---

## 8. Further Reading

### Official Documentation

- **SHAP**: https://shap.readthedocs.io/
- **LIME**: https://github.com/marcotcr/lime
- **Captum**: https://captum.ai/
- **InterpretML**: https://interpret.ml/
- **AIF360**: https://aif360.mybluemix.net/
- **Fairlearn**: https://fairlearn.org/
- **What-If Tool**: https://pair-code.github.io/what-if-tool/

### Key Papers

- Lundberg & Lee (2017) "A Unified Approach to Interpreting Model Predictions" -- SHAP foundations
- Ribeiro et al. (2016) "Why Should I Trust You? Explaining the Predictions of Any Classifier" -- LIME
- Sundararajan et al. (2017) "Axiomatic Attribution for Deep Networks" -- Integrated Gradients
- Chouldechova (2017) "Fair Prediction with Disparate Impact: A Study of Bias in Recidivism Prediction" -- impossibility theorem
- Kleinberg et al. (2016) "Inherent Trade-offs in the Fair Determination of Risk Scores" -- calibration vs. fairness
- Hardt et al. (2016) "Equality of Opportunity in Supervised Learning" -- equalized odds

### Regulatory References

- **EU AI Act**: https://artificialintelligenceact.eu/
- **ECOA / Regulation B**: https://www.consumerfinance.gov/rules-policy/regulations/1002/
- **SR 11-7**: https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm
- **NIST AI RMF**: https://www.nist.gov/artificial-intelligence/ai-risk-management-framework
- **CFPB on AI/ML in Credit**: https://www.consumerfinance.gov/about-us/blog/
