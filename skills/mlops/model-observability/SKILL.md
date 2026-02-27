---
name: model-observability
description: >
  Implement ML observability for production systems. Covers prediction logging, model explainability (SHAP, LIME,
  Integrated Gradients, Anchors, counterfactuals), feature attribution, OpenTelemetry integration for ML services,
  distributed tracing across pipeline components, structured logging, debugging mispredictions, slice-based analysis
  by cohort/segment, fairness and bias detection, model interpretability dashboards, root cause analysis, data
  lineage visualization, prediction audit trails, Arize/Fiddler/WhyLabs integration, and custom observability metrics.
  Use when debugging models, explaining predictions, setting up tracing, or building interpretability dashboards.
license: Apache-2.0
metadata:
  author: mlops-skills
  version: "1.0"
  category: mlops
---

# Model Observability

## Overview

ML observability goes beyond monitoring by providing the ability to understand WHY a model
is behaving a certain way - through explainability, tracing, logging, and slice analysis.

## When to Use This Skill

- Debugging why a model makes certain predictions
- Adding explainability to model outputs
- Setting up distributed tracing for ML pipelines
- Building fairness and bias detection systems
- Creating audit trails for compliance

## Three Pillars of ML Observability

```
                ML Observability
         ┌──────────┬──────────┐
         │  Metrics  │   Logs   │  Traces
         │          │          │
         │ accuracy │ predict  │ request →
         │ latency  │ inputs   │ preprocess →
         │ drift    │ outputs  │ inference →
         │ fairness │ errors   │ postprocess →
         │          │          │ response
         └──────────┴──────────┘
```

## Step-by-Step Instructions

### 1. SHAP Explanations

```python
import shap
import numpy as np

class ModelExplainer:
    def __init__(self, model, model_type="tree"):
        self.model = model
        if model_type == "tree":
            self.explainer = shap.TreeExplainer(model)
        elif model_type == "linear":
            self.explainer = shap.LinearExplainer(model, X_train)
        else:
            self.explainer = shap.KernelExplainer(model.predict, shap.sample(X_train, 100))

    def explain_single(self, instance):
        """Explain a single prediction."""
        shap_values = self.explainer.shap_values(instance.reshape(1, -1))
        return {
            "base_value": float(self.explainer.expected_value),
            "prediction": float(self.model.predict(instance.reshape(1, -1))[0]),
            "feature_contributions": dict(zip(
                feature_names,
                [float(v) for v in shap_values[0]]
            )),
            "top_positive": sorted(
                zip(feature_names, shap_values[0]),
                key=lambda x: x[1], reverse=True
            )[:5],
            "top_negative": sorted(
                zip(feature_names, shap_values[0]),
                key=lambda x: x[1]
            )[:5],
        }

    def explain_batch(self, X, save_path=None):
        """Generate SHAP summary for a batch."""
        shap_values = self.explainer.shap_values(X)

        if save_path:
            shap.summary_plot(shap_values, X, feature_names=feature_names,
                            show=False)
            import matplotlib.pyplot as plt
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            plt.close()

        # Feature importance ranking
        importance = np.abs(shap_values).mean(axis=0)
        return dict(sorted(
            zip(feature_names, importance),
            key=lambda x: x[1], reverse=True
        ))
```

### 2. LIME Explanations

```python
from lime.lime_tabular import LimeTabularExplainer

explainer = LimeTabularExplainer(
    X_train.values,
    feature_names=feature_names,
    class_names=class_names,
    mode="classification",
)

explanation = explainer.explain_instance(
    instance,
    model.predict_proba,
    num_features=10,
    num_samples=5000,
)

# Get feature contributions
contributions = explanation.as_list()
# [('feature_a > 0.5', 0.15), ('feature_b = 1', -0.08), ...]
```

### 3. Structured Prediction Logging

```python
import json
import logging
from datetime import datetime
from uuid import uuid4

class PredictionLogger:
    def __init__(self, log_path="predictions.jsonl"):
        self.logger = logging.getLogger("predictions")
        handler = logging.FileHandler(log_path)
        handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def log(self, features, prediction, model_version, metadata=None):
        record = {
            "prediction_id": str(uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "model_version": model_version,
            "features": features if isinstance(features, dict) else features.tolist(),
            "prediction": float(prediction) if not isinstance(prediction, list) else prediction,
            "metadata": metadata or {},
        }
        self.logger.info(json.dumps(record))
        return record["prediction_id"]
```

### 4. OpenTelemetry Tracing

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Setup
provider = TracerProvider()
processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="http://jaeger:4317"))
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("ml-service")

def predict_with_tracing(features, model, preprocessor):
    with tracer.start_as_current_span("prediction_pipeline") as span:
        span.set_attribute("model.version", "v1.0")
        span.set_attribute("input.feature_count", len(features))

        with tracer.start_as_current_span("preprocessing"):
            processed = preprocessor.transform(features)
            span.set_attribute("preprocessing.output_shape", str(processed.shape))

        with tracer.start_as_current_span("inference"):
            prediction = model.predict(processed)
            span.set_attribute("prediction.value", float(prediction[0]))

        with tracer.start_as_current_span("postprocessing"):
            result = postprocess(prediction)

        return result
```

### 5. Slice-Based Analysis

```python
def slice_analysis(df, prediction_col, target_col, slice_columns):
    """Analyze model performance across data slices."""
    results = []

    for col in slice_columns:
        for slice_value in df[col].unique():
            mask = df[col] == slice_value
            subset = df[mask]

            if len(subset) < 30:  # Skip small slices
                continue

            metrics = {
                "slice_column": col,
                "slice_value": str(slice_value),
                "sample_size": len(subset),
                "accuracy": accuracy_score(subset[target_col], subset[prediction_col]),
                "f1": f1_score(subset[target_col], subset[prediction_col], average="weighted"),
                "positive_rate": subset[prediction_col].mean(),
            }
            results.append(metrics)

    results_df = pd.DataFrame(results)
    # Flag underperforming slices
    overall_acc = accuracy_score(df[target_col], df[prediction_col])
    results_df["underperforming"] = results_df["accuracy"] < overall_acc * 0.9
    return results_df
```

### 6. Fairness Metrics

```python
def compute_fairness_metrics(y_true, y_pred, sensitive_attribute):
    """Compute group fairness metrics."""
    groups = sensitive_attribute.unique()
    metrics = {}

    for group in groups:
        mask = sensitive_attribute == group
        metrics[group] = {
            "accuracy": accuracy_score(y_true[mask], y_pred[mask]),
            "positive_rate": y_pred[mask].mean(),
            "true_positive_rate": recall_score(y_true[mask], y_pred[mask], zero_division=0),
            "false_positive_rate": (
                ((y_pred[mask] == 1) & (y_true[mask] == 0)).sum() /
                max((y_true[mask] == 0).sum(), 1)
            ),
            "sample_size": mask.sum(),
        }

    # Disparate impact ratio
    rates = [m["positive_rate"] for m in metrics.values()]
    metrics["disparate_impact"] = min(rates) / max(rates) if max(rates) > 0 else 0
    metrics["equalized_odds_gap"] = max(
        [m["true_positive_rate"] for m in metrics.values() if isinstance(m, dict)]
    ) - min(
        [m["true_positive_rate"] for m in metrics.values() if isinstance(m, dict)]
    )

    return metrics
```

## Best Practices

1. **Log every prediction** with a unique ID for traceability
2. **Explain outlier predictions** automatically (high confidence wrong, low confidence)
3. **Slice by business segments** - not just overall metrics
4. **Use SHAP for global, LIME for local** explanations
5. **Add OpenTelemetry** early - retrofitting tracing is painful
6. **Monitor fairness metrics** alongside performance metrics
7. **Build audit trails** for regulated industries
8. **Set retention policies** for prediction logs
9. **Profile explanations** - SHAP can be slow for large models

## Scripts

- `scripts/explain_predictions.py` - SHAP/LIME explainability engine
- `scripts/prediction_logger.py` - Structured prediction logging with tracing

## References

See [references/REFERENCE.md](references/REFERENCE.md) for platform comparisons.
