# Artifact Contract

The repo-wide convention for what each pipeline stage writes and what the next stage
expects. Every path and key below is what the scripts in this repo actually produce or
consume today — treat this file as the interface definition between skills.

## Directory Layout

```
<output-dir>/                        # --output of model-training/scripts/train_model.py
  model.joblib                       # trained model (joblib for sklearn/xgb/lgbm/svm)
  training_report.json               # run metadata + metrics (schema below)

<artifact-root>/<run_name>/<stage>/  # ml-pipeline-orchestration scripts
  model/model.pkl                    #   (default artifact_root: /tmp/ml_pipeline/artifacts)
  evaluation/metrics.json            # flat {metric_name: value} dict
  evaluation/classification_report.txt
```

## Training Outputs

`model-training/scripts/train_model.py --output <dir>` writes:

- **`model.joblib`** — the fitted model, saved with `joblib.dump`.
- **`training_report.json`** — keys: `timestamp`, `model_type`, `task`, `input_file`,
  `target_column`, `dataset_rows`, `dataset_features`, `cv_folds`, `cv_strategy`,
  `n_trials`, `best_trial`, `best_cv_score`, `best_params`,
  `train_resubstitution_metrics`, `model_path`, `training_time_seconds`.

Metrics dicts are always flat, with **metric names as keys** and float values:

```json
{"accuracy": 0.94, "f1_weighted": 0.93}            // classification
{"rmse": 3.21, "mae": 2.10, "r2": 0.87}            // regression
```

The orchestration pipelines (`ml-pipeline-orchestration/scripts/{prefect,airflow}_pipeline.py`)
write the same flat shape to `evaluation/metrics.json`
(`accuracy`, `precision`, `recall`, `f1`, `auc_roc`). Use `best_cv_score` for
generalization estimates — `train_resubstitution_metrics` are computed on training data
and are optimistic.

## What Registration Expects

`model-registry/scripts/registry_manager.py --action register` does **not** take a file
path — it registers from an MLflow run:

- `--run-id` — a completed MLflow run that logged the model
  (e.g. via `mlflow.sklearn.log_model(model, "model")`).
- `--artifact-path` — the artifact sub-path within the run (default `model`).

Consequence: log metrics to the run with the same flat metric names as above
(`mlflow.log_metrics(metrics)`), because `--action compare` and `--action lineage` read
`run.data.metrics` — a version registered without run metrics compares as all `N/A`.
Promotion assigns aliases (`--action promote --alias champion`); serving resolves them.

## What Serving Expects

`model-serving/scripts/serve_model.py` accepts either:

- `--model-path model.joblib --framework sklearn` — the local file exactly as written
  by `train_model.py` (also `.onnx`, TorchScript `.pt`, or a TF SavedModel dir), or
- `--model-uri "models:/<name>@champion"` — the registry alias (requires mlflow).

`model-serving/scripts/batch_inference.py --input data.parquet --model-path model.joblib`
appends a **`prediction`** column (and optional `probabilities` column of JSON-encoded
lists) to the input frame and writes `<input>_predictions.<ext>`.

## What Monitoring Expects

`model-monitoring/scripts/monitor_model.py --reference ref.parquet --current cur.parquet`:

- **Reference and current files** — Parquet (preferred) or CSV, same feature columns.
  Reference = a scored snapshot of training/validation data; current = recent
  production data, i.e. the direct output of `batch_inference.py`.
- **Prediction column** — named **`prediction`** by default (`--prediction` to override);
  must exist in both files for prediction-distribution checks.
- **Target column** — optional `--target`; when present alongside `prediction`,
  performance metrics are computed with the same names as training
  (`accuracy`/`f1_weighted`/... or `rmse`/`mae`/`r2`/`mape`).
- **Thresholds file** — `--thresholds thresholds.json`, keyed by the same metric names:
  `{"accuracy": 0.9}` (minimum) or `{"rmse": {"max": 5.0}}` (min/max bounds).

`model-drift-detection/scripts/detect_drift.py` uses the identical
`--reference`/`--current` file pair.

## Contract Summary

| Producer | Artifact | Consumer | Key/column it relies on |
|----------|----------|----------|-------------------------|
| train_model.py | `model.joblib` | serve_model.py, batch_inference.py | `--model-path` + `--framework sklearn` |
| train_model.py | `training_report.json` | humans, CI gates | `best_cv_score`, flat metric names |
| MLflow run (`log_model` + `log_metrics`) | registered version | registry_manager.py | `--run-id`, `run.data.metrics` |
| registry (alias `@champion`) | model URI | serve_model.py | `--model-uri models:/<name>@champion` |
| batch_inference.py | `*_predictions.parquet` | monitor_model.py, detect_drift.py | `prediction` column |
| monitor_model.py | report JSON | setup_alerts.py | flat metric names in thresholds |
