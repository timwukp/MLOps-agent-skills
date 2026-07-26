# MLOps batch 1 findings (data-ingestion, data-validation, feature-engineering, feature-store, ml-experiment-tracking)

## data-ingestion
- 1.1 SKILL.md:130: "Manual commit for exactly-once" false → "at-least-once; pair with idempotent writes".
- 1.2 SKILL.md:163-166: Delta Spark session missing spark.sql.extensions + catalog config; stale delta-spark 3.1.0 pin → add both configs (or configure_spark_with_delta_pip), bump pin note.
- 1.3 SKILL.md:213-233: LakeFS example broken (module-as-object calls; commits_api undefined; deprecated lakefs_client) → rewrite with current `lakefs` high-level SDK.
- 1.4 SKILL.md:101: SQL injection f-string in incremental load → bound parameters.
- 1.5 SKILL.md:188: note days() import (pyspark.sql.functions).
- 1.6 REFERENCE:138: pd.read_gbq removed in pandas 3.0 → pandas_gbq.read_gbq or bigquery to_dataframe().
- 1.7 REFERENCE:181: missing Attr import.
- 1.8 ingest_batch.py:154-158 vs 210: watermark persisted BEFORE save → move watermark write after successful save.
- 1.9/1.12 datetime.utcnow() → datetime.now(timezone.utc) (ingest_batch.py:219, ingest_streaming.py:93,97,111).
- 1.10 ingest_batch.py:199-203: --validate never fails run → add --strict or fail on issues.
- 1.11 ingest_streaming.py:139-140: timer-based flush skips consumer.commit() → commit after that flush too.

## data-validation
- 2.2 SKILL.md:81 + REFERENCE:84: `import pandera as pa` → `import pandera.pandas as pa` (0.20+).
- 2.3 REFERENCE:30-75: GX suite/checkpoint section is removed 0.x API → rewrite to 1.x: context.suites.add(gx.ExpectationSuite(name=...)), gx.Checkpoint(validation_definitions=[...], actions=[...]), checkpoint.run().
- 2.4 REFERENCE:139-149: Airflow GreatExpectationsOperator legacy → GXValidateDataFrameOperator/GXValidateCheckpointOperator (provider 1.0+).
- 2.5 validate_data.py:2 + SKILL.md:268: docstring claims GX/Pandera but script is hand-rolled → fix descriptions.
- 2.6 validate_data.py:168: datetime.utcnow.
- 2.8 data_contract.py:166-169: one bad value empties nums → no check emitted → per-value filter + report unparseable as failures.
- 2.9 data_contract.py:158,219: v may be None from DictReader → (v or "").

## feature-engineering
- 3.1 SKILL.md:211-215: SHAP importance wrong for classifiers (3-D/list output) → np.abs(sv).mean(axis=(0,2)) or normalize sv[...,1].
- 3.2 SKILL.md:94: TargetEncoder fit_transform returns DataFrame → .iloc[:,0]; also note sklearn>=1.3 built-in TargetEncoder with CV (3.3).
- 3.4 transform_features.py:85-88: log/power transform applied to ALL numeric cols though config docs say per-column → implement per-column selection (separate pipeline for listed cols) or fix docs; guard log1p negatives.
- 3.5 transform_features.py:194-199,221-226: sparse output crashes pd.DataFrame → sparse_threshold=0 on ColumnTransformer.
- 3.6 transform_features.py:115: lambda in FunctionTransformer unpicklable → module-level function.
- 3.7 select_features.py:70-72: chi2 fallback returns first-N as "ranking"; chi2 run for regression targets → skip chi2 for regression, return empty on failure.
- 3.8 select_features.py:37: document _is_clf heuristic.

## feature-store
- 4.1 SKILL.md:96-101 + feast_setup.py:80-84: Entity(value_type=ValueType.INT64) deprecated → Entity(name=..., join_keys=[...]) (REFERENCE already correct).
- 4.2 SKILL.md:165-183: freshness uses last_updated_timestamp (apply time) not materialization → use fv.materialization_intervals[-1][1].
- 4.3 SKILL.md:90 + feast_setup.py:39: entity_key_serialization_version 2 → 3.
- 4.4 SKILL.md:226-252: missing imports; store.push needs PushSource defined → add PushSource to example.
- 4.5 SKILL.md:61: quote 'feast[redis]'.
- 4.6 feast_setup.py:130: datetime.utcnow → now(timezone.utc).
- feature_registry.py:95-101: re-register overwrites created_at → preserve original created_at.

## ml-experiment-tracking
- 5.1/5.6 SKILL.md:69 + REFERENCE:59-60 + mlflow_tracker.py:201-214: log_model positional artifact_path deprecated MLflow 3 → name= kwarg.
- 5.2 SKILL.md:97: mlflow.transformers.autolog() doesn't exist → use mlflow.autolog() or transformers MLflowCallback.
- 5.3 SKILL.md:214-218: Docker env vars wrong → MLFLOW_BACKEND_STORE_URI / MLFLOW_DEFAULT_ARTIFACT_ROOT.
- 5.7 mlflow_tracker.py:351-374: cleanup deletes RUNNING runs by default → default delete_unfinished=False or add age threshold.
- 5.8/5.11 datetime.utcnow (mlflow_tracker.py:107, experiment_compare.py:269,321).
- 5.9 missing metrics rendered as 0 → N/A (mlflow_tracker.py:344, experiment_compare.py:251,292).
- 5.10 experiment_compare.py:277-281 + 468-473: 'N/A' string through :.5f crashes → guard stats["error"] before markdown; conditional formatting.
- 5.12 note: parallel-coords params from first run only; higher-is-better assumption → add note/direction flag.

## Hygiene
- Remove committed __pycache__/*.pyc under skills/*/scripts/ and add .gitignore entry.
