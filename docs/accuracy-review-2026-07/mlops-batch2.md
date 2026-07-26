# MLOps batch 2 findings (model-training, model-registry, model-serving, model-monitoring, model-drift-detection)

## model-training
- SKILL.md:101-103 vs 141: CosineAnnealingLR T_max sized in steps but scheduler.step() per epoch → set T_max=epochs OR step per batch.
- SKILL.md:90 + REFERENCE:126: torch.cuda.amp imports deprecated → torch.amp.autocast("cuda")/GradScaler("cuda") (script already uses new API).
- SKILL.md:215: DDP `loss = model(X, y)` wrong → output = model(X); loss = criterion(output, y).
- SKILL.md:196-198 vs 225: mp.spawn-style init vs torchrun launch inconsistent → env-based init + LOCAL_RANK for torchrun.
- SKILL.md:295-301: grad accumulation no final flush (script handles correctly) → add `or (step+1)==len(loader)`.
- SKILL.md:178-183: Optuna MedianPruner ineffective without trial.report() — note or fix.
- train_model.py:171,236-237,348: XGBoost early_stopping_rounds without eval_set → EVERY xgb trial fails. Drop early_stopping_rounds for CV path.
- train_model.py:349-352: "final metrics" are resubstitution on train data → rename key/log clearly or compute on held-out.
- distributed_train.py:26 vs 355: torch.amp.GradScaler needs torch>=2.3 → bump stated requirement.
- distributed_train.py:332-376: val metrics not all-reduced under DDP → dist.all_reduce counters.

## model-registry
- SKILL.md:63-81,163,176,260-265 + registry_manager.py:97-102,184,225: MLflow stages API deprecated 2.9/removed 3.x → migrate to aliases (set_registered_model_alias, get_model_version_by_alias, models:/name@champion). REFERENCE.md already shows correct API — make consistent.
- SKILL.md:171: ModelVersion.tags is a dict; `{t.key: t.value for t in ...}` raises AttributeError → use dict directly.
- SKILL.md:102: datetime.utcnow() → datetime.now(timezone.utc).
- SKILL.md:115-120: log_model artifact_path positional deprecated in MLflow 3 → name= (version note).
- model_packager.py:70-74: `mlflow models serve --no-conda` removed in MLflow 2.0 → --env-manager local.
- model_packager.py:90: stale fallback pin mlflow==2.12.1 → current 3.x (or >=2.9).

## model-serving
- SKILL.md:88-90: /metrics returns bytes via JSONResponse → Response(generate_latest(), media_type=CONTENT_TYPE_LATEST).
- SKILL.md:95-115: BentoML 1.1 runner/bentoml.io API deprecated → rewrite to 1.2+ @bentoml.service class API.
- SKILL.md:273-298: BatchPredictor stranded-queue + race; np not imported → fix loop (drain until empty / re-arm task safely) + import.
- SKILL.md:130-135: read_parquet loads all — fix "chunks manage memory" claim (pyarrow iter_batches) ; remove unused ProcessPoolExecutor.
- SKILL.md:181,193-194: HEALTHCHECK curl on python:3.11-slim (no curl) → python urllib check or install curl.
- SKILL.md:59: Optional[List[float]]; 64-69: blocking predict in async def → def or run_in_executor (note).
- REFERENCE:7: Triton supports more backends (TensorRT, OpenVINO, Python, FIL, vLLM) → fix row.
- REFERENCE:13,21,215: TorchServe limited-maintenance note.
- serve_model.py:264,293: uvicorn.run(app_object, workers=N) silently ignored → note/import-string or remove flag.
- serve_model.py:239: on_event deprecated → lifespan; latencies list unbounded → cap deque.
- batch_inference.py:133: df.values feeds ALL columns → add --feature-columns/--drop-columns option or documented contract.

## model-monitoring
- SKILL.md:37-64 + monitor_model.py:96-99: Evidently ≤0.4 legacy API → rewrite to 0.7+: `from evidently import Report, Dataset, DataDefinition`; `from evidently.presets import DataDriftPreset,...`; TestSuite removed (Report include_tests=True); run() returns snapshot. TargetDriftPreset gone.
- SKILL.md:90-91: whylogs generate_constraints_report returns list of ReportResult → fix aggregate usage.
- SKILL.md:131: datetime.utcnow no import + deprecated.
- SKILL.md:152-156: Histogram default buckets meaningless for prediction values → set buckets=.
- REFERENCE:78-81: PSI thresholds inconsistent with drift skill (0.1/0.25 vs 0.1/0.2) → align, note conventions.
- monitor_model.py:41: MAPE zero-denominator hack → document/exclude zeros.
- setup_alerts.py:87,114-117: cooldown in-memory only, lost per CLI run → persist last-fired in SQLite.
- setup_alerts.py:230: AlertRule(**r) TypeErrors on SKILL.md's `window:` key → align YAML schema.

## model-drift-detection
- SKILL.md:114-118 + detect_drift.py:56-58: chisquare + clip(lower=1) violates scipy sum constraint → ValueError exactly when categories differ. Fix: chi2_contingency on 2×k table OR rescale expected to sum equality.
- SKILL.md:130-158: Evidently legacy API + version-specific result shape → update to 0.7+ or pin.
- SKILL.md:179: pd not imported; :206 datetime.utcnow.
- REFERENCE:79: KS "less powerful for large samples" backwards → "overly sensitive".
- drift_monitor.py:60-63: PSI bins derived per-dataset → blind to location shift (MAJOR math bug). Fix: reference-derived shared edges.
- drift_monitor.py:175,279: single threshold for PSI and KS incommensurate → separate --psi-threshold/--ks-threshold.
- Minor: O(n·m) KS note; silent cell drops in _read_csv_numeric.
