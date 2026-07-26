# MLOps batch 3 findings (model-observability, ml-pipeline-orchestration, ml-testing, ml-security, ml-cost-optimization)

## model-observability
- A1 explain_predictions.py:134: LIME intercept is dict keyed by class → KeyError:0 on every lime run. Fix: `label=next(iter(exp.intercept))`.
- A2 explain_predictions.py:39: torch.load needs weights_only=False on PyTorch>=2.6 (or load state_dict).
- A3 explain_predictions.py:89-90,110,162-163: SHAP>=0.45 returns 3-D ndarray, list-guard never fires → wrong importances. Fix: `if sv.ndim==3: sv=sv[...,1]`.
- A4 SKILL.md:69: float(expected_value) fails for classifiers (ndarray).
- A5 SKILL.md:61,72,90,99: X_train/feature_names undefined in ModelExplainer class.
- B1 REFERENCE.md:57: TreeSHAP complexity O(TLD) → O(TLD^2).
- B2 SKILL.md:247-251: "equalized_odds_gap" only computes TPR gap = equal opportunity; add FPR or rename.
- B3 REFERENCE.md:106: calibration prose conflates predictive parity.
- B4 REFERENCE.md:355: dead AIF360 URL → https://aif360.readthedocs.io/.
- B5 SKILL.md:261: "SHAP global, LIME local" misleading.
- B6 shap.summary_plot legacy → shap.plots.beeswarm/bar (note).
- B7 SKILL.md:132,147: datetime.utcnow() deprecated/naive → datetime.now(timezone.utc).
- Minor: prediction_logger.py:97-99 conn leak; logging propagate.

## ml-pipeline-orchestration (Airflow 3 / Prefect 3 sweep)
- A1 SKILL.md:64: days_ago removed in Airflow 3 → static start_date.
- A2 schedule_interval removed in Airflow 3 → schedule= (SKILL.md:83,210; REFERENCE:783,1876; airflow_pipeline.py:559).
- A3 SKILL.md:109: KubernetesPodOperator import → airflow.providers.cncf.kubernetes.operators.pod.
- A4 SKILL.md:117-120 + REFERENCE:1811: resources= dict removed → container_resources=k8s.V1ResourceRequirements.
- A5 SKILL.md:122: is_delete_operator_pod → on_finish_action="delete_pod".
- A6 SKILL.md:188: Dataset → Asset in Airflow 3 (note rename).
- A7 airflow_pipeline.py:100,114: context["execution_date"] removed → logical_date.
- A8 REFERENCE:280: task SLAs removed in Airflow 3 (Deadline Alerts) — add caveat.
- A9 SKILL.md:264: TriggerDagRunOperator → airflow.providers.standard.operators.trigger_dagrun in 3.x.
- B1 airflow_pipeline.py:215/601/644-661: notify_failure not downstream of branch → validation failures produce NO alert. Fix: add t_branch_validation >> t_notify_failure.
- B2 REFERENCE:357: df.to_parquet(mode="overwrite") — no such param → TypeError.
- B3 prefect_pipeline.py:93-94,113: cache keyed on per-second run_name → never hits; ingest cache returns stale run_name. Fix caching keys.
- B4 prefect_pipeline.py:832-868: claims .map() parallel but sequential loop — fix docstring or use .submit().
- B5 prefect_pipeline.py:853,863: max_depth=None silently becomes 100.
- C1 prefect_pipeline.py:781-803 + REFERENCE:1199-1216: Deployment.build_from_flow removed in Prefect 3 → flow.deploy()/flow.serve().
- C2 "default-agent-pool" stale naming.
- C3 task_input_hash legacy → cache_policy=INPUTS (Prefect 3).
- C4 REFERENCE:1842-1850: prefect.infrastructure removed in Prefect 3.
- D1 REFERENCE:1571,1823: KFP Output[Model] as return annotation invalid (v2 wants param).
- D2 REFERENCE:1076: set_gpu_limit deprecated → set_accelerator_type + set_accelerator_limit.
- D3 REFERENCE:1925-1948: KFP v1 YAML in v2 doc.
- D4 REFERENCE:1322,1329-30: Dagster FreshnessPolicy/AutoMaterializePolicy deprecated → AutomationCondition.eager().
- D5 REFERENCE:47: "Dagit" retired → Dagster UI / dagster-webserver.
- D6 REFERENCE:1491: ZenML Output() removed → Tuple[Annotated[...]].
- E1 REFERENCE:241: parse interval attributed to dag_dir_list_interval; actual min_file_process_interval 30s.
- E2 SKILL.md:151 + REFERENCE:1541: "XCom ~48KB" invented → DB-dependent (MySQL ~64KB, Postgres ~1GB).
- E3 SKILL.md:384-394: fabricated Prometheus metric names → real statsd names (dagrun.duration.failed.<dag_id>).
- E4 REFERENCE:1613: pkg_resources removed → importlib.metadata.
- Minor: unused imports, datetime.utcnow, missing imports in snippets.

## ml-testing
- A1 REFERENCE:135-136: GX context.sources → context.data_sources (GX 1.0+); validator flow → ValidationDefinition/Checkpoint.
- A2 pandera: recommend `import pandera.pandas as pa` (0.20+).
- B1 test_model.py:191-201: "global_metric" is unweighted mean of slice accuracies → use full-set accuracy.
- B2 test_data_pipeline.py:94: dtype substring match ("int" in "uint8") → exact pandas_dtype compare.
- B3 test_data_pipeline.py:188-189: KS p-value gate fails at scale → threshold on statistic/PSI or subsample (add note).
- B4 test_model.py:327-328: "N/A" default truthy → prints PASS.
- C: SKILL.md:212-219 single un-warmed latency assert flaky (script does it right — align); SKILL.md:151-157 array == in assert; REFERENCE:258 predict_proba array compare.

## ml-security (highest severity batch)
- A1 REFERENCE:241-257: MITRE ATLAS technique table FABRICATED (AML.T0000 doesn't exist; wrong IDs throughout) → replace with real IDs (AML.T0006 Active Scanning, T0020 Poison Training Data, T0043 Craft Adversarial Data, T0018 Backdoor, T0024.x extraction/membership, T0029 DoS).
- A2 security_scan.py:550,569: .h5 marked safe — Keras Lambda layers = code exec → move to medium.
- A3 REFERENCE:605: CCPA 72-hour breach deadline wrong → "expedient/without unreasonable delay; AG if >500 CA residents".
- A4 REFERENCE:534,607: GDPR "right to explanation Art.22 required" overstated.
- A5 privacy_guard.py:233-243: k_anonymize_* is just bucketing, no group-size guarantee → rename/verify min group size.
- A6 privacy_guard.py:352-355: "rdp" = sum(eps)*0.7 fail-open → remove or use opacus RDPAccountant.
- A7 REFERENCE:284: advertorch license LGPL-3.0 not MIT.
- B1 security_scan.py:765-786: pip-audit exit 1 = vulns found but code only parses on 0 → all real vulns dropped, reports "passed". Parse stdout for rc in (0,1).
- B2 security_scan.py:815-825: safety 2/3.x JSON is object not list → findings dropped; safety check deprecated → safety scan.
- B3 security_scan.py:295-302: named_modules name is attr path not class → use type(m).__name__.
- B4 security_scan.py:278: scanner itself uses weights_only=False (self-flagging) → gate behind flag.
- B5 privacy_guard.py:660: read_json nrows without lines=True raises → silently "no PII".
- B6 privacy_guard.py:349: log(1/0) when delta=0; sum_eps_sq unused.
- B8 SKILL.md:386-391: slowapi needs request: Request param + app.state.limiter + exception handler — example can't start.
- B9 SKILL.md:368-383: pydantic v1/v2 mixed → field_validator + v2 syntax.
- B10 SKILL.md:213-219: TF Privacy needs reduction=NONE loss.
- B11 SKILL.md:100-112: "pgd_attack" lacks random start = BIM not PGD → add uniform init.
- B12 SKILL.md:278-283: ADMIN bypasses explicit deny → reorder checks.
- B13/B14 REFERENCE: model.num_classes undefined; mask_pii undefined in snippet.
- C1 security_scan.py:172-372: "adversarial robustness scanner" runs NO attacks (imports FGM/PGD unused) → relabel as load smoke-check or implement.
- C2: every .pkl model hard-fails scan exit 2 → make severity configurable.
- C3: regex PII presented as compliance-grade → add Luhn check for credit card, note Presidio/Comprehend for HIPAA Safe Harbor.
- C4 REFERENCE:1380-94: ART FGSM on tree ensembles raises NotImplementedError → use HopSkipJump or linear model.
- D1 REFERENCE:403-409: DP noise→epsilon table overstates eps ~3-5x → regenerate with opacus accountant.
- Minor: section numbering collision (two "10"s), dead code, magic delta*10.

## ml-cost-optimization
- A1 SKILL.md:113-118: SIGTERM handler calls save_checkpoint(emergency=True) missing 5 required args → TypeError at preemption. Fix: store state on instance.
- A2 model_compress.py:462-496: iterative pruning never accumulates (make_permanent + re-selecting zeros) → 37% not 90%. Fix cumulative amount or keep masks.
- A3 model_compress.py:238-264: static quantization mutates caller's model, no QuantStub/fusion → non-runnable. Use deepcopy + stubs or note limitation.
- A4 cost_analyzer.py:237-259: mixed_precision baseline excludes HPO hours → negative savings with --hpo-trials.
- A5 model_compress.py:432-456 + REFERENCE:71: structured pruning mask ≠ compute reduction — fix notes.
- A6 model_compress.py CLI subcommands hard-code MLP arch → document demo-only or accept arch.
- A7 quantize output can't be loaded by compare (packed params) → fix workflow doc.
- B1 SKILL.md:150-174: torch.cuda.amp deprecated → torch.amp autocast("cuda")/GradScaler("cuda").
- B2 torch.quantization → torch.ao.quantization (+ mention FX/PT2E).
- B3 SKILL.md:170-178: grad accumulation snippet: model(batch) is logits not loss.
- C1 SKILL.md:181: "1x A100 40GB + accumulation replaces 4x A100 80GB" WRONG (weights/optimizer memory; 4x slower) → fix claim.
- C2 H100 990 TFLOPS is sparse figure in dense table → 494 dense (SXM); adjust throughput_factor.
- C3 cost_analyzer.py prices contradict REFERENCE table (V100 1.50 vs 3.06; H100 6 vs 8+/12.3) → reconcile.
- C4 REFERENCE:11: p3.2xlarge listed as 32GB V100 → 16GB; 32GB only p3dn.24xlarge.
- C5 GCP T4 $0.35 is accelerator-only price, excludes VM.
- C6 H100 spot $2.50 unrealistic on big-3 → note neocloud figure.
- C8 REFERENCE:159-166: TCO example uses $200k staff then $50k in math → fix arithmetic; conclusion sensitive.
- C9 SKILL.md:105-111: EC2 spot warning is IMDS/EventBridge, NOT SIGTERM on bare EC2 → add IMDS poll or scope claim to ECS/EKS/SageMaker.
- C10: GPU catalog tops out at H100 → add H200/B200/L40S/MI300X; Cloud Run GPU is GA (2025).
- C11 "free accuracy" contradicts own 0.1-1% loss table.
- D1 cost_analyzer.py:590-595: additive savings can exceed 100% → cap/note overlap.
- D3: right-sizing rule uses dataset size as proxy for model size → note limitation.
- Content gap: no hosted-LLM token economics coverage.
