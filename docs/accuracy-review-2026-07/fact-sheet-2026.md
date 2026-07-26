# MLOps/LLMOps 2026-07 fact sheet (deep research, verified vs primary sources)

## MLflow
- Current 3.14.0. MLflow 3.0 GA Jun 2025. LoggedModel first-class; tracing @mlflow.trace; GenAI eval.
- Stages deprecated NOT removed; recommend aliases (set_registered_model_alias, models:/MyModel@champion).
- Removed in 3.x: Recipes, fastai/mleap flavors, RunInfo.run_uuid (use run_id), evaluate baseline_model.

## Feast
- Current 0.65.0. Core API stable: apply/get_online_features/get_historical_features/materialize/materialize_incremental/push + write_to_online_store CURRENT. PushSource current. Registry now has remote gRPC + RBAC; vector search in Milvus/Mongo/Scylla online stores.

## Evidently
- Current 0.7.21. 0.7 made rewritten API default (BREAKING). New: `from evidently import Report, Dataset, DataDefinition`; presets from `evidently.presets` (DataDriftPreset, DataSummaryPreset, ClassificationPreset, TextEvals). Dataset+DataDefinition replace ColumnMapping. report.run(current, reference) returns snapshot. TestSuite GONE — merged into Report (include_tests=True; per-metric tests=[gte(...),lte(...)]).

## Great Expectations
- Current 1.19.1. 1.x flow: gx.get_context() → data_sources → asset → Batch Definition → gxe.Expect* in ExpectationSuite → ValidationDefinition → Checkpoint. validator.expect_* fluent style GONE. context.sources → context.data_sources.

## Airflow 3.3 / Prefect 3.8
- Airflow: authoring from `airflow.sdk` (dag, task, DAG, Asset). Core operators → apache-airflow-providers-standard. Removed: SubDAGs, execution_date, schedule_interval (use schedule=), REST v1, SLAs (→Deadline Alerts). catchup default False.
- Prefect 3.8: agents removed → workers/work pools; Deployment.build_from_flow gone → flow.deploy()/flow.serve(); task failures don't auto-fail flow; cache_policy replaces task_input_hash idiom; schedule= → schedules=[...].

## Inference servers
- vLLM 0.26.0; V1 engine default since 0.8, V0 removed 0.11 (AsyncLLMEngine/LLMEngine gone). gpu-memory-utilization default 0.92. Chunked prefill + prefix caching ON by default. Tool calling: --enable-auto-tool-choice --tool-call-parser. reasoning_content → reasoning.
- TGI: MAINTENANCE MODE — do not recommend for new deployments; README points to vLLM/SGLang.
- Ollama v0.32.x: OpenAI + Anthropic compat, tools, structured outputs, multimodal engine, cloud models.

## HF stack
- transformers v5 (current 5.14.1): eval_strategy only; torch_dtype → dtype; load_in_4bit/8bit kwargs REMOVED (quantization_config mandatory); apply_chat_template returns BatchEncoding.
- TRL 1.9 (1.0 Mar 2026): SFTConfig.max_length (renamed from max_seq_length), dataset_text_field in SFTConfig, processing_class, packing_strategy="bfd"; peft_config+quantization_config at constructor. DPOConfig loss_type is list[str]. GRPO stable; ORPO/CPO/PPO → trl.experimental.
- PEFT 0.19: target_modules="all-linear", use_dora, use_rslora; QLoRA recipe unchanged (nf4 + prepare_model_for_kbit_training).

## RAGAS
- Current 0.4.3. AnswerRelevancy → ResponseRelevancy (0.2) → back to AnswerRelevancy in ragas.metrics.collections (0.4 recommended path). LangchainLLMWrapper discontinued → llm_factory("..."). evaluate() deprecated toward @experiment but still works with EvaluationDataset/SingleTurnSample (user_input/response/retrieved_contexts/reference).

## Agents
- LangGraph 1.2.x (1.0 Oct 2025): StateGraph API unchanged; langgraph.prebuilt create_react_agent DEPRECATED → `from langchain.agents import create_agent` (model string "openai:gpt-5.5", middleware, checkpointer). InMemorySaver successor to MemorySaver. langchain-classic for legacy chains.
- CrewAI 1.15 (1.0 Oct 2025): independent of LangChain; Flows event-driven layer; CrewAgentExecutor→AgentExecutor.

## Pricing (per 1M tok, Jul 2026)
- OpenAI: gpt-5.6-sol $5/$30, -terra $2.50/$15, -luna $1/$6. gpt-5 legacy $1.25/$10 (EOL 2026-12-11); gpt-5-mini $0.25/$2. gpt-4.1 $2/$8, 4.1-mini $0.40/$1.60 still active. gpt-4o $2.50/$10 legacy. o3/o4-mini retiring 2026. Assistants API sunsets 2026-08-26; Responses API primary. Batch 50% off.
- Anthropic CONFIRMED: claude-opus-5 $5/$25; claude-sonnet-5 $3/$15 (intro $2/$10 to 2026-08-31); claude-haiku-4-5 $1/$5; claude-fable-5 $10/$50. Cache reads 0.1x; batch 50% off.
- Bedrock: Claude 5.x line, Nova 2, OpenAI GPT-5.5 on Bedrock, Llama 4 Scout/Maverick (~$0.80/$2.40), DeepSeek V3.x ($0.62/$1.85), Qwen3, Mistral Large 3.
- Open-weight: Llama 4 Scout/Maverick; Qwen3 / Qwen3.5-397B; DeepSeek V4 (v4-flash $0.14/$0.28); gpt-oss-120b/20b Apache-2.0.

## SageMaker
- Python SDK v3 (3.17.0): Estimator/Model/Predictor REMOVED → ModelTrainer (from sagemaker.train import ModelTrainer; SourceCode/Compute configs) + ModelBuilder for deploy. v2 (2.257.x) maintained in parallel. sagemaker-core merged in.
- Managed MLflow GA (tracking servers + serverless MLflow Apps preferred). HyperPod: EKS/Slurm, recipes v2 (SFT/LoRA/DPO/RLAIF), HyperPod inference GA.
- Naming: "SageMaker AI" classic; umbrella = Unified Studio + lakehouse.

## Guardrails/moderation
- GuardrailsAI 0.10.x active; Guard().use(...) unchanged. NeMo Guardrails 0.23; Colang 2.0 still beta. Llama Guard 4 (12B) current, MLCommons S1-S14; Prompt Guard 2 for injection. OpenAI moderation: omni-moderation-latest (free).

## Embeddings / vector DBs
- OpenAI text-embedding-3 still latest but not leaderboard-leading. Cohere embed-v4.0 (multimodal, 128K). Voyage voyage-4. Qwen3-Embedding (Apache) top open family; bge-m3 multilingual workhorse.
- Chroma 1.x Rust rewrite (sparse vectors/BM25 now supported). Qdrant 1.18 (GPU HNSW). pgvector 0.8.x (iterative index scans, halfvec/sparsevec). Milvus 2.6/3.0-beta.
