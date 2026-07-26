# LLMOps batch 1 findings (llm-fine-tuning, llm-evaluation, llm-deployment, llm-prompt-engineering, llm-rag)

## 1. llm-fine-tuning
- 1.1 SKILL.md:99-106 + scripts/finetune_lora.py:133-140: SFTTrainer arg mix fails on every TRL version (`max_seq_length` on trainer + `processing_class` + plain TrainingArguments). Fix: use `SFTConfig(max_length=2048,...)`, drop max_seq_length kwarg.
- 1.2 SKILL.md:60-61 + finetune_lora.py:78: `torch_dtype` deprecated (transformers>=4.56 → `dtype=`); missing `import torch` in snippet.
- 1.3 SKILL.md:112-133 + finetune_lora.py:69-80: QLoRA missing `prepare_model_for_kbit_training()`.
- 1.4 SKILL.md:207-208: wrong GGUF cmd — `convert_hf_to_gguf.py --outtype q4_k_m` invalid; needs f16 convert then `llama-quantize ... Q4_K_M`.
- 1.5 SKILL.md:231: `np` used without import.
- 1.6 Llama-3.1-8B dated exemplar → mention Llama 4/Qwen3.
- 1.7 REFERENCE.md:183: Axolotl repo moved to axolotl-ai-cloud/axolotl.
- 1.8 PLAUSIBLE REFERENCE.md:66: ChatML attributed to Mistral (wrong; Mistral uses [INST]).

## 2. llm-evaluation
- 2.1 SKILL.md:143-169 + evaluate_llm.py:120-146: RAGAS legacy pre-0.2 API (function metrics, question/answer/contexts/ground_truth schema). Fix: class metrics + EvaluationDataset/SingleTurnSample (user_input/response/retrieved_contexts/reference), explicit llm=/embeddings=.
- 2.2 REFERENCE.md:186: `scipy.stats.binom_test` removed SciPy 1.12 → `binomtest`.
- 2.3 SKILL.md:92-137: module-level openai call legacy; `json` not imported; gpt-4o old defaults (also evaluate_llm.py:6/149/207/241, safety_eval.py:5-6/223).
- 2.4 evaluate_llm.py:149-188: Anthropic fallback still passes gpt-4o-mini model name → API rejects. Default should map to claude model on fallback.
- 2.5 safety_eval.py:34+127: canary "dan" substring matches "dangerous" etc → false positives; use \bdan\b.
- 2.6 safety_eval.py:49: setdefault no-op if OPENAI_API_KEY set; --api-key-env silently ignored.
- 2.8 evaluate_llm.py:180-189: judge JSON parse without response_format → markdown-fenced JSON fails silently.
- 2.9 REFERENCE.md:73/135/170: stale judge model recs (GPT-4, Claude Sonnet/Opus by old names).

## 3. llm-deployment
- 3.1 SKILL.md:164-176: Dockerfile CMD broken — exec-form no env expansion + vllm-openai image has ENTRYPOINT. Fix: CMD as entrypoint args with inline model name.
- 3.2 deploy_vllm.py:53-75: generated docker-compose uses ${MODEL} host-interpolation that's never set → inline values into command:.
- 3.3 deploy_vllm.py:210: throughput_rps = count/max(latencies) wrong → measure wall clock.
- 3.4 deploy_vllm.py:132-159 + benchmark_inference.py:74-86: "tokens/sec" counts SSE chunks; add caveat.
- 3.5 SKILL.md:137-148: AutoAWQ deprecated → llm-compressor; also repeats wrong GGUF cmd at :148.
- 3.6 REFERENCE.md:8: PagedAttention row self-contradictory (Ollama "Via llama.cpp" vs llama.cpp "No").
- 3.7 REFERENCE.md:9 + SKILL.md:251: continuous batching/multi-GPU "No" for Ollama/llama.cpp stale (OLLAMA_NUM_PARALLEL etc).
- 3.8 REFERENCE.md:139-148: pricing stale — Claude 3.5 Sonnet retired → claude-sonnet-5 $3/$15, claude-opus-5 $5/$25, claude-haiku-4-5 $1/$5; gpt-4o old; Llama 3.1 dated.
- 3.9 SKILL.md:112: `ollama pull` before `ollama serve` backwards; dated tag.
- 3.10 SKILL.md:257: blanket "AWQ 4-bit best" contradicts own REFERENCE (FP8 on H100).
- 3.11 PLAUSIBLE: --enable-prefix-caching default-on in vLLM V1.

## 4. llm-prompt-engineering
- 4.1 prompt_optimizer.py:31-37: COST_PER_1K stale, no Anthropic models; unknown model silently falls back to gpt-4o-mini rates (:78).
- 4.2 SKILL.md:118: `client.beta.chat.completions.parse` → `client.chat.completions.parse` (GA).
- 4.3 REFERENCE.md:148-154: Outlines v0 API removed in 1.0 → `outlines.from_transformers(...)`.
- 4.4 REFERENCE.md:130: "cache hits 75-90% cheaper" — Anthropic 90%, OpenAI 50% (outside range).
- 4.5 SKILL.md:142-147: PromptManager.load version="latest" builds vlatest.yaml → FileNotFoundError; add latest-resolution.
- 4.6 prompt_optimizer.py:209-218: injection flag matches "system prompt" in refusals → false positives.
- 4.7 SKILL.md:185-202: sanitize only lower/upper case; title case bypasses → case-insensitive regex.
- 4.8 prompt_optimizer.py:177-179: cost estimate ignores user-input tokens.
- 4.10 gpt-4o/gpt-4o-mini defaults throughout (:271 etc).

## 5. llm-rag
- 5.1 SKILL.md:55: `langchain.text_splitter` removed → `langchain_text_splitters`.
- 5.2 SKILL.md:113/123: `langchain_community.vectorstores.Chroma` deprecated → `langchain_chroma`; :184 `langchain.prompts` → `langchain_core.prompts`.
- 5.3 SKILL.md:71: `datetime.utcnow()` deprecated; datetime not imported.
- 5.4 SKILL.md:218-239: `json.loads(llm.invoke(...))` on AIMessage → needs `.content`.
- 5.5 evaluate_rag.py:199: predicted_answer falls back to ground_truth → perfect scores; skip/warn instead.
- 5.6 SKILL.md:258: claims evaluate_rag.py uses RAGAS — false; fix description.
- 5.7 build_rag.py:44-59: chunk_size words-vs-tokens mismatch; score 1-distance can go negative (also evaluate_rag.py:46).
- 5.8 gpt-4o defaults (SKILL.md:188, build_rag.py:128/166).
- 5.9 REFERENCE.md:29-41: embedding table dated (add v4-era models).
- 5.10 PLAUSIBLE: Chroma/pgvector hybrid search "No" drifting stale.
