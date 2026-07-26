# LLMOps batch 2 findings (llm-guardrails, llm-observability, llm-agent-orchestration, llm-cost-optimization, llm-data-preparation)

## llm-guardrails
- guardrails_pipeline.py:117: injection regex vs text.lower() — uppercase literals (DAN, <<SYS>>, [INST]) NEVER match → drop .lower(), use re.IGNORECASE. (critical check silently disabled)
- guardrails_pipeline.py:42-47,269-271 + SKILL.md:46-76: PII redaction order corrupts credit cards (phone pattern eats 10 digits) → redact credit_card→ssn→phone; add \b anchors to phone regex.
- guardrails_pipeline.py:44: unanchored phone regex blocks any 10-digit run → \b anchors.
- guardrails_pipeline.py:238-257,303: failure_action warn/fallback not implemented (warn blocks!) → implement or remove choices.
- guardrails_pipeline.py:239: redact path skips output guardrails → run output_guard on redacted-path responses.
- guardrails_pipeline.py:141-151: _language fails open, contradicts REFERENCE "fail closed" → make configurable/fail closed or note.
- red_team.py:130-140: API errors counted as jailbreak successes → separate errored bucket, exclude from denominator.
- red_team.py:122: unused prompt param.
- red_team.py:66: malformed base64 payload (trailing \x08) → fix to ...SEQi.
- red_team.py:31: "as an ai" counted as refusal signal → remove/deweight.
- REFERENCE.md:249-251: garak probe `knowledgegraph` doesn't exist → use real probes (encoding,dan,promptinject); update gpt-4o.
- SKILL.md:186-191: NeMo expects config.yml not rails.yaml; model stale.
- SKILL.md:106-114: toxic-bert pipeline instantiated per call (violates own <200ms rule) → hoist to __init__; note [:512] is chars.
- SKILL.md:99-152: missing import json.
- SKILL.md:52-60: substring jailbreak blocklist needs caveat (defence-in-depth supplement).
- Missing best practice: structural defense — separate trusted system instructions from untrusted content (system role channel).
- REFERENCE.md:205-218: streaming guard drops tail buffer → final `if buffer: yield buffer`.
- REFERENCE.md:155: omni-moderation has 13 categories not 11.
- SKILL.md:172-176 + REFERENCE:116-120: guard(llm_api=...) legacy → guard(model=..., messages=[...]) (Guardrails >=0.5).

## llm-observability
- SKILL.md:57: "claude-sonnet-4-6-20250514" wrong (no date suffix; that's Sonnet 4 date) → claude-sonnet-4-6 or claude-sonnet-5.
- SKILL.md:58: claude-haiku-4-5 pricing $0.80/$4 wrong → $1.00/$5.00; use bare alias.
- llm_monitor.py:24-35: DEFAULT_PRICING nearly all retired models → replace with claude-opus-5 (5/25), claude-sonnet-5 (3/15), claude-haiku-4-5 (1/5), gpt-5 family; warn on unknown model instead of silent $0.
- llm_monitor.py:126-132: latency alert compares sorted values to own p95 — never fires; recency slice wrong → fix percentile index + keep chronological order for recent slice.
- llm_monitor.py:43: hardcoded cl100k_base wrong for gpt-4o (o200k); tiktoken categorically wrong for Claude → encoding_for_model + note messages.count_tokens for Claude.
- SKILL.md:80-87: groupby agg on grouping key fragile → named aggregation.
- SKILL.md:69,202,213: datetime.utcnow() deprecated → datetime.now(timezone.utc).
- SKILL.md:169-179: streaming counts chunks as tokens; input_tokens=0 stub → read final usage chunk.
- quality_tracker.py:126: A/B winner ignores ties/significance → note min sample + tie handling.
- SKILL.md:94-99 + REFERENCE:81-84: LANGCHAIN_* env vars legacy → LANGSMITH_TRACING/API_KEY/PROJECT; key prefix lsv2_pt_.
- SKILL.md:99: langchain.callbacks import deprecated + unused → remove.
- SKILL.md:120-142 + REFERENCE:96-123: Langfuse v2 API removed in v3 → `from langfuse import observe, get_client`; start_as_current_generation etc.
- REFERENCE:105: get_relevant_documents deprecated → retriever.invoke.
- REFERENCE.md:7: Phoenix is Elastic License 2.0 not Apache 2.0.
- REFERENCE.md:8: LangSmith self-hosted exists (Enterprise) → "Enterprise only".
- gpt-4o refs → current models.

## llm-agent-orchestration
- build_agent.py:49-60: eval "sandbox" escapable — replace with ast-based safe evaluator (ast.literal_eval/operator whitelist); fix docstring.
- build_agent.py:63-85: python_executor blocklist trivially bypassed → label clearly UNSAFE-for-untrusted-input, recommend subprocess+container isolation; use sys.executable.
- REFERENCE:144-161: "isolated subprocess" claim false → fix prose (PATH-only restriction ≠ sandbox), sys.executable, cleanup temp file, add timeouts/limits note.
- multi_agent.py:199-226: run_parallel is sequential → ThreadPoolExecutor.
- multi_agent.py:277: worker results overwritten → append list.
- multi_agent.py:235/374: default hierarchical manager mismatch → pick agent with can_delegate_to.
- multi_agent.py:307-308: parse fallback assigns whole plan to every worker → fail loudly / retry.
- build_agent.py:241: greedy DOTALL action regex → non-greedy, no DOTALL.
- build_agent.py:249-253: arg fallback wrong for python_executor/file_reader → map from tool's first required param.
- build_agent.py:212 + multi_agent.py:100: OPENAI_API_BASE is v0 var → OPENAI_BASE_URL.
- build_agent.py:213: empty-string api_key → pass None.
- build_agent.py:223: client per call → hoist.
- build_agent.py:227: .content may be None → guard.
- build_agent.py:270: catch Exception in tool exec, return as observation.
- SKILL.md:120-168: LangGraph — define llm; use add_edge(START,...)/plain add_edge; modern idiom note; create_react_agent deprecated → `from langchain.agents import create_agent` note in REFERENCE:26-36.
- REFERENCE:286-307: interrupt_before legacy → show interrupt()/Command(resume=...) as primary.
- SKILL.md:55-86: add "strict": True + additionalProperties:false to tool schema; note default not honored.
- SKILL.md:101-110: model_dump() for message append; check finish_reason.
- REFERENCE.md:7: "MIT (Creative Commons)" → "MIT".
- gpt-4o refs → current models.

## llm-cost-optimization
- SKILL.md:251: Haiku 4.5 $0.80/$4 wrong → $1.00/$5.00. (also contradicts cost_optimizer.py:25 $0.25/$1.25 = Claude 3 Haiku)
- SKILL.md:250: add claude-sonnet-5/claude-opus-5/claude-fable-5 rows; note 4.6 previous-gen.
- SKILL.md:57-70: o3-mini superseded + "reasoning" tier unreachable dead code → wire reasoning branch or remove.
- REFERENCE:11-12: Claude 3.5 Sonnet/Haiku RETIRED → replace rows (claude-sonnet-5 3/15, claude-haiku-4-5 1/5).
- REFERENCE:17-18: Gemini 1.5 retired → current Gemini.
- REFERENCE:27-31 + cost_optimizer.py:21-28: invented model keys ("claude-sonnet","llama-3") → real IDs; fix Claude pricing.
- SKILL.md:208-241: CostAwareLLMPipeline calls undefined select_top_context/calculate_cost → define or simplify.
- SKILL.md:218: cache-hit cost 0 wrong for semantic cache (embedding cost) — own REFERENCE lists this pitfall.
- SKILL.md:128-165: compress_prompt dead strategies list, ZeroDivision on empty, unused target_reduction.
- SKILL.md:174-202: batch ignores output_file param; missing OpenAI import; add custom_id ordering warning.
- cost_optimizer.py:36-42: tiktoken KeyError for Claude → word-count fallback silently; log path; note count_tokens API.
- cost_optimizer.py:30-33: complexity regex overbroad (bare "explain") → require more signal or lower weight.
- cost_optimizer.py:86: hardcoded 1.5 output ratio inside route_request → use avg_output_ratio param.
- cost_optimizer.py:91-93: budget fallback ignores output price; return dict shape inconsistent.
- cost_optimizer.py:225-227: compress action writes different shape to file vs stdout → make consistent.
- cache_manager.py:31-35: builtin hash() randomized per process → persisted fallback embeddings garbage after restart → hashlib stable hash or don't persist.
- cache_manager.py:23-29: catch OSError too for model download failure.
- cache_manager.py:53-96: stats never persisted → always 0 from CLI → persist counters in DB.
- cache_manager.py:76-81: linear scan per miss — note vector index need.
- cache_manager.py:130-153: SQLite "LRU" is FIFO (timestamp never updated on read) → update timestamp on get or fix docs.
- cache_manager.py:82: no model/system-prompt keying → include model in cache key.
- REFERENCE:97-100: openai.ChatCompletion.create v0 API removed → current GPTCache adapter form.
- REFERENCE:118-128: missing import time.
- Missing: provider prompt caching section (90% reads discount, 1.25x write, prefix-match invariant, interaction with prompt compression).

## llm-data-preparation
- generate_synthetic.py:72-99: two infinite-loop paths on LLM failure/wrong keys → max-attempts guard, unconditional decrement.
- generate_synthetic.py:121-134: brute-force minhash O(n²), no LSH banding → use datasketch MinHashLSH or fix claim.
- curate_dataset.py:22: credit-card regex matches any 13-16 digit run → add Luhn check.
- curate_dataset.py:23: IP regex matches version strings → octet-constrained pattern.
- curate_dataset.py:137-140: remove_pii mutates in place → copy.
- curate_dataset.py:155-172: stratified split gives 0 train for groups <4 → guard small groups into train.
- curate_dataset.py:95-118: score_quality bisects text mid-word; magic weights; "1." matches decimals → fix/document.
- SKILL.md:61-75: json_object mode vs "return JSON array" prompt vs pairs/data parser 3-way mismatch → ask for {"pairs":[...]} (like the script) or use json_schema.
- SKILL.md:158-218: missing imports (np, random), undefined client, ZeroDivision on empty → fix.
- SKILL.md:226-254: lsh.insert ValueError comment wrong (key exists ≠ duplicate); branch unreachable → use lsh.query like REFERENCE.
- REFERENCE:218-226: create_minhash empty sig for <3 words → guard like scripts.
- REFERENCE:252-270: semantic_dedup needs normalized embeddings; DBSCAN min_samples=1 chains clusters; O(N²) memory → note all three.
- SKILL.md:122-152 + REFERENCE:119-149: Argilla 1.x API removed in 2.0 → rg.Argilla client, rg.Dataset+rg.Settings, rg.Record, dataset.records.log, rg.Suggestion.
- REFERENCE:79-88: MAGPIE stop_token param not real; note model-family-specific.
- SKILL.md:86-117: DPO rejected-from-weaker-model caveat (length bias; on-policy + judge standard) → add note.
- REFERENCE:465: model collapse vs filtered self-distillation distinction → refine.
- Missing: benchmark decontamination implementation → at least note.
- gpt-4o/gpt-4o-mini refs → current models.

## Cross-cutting
- Add version pins/targets in SKILL.md frontmatter or a "Tested with" line (argilla 2.x, langfuse 3.x, openai 1.x, langgraph 1.x, guardrails 0.10).
