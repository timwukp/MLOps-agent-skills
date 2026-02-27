# LLM Data Preparation Reference

## Synthetic Data Generation Approaches

### Method Comparison

| Method | Source Paper | Core Idea | Quality | Diversity | Cost |
|---|---|---|---|---|---|
| **Self-Instruct** | Wang et al. 2023 | LLM generates instructions from seed tasks | Medium | Medium | Low |
| **Evol-Instruct** | Xu et al. 2023 | Iteratively evolve instructions for complexity | High | High | Medium |
| **MAGPIE** | Xu et al. 2024 | Extract instructions from LLM's own prefill | High | High | Low |
| **UltraChat** | Ding et al. 2023 | Multi-turn dialog generation via two LLMs | High | High | Medium |
| **OSS-Instruct** | Wei et al. 2024 | Generate from open-source code snippets | High (code) | High | Low |
| **Distillation** | Various | Strong model teaches weak model | Varies | Limited | Medium-High |

### Self-Instruct Pipeline

```python
# Self-Instruct: Generate instructions from seed tasks
seed_tasks = [
    {"instruction": "Write a poem about autumn", "output": "Golden leaves..."},
    {"instruction": "Explain photosynthesis", "output": "Photosynthesis is..."},
    # 10-20 diverse seed tasks
]

def self_instruct_generate(seed_tasks, num_to_generate=1000):
    generated = []
    for i in range(num_to_generate):
        # Sample a few seed tasks as examples
        examples = random.sample(seed_tasks + generated[:100], k=3)
        prompt = f"""Generate a new, diverse instruction-response pair.

Examples:
{format_examples(examples)}

New instruction and response:"""

        result = llm.invoke(prompt)
        instruction, output = parse_instruction_output(result)

        # Filter for quality and uniqueness
        if is_unique(instruction, generated) and passes_quality_check(output):
            generated.append({"instruction": instruction, "output": output})

    return generated
```

### Evol-Instruct Pipeline

```python
# Evol-Instruct: Evolve instructions through complexity dimensions
EVOLUTION_PROMPTS = {
    "add_constraints": "Add specific constraints or conditions to this instruction: {inst}",
    "deepen": "Make this instruction require deeper reasoning or analysis: {inst}",
    "concretize": "Make this instruction more specific and concrete: {inst}",
    "increase_steps": "Rewrite to require more reasoning steps: {inst}",
    "broaden": "Broaden the scope of this instruction while keeping it specific: {inst}",
}

def evol_instruct(instruction: str, num_evolutions: int = 3) -> list:
    evolved = [instruction]
    current = instruction
    for _ in range(num_evolutions):
        evolution_type = random.choice(list(EVOLUTION_PROMPTS.keys()))
        prompt = EVOLUTION_PROMPTS[evolution_type].format(inst=current)
        current = llm.invoke(prompt)
        if passes_quality_filter(current):
            evolved.append(current)
    return evolved
```

### MAGPIE Approach

```python
# MAGPIE: Extract instructions by sampling from LLM's pre-fill distribution
# The key insight: feed the LLM a chat template prefix and let it complete
# with a "user" message, extracting natural instructions

template_prefix = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"

# The LLM will naturally complete this with a plausible user instruction
response = llm.generate(
    prompt=template_prefix,
    temperature=1.0,
    stop_token="<|eot_id|>"
)
extracted_instruction = response.text
# Then generate a response to this instruction using the same or stronger model
```

### UltraChat Multi-Turn Generation

```
LLM-A (User Simulator) <--> LLM-B (Assistant)

Round 1: LLM-A generates an initial question about a topic
Round 2: LLM-B responds, LLM-A asks a follow-up
Round 3: Continue for 3-8 turns
Round 4: Quality filter the entire conversation
```

## Data Annotation Tools Comparison

| Feature | Argilla | Label Studio | Prodigy | Doccano |
|---|---|---|---|---|
| **License** | Apache 2.0 | Apache 2.0 | Commercial | MIT |
| **Deployment** | Self-hosted / Cloud | Self-hosted / Cloud | Local | Self-hosted |
| **LLM Integration** | Native (suggestions) | Via ML backend | Via recipes | Limited |
| **Active Learning** | Yes | Via ML backend | Yes (core feature) | No |
| **NER Annotation** | Yes | Yes | Yes | Yes |
| **Text Classification** | Yes | Yes | Yes | Yes |
| **Preference / RLHF** | Yes (native) | Custom template | Custom recipe | No |
| **Multi-User** | Yes | Yes | Single user | Yes |
| **Programmatic Access** | Python SDK | Python SDK | Python API | REST API |
| **Best For** | LLM fine-tuning data | General annotation | Rapid NLP annotation | Simple projects |

### Argilla for RLHF Data Collection

```python
import argilla as rg

# Create a preference dataset for RLHF
dataset = rg.FeedbackDataset(
    fields=[
        rg.TextField(name="instruction"),
        rg.TextField(name="response_a"),
        rg.TextField(name="response_b"),
    ],
    questions=[
        rg.RatingQuestion(name="preference", values=[1, 2, 3, 4, 5],
                          description="Rate response A vs B (1=A much better, 5=B much better)"),
        rg.TextQuestion(name="rationale", description="Explain your preference"),
    ]
)

# Add records with model-generated suggestions
for item in data:
    dataset.add_records([
        rg.FeedbackRecord(
            fields={
                "instruction": item["instruction"],
                "response_a": item["response_a"],
                "response_b": item["response_b"],
            },
            suggestions=[
                rg.SuggestionSchema(question_name="preference", value=3)
            ]
        )
    ])
```

## Dataset Quality Metrics and Filtering

### Quality Dimensions

| Metric | Description | Measurement Method |
|---|---|---|
| **Instruction Clarity** | Is the instruction unambiguous? | LLM-as-judge scoring |
| **Response Correctness** | Is the response factually accurate? | Human review or LLM verification |
| **Response Completeness** | Does the response fully address the instruction? | Checklist evaluation |
| **Instruction-Response Alignment** | Does the response match the instruction? | Embedding similarity |
| **Toxicity** | Is the content free from harmful material? | Toxicity classifier |
| **Language Quality** | Grammar, coherence, and readability | Perplexity scoring |
| **Diversity** | Does the dataset cover varied topics and styles? | Embedding clustering analysis |

### Quality Filtering Pipeline

```python
def quality_filter_pipeline(dataset: list) -> list:
    filtered = []
    for item in dataset:
        # 1. Length filters
        if len(item["instruction"].split()) < 3:
            continue  # Too short
        if len(item["output"].split()) < 10:
            continue  # Response too short
        if len(item["output"].split()) > 4000:
            continue  # Response too long

        # 2. Language quality (perplexity-based)
        if compute_perplexity(item["output"]) > 100:
            continue  # Likely incoherent

        # 3. Toxicity check
        if toxicity_score(item["output"]) > 0.7:
            continue

        # 4. Instruction-response alignment
        if embedding_similarity(item["instruction"], item["output"]) < 0.3:
            continue  # Response not related to instruction

        # 5. Deduplication (see next section)
        if is_near_duplicate(item, filtered):
            continue

        filtered.append(item)

    return filtered
```

## Deduplication Methods

### Method Comparison

| Method | Speed | Accuracy | Memory | Best For |
|---|---|---|---|---|
| **Exact Match** | Very fast | Perfect (exact) | Low | Identical duplicates |
| **MinHash LSH** | Fast | Good (near-duplicate) | Medium | Large-scale fuzzy dedup |
| **SimHash** | Very fast | Moderate | Low | Quick approximate dedup |
| **Embedding-Based** | Slow | Excellent (semantic) | High | Semantic deduplication |
| **Suffix Array** | Medium | Good (substring) | High | Training data contamination |

### MinHash LSH Implementation

```python
from datasketch import MinHash, MinHashLSH

def create_minhash(text: str, num_perm: int = 128) -> MinHash:
    """Create MinHash signature for a text."""
    m = MinHash(num_perm=num_perm)
    # Use word-level n-grams
    words = text.lower().split()
    for i in range(len(words) - 2):
        ngram = " ".join(words[i:i+3])
        m.update(ngram.encode("utf-8"))
    return m

def deduplicate_dataset(dataset: list, threshold: float = 0.8) -> list:
    """Remove near-duplicates using MinHash LSH."""
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    deduplicated = []

    for i, item in enumerate(dataset):
        text = item["instruction"] + " " + item["output"]
        minhash = create_minhash(text)

        # Check if similar document already exists
        duplicates = lsh.query(minhash)
        if not duplicates:
            lsh.insert(f"doc_{i}", minhash)
            deduplicated.append(item)

    return deduplicated
```

### Embedding-Based Semantic Deduplication

```python
import numpy as np
from sklearn.cluster import DBSCAN

def semantic_dedup(dataset: list, similarity_threshold: float = 0.95) -> list:
    """Remove semantically duplicate entries using embeddings."""
    texts = [item["instruction"] for item in dataset]
    embeddings = embed_batch(texts)  # shape: (N, dim)

    # Compute pairwise cosine similarity via DBSCAN
    distance_matrix = 1 - np.dot(embeddings, embeddings.T)
    clustering = DBSCAN(eps=1 - similarity_threshold, min_samples=1, metric="precomputed")
    labels = clustering.fit_predict(distance_matrix)

    # Keep one representative per cluster
    deduplicated = []
    seen_clusters = set()
    for i, label in enumerate(labels):
        if label not in seen_clusters:
            seen_clusters.add(label)
            deduplicated.append(dataset[i])

    return deduplicated
```

## Data Mixing Strategies for Fine-Tuning

### Recommended Mixing Ratios

| Data Type | Ratio Range | Purpose | Example Source |
|---|---|---|---|
| **Task-Specific** | 40-60% | Primary capability | Custom curated for your use case |
| **General Instruction** | 20-30% | Maintain general capability | OpenHermes, SlimOrca |
| **Code** | 5-15% | Reasoning and structure | Code Alpaca, OSS-Instruct |
| **Math/Logic** | 5-10% | Logical reasoning | MetaMathQA, GSM8K |
| **Safety/Alignment** | 5-10% | Maintain safety properties | Safety-tuned examples |
| **Multi-Turn Dialog** | 5-15% | Conversational ability | UltraChat, ShareGPT |

### Data Mixing Configuration

```python
# Example mixing configuration for a customer support fine-tune
mixing_config = {
    "task_specific": {
        "path": "data/customer_support_conversations.jsonl",
        "ratio": 0.50,
        "description": "Domain-specific customer support examples"
    },
    "general_instruction": {
        "path": "data/openhermes_filtered.jsonl",
        "ratio": 0.20,
        "description": "General instruction following"
    },
    "multi_turn": {
        "path": "data/ultrachat_sampled.jsonl",
        "ratio": 0.10,
        "description": "Multi-turn conversational ability"
    },
    "code": {
        "path": "data/code_alpaca.jsonl",
        "ratio": 0.05,
        "description": "Structured reasoning via code"
    },
    "safety": {
        "path": "data/safety_examples.jsonl",
        "ratio": 0.10,
        "description": "Refusal and safety behaviors"
    },
    "math": {
        "path": "data/metamath_sampled.jsonl",
        "ratio": 0.05,
        "description": "Mathematical reasoning"
    }
}

def create_mixed_dataset(config: dict, total_examples: int = 50000) -> list:
    mixed = []
    for name, cfg in config.items():
        n = int(total_examples * cfg["ratio"])
        data = load_jsonl(cfg["path"])
        sampled = random.sample(data, min(n, len(data)))
        mixed.extend(sampled)
    random.shuffle(mixed)
    return mixed
```

## Dataset Formats

### Alpaca Format

```json
{
  "instruction": "Summarize the following article in 3 bullet points.",
  "input": "The article text goes here...",
  "output": "- Point 1\n- Point 2\n- Point 3"
}
```

### ShareGPT Format (Multi-Turn)

```json
{
  "conversations": [
    {"from": "human", "value": "What is reinforcement learning?"},
    {"from": "gpt", "value": "Reinforcement learning is a type of ML..."},
    {"from": "human", "value": "How does it differ from supervised learning?"},
    {"from": "gpt", "value": "The key differences are..."}
  ]
}
```

### OpenAI Chat Format

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum computing."},
    {"role": "assistant", "content": "Quantum computing uses..."}
  ]
}
```

### OASST Format (Conversation Tree)

```json
{
  "message_id": "msg_001",
  "parent_id": null,
  "text": "What is the capital of France?",
  "role": "prompter",
  "labels": {"quality": 0.9, "toxicity": 0.0},
  "replies": [
    {
      "message_id": "msg_002",
      "parent_id": "msg_001",
      "text": "The capital of France is Paris.",
      "role": "assistant",
      "labels": {"quality": 0.95, "helpfulness": 0.9}
    }
  ]
}
```

### Format Conversion Utility

```python
def alpaca_to_chat(example: dict) -> dict:
    """Convert Alpaca format to OpenAI chat format."""
    user_content = example["instruction"]
    if example.get("input"):
        user_content += f"\n\n{example['input']}"
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": example["output"]}
        ]
    }

def sharegpt_to_chat(example: dict) -> dict:
    """Convert ShareGPT format to OpenAI chat format."""
    role_map = {"human": "user", "gpt": "assistant"}
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for turn in example["conversations"]:
        messages.append({
            "role": role_map[turn["from"]],
            "content": turn["value"]
        })
    return {"messages": messages}
```

## Legal and Ethical Considerations

### Data Licensing Checklist

| Concern | Question to Ask | Risk Level |
|---|---|---|
| **License Compliance** | Does the dataset license allow commercial use and model training? | High |
| **Attribution** | Does the license require attribution? | Medium |
| **Derivative Works** | Are fine-tuned models considered derivative works? | High |
| **PII Content** | Does the dataset contain personal identifiable information? | High |
| **Copyrighted Content** | Does the dataset contain copyrighted material? | High |
| **Synthetic Data Provenance** | Was synthetic data generated from a model whose ToS restricts this? | Medium |
| **Opt-Out Compliance** | Have data subjects been given the opportunity to opt out? | Medium |

### Model Terms of Service for Synthetic Data

- **OpenAI**: Output can be used to train models, but review current ToS for restrictions.
- **Anthropic**: Check usage policies for synthetic data generation at scale.
- **Google**: Review Gemini API ToS for model distillation clauses.
- **Meta (Llama)**: Community license generally permits, but review acceptable use policy.
- **Mistral**: Review commercial license terms for output usage.

Always check the most current terms of service, as they are subject to change.

### Ethical Data Practices

1. Audit datasets for demographic bias and harmful stereotypes before training.
2. Include diverse annotators in preference data collection.
3. Document dataset composition, sources, and known limitations (datasheet/data card).
4. Implement PII detection and removal before using data for training.
5. Respect robots.txt and terms of service when collecting web data.
6. Obtain informed consent when collecting data from human participants.

## Best Practices

1. **Quality over quantity**: A small, high-quality dataset outperforms a large, noisy one.
2. **Diversify sources**: Mix synthetic, curated, and real-world data for robustness.
3. **Deduplicate aggressively**: Near-duplicates waste compute and can cause memorization.
4. **Filter systematically**: Apply multiple quality filters in a pipeline, not ad hoc.
5. **Validate with humans**: Spot-check a random sample of the final dataset before training.
6. **Version your datasets**: Track dataset versions alongside model versions.
7. **Document everything**: Create a data card describing sources, processing, and known limitations.

## Common Pitfalls

- Using synthetic data from a model to train the same model (model collapse).
- Not deduplicating between training and evaluation sets (data contamination).
- Over-representing one topic or style in the training mix.
- Ignoring license restrictions on source datasets or model-generated outputs.
- Not filtering for quality, leading to noisy training signal.
- Setting mixing ratios without empirical validation on a held-out set.
- Generating synthetic data without diversity prompting, producing homogeneous outputs.

## Further Reading

- [Self-Instruct Paper](https://arxiv.org/abs/2212.10560)
- [Evol-Instruct / WizardLM Paper](https://arxiv.org/abs/2304.12244)
- [MAGPIE Paper](https://arxiv.org/abs/2406.08464)
- [UltraChat Paper](https://arxiv.org/abs/2305.14233)
- [Argilla Documentation](https://docs.argilla.io/)
- [Label Studio Documentation](https://labelstud.io/guide/)
- [Hugging Face Datasets Library](https://huggingface.co/docs/datasets/)
- [Lilac Dataset Curation Tool](https://lilacml.com/)
- [Data-Centric AI Resource Hub](https://github.com/daochenzha/data-centric-AI)
- [Databricks Dolly Paper - Democratizing LLM Training Data](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm)
