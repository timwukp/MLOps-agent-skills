# LLM RAG (Retrieval-Augmented Generation) Reference Guide

## RAG Architecture Patterns

| Pattern | Complexity | Quality | Latency | Best For |
|---------|-----------|---------|---------|----------|
| Naive RAG | Low | Baseline | Low | Simple Q&A, prototyping |
| Sentence-Window | Medium | Good | Low-Medium | Precise retrieval with broader context |
| Auto-Merging | Medium-High | High | Medium | Hierarchical documents |
| Parent-Child (Small-to-Big) | Medium | High | Medium | Long documents, section-level answers |
| Fusion RAG | High | Very High | High | Complex multi-facet queries |
| Corrective RAG (CRAG) | High | Very High | High | High-accuracy requirements |
| Self-RAG | Very High | Highest | Very High | Self-correcting generation |

**Naive RAG**: Chunk -> embed -> retrieve top-k -> generate. Simple and effective for straightforward factual queries.

**Sentence-Window**: Embed individual sentences for precise matching, but expand the context window (+/- 2 sentences) before sending to the LLM.

**Auto-Merging**: Hierarchical chunks (sentences -> paragraphs -> sections). If multiple child chunks from the same parent are retrieved, merge up to the parent level.

**Parent-Child**: Index small chunks for retrieval but return the larger parent chunk to the LLM. Balances retrieval precision with generation context.

**Fusion RAG**: Generate multiple query variants, retrieve for each, then apply reciprocal rank fusion to combine and rerank results.

**Corrective RAG**: An evaluator assesses retrieved document relevance. Irrelevant documents are filtered or the query is reformulated.

**Self-RAG**: The LLM decides when to retrieve, evaluates relevance, generates, and self-critiques for faithfulness.

## Embedding Models Comparison

| Model | Dimensions | Max Tokens | MTEB Score | Speed | Cost |
|-------|-----------|------------|------------|-------|------|
| OpenAI text-embedding-3-large | 3072 | 8191 | ~64.6 | Fast (API) | $0.13/1M tokens |
| OpenAI text-embedding-3-small | 1536 | 8191 | ~62.3 | Fast (API) | $0.02/1M tokens |
| Cohere embed-v3.0 | 1024 | 512 | ~64.5 | Fast (API) | $0.10/1M tokens |
| BGE-large-en-v1.5 | 1024 | 512 | ~64.0 | Medium | Free (self-hosted) |
| BGE-M3 | 1024 | 8192 | ~64.0 | Medium | Free (self-hosted) |
| E5-mistral-7b | 4096 | 32768 | ~66.6 | Slow | Free (GPU required) |
| all-MiniLM-L6-v2 | 384 | 256 | ~56.3 | Very Fast | Free (self-hosted) |

**Selection**: API best quality = OpenAI large or Cohere v3. Self-hosted best quality = E5-mistral-7b (GPU) or BGE-M3. Speed/resource constrained = all-MiniLM-L6-v2. Multi-lingual = BGE-M3.

## Vector Databases Comparison

| Feature | Chroma | Pinecone | Weaviate | Qdrant | Milvus | pgvector |
|---------|--------|----------|----------|--------|--------|----------|
| Type | Embedded | Managed SaaS | Self-hosted/Cloud | Self-hosted/Cloud | Self-hosted/Cloud | Postgres ext |
| Scalability | Small-Medium | Large | Large | Large | Very Large | Medium |
| Hybrid Search | No | Yes | Yes (BM25+vector) | Yes | Yes | No |
| Multi-Tenancy | Via collections | Via namespaces | Native | Via collections | Via partitions | Via schemas/RLS |
| Pricing | Free (OSS) | Per-pod/serverless | Free (OSS)/Cloud | Free (OSS)/Cloud | Free (OSS)/Cloud | Free |
| Best For | Prototyping | Zero-ops production | Feature-rich apps | Performance-critical | Massive scale | Existing Postgres |

## Chunking Strategies

| Strategy | Chunk Size | Overlap | Quality | Best For |
|----------|-----------|---------|---------|----------|
| Fixed-Size | 256-1024 tokens | 10-20% | Baseline | Quick prototyping |
| Recursive Character | 256-1024 tokens | 10-20% | Good | General-purpose text |
| Sentence-Based | 1-5 sentences | 0-1 sentence | Good | Precise retrieval |
| Semantic | Variable | Adaptive | High | Topic-coherent chunks |
| Document-Structure | Section/heading | None | High | Structured documents (HTML, MD) |

### Chunk Size by Document Type

- **FAQs, short docs**: 128-256 tokens, 0-10% overlap
- **Technical docs**: 256-512 tokens, 10-15% overlap
- **Long-form articles**: 512-1024 tokens, 15-20% overlap
- **Legal/regulatory**: 256-512 tokens, 20% overlap
- **Code**: Function/class-level, AST-based splitting

## Retrieval Strategies

| Strategy | Strengths | Weaknesses |
|----------|-----------|------------|
| Dense Retrieval | Semantic understanding | Misses exact terms |
| Sparse (BM25) | Exact keyword matching | No semantic understanding |
| Hybrid (dense+sparse) | Best of both worlds | More complex, tuning needed |
| HyDE | Better query-doc alignment | Extra LLM call, higher latency |
| Multi-Query | Handles ambiguity | Higher latency and cost |
| Reranking | Higher precision in top results | Additional latency |

### Hybrid Search

Combine dense and sparse scores with a weighting parameter alpha:
- **alpha=0.7-0.8**: Default, emphasize semantic matching
- **alpha=0.3-0.5**: Keyword-heavy domains (legal, medical, technical)
- **alpha=0.9**: Conversational queries, natural language questions

Tune alpha on your evaluation set using retrieval metrics (MRR, NDCG).

## Reranking Models

| Model | Type | Quality | Speed | Cost |
|-------|------|---------|-------|------|
| Cohere Rerank v3 | API | Very High | Fast | $1/1K searches |
| BGE-reranker-v2-m3 | Cross-encoder | High | Medium | Free (self-hosted) |
| ms-marco-MiniLM-L-12 | Cross-encoder | Good | Fast | Free (self-hosted) |
| ColBERTv2 | Late interaction | High | Fast (post-indexing) | Free (self-hosted) |

**Pipeline**: Retrieve top 20-50 candidates with fast dense/hybrid search, rerank with cross-encoder, take top 3-5 for the LLM context. Typical improvement: 10-30% in retrieval precision.

## Evaluation Metrics for RAG

### Retrieval Metrics

| Metric | Good Score | Measures |
|--------|------------|----------|
| Hit Rate @k | >0.85 | Basic retrieval success |
| MRR | >0.7 | Ranking quality |
| NDCG @k | >0.7 | Full ranking quality |
| Context Precision | >0.8 | Signal-to-noise in context |
| Context Recall | >0.8 | Completeness of retrieval |

### Generation Metrics

| Metric | Good Score | Measures |
|--------|------------|----------|
| Faithfulness | >0.9 | Answer grounded in retrieved context |
| Answer Relevancy | >0.85 | Answer addresses the question |
| Answer Correctness | >0.8 | Factual accuracy vs ground truth |
| Hallucination Rate | <0.1 | Claims not supported by context |

Build a golden test set of 50-200 question/answer/context triples. Evaluate retrieval and generation separately to diagnose failures. Use RAGAS or DeepEval for automated evaluation.

## Common RAG Failure Modes and Solutions

### Retrieval Failures

| Failure | Root Cause | Solution |
|---------|------------|----------|
| Missed relevant docs | Poor chunking or embedding mismatch | Hybrid search, adjust chunk size, domain-specific embeddings |
| Retrieved irrelevant docs | Overly broad retrieval | Add reranking, use metadata filtering |
| Keyword mismatch | Dense retrieval misses exact terms | Add BM25/hybrid search, query expansion |
| Cross-document reasoning fail | Chunks too isolated | Larger chunks, auto-merging, multi-hop retrieval |

### Generation Failures

| Failure | Root Cause | Solution |
|---------|------------|----------|
| Hallucination | LLM using parametric knowledge | Strengthen grounding instructions, lower temperature |
| Lost in the middle | LLM attention limitations | Put relevant chunks first/last, reduce context length |
| Refusal to answer | Context insufficient or prompt too restrictive | Increase top-k, relax grounding constraints |
| Verbose/unfocused | Too much context, vague prompt | Reduce context, add "be concise" |

### Debugging Checklist

1. **Check retrieval**: Are the right documents being retrieved? Log and inspect top-k results.
2. **Inspect chunks**: Are they semantically coherent and contain complete information?
3. **Test embedding similarity**: If query-to-expected-chunk similarity is low, try a different embedding model.
4. **Evaluate the prompt**: Test generation with perfect context to isolate prompt issues.
5. **Measure end-to-end**: Track metrics over time across chunking, embedding, and prompt changes.

## Further Reading

- [RAG Survey (Gao et al., 2024)](https://arxiv.org/abs/2312.10997)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [LangChain RAG Documentation](https://python.langchain.com/docs/tutorials/rag/)
- [RAGAS: Automated Evaluation of RAG (Es et al., 2023)](https://arxiv.org/abs/2309.15217)
- [Sentence Transformers Library](https://www.sbert.net/)
- [Chroma Documentation](https://docs.trychroma.com/)
- [Pinecone Learning Center](https://www.pinecone.io/learn/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Self-RAG (Asai et al., 2023)](https://arxiv.org/abs/2310.11511)
- [Corrective RAG (Yan et al., 2024)](https://arxiv.org/abs/2401.15884)
- [ColBERT (Khattab & Zaharia, 2020)](https://arxiv.org/abs/2004.12832)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
