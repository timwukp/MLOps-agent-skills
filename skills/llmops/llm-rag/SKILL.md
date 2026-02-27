---
name: llm-rag
description: >
  Build and optimize Retrieval-Augmented Generation (RAG) pipelines. Covers document loading and parsing,
  chunking strategies (fixed-size, semantic, recursive, sentence-based), embedding models (OpenAI, Sentence
  Transformers, Cohere), vector stores (Chroma, Pinecone, Weaviate, Qdrant, pgvector, FAISS), hybrid search
  (dense + sparse, BM25 + embeddings), reranking (Cohere, cross-encoders), query transformation, multi-step
  retrieval, agentic RAG, evaluation (RAGAS), context window optimization, metadata filtering, multi-modal RAG,
  and production RAG architecture. Use when building RAG systems, improving retrieval quality, or deploying
  knowledge-augmented LLM applications.
license: Apache-2.0
metadata:
  author: llmops-skills
  version: "1.0"
  category: llmops
---

# RAG (Retrieval-Augmented Generation)

## Overview

RAG combines retrieval from a knowledge base with LLM generation, enabling accurate,
up-to-date, and grounded responses without fine-tuning.

## RAG Architecture

```
User Query
    │
    ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Query      │     │   Retriever  │     │  Generator  │
│   Transform  │────▶│  (Vector DB) │────▶│  (LLM)      │
└─────────────┘     └──────────────┘     └─────────────┘
                           │                     │
                    Retrieved Chunks      Grounded Response
```

## When to Use This Skill

- Building knowledge-base Q&A systems
- Adding company documents to LLM applications
- Improving LLM accuracy with retrieval
- Optimizing RAG pipeline quality and latency
- Deploying production RAG systems

## Step-by-Step Instructions

### 1. Document Loading and Parsing

```python
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader, UnstructuredHTMLLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load documents
loaders = [
    PyPDFLoader("docs/manual.pdf"),
    TextLoader("docs/faq.txt"),
    CSVLoader("docs/products.csv"),
]

documents = []
for loader in loaders:
    documents.extend(loader.load())

# Add metadata
for doc in documents:
    doc.metadata["source_type"] = doc.metadata.get("source", "").split(".")[-1]
    doc.metadata["ingested_at"] = datetime.utcnow().isoformat()
```

### 2. Chunking Strategies

```python
# Recursive character splitting (default, good for most cases)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)
chunks = splitter.split_documents(documents)

# Semantic chunking (split by meaning changes)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

semantic_splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95,
)
semantic_chunks = semantic_splitter.split_documents(documents)
```

**Chunking Strategy Guide:**

| Strategy | Best For | Chunk Size |
|----------|----------|------------|
| Fixed-size | General purpose | 256-1024 tokens |
| Recursive | Structured documents | 512 tokens |
| Semantic | Dense/technical docs | Variable |
| Sentence | Short-form content | 3-5 sentences |
| Paragraph | Well-structured docs | Natural boundaries |
| Document | Short documents | Full document |

### 3. Embedding and Indexing

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer

# OpenAI embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Open-source embeddings (free, private)
local_embeddings = SentenceTransformer("BAAI/bge-small-en-v1.5")

# Create vector store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_metadata={"hnsw:space": "cosine"},
)

# Similarity search
results = vectorstore.similarity_search_with_score(
    "What is the return policy?",
    k=5,
    filter={"source_type": "pdf"},
)
```

### 4. Hybrid Search (Dense + Sparse)

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# BM25 (sparse/keyword retrieval)
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 5

# Dense retrieval
dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Combine with Reciprocal Rank Fusion
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.4, 0.6],  # Adjust based on your data
)

results = ensemble_retriever.invoke("What is the return policy?")
```

### 5. Reranking

```python
# Cross-encoder reranking (most accurate)
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, documents, top_k=3):
    pairs = [(query, doc.page_content) for doc in documents]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in ranked[:top_k]]

# Retrieve more, rerank to top_k
initial_results = vectorstore.similarity_search(query, k=20)
reranked = rerank(query, initial_results, top_k=5)
```

### 6. Full RAG Pipeline

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# RAG prompt
rag_prompt = ChatPromptTemplate.from_template("""Answer the question based only on the provided context.
If the context doesn't contain enough information, say "I don't have enough information to answer this."

Context:
{context}

Question: {question}

Answer:""")

# Build chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("What is the return policy?")
```

### 7. Query Transformation

```python
def transform_query(original_query, llm):
    """Improve retrieval by transforming the query."""

    # HyDE: Generate hypothetical answer, then search for similar
    hyde_prompt = f"Write a short passage that would answer: {original_query}"
    hypothetical_answer = llm.invoke(hyde_prompt)

    # Multi-query: Generate multiple search queries
    multi_prompt = f"""Generate 3 different search queries for: {original_query}
    Return as a JSON list of strings."""
    queries = json.loads(llm.invoke(multi_prompt))

    # Step-back: Abstract the question
    stepback_prompt = f"What is a more general question that would help answer: {original_query}"
    general_query = llm.invoke(stepback_prompt)

    return {
        "original": original_query,
        "hyde": hypothetical_answer,
        "multi_query": queries,
        "step_back": general_query,
    }
```

## Best Practices

1. **Start with recursive chunking** at 512 tokens, then optimize
2. **Use hybrid search** (BM25 + dense) for best recall
3. **Always rerank** - Cheap quality improvement
4. **Include metadata** in chunks for filtering
5. **Evaluate with RAGAS** - Measure faithfulness, relevancy, precision
6. **Use citation/source tracking** - Users need to verify answers
7. **Handle "I don't know"** - Prompt the LLM to say when context is insufficient
8. **Monitor retrieval quality** separately from generation quality
9. **Update index regularly** - Stale data = wrong answers
10. **Chunk overlap** prevents cutting context at boundaries

## Scripts

- `scripts/build_rag.py` - End-to-end RAG pipeline builder
- `scripts/evaluate_rag.py` - RAG evaluation with RAGAS

## References

See [references/REFERENCE.md](references/REFERENCE.md) for vector store comparisons and optimization guides.
