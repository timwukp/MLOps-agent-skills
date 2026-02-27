#!/usr/bin/env python3
"""RAG pipeline builder - index documents and query with retrieval-augmented generation.

Usage:
    python build_rag.py index --docs docs/ --db chroma_db/
    python build_rag.py query --db chroma_db/ --question "What is MLOps?"
    python build_rag.py evaluate --db chroma_db/ --eval-data eval.jsonl
"""
import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_documents(docs_dir):
    """Load documents from a directory."""
    docs = []
    for ext in ["*.txt", "*.md", "*.pdf"]:
        for path in Path(docs_dir).rglob(ext):
            try:
                if path.suffix == ".pdf":
                    try:
                        from pypdf import PdfReader
                        reader = PdfReader(str(path))
                        text = "\n".join(page.extract_text() or "" for page in reader.pages)
                    except ImportError:
                        logger.warning(f"pypdf not installed, skipping {path}")
                        continue
                else:
                    text = path.read_text(encoding="utf-8", errors="ignore")

                if text.strip():
                    docs.append({"text": text, "source": str(path), "filename": path.name})
            except Exception as e:
                logger.warning(f"Error loading {path}: {e}")
    logger.info(f"Loaded {len(docs)} documents")
    return docs


def chunk_documents(docs, chunk_size=512, chunk_overlap=50):
    """Split documents into chunks."""
    chunks = []
    for doc in docs:
        text = doc["text"]
        words = text.split()
        for i in range(0, len(words), chunk_size - chunk_overlap):
            chunk_text = " ".join(words[i:i + chunk_size])
            if len(chunk_text.strip()) > 50:
                chunks.append({
                    "text": chunk_text,
                    "source": doc["source"],
                    "chunk_index": len(chunks),
                })
    logger.info(f"Created {len(chunks)} chunks from {len(docs)} documents")
    return chunks


def create_index(chunks, db_path, embedding_model="all-MiniLM-L6-v2"):
    """Create a vector store index from chunks."""
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error("Install: pip install chromadb sentence-transformers")
        sys.exit(1)

    model = SentenceTransformer(embedding_model)
    client = chromadb.PersistentClient(path=str(db_path))

    collection = client.get_or_create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]
        embeddings = model.encode(texts).tolist()
        ids = [f"chunk_{c['chunk_index']}" for c in batch]
        metadatas = [{"source": c["source"]} for c in batch]

        collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        logger.info(f"Indexed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")

    logger.info(f"Index created at {db_path} with {collection.count()} chunks")
    return collection


def query_rag(question, db_path, top_k=5, embedding_model="all-MiniLM-L6-v2"):
    """Query the RAG pipeline."""
    import chromadb
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(embedding_model)
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_collection("documents")

    query_embedding = model.encode(question).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    retrieved_docs = []
    for i, (doc, metadata, distance) in enumerate(zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    )):
        retrieved_docs.append({
            "rank": i + 1,
            "text": doc,
            "source": metadata.get("source", "unknown"),
            "score": round(1 - distance, 4),
        })

    # Build context
    context = "\n\n---\n\n".join(d["text"] for d in retrieved_docs)

    return {
        "question": question,
        "context": context,
        "retrieved_documents": retrieved_docs,
        "num_results": len(retrieved_docs),
    }


def generate_answer(question, context, model_name="gpt-4o-mini"):
    """Generate answer using LLM with retrieved context."""
    try:
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Answer based only on the provided context. If the context doesn't contain enough information, say so."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"},
            ],
            temperature=0,
            max_tokens=500,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.warning(f"LLM generation failed: {e}. Returning context only.")
        return None


def main():
    parser = argparse.ArgumentParser(description="RAG pipeline builder")
    subparsers = parser.add_subparsers(dest="command")

    # Index command
    idx_parser = subparsers.add_parser("index", help="Index documents")
    idx_parser.add_argument("--docs", required=True, help="Documents directory")
    idx_parser.add_argument("--db", required=True, help="Vector DB path")
    idx_parser.add_argument("--chunk-size", type=int, default=512)
    idx_parser.add_argument("--chunk-overlap", type=int, default=50)
    idx_parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")

    # Query command
    q_parser = subparsers.add_parser("query", help="Query the RAG pipeline")
    q_parser.add_argument("--db", required=True, help="Vector DB path")
    q_parser.add_argument("--question", required=True, help="Question to ask")
    q_parser.add_argument("--top-k", type=int, default=5)
    q_parser.add_argument("--generate", action="store_true", help="Generate answer with LLM")
    q_parser.add_argument("--model", default="gpt-4o-mini")

    args = parser.parse_args()

    if args.command == "index":
        docs = load_documents(args.docs)
        chunks = chunk_documents(docs, args.chunk_size, args.chunk_overlap)
        create_index(chunks, args.db, args.embedding_model)

    elif args.command == "query":
        result = query_rag(args.question, args.db, args.top_k)
        print(f"\nQuestion: {result['question']}")
        print(f"Retrieved {result['num_results']} documents\n")

        for doc in result["retrieved_documents"]:
            print(f"  [{doc['rank']}] Score: {doc['score']} | Source: {doc['source']}")
            print(f"      {doc['text'][:200]}...\n")

        if args.generate:
            answer = generate_answer(args.question, result["context"], args.model)
            if answer:
                print(f"\nAnswer: {answer}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
