#!/usr/bin/env python3
"""Semantic caching system for LLM responses - reduce redundant API calls via
exact-match and embedding-based similarity lookup.

Usage:
    python cache_manager.py --action store --db-path cache.db --query "What is MLOps?" --response "MLOps is..."
    python cache_manager.py --action lookup --db-path cache.db --query "Explain MLOps"
    python cache_manager.py --action stats --db-path cache.db
    python cache_manager.py --action warm --db-path cache.db --logs historical.jsonl
    python cache_manager.py --action clear --db-path cache.db
"""
import argparse
import hashlib
import json
import logging
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


EMBED_DIM_FALLBACK = 256


def _stable_bucket(trigram, buckets=EMBED_DIM_FALLBACK):
    """Deterministic bucket index for a trigram.

    Python's builtin hash() is randomized per process (PYTHONHASHSEED), so using
    it here would make every persisted fallback embedding meaningless after a
    restart - the same text would hash into different buckets. blake2b is stable
    across processes and machines.
    """
    digest = hashlib.blake2b(trigram.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") % buckets


def _embed(text, _model={}):
    """Embed text via sentence-transformers; falls back to trigram frequency vector.

    The fallback is a lexical bag-of-trigrams, NOT a semantic embedding: it will
    miss paraphrases that a real model catches. Install sentence-transformers for
    genuine semantic caching.
    """
    try:
        from sentence_transformers import SentenceTransformer
        if "m" not in _model:
            _model["m"] = SentenceTransformer("all-MiniLM-L6-v2")
        return _model["m"].encode(text).tolist()
    except (ImportError, OSError) as exc:
        # OSError covers a failed model download / missing cache, which otherwise
        # crashed instead of degrading to the fallback.
        if "warned" not in _model:
            _model["warned"] = True
            logger.warning(f"sentence-transformers unavailable ({type(exc).__name__}); "
                           f"using lexical trigram fallback - paraphrases will miss.")
        vec = [0.0] * EMBED_DIM_FALLBACK
        for i in range(len(text) - 2):
            vec[_stable_bucket(text[i:i + 3])] += 1.0
        norm = max(sum(v * v for v in vec) ** 0.5, 1e-9)
        return [v / norm for v in vec]


def _cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na > 1e-9 and nb > 1e-9 else 0.0


class SemanticCache:
    """Two-tier cache: exact hash lookup (fast path) then embedding similarity.

    Cache keys include the model and system prompt: the same user question asked
    of a different model, or under a different system prompt, is a different
    question and must not share a cached answer.

    Scaling note: the semantic path is a linear scan over every live entry, which
    is fine to a few thousand rows and unusable beyond that. For production
    volume put the embeddings in a vector index (pgvector, Qdrant, Chroma,
    FAISS) and do an ANN query instead of all_entries().
    """

    def __init__(self, db_path=None, ttl_hours=24, max_size=10000,
                 similarity_threshold=0.90, model="", system_prompt=""):
        self.ttl_seconds = ttl_hours * 3600
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.model = model
        self.system_prompt = system_prompt
        self._backend = _SQLiteBackend(db_path) if db_path else _MemoryBackend()

    def store(self, prompt, response):
        """Store a prompt-response pair with its embedding."""
        key = self._hash(prompt)
        self._backend.put(key, prompt, response, _embed(prompt), time.time(),
                          self._context())
        self._evict_if_needed()
        logger.debug(f"Stored cache entry: {key[:12]}...")

    def lookup(self, query):
        """Look up a cached response. Returns (response, score) or (None, 0.0)."""
        # Fast path: exact hash match
        key = self._hash(query)
        entry = self._backend.get_by_key(key)
        if entry and not self._expired(entry):
            self._backend.touch(key)
            self._backend.bump_stat("hits")
            logger.info("Cache hit (exact match)")
            return entry["response"], 1.0
        # Slow path: semantic similarity search (linear scan - see class docstring).
        # Restricted to the same (model, system_prompt) context as the exact path.
        qemb = _embed(query)
        best_score, best_entry = 0.0, None
        for e in self._backend.all_entries(context=self._context()):
            if self._expired(e):
                continue
            s = _cosine_similarity(qemb, e["embedding"])
            if s > best_score:
                best_score, best_entry = s, e
        if best_entry and best_score >= self.similarity_threshold:
            self._backend.touch(best_entry["key"])
            self._backend.bump_stat("hits")
            logger.info(f"Cache hit (semantic, score={best_score:.4f})")
            return best_entry["response"], round(best_score, 4)
        self._backend.bump_stat("misses")
        logger.info("Cache miss")
        return None, 0.0

    def stats(self):
        """Return cache statistics: size, hits, misses, hit rate.

        Counters live in the backend, so a persistent cache reports cumulative
        numbers across CLI invocations instead of always showing 0.
        """
        hits, misses = self._backend.get_stat("hits"), self._backend.get_stat("misses")
        total = hits + misses
        return {"size": self._backend.size(), "hits": hits,
                "misses": misses,
                "hit_rate": round(hits / total, 4) if total else 0.0,
                "total_lookups": total}

    def warm(self, logs_path):
        """Pre-populate cache from a JSONL file of historical prompt/response pairs."""
        count = 0
        with open(logs_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    prompt = obj.get("prompt", obj.get("query", ""))
                    response = obj.get("response", obj.get("answer", ""))
                    if prompt and response:
                        self.store(prompt, response)
                        count += 1
                except json.JSONDecodeError:
                    continue
        logger.info(f"Cache warmed with {count} entries from {logs_path}")

    def clear(self):
        """Remove all entries and reset stats."""
        self._backend.clear()
        logger.info("Cache cleared")

    def _context(self):
        """Stable id for the (model, system_prompt) pair an entry belongs to."""
        return hashlib.sha256(
            f"{self.model}\x00{self.system_prompt}".encode("utf-8")).hexdigest()[:16]

    def _hash(self, text):
        """Key on (model, system_prompt, prompt) so contexts never collide."""
        material = "\x00".join((self.model, self.system_prompt, text))
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def _expired(self, entry):
        return (time.time() - entry["timestamp"]) > self.ttl_seconds

    def _evict_if_needed(self):
        while self._backend.size() > self.max_size:
            self._backend.evict_oldest()


class _MemoryBackend:
    """In-memory dict backend with LRU eviction order."""
    def __init__(self):
        self._store = {}
        self._order = []
        self._stats = {"hits": 0, "misses": 0}

    def put(self, key, prompt, response, embedding, ts, context=""):
        self._store[key] = {"key": key, "prompt": prompt, "response": response,
                            "embedding": embedding, "timestamp": ts,
                            "context": context}
        if key in self._order:
            self._order.remove(key)
        self._order.append(key)

    def get_by_key(self, key):
        return self._store.get(key)

    def touch(self, key):
        """Mark an entry as most-recently used (called on every hit)."""
        if key in self._store:
            if key in self._order:
                self._order.remove(key)
            self._order.append(key)

    def all_entries(self, context=None):
        if context is None:
            return list(self._store.values())
        return [e for e in self._store.values() if e.get("context") == context]

    def size(self):
        return len(self._store)

    def evict_oldest(self):
        if self._order:
            self._store.pop(self._order.pop(0), None)

    def bump_stat(self, name):
        self._stats[name] = self._stats.get(name, 0) + 1

    def get_stat(self, name):
        return self._stats.get(name, 0)

    def clear(self):
        self._store.clear()
        self._order.clear()
        self._stats = {"hits": 0, "misses": 0}


class _SQLiteBackend:
    """SQLite-backed persistent cache with JSON-serialised embeddings.

    `created_at` drives TTL expiry (a cached answer does not become fresher by
    being read); `last_used` drives LRU eviction and is updated on every hit.
    Keeping them separate is what makes eviction genuinely LRU rather than FIFO.
    """
    def __init__(self, db_path):
        import sqlite3
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache ("
            "  key TEXT PRIMARY KEY, prompt TEXT NOT NULL,"
            "  response TEXT NOT NULL, embedding TEXT NOT NULL,"
            "  created_at REAL NOT NULL, last_used REAL NOT NULL,"
            "  context TEXT NOT NULL DEFAULT '')")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache_stats ("
            "  name TEXT PRIMARY KEY, value INTEGER NOT NULL DEFAULT 0)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_last_used ON cache(last_used)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_context ON cache(context)")
        self._conn.commit()

    def put(self, key, prompt, response, embedding, ts, context=""):
        self._conn.execute(
            "INSERT OR REPLACE INTO cache "
            "(key, prompt, response, embedding, created_at, last_used, context) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (key, prompt, response, json.dumps(embedding), ts, ts, context))
        self._conn.commit()

    def get_by_key(self, key):
        row = self._conn.execute(
            "SELECT * FROM cache WHERE key = ?", (key,)).fetchone()
        return self._to_entry(row) if row else None

    def touch(self, key):
        """Refresh last_used so eviction is LRU, not FIFO. TTL is unaffected."""
        self._conn.execute("UPDATE cache SET last_used = ? WHERE key = ?",
                           (time.time(), key))
        self._conn.commit()

    def all_entries(self, context=None):
        if context is None:
            rows = self._conn.execute("SELECT * FROM cache").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM cache WHERE context = ?", (context,)).fetchall()
        return [self._to_entry(r) for r in rows]

    def size(self):
        return self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]

    def evict_oldest(self):
        self._conn.execute(
            "DELETE FROM cache WHERE key = ("
            "  SELECT key FROM cache ORDER BY last_used ASC LIMIT 1)")
        self._conn.commit()

    def bump_stat(self, name):
        self._conn.execute(
            "INSERT INTO cache_stats(name, value) VALUES (?, 1) "
            "ON CONFLICT(name) DO UPDATE SET value = value + 1", (name,))
        self._conn.commit()

    def get_stat(self, name):
        row = self._conn.execute(
            "SELECT value FROM cache_stats WHERE name = ?", (name,)).fetchone()
        return row["value"] if row else 0

    def clear(self):
        self._conn.execute("DELETE FROM cache")
        self._conn.execute("DELETE FROM cache_stats")
        self._conn.commit()

    @staticmethod
    def _to_entry(row):
        return {"key": row["key"], "prompt": row["prompt"],
                "response": row["response"],
                "embedding": json.loads(row["embedding"]),
                "context": row["context"],
                # TTL uses creation time, not last access.
                "timestamp": row["created_at"]}


def main():
    parser = argparse.ArgumentParser(
        description="Semantic cache manager for LLM responses")
    parser.add_argument("--action", required=True,
                        choices=["lookup", "store", "stats", "warm", "clear"])
    parser.add_argument("--db-path", default=None,
                        help="SQLite database path (omit for in-memory)")
    parser.add_argument("--query", default=None, help="Prompt to look up or store")
    parser.add_argument("--response", default=None, help="Response to cache (store)")
    parser.add_argument("--similarity-threshold", type=float, default=0.90,
                        help="Min cosine similarity for semantic hit (default 0.90)")
    parser.add_argument("--ttl-hours", type=float, default=24,
                        help="Cache entry TTL in hours (default 24)")
    parser.add_argument("--max-size", type=int, default=10000,
                        help="Max entries before LRU eviction")
    parser.add_argument("--logs", default=None,
                        help="JSONL log file for cache warming")
    parser.add_argument("--model", default="",
                        help="Model the entry belongs to (part of the cache key)")
    parser.add_argument("--system-prompt", default="",
                        help="System prompt the entry belongs to (part of the cache key)")
    args = parser.parse_args()

    cache = SemanticCache(db_path=args.db_path, ttl_hours=args.ttl_hours,
                          max_size=args.max_size,
                          similarity_threshold=args.similarity_threshold,
                          model=args.model, system_prompt=args.system_prompt)

    if args.action == "store":
        if not args.query or not args.response:
            logger.error("--query and --response required for store")
            sys.exit(1)
        cache.store(args.query, args.response)
        print(json.dumps({"status": "stored", "query": args.query}))

    elif args.action == "lookup":
        if not args.query:
            logger.error("--query required for lookup")
            sys.exit(1)
        response, score = cache.lookup(args.query)
        print(json.dumps({"hit": response is not None, "score": score,
                          "response": response}, indent=2))

    elif args.action == "stats":
        print(json.dumps(cache.stats(), indent=2))

    elif args.action == "warm":
        if not args.logs:
            logger.error("--logs required for warm")
            sys.exit(1)
        cache.warm(args.logs)
        print(json.dumps(cache.stats(), indent=2))

    elif args.action == "clear":
        cache.clear()
        print(json.dumps({"status": "cleared"}))


if __name__ == "__main__":
    main()
