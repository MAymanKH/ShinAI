import asyncio

from sentence_transformers import SentenceTransformer
from shin_ai.utils.db import client
from shin_ai.config import EMBEDDING_MODEL

# Lazy-initialized to avoid loading the model at import time
_collection = None
_embedder = None


def _get_collection():
    global _collection
    if _collection is None:
        _collection = client.get_or_create_collection("style_group")
    return _collection


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


# Keep the module-level reference for backward compatibility with embedder.encode()
# This is a lazy proxy that loads the model on first attribute access.
class _LazyEmbedder:
    def __getattr__(self, name):
        return getattr(_get_embedder(), name)


embedder = _LazyEmbedder()


def get_style_examples(query, k=10):
    q_emb = embedder.encode(f"query: {query}").tolist()
    res = _get_collection().query(query_embeddings=[q_emb], n_results=k)
    return res["documents"][0]
