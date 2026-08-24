import asyncio

from shin_ai.services.embeddings import get_embedding_service
from shin_ai.utils.db import client

# Lazy-initialized to avoid loading the model at import time
_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        _collection = client.get_or_create_collection("style_group")
    return _collection


async def get_style_examples(query: str, k: int = 10) -> list[str]:
    q_emb = (await get_embedding_service().encode(f"query: {query}")).tolist()
    res = await asyncio.to_thread(
        _get_collection().query,
        query_embeddings=[q_emb],
        n_results=k,
    )
    return res["documents"][0]
