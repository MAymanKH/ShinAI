import asyncio
import time
import uuid
from datetime import datetime

from shin_ai.config import MEMORY_MAX_DISTANCE
from shin_ai.services.embeddings import get_embedding_service
from shin_ai.utils.db import get_chroma_client
from shin_ai.utils.logger_config import logger
from shin_ai.utils.memory_time import detect_time_filter
from shin_ai.utils.similarity import select_mmr_indices_async, within_distance

# Lazy-initialized to avoid import-time side effects
_memory_collection = None


def _get_memory_collection():
    """Return the chat memories collection, creating it on first use."""
    global _memory_collection
    if _memory_collection is None:
        _memory_collection = get_chroma_client().get_or_create_collection("chat_memories")
    return _memory_collection


# Memory Storage
async def save_memory(
    platform: str,
    user_id: int | str,
    username: str,
    prompt: str,
    response: str,
    context: str = "",
    chat_id: int | str = 0,
    chat_title: str = "",
):
    """
    Saves a user-bot interaction to the vector database.
    """
    try:
        if not response or not prompt:
            return

        # Get formatted timestamp
        now_str = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

        # Format the memory text.
        if context:
            # If there is context (previous reply), include it so the memory stands on its own
            memory_text = f"Context: {context}\nUser ({username}) said: {prompt}\nBot replied: {response}"
        else:
            memory_text = f"User ({username}) said: {prompt}\nBot replied: {response}"

        # Clean up reaction responses for better reading in future
        if response.startswith("react:"):
            reaction = response.split(":")[1]
            memory_text = f"User ({username}) said: {prompt}\nBot reacted with: {reaction}"
        elif response.startswith("sticker:"):
            memory_text = f"User ({username}) said: {prompt}\nBot sent a sticker."

        # Add timestamp and chat title to the readable memory text
        chat_prefix = (
            f" [Chat: {chat_title} on {platform.title()}]"
            if chat_title
            else f" [Platform: {platform.title()}]"
        )
        memory_text = f"[{now_str}]{chat_prefix}\n{memory_text}"

        # Metadata for filtering/context
        meta = {
            "platform": platform,
            "user_id": str(user_id),
            "username": username or "Unknown",
            "timestamp": int(time.time()),
            "date_string": now_str,
            "type": "conversation",
        }
        if chat_id:
            meta["chat_id"] = str(chat_id)
        if chat_title:
            meta["chat_title"] = chat_title

        # Unique Memory ID
        mem_id = str(uuid.uuid4())

        # Create embedding.
        # Only the interaction itself is embedded. The timestamp/chat header is
        # identical across every memory, so including it pulls all documents
        # toward each other and measurably flattens retrieval separation --
        # time and chat are filterable through metadata instead.
        # E5 requires the "passage: " prefix for stored documents.
        searchable_text = f"passage: User ({username}) said: {prompt}\nBot replied: {response}"
        # Off-thread to avoid blocking event loop
        embedding_tensor = await get_embedding_service().encode(searchable_text)
        embedding = embedding_tensor.tolist()

        await asyncio.to_thread(
            _get_memory_collection().add,
            ids=[mem_id],
            documents=[memory_text],
            embeddings=[embedding],
            metadatas=[meta],
        )
        logger.debug("Memory saved for user %s (chat=%s platform=%s)", username, chat_id, platform)
    except Exception as e:
        logger.error("Failed to save memory for user %s: %s", username, e, exc_info=True)


# Memory Retrieval


def _filter_by_relevance(results: dict, max_cosine_distance: float) -> tuple[list, list]:
    """Keep only candidates whose distance passes the relevance gate."""
    if not results.get("documents"):
        return [], []

    documents = results["documents"][0]
    distances = results["distances"][0]
    embeddings = results["embeddings"][0]

    kept_documents = []
    kept_embeddings = []
    for document, distance, embedding in zip(documents, distances, embeddings, strict=False):
        if within_distance(distance, max_cosine_distance):
            kept_documents.append(document)
            kept_embeddings.append(embedding)

    if len(documents) and not kept_documents:
        logger.debug(
            "All %d memory candidates failed the relevance gate (max_distance=%.3f)",
            len(documents),
            max_cosine_distance,
        )
    return kept_documents, kept_embeddings


async def _rank_diverse(query_embedding: list, documents: list, embeddings: list, limit: int) -> list:
    """Re-rank relevant candidates for diversity, off the event loop."""
    if not documents:
        return []
    indices = await select_mmr_indices_async(query_embedding, embeddings, limit)
    return [documents[index] for index in indices]


async def retrieve_memories(query: str, limit: int = 15):
    """
    Retrieves semantically relevant past interactions.
    If the query contains a time reference (e.g. "2 days ago", "قبل ساعة"),
    results are constrained to that time window via ChromaDB metadata filtering.
    """
    try:
        # E5 requires "query: " prefix for search queries, off-thread
        query_emb_tensor = await get_embedding_service().encode(f"query: {query}")
        query_emb = query_emb_tensor.tolist()

        # Check for time references in the query
        start_epoch, end_epoch = await detect_time_filter(query)

        where_filter = None
        if start_epoch is not None and end_epoch is not None:
            where_filter = {
                "$and": [
                    {"timestamp": {"$gte": start_epoch}},
                    {"timestamp": {"$lte": end_epoch}},
                ]
            }
            logger.debug("Time-filtered memory search: %s → %s", start_epoch, end_epoch)

        # Fetch a large pool for MMR deduplication
        results = await asyncio.to_thread(
            _get_memory_collection().query,
            query_embeddings=[query_emb],
            n_results=40,
            where=where_filter,
            include=["documents", "distances", "embeddings"],
        )

        # A time-filtered search has already narrowed candidates by metadata, so
        # relevance can be graded a little more loosely there.
        max_distance = MEMORY_MAX_DISTANCE * (1.25 if where_filter else 1.0)
        filtered_docs, filtered_embs = _filter_by_relevance(results, max_distance)

        final_memories = await _rank_diverse(query_emb, filtered_docs, filtered_embs, limit)

        # If time filter was applied but returned nothing, fall back to unfiltered
        if where_filter and not final_memories:
            logger.debug("Time-filtered search returned no results — falling back to unfiltered")
            return await _retrieve_memories_unfiltered(query_emb, limit)

        return final_memories
    except Exception as e:
        logger.error("Failed to retrieve memories: %s", e, exc_info=True)
        return []


async def _retrieve_memories_unfiltered(query_emb: list, limit: int = 15):
    """
    Fallback: pure semantic retrieval without any time filter.
    Accepts a pre-computed embedding to avoid re-encoding.
    """
    try:
        results = await asyncio.to_thread(
            _get_memory_collection().query,
            query_embeddings=[query_emb],
            n_results=40,
            include=["documents", "distances", "embeddings"],
        )
        filtered_docs, filtered_embs = _filter_by_relevance(results, MEMORY_MAX_DISTANCE)
        return await _rank_diverse(query_emb, filtered_docs, filtered_embs, limit)
    except Exception as e:
        logger.error("Failed to retrieve memories (unfiltered fallback): %s", e, exc_info=True)
        return []
