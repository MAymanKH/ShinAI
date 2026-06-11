"""
Memory Lookup Tool

Provides the bot with an explicit tool to query its long-term memory
with fine-grained filters
"""

import asyncio
import json
import numpy as np
from typing import Optional

from shin_ai.stylers.style_retriever import embedder
from shin_ai.utils.logger_config import logger
from shin_ai.utils.memory import memory_collection
from shin_ai.utils.memory_lookup_filters import (
    build_filter_summary,
    build_memory_where_filter,
    sort_memory_results_by_timestamp,
)

# Core lookup function
async def memory_lookup_tool(
    keywords: Optional[str] = None,
    usernames: Optional[list[str]] = None,
    chat_titles: Optional[list[str]] = None,
    platform: Optional[str] = None,
    time_start: Optional[str] = None,
    time_end: Optional[str] = None,
    limit: int = 30,
) -> str:
    """
    Search the bot's long-term memory with optional filters.

    At least one filter parameter must be provided.
    Returns a JSON string with the matching memories or an error object.
    """
    logger.debug(
        "Memory lookup — keywords=%r, usernames=%r, chat_titles=%r, platform=%r, "
        "time_start=%r, time_end=%r, limit=%d",
        keywords, usernames, chat_titles, platform, time_start, time_end, limit,
    )

    # Validation
    if not any([keywords, usernames, chat_titles, platform, time_start]):
        return json.dumps(
            {"error": "At least one filter parameter must be provided (keywords, usernames, chat_titles, platform, or time_start)."},
            ensure_ascii=False,
        )

    # Clamp limit
    limit = max(1, min(limit, 200))

    try:
        where_filter = build_memory_where_filter(
            usernames=usernames,
            chat_titles=chat_titles,
            platform=platform,
            time_start=time_start,
            time_end=time_end,
        )

        # Two paths depending on whether keywords are provided
        if keywords:
            # Path A: metadata filter first via get(), then semantic rank
            results = await _lookup_with_keywords(keywords, where_filter, limit)
        else:
            # Path B: metadata-only lookup
            results = await _lookup_metadata_only(where_filter, limit)

        if not results:
            return json.dumps(
                {"query_filters": build_filter_summary(keywords, usernames, chat_titles, platform, time_start, time_end),
                "results": [],
                "message": "No memories matched the given filters."},
                ensure_ascii=False,
            )

        return json.dumps(
            {"query_filters": build_filter_summary(keywords, usernames, chat_titles, platform, time_start, time_end),
            "count": len(results),
            "results": results},
            ensure_ascii=False,
        )

    except Exception as e:
        logger.error(f"Memory lookup tool failed: {e}", exc_info=True)
        return json.dumps({"error": f"Memory lookup failed: {str(e)}"}, ensure_ascii=False)


def _mmr_indices(
    query_emb: list,
    embs_arr: np.ndarray,
    limit: int,
    lambda_param: float = 0.65,
) -> list[int]:
    """
    MMR selection returning selected indices (not docs), so callers can
    zip docs and metadatas together after selection.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    query_tensor = np.array(query_emb).reshape(1, -1)
    sim_to_query = cosine_similarity(query_tensor, embs_arr)[0]
    cand_sim_matrix = cosine_similarity(embs_arr)

    selected: list[int] = []
    available = list(range(len(embs_arr)))

    while len(selected) < limit and available:
        best_score = -float("inf")
        best_idx = -1
        for idx in available:
            rel = sim_to_query[idx]
            div = max(cand_sim_matrix[idx][s] for s in selected) if selected else 0.0
            score = lambda_param * rel - (1.0 - lambda_param) * div
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx != -1:
            selected.append(best_idx)
            available.remove(best_idx)
        else:
            break
    return selected


async def _lookup_with_keywords(
    keywords: str,
    where_filter: Optional[dict],
    limit: int,
) -> list[dict]:
    """
    First fetch candidates via metadata filters using get(), then
    re-rank those candidates semantically using the E5 embedder.
    If no metadata filter is provided, fall back to a direct semantic query.
    Results are sorted newest-to-oldest.
    """
    if where_filter is not None:
        # Step 1: Get candidates matching metadata filters
        pool_size = min(limit * 5, 500)
        try:
            candidates = memory_collection.get(
                where=where_filter,
                limit=pool_size,
                include=["documents", "embeddings", "metadatas"],
            )
        except Exception as e:
            logger.error("ChromaDB get() failed: %s", e, exc_info=True)
            return []

        docs = candidates.get("documents") or []
        embs = candidates.get("embeddings")
        metas = candidates.get("metadatas") or [{}] * len(docs)
        if embs is None or (hasattr(embs, "__len__") and len(embs) == 0):
            embs = []

        if not docs:
            return []

        # Step 2: Semantic re-rank with E5
        query_emb = await asyncio.to_thread(embedder.encode, f"query: {keywords}")
        query_emb_list = query_emb.tolist()
        query_arr = np.array(query_emb_list).reshape(1, -1)
        embs_arr = np.array(embs)

        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_arr, embs_arr)[0]

        # Filter by generous threshold
        filtered: list[tuple[str, list, dict]] = []
        for doc, emb, meta, sim in zip(docs, embs, metas, similarities):
            if sim > 0.3:
                filtered.append((doc, emb, meta or {}))

        if not filtered:
            # Fall back: all metadata-matched docs, sorted by time
            pairs = list(zip(docs[:limit], metas[:limit]))
            return sort_memory_results_by_timestamp([(d, m or {}) for d, m in pairs])

        f_docs, f_embs, f_metas = zip(*filtered)
        selected_indices = _mmr_indices(query_emb_list, np.array(f_embs), limit)
        selected_pairs = [(f_docs[i], f_metas[i]) for i in selected_indices]
        return sort_memory_results_by_timestamp(selected_pairs)

    else:
        # No metadata filter — pure semantic search
        query_emb = await asyncio.to_thread(embedder.encode, f"query: {keywords}")
        query_emb_list = query_emb.tolist()

        results = memory_collection.query(
            query_embeddings=[query_emb_list],
            n_results=min(limit * 3, 300),
            include=["documents", "distances", "embeddings", "metadatas"],
        )

        docs = results.get("documents", [[]])[0]
        dists = results.get("distances", [[]])[0]
        embs = results.get("embeddings", [[]])[0]
        metas_raw = results.get("metadatas", [[]])[0]

        filtered: list[tuple[str, list, dict]] = []
        for doc, dist, emb, meta in zip(docs, dists, embs, metas_raw):
            if dist < 1.5:
                filtered.append((doc, emb, meta or {}))

        if not filtered:
            return []

        f_docs, f_embs, f_metas = zip(*filtered)
        selected_indices = _mmr_indices(query_emb_list, np.array(f_embs), limit)
        selected_pairs = [(f_docs[i], f_metas[i]) for i in selected_indices]
        return sort_memory_results_by_timestamp(selected_pairs)


async def _lookup_metadata_only(
    where_filter: Optional[dict],
    limit: int,
) -> list[dict]:
    """Retrieve memories using only metadata filters, sorted newest-to-oldest."""
    if where_filter is None:
        return []

    try:
        results = memory_collection.get(
            where=where_filter,
            limit=limit,
            include=["documents", "metadatas"],
        )
        docs = results.get("documents") or []
        metas = results.get("metadatas") or [{}] * len(docs)
        pairs = [(doc, meta or {}) for doc, meta in zip(docs, metas)]
        return sort_memory_results_by_timestamp(pairs)
    except Exception as e:
        logger.error("ChromaDB metadata-only get() failed: %s", e, exc_info=True)
        return []
# Tool schema (OpenAI function-calling format)
# Used by Groq, Cerebras, OpenRouter, and Local LLM providers.

MEMORY_LOOKUP_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "memory_lookup_tool",
        "description": (
            "Search the bot's long-term conversation memory with flexible filters. "
            "Use this tool whenever you need to recall past conversations, look up what "
            "someone said, find discussions from a specific chat or platform, or search "
            "by time range. You can combine any filters together. "
            "At least one parameter must be provided."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "string",
                    "description": (
                        "Free-text keywords for semantic similarity search across memory content. "
                        "Use this to find memories about a specific topic or containing specific phrases."
                    ),
                },
                "usernames": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Filter by usernames or WhatsApp phone numbers. "
                        "Examples: ['ahmed', 'john_doe'] or ['+201234567890']. "
                        "Case-insensitive. Matches the username field of stored memories."
                    ),
                },
                "chat_titles": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Filter by chat or group titles. "
                        "Examples: ['Dev Team', 'Family Group']. "
                        "Must match the exact chat title as stored."
                    ),
                },
                "platform": {
                    "type": "string",
                    "enum": ["telegram", "whatsapp", "discord"],
                    "description": "Filter by messaging platform.",
                },
                "time_start": {
                    "type": "string",
                    "description": (
                        "Start of time range in ISO 8601 format (e.g. '2025-01-15' or '2025-01-15T14:30:00'). "
                        "Only memories from this time onward will be returned."
                    ),
                },
                "time_end": {
                    "type": "string",
                    "description": (
                        "End of time range in ISO 8601 format (e.g. '2025-01-20' or '2025-01-20T23:59:59'). "
                        "Only memories up to this time will be returned."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Defaults to 30. Max 200.",
                },
            },
            "required": [],
        },
    },
}
