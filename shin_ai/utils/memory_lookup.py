"""
Memory Lookup Tool

Provides the bot with an explicit tool to query its long-term memory
with fine-grained filters
"""

import asyncio
import json
import numpy as np
from datetime import datetime
from typing import Optional

from shin_ai.utils.db import client
from shin_ai.stylers.style_retriever import embedder
from shin_ai.utils.logger_config import logger
from shin_ai.utils.memory import memory_collection, _apply_mmr

# Core lookup function
async def memory_lookup_tool(
    keywords: Optional[str] = None,
    usernames: Optional[list[str]] = None,
    chat_titles: Optional[list[str]] = None,
    platform: Optional[str] = None,
    time_start: Optional[str] = None,
    time_end: Optional[str] = None,
    limit: int = 100,
) -> str:
    """
    Search the bot's long-term memory with optional filters.

    At least one filter parameter must be provided.
    Returns a JSON string with the matching memories or an error object.
    """
    logger.info(
        f"Memory lookup called — keywords={keywords!r}, usernames={usernames!r}, "
        f"chat_titles={chat_titles!r}, platform={platform!r}, "
        f"time_start={time_start!r}, time_end={time_end!r}, limit={limit}"
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
        # Build ChromaDB metadata filter (WHERE clause)
        where_clauses: list[dict] = []

        # usernames (case-insensitive, OR across list)
        if usernames:
            cleaned = [u.strip().lstrip("@").lower() for u in usernames if u.strip()]
            if cleaned:
                if len(cleaned) == 1:
                    where_clauses.append({"username": {"$eq": cleaned[0]}})
                else:
                    where_clauses.append({"$or": [{"username": {"$eq": u}} for u in cleaned]})

        # chat_titles (case-insensitive substring would be ideal,
        # but ChromaDB only supports exact match on metadata.
        # We do exact match; callers should use short distinctive titles.)
        if chat_titles:
            cleaned = [t.strip() for t in chat_titles if t.strip()]
            if cleaned:
                if len(cleaned) == 1:
                    where_clauses.append({"chat_title": {"$eq": cleaned[0]}})
                else:
                    where_clauses.append({"$or": [{"chat_title": {"$eq": t}} for t in cleaned]})

        # platform
        if platform:
            where_clauses.append({"platform": {"$eq": platform.strip().lower()}})

        # time range
        start_epoch = _parse_iso_to_epoch(time_start) if time_start else None
        end_epoch = _parse_iso_to_epoch(time_end) if time_end else None

        if start_epoch is not None:
            where_clauses.append({"timestamp": {"$gte": start_epoch}})
        if end_epoch is not None:
            where_clauses.append({"timestamp": {"$lte": end_epoch}})

        # Combine all clauses
        where_filter: Optional[dict] = None
        if len(where_clauses) == 1:
            where_filter = where_clauses[0]
        elif len(where_clauses) > 1:
            where_filter = {"$and": where_clauses}

        # Two paths depending on whether keywords are provided
        if keywords:
            # Path A: metadata filter first via get(), then semantic rank
            results = await _lookup_with_keywords(keywords, where_filter, limit)
        else:
            # Path B: metadata-only lookup
            results = await _lookup_metadata_only(where_filter, limit)

        if not results:
            return json.dumps(
                {"query_filters": _build_filter_summary(keywords, usernames, chat_titles, platform, time_start, time_end),
                "results": [],
                "message": "No memories matched the given filters."},
                ensure_ascii=False,
            )

        return json.dumps(
            {"query_filters": _build_filter_summary(keywords, usernames, chat_titles, platform, time_start, time_end),
            "count": len(results),
            "results": results},
            ensure_ascii=False,
        )

    except Exception as e:
        logger.error(f"Memory lookup tool failed: {e}", exc_info=True)
        return json.dumps({"error": f"Memory lookup failed: {str(e)}"}, ensure_ascii=False)


# Internal helpers

def _parse_iso_to_epoch(iso_str: str) -> Optional[int]:
    """Parse an ISO-8601 datetime string to a Unix epoch integer."""
    try:
        # Try full ISO with timezone
        dt = datetime.fromisoformat(iso_str)
        return int(dt.timestamp())
    except ValueError:
        pass
    # Try common date-only format
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y"):
        try:
            dt = datetime.strptime(iso_str, fmt)
            return int(dt.timestamp())
        except ValueError:
            continue
    logger.warning(f"Could not parse time string: {iso_str!r}")
    return None


async def _lookup_with_keywords(
    keywords: str,
    where_filter: Optional[dict],
    limit: int,
) -> list[str]:
    """
    First fetch candidates via metadata filters using get(), then
    re-rank those candidates semantically using the E5 embedder.
    If no metadata filter is provided, fall back to a direct semantic query.
    """
    if where_filter is not None:
        # Step 1: Get candidates matching metadata filters
        # Fetch a generous pool for semantic re-ranking
        pool_size = min(limit * 5, 500)
        try:
            candidates = memory_collection.get(
                where=where_filter,
                limit=pool_size,
                include=["documents", "embeddings"],
            )
        except Exception as e:
            logger.error(f"ChromaDB get() failed: {e}")
            return []

        docs = candidates.get("documents")
        embs = candidates.get("embeddings")
        if docs is None:
            docs = []
        if embs is None:
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

        # Filter by a generous threshold
        filtered_docs = []
        filtered_embs = []
        for doc, emb, sim in zip(docs, embs, similarities):
            if sim > 0.3:  # lenient — MMR will handle diversity
                filtered_docs.append(doc)
                filtered_embs.append(emb)

        if not filtered_docs:
            # Fall back to returning all metadata-matched docs (unsorted)
            return docs[:limit]

        return _apply_mmr(query_emb_list, filtered_docs, filtered_embs, limit)

    else:
        # No metadata filter — pure semantic search (same as retrieve_memories)
        query_emb = await asyncio.to_thread(embedder.encode, f"query: {keywords}")
        query_emb_list = query_emb.tolist()

        results = memory_collection.query(
            query_embeddings=[query_emb_list],
            n_results=min(limit * 3, 300),
            include=["documents", "distances", "embeddings"],
        )

        docs = results.get("documents", [[]])[0]
        dists = results.get("distances", [[]])[0]
        embs = results.get("embeddings", [[]])[0]

        filtered_docs = []
        filtered_embs = []
        for doc, dist, emb in zip(docs, dists, embs):
            if dist < 1.5:
                filtered_docs.append(doc)
                filtered_embs.append(emb)

        if not filtered_docs:
            return []

        return _apply_mmr(query_emb_list, filtered_docs, filtered_embs, limit)


async def _lookup_metadata_only(
    where_filter: Optional[dict],
    limit: int,
) -> list[str]:
    """Retrieve memories using only metadata filters (no semantic search)."""
    if where_filter is None:
        return []

    try:
        results = memory_collection.get(
            where=where_filter,
            limit=limit,
            include=["documents"],
        )
        docs = results.get("documents")
        return docs if docs is not None else []
    except Exception as e:
        logger.error(f"ChromaDB metadata-only get() failed: {e}")
        return []


def _build_filter_summary(
    keywords: Optional[str],
    usernames: Optional[list[str]],
    chat_titles: Optional[list[str]],
    platform: Optional[str],
    time_start: Optional[str],
    time_end: Optional[str],
) -> dict:
    """Build a human-readable summary of the applied filters."""
    summary = {}
    if keywords:
        summary["keywords"] = keywords
    if usernames:
        summary["usernames"] = usernames
    if chat_titles:
        summary["chat_titles"] = chat_titles
    if platform:
        summary["platform"] = platform
    if time_start:
        summary["time_start"] = time_start
    if time_end:
        summary["time_end"] = time_end
    return summary


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
                    "description": "Maximum number of results to return. Defaults to 100. Max 200.",
                },
            },
            "required": [],
        },
    },
}
