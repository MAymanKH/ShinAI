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
from shin_ai.utils.memory import memory_collection

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


def _sort_by_timestamp(pairs: list[tuple[str, dict]]) -> list[dict]:
    """
    Sort (doc, meta) pairs newest-to-oldest and format each as a result dict
    with all metadata fields visible to the bot.
    """
    def ts(pair):
        return pair[1].get("timestamp", 0) if pair[1] else 0

    sorted_pairs = sorted(pairs, key=ts, reverse=True)
    results = []
    for doc, meta in sorted_pairs:
        entry = {
            "timestamp": meta.get("date_string", meta.get("timestamp", "Unknown")),
            "platform": meta.get("platform", "Unknown"),
            "username": meta.get("username", "Unknown"),
            "user_id": meta.get("user_id", "Unknown"),
            "chat_title": meta.get("chat_title", "Unknown"),
            "chat_id": meta.get("chat_id", "Unknown"),
            "text": doc,
        }
        results.append(entry)
    return results


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
            logger.error(f"ChromaDB get() failed: {e}")
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
            return _sort_by_timestamp([(d, m or {}) for d, m in pairs])

        f_docs, f_embs, f_metas = zip(*filtered)
        selected_indices = _mmr_indices(query_emb_list, np.array(f_embs), limit)
        selected_pairs = [(f_docs[i], f_metas[i]) for i in selected_indices]
        return _sort_by_timestamp(selected_pairs)

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
        return _sort_by_timestamp(selected_pairs)


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
        return _sort_by_timestamp(pairs)
    except Exception as e:
        logger.error(f"ChromaDB metadata-only get() failed: {e}")
        return []

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
                    "description": "Maximum number of results to return. Defaults to 30. Max 200.",
                },
            },
            "required": [],
        },
    },
}
