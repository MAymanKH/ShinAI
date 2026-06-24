from datetime import datetime
from typing import Optional

from shin_ai.utils.logger_config import logger


def parse_iso_to_epoch(iso_str: str) -> Optional[int]:
    """Parse an ISO-8601 datetime string to a Unix epoch integer."""
    try:
        dt = datetime.fromisoformat(iso_str)
        return int(dt.timestamp())
    except ValueError:
        pass

    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y"):
        try:
            dt = datetime.strptime(iso_str, fmt)
            return int(dt.timestamp())
        except ValueError:
            continue

    logger.warning(f"Could not parse time string: {iso_str!r}")
    return None


def build_memory_where_filter(
    usernames: Optional[list[str]] = None,
    chat_titles: Optional[list[str]] = None,
    platform: Optional[str] = None,
    time_start: Optional[str] = None,
    time_end: Optional[str] = None,
) -> Optional[dict]:
    where_clauses: list[dict] = []

    if usernames:
        cleaned = [u.strip().lstrip("@").lower() for u in usernames if u.strip()]
        if cleaned:
            if len(cleaned) == 1:
                where_clauses.append({"username": {"$eq": cleaned[0]}})
            else:
                where_clauses.append({"$or": [{"username": {"$eq": u}} for u in cleaned]})

    if chat_titles:
        cleaned = [t.strip() for t in chat_titles if t.strip()]
        if cleaned:
            if len(cleaned) == 1:
                where_clauses.append({"chat_title": {"$eq": cleaned[0]}})
            else:
                where_clauses.append({"$or": [{"chat_title": {"$eq": t}} for t in cleaned]})

    if platform:
        where_clauses.append({"platform": {"$eq": platform.strip().lower()}})

    start_epoch = parse_iso_to_epoch(time_start) if time_start else None
    end_epoch = parse_iso_to_epoch(time_end) if time_end else None

    if start_epoch is not None:
        where_clauses.append({"timestamp": {"$gte": start_epoch}})
    if end_epoch is not None:
        where_clauses.append({"timestamp": {"$lte": end_epoch}})

    if len(where_clauses) == 1:
        return where_clauses[0]
    if len(where_clauses) > 1:
        return {"$and": where_clauses}
    return None


def sort_memory_results_by_timestamp(pairs: list[tuple[str, dict]]) -> list[dict]:
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
            "timestamp_epoch": meta.get("timestamp", 0),
            "platform": meta.get("platform", "Unknown"),
            "username": meta.get("username", "Unknown"),
            "user_id": meta.get("user_id", "Unknown"),
            "chat_title": meta.get("chat_title", "Unknown"),
            "chat_id": meta.get("chat_id", "Unknown"),
            "text": doc,
        }
        results.append(entry)
    return results


def build_filter_summary(
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
