"""One-off Telegram style corpus indexer."""

import asyncio

from pyrogram import Client

from shin_ai.services.embeddings import close_embedding_service, get_embedding_service
from shin_ai.settings import get_settings
from shin_ai.utils.db import get_chroma_client
from shin_ai.utils.logger_config import logger


async def _upsert_batch(collection, ids: list[str], documents: list[str]) -> None:
    embeddings = await get_embedding_service().encode([f"passage: {text}" for text in documents])
    await asyncio.to_thread(
        collection.upsert,
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
    )


async def main() -> None:
    if not get_settings().style_group_id:
        raise ValueError("style_group_id must be configured before running the style indexer")

    collection = get_chroma_client().get_or_create_collection("style_group")
    telegram = Client(
        "style_session",
        api_id=get_settings().platform.telegram_api_id,
        api_hash=get_settings().platform.telegram_api_hash,
    )
    seen_ids: set[str] = set()
    batch_ids: list[str] = []
    batch_documents: list[str] = []

    logger.info("Starting style indexer...")
    try:
        async with telegram:
            async for message in telegram.get_chat_history(get_settings().style_group_id, limit=100_000):
                text = (message.text or "").strip()
                if len(text) < 6:
                    continue
                message_id = str(message.id)
                seen_ids.add(message_id)
                batch_ids.append(message_id)
                batch_documents.append(text)
                if len(batch_ids) >= get_settings().embedding.batch_size:
                    await _upsert_batch(collection, batch_ids, batch_documents)
                    batch_ids, batch_documents = [], []

        if batch_ids:
            await _upsert_batch(collection, batch_ids, batch_documents)

        existing = await asyncio.to_thread(collection.get, include=["metadatas"])
        stale_ids = list(set(existing.get("ids") or []) - seen_ids)
        if stale_ids:
            await asyncio.to_thread(collection.delete, ids=stale_ids)
        logger.info("Style index complete — indexed=%d removed=%d", len(seen_ids), len(stale_ids))
    finally:
        await close_embedding_service()


if __name__ == "__main__":
    asyncio.run(main())
