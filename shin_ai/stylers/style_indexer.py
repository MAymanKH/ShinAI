from pyrogram import Client
from sentence_transformers import SentenceTransformer
import asyncio
from shin_ai.utils.db import client
from shin_ai.utils.logger_config import logger
from shin_ai.config import TELEGRAM_API_ID, TELEGRAM_API_HASH, STYLE_GROUP_ID, EMBEDDING_MODEL

if not STYLE_GROUP_ID:
    raise ValueError("STYLE_GROUP_ID must be set in .env file. Get your group ID by forwarding a message to @RawDataBot")

# Configurable embedding model
embedder = SentenceTransformer(EMBEDDING_MODEL)

# Force delete existing collection to ensure dimensions are correct
try:
    client.delete_collection("style_group")
    logger.info("🗑️ Deleted old style_group collection to ensure correct dimensions.")
except Exception:
    pass

collection = client.get_or_create_collection("style_group")

async def main():
    app = Client("style_session", api_id=TELEGRAM_API_ID, api_hash=TELEGRAM_API_HASH)
    logger.info("🚀 Starting style indexer...")
    async with app:
        logger.info("📥 Fetching messages from style group...")
        ctn = 1
        async for msg in app.get_chat_history(STYLE_GROUP_ID, limit=100000):
            logger.debug("Indexing style message %d...", ctn)
            ctn += 1
            if not msg.text:
                continue

            text = msg.text.strip()
            if len(text) < 6:
                continue

            # E5 requires "passage: " prefix for stored documents
            collection.add(
                ids=[str(msg.id)],
                documents=[text],
                embeddings=[embedder.encode(f"passage: {text}").tolist()]
            )

    logger.info("✅ Style index completed")

if __name__ == "__main__":
    asyncio.run(main())
