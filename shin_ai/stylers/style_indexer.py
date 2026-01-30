from pyrogram import Client
from sentence_transformers import SentenceTransformer
import asyncio
from shin_ai.utils.db import client
from shin_ai.utils.logger_config import logger
from shin_ai.config import API_ID, API_HASH

STYLE_GROUP_ID = -1001894386311  # source group

# Updated to E5 model for consistent embeddings
embedder = SentenceTransformer("intfloat/multilingual-e5-large")

# Force delete existing collection to ensure dimensions are correct
try:
    client.delete_collection("style_group")
    logger.info("🗑️ Deleted old style_group collection to ensure correct dimensions.")
except Exception:
    pass

collection = client.get_or_create_collection("style_group")

async def main():
    app = Client("style_session", api_id=API_ID, api_hash=API_HASH)
    logger.info("🚀 Starting style indexer...")
    async with app:
        logger.info("📥 Fetching messages from style group...")
        ctn = 1
        async for msg in app.get_chat_history(STYLE_GROUP_ID, limit=100000):
            logger.info(f"Processing message {ctn}")
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
