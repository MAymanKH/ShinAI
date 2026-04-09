import asyncio
from pyrogram import idle
from shin_ai.bot import app
from shin_ai.utils.logger_config import logger
from shin_ai.services.social import index_social_context
from shin_ai.handlers.discord_chat import discord_platform
from shin_ai.handlers.whatsapp_chat import whatsapp_platform

async def main():
    # Initialize the social context database
    try: 
        index_social_context()
    except Exception as e: 
        logger.error(f"Failed to index social context: {e}")

    logger.info("Starting Telegram Platform...")
    await app.start()
    
    if discord_platform:
        logger.info("Starting Discord Platform...")
        # Since Discord relies on discord.py, discord._get_running_loop() 
        # is okay because we are in an async function running within asyncio loop.
        await discord_platform.start()

    if whatsapp_platform:
        logger.info("Starting WhatsApp Platform...")
        await whatsapp_platform.start()

    logger.info("ShinAI Started Successfully. Listening for messages...")
    
    # Wait until interrupted
    await idle()
    
    logger.info("Stopping platforms...")
    if whatsapp_platform:
        await whatsapp_platform.stop()
    if discord_platform:
        await discord_platform.stop()
    await app.stop()

if __name__ == "__main__":
    app.run(main())
