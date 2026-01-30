from pyrogram import Client
from shin_ai.config import API_ID, API_HASH, BOT_TOKEN

app = Client(
    "shin_ai_bot",
    api_id=API_ID,
    api_hash=API_HASH,
    bot_token=BOT_TOKEN,
    workdir="." # Important to keep session in root or intended location
)
