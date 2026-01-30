"""
Fun Handler Module

Handles fun/greeting patterns like "ثبح" (morning greeting).
"""
from pyrogram import Client, filters
from pyrogram.types import Message
from shin_ai.core.client import app
from shin_ai.handlers.chat import yalbot


@app.on_message(filters.regex("ثبح"))
async def thbh(client: Client, msg: Message):
    """Respond to morning greeting pattern."""
    text = msg.text or msg.caption
    if text and (text.startswith("ثبح ") or text == "ثبح"):
        await msg.reply_text("ثباحو")
    else: await yalbot(client, msg)

@app.on_message(filters.regex("ثباحو"))
async def thbaho(client: Client, msg: Message):
    text = msg.text or msg.caption
    if text and (text.startswith("ثباحو ") or text == "ثباحو"):
        await msg.reply_text("ثبح")
    else: await yalbot(client, msg)

@app.on_message(filters.regex("مثائو"))
async def mthao(client: Client, msg: Message):
    text = msg.text or msg.caption
    if text and (text.startswith("مثائو ") or text == "مثائو"):
        await msg.reply_text("مثا")
    else: await yalbot(client, msg)

@app.on_message(filters.regex("مثا"))
async def mtha(client: Client, msg: Message):
    text = msg.text or msg.caption
    if text and (text.startswith("مثا ") or text == "مثا"):
        await msg.reply_text("مثائو")
    else: await yalbot(client, msg)
