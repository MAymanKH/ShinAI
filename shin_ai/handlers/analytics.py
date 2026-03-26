import time
from collections import Counter
from pyrogram import filters, Client
from pyrogram.types import Message

from shin_ai.core.client import app
from shin_ai.config import ADMIN_USER_ID
from shin_ai.utils.memory import memory_collection
from shin_ai.utils.logger_config import logger

@app.on_message(filters.command("analytics"))
async def show_analytics(client: Client, msg: Message):
    if msg.from_user.id != ADMIN_USER_ID:
        return

    try:
        # Fetch all metadatas from ChromaDB
        # We can just request metadatas and empty documents to save bandwidth
        # We might have a lot, so we just get all by calling get()
        data = memory_collection.get(include=["metadatas"])
        metadatas = data.get("metadatas", [])
        
        if not metadatas:
            await msg.reply("No interaction data found in database.")
            return
            
        total_interactions = len(metadatas)
        
        user_counts = Counter()
        chat_counts = Counter()
        last_24h_interactions = 0
        current_time = int(time.time())
        
        for meta in metadatas:
            # Process user stats
            user_id = meta.get("user_id", "Unknown")
            username = meta.get("username", "Unknown")
            user_counts[(user_id, username)] += 1
            
            # Process chat stats
            chat_id = meta.get("chat_id", "Unknown")
            chat_title = meta.get("chat_title", "Unknown")
            if chat_id != "Unknown":
                chat_counts[(chat_id, chat_title)] += 1

            # Process recent activity
            try:
                timestamp = int(meta.get("timestamp", 0))
                if current_time - timestamp <= 86400:
                    last_24h_interactions += 1
            except ValueError:
                pass
                
        # Format Top 10 Users
        top_users = user_counts.most_common(10)
        user_text = "\n".join(
            f"• {usr[1]} (`{usr[0]}`): {count} msgs" 
            for (usr, count) in top_users
        )
        
        # Format Top 10 Chats
        top_chats = chat_counts.most_common(10)
        if top_chats:
            chat_text = "\n".join(
                f"• {chat[1]} (`{chat[0]}`): {count} intrx" 
                for (chat, count) in top_chats
            )
        else:
            chat_text = "No grouped chat data available yet."
            
        # Get 5 most recent interactions
        recent_metas = sorted(metadatas, key=lambda x: int(x.get("timestamp", 0)), reverse=True)[:5]
        recent_text = "\n".join(
            f"• {m.get('username', 'Unknown')} in {m.get('chat_title', 'Unknown')} at {m.get('date_string', 'Unknown')}"
            for m in recent_metas
        )

        if not recent_text:
            recent_text = "No recent activity."

        report = (
            f"📊 **Bot Analytics**\n\n"
            f"**Total Interactions:** {total_interactions}\n"
            f"**Last 24 Hours:** {last_24h_interactions}\n\n"
            f"👤 **Top 10 Users:**\n{user_text}\n\n"
            f"💬 **Top 10 Chats/Groups:**\n{chat_text}\n\n"
            f"🕒 **Recent Activity:**\n{recent_text}\n"
        )
        
        await msg.reply(report)
        
    except Exception as e:
        logger.error(f"Error generating analytics: {e}", exc_info=True)
        await msg.reply("An error occurred while generating analytics.")
