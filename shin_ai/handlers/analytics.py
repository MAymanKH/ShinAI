import time
from collections import Counter
from pyrogram import filters, Client
from pyrogram.types import Message, CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup

from shin_ai.core.client import app
from shin_ai.config import ADMIN_USER_ID
from shin_ai.utils.memory import memory_collection
from shin_ai.utils.logger_config import logger

PAGE_SIZE = 20
MAX_RECENT_ACTIVITY = 100


def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_str(value, default="Unknown"):
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _format_username(username: str) -> str:
    username = _safe_str(username, "Unknown")
    if username == "Unknown":
        return username
    if username.startswith("@"):
        return username
    return f"@{username}"


def _format_platform(platform: str) -> str:
    platform = _safe_str(platform, "Unknown")
    if platform.lower() == "unknown":
        return "Telegram"
    return platform.title()


def _build_chat_labels(metadatas):
    chat_title_by_id = {}
    private_usernames_by_chat = {}

    for meta in metadatas:
        platform = _format_platform(_safe_str(meta.get("platform"), "Unknown"))
        chat_id = _safe_str(meta.get("chat_id"), "Unknown")
        if chat_id == "Unknown":
            continue

        chat_key = (platform, chat_id)

        chat_title = _safe_str(meta.get("chat_title"), "")
        username = _safe_str(meta.get("username"), "Unknown")

        if chat_title:
            chat_title_by_id[chat_key] = chat_title

        if username != "Unknown":
            private_usernames_by_chat.setdefault(chat_key, Counter())
            private_usernames_by_chat[chat_key][username] += 1

    chat_label_by_id = {}
    for chat_key in set(list(chat_title_by_id.keys()) + list(private_usernames_by_chat.keys())):
        chat_title = chat_title_by_id.get(chat_key, "")
        if chat_title:
            chat_label_by_id[chat_key] = chat_title
            continue

        if chat_key in private_usernames_by_chat and private_usernames_by_chat[chat_key]:
            top_username, _ = private_usernames_by_chat[chat_key].most_common(1)[0]
            chat_label_by_id[chat_key] = f"Private with {_format_username(top_username)}"
            continue

        _, chat_id = chat_key
        chat_label_by_id[chat_key] = f"Chat {chat_id}"

    return chat_label_by_id


def _load_analytics_data():
    metadatas = []
    batch_size = 5000
    offset = 0
    while True:
        try:
            batch = memory_collection.get(
                limit=batch_size,
                offset=offset,
                include=["metadatas"]
            )
            batch_ids = batch.get("ids", [])
            if not batch_ids:
                break
            
            batch_metadatas = batch.get("metadatas", [])
            metadatas.extend(batch_metadatas)
            
            if len(batch_ids) < batch_size:
                break
            offset += batch_size
        except Exception as e:
            logger.error(f"Error loading analytics data batch at offset {offset}: {e}")
            break

    if not metadatas:
        return {
            "metadatas": [],
            "user_counts": Counter(),
            "chat_counts": Counter(),
            "chat_label_by_id": {},
            "last_24h_interactions": 0,
            "recent_activities": [],
            "total_interactions": 0,
        }

    current_time = int(time.time())
    user_counts = Counter()
    chat_counts = Counter()
    platform_counts = Counter()
    last_1h_interactions = 0
    last_7d_interactions = 0
    last_30d_interactions = 0
    last_year_interactions = 0
    last_24h_interactions = 0

    for meta in metadatas:
        platform = _format_platform(_safe_str(meta.get("platform"), "Unknown"))
        platform_counts[platform] += 1

        user_id = _safe_str(meta.get("user_id"), "Unknown")
        username = _safe_str(meta.get("username"), "Unknown")
        user_counts[(platform, user_id, username)] += 1

        chat_id = _safe_str(meta.get("chat_id"), "Unknown")
        if chat_id != "Unknown":
            chat_counts[(platform, chat_id)] += 1

        timestamp = _safe_int(meta.get("timestamp", 0), 0)
        if timestamp:
            elapsed = current_time - timestamp
            if elapsed <= 3600:
                last_1h_interactions += 1
            if elapsed <= 86400:
                last_24h_interactions += 1
            if elapsed <= 604800:
                last_7d_interactions += 1
            if elapsed <= 2592000:
                last_30d_interactions += 1
            if elapsed <= 31536000:
                last_year_interactions += 1

    chat_label_by_id = _build_chat_labels(metadatas)
    recent_activities = sorted(
        metadatas,
        key=lambda x: _safe_int(x.get("timestamp", 0), 0),
        reverse=True,
    )

    return {
        "metadatas": metadatas,
        "user_counts": user_counts,
        "chat_counts": chat_counts,
        "chat_label_by_id": chat_label_by_id,
        "platform_counts": platform_counts,
        "last_1h_interactions": last_1h_interactions,
        "last_24h_interactions": last_24h_interactions,
        "last_7d_interactions": last_7d_interactions,
        "last_30d_interactions": last_30d_interactions,
        "last_year_interactions": last_year_interactions,
        "recent_activities": recent_activities,
        "total_interactions": len(metadatas),
    }


def _main_keyboard():
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("All Users", callback_data="analytics:users:0")],
            [InlineKeyboardButton("All Chats/Groups", callback_data="analytics:chats:0")],
            [InlineKeyboardButton("Last 100 Recent Activity", callback_data="analytics:activity:0")],
        ]
    )


def _subview_keyboard(view: str, page: int, total_items: int):
    prev_page = page - 1
    next_page = page + 1

    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("Previous", callback_data=f"analytics:{view}:{prev_page}"),
                InlineKeyboardButton("Next", callback_data=f"analytics:{view}:{next_page}"),
            ],
            [InlineKeyboardButton("Back to Main", callback_data="analytics:main:0")],
        ]
    )


def _view_item_count(analytics, view: str) -> int:
    if view == "users":
        return len(analytics["user_counts"])
    if view == "chats":
        return len(analytics["chat_counts"])
    if view == "activity":
        return len(analytics["recent_activities"][:MAX_RECENT_ACTIVITY])
    return 0


def _main_view_text(analytics):
    top_users = analytics["user_counts"].most_common(10)
    user_text = "\n".join(
        f"• {_format_platform(platform)} | {_format_username(username)} ({user_id}): {count} msgs"
        for (platform, user_id, username), count in top_users
    )
    if not user_text:
        user_text = "No user interactions yet."

    top_chats = analytics["chat_counts"].most_common(10)
    chat_label_by_id = analytics["chat_label_by_id"]
    chat_text = "\n".join(
        f"• {_format_platform(platform)} | {chat_label_by_id.get((platform, chat_id), f'Chat {chat_id}')} ({chat_id}): {count} intrx"
        for (platform, chat_id), count in top_chats
    )
    if not chat_text:
        chat_text = "No grouped chat data available yet."

    top_platforms = analytics["platform_counts"].most_common(5)
    platform_text = "\n".join(
        f"• {_format_platform(platform)}: {count} intrx"
        for platform, count in top_platforms
    )
    if not platform_text:
        platform_text = "No platform activity yet."

    recent_metas = analytics["recent_activities"][:5]
    recent_text = "\n".join(
        f"• {_format_platform(_safe_str(m.get('platform'), 'Unknown'))} | {_format_username(_safe_str(m.get('username'), 'Unknown'))} in {analytics['chat_label_by_id'].get((_format_platform(_safe_str(m.get('platform'), 'Unknown')), _safe_str(m.get('chat_id'), 'Unknown')), _safe_str(m.get('chat_title'), 'Unknown'))} at {_safe_str(m.get('date_string'), 'Unknown')}"
        for m in recent_metas
    )
    if not recent_text:
        recent_text = "No recent activity."

    return (
        f"📊 **Bot Analytics**\n\n"
        f"**Total Interactions:** {analytics['total_interactions']}\n"
        f"**Last Year:** {analytics['last_year_interactions']}\n"
        f"**Last 30 Days:** {analytics['last_30d_interactions']}\n"
        f"**Last 7 Days:** {analytics['last_7d_interactions']}\n"
        f"**Last 24 Hours:** {analytics['last_24h_interactions']}\n"
        f"**Last 1 Hour:** {analytics['last_1h_interactions']}\n\n"
        f"👤 **Top 10 Users:**\n{user_text}\n\n"
        f"💬 **Top 10 Chats/Groups:**\n{chat_text}\n\n"
        f"🧭 **Top Platforms:**\n{platform_text}\n\n"
        f"🕒 **Recent Activity:**\n{recent_text}\n"
    )


def _render_users_view(analytics, requested_page: int):
    rows = sorted(
        analytics["user_counts"].items(),
        key=lambda item: (-item[1], item[0][0], item[0][2], item[0][1]),
    )
    total_items = len(rows)
    max_page = 0 if total_items == 0 else (total_items - 1) // PAGE_SIZE
    page = min(max(requested_page, 0), max_page)
    page_label = f"Page {page + 1}/{max_page + 1}"

    start = page * PAGE_SIZE
    end = start + PAGE_SIZE
    page_rows = rows[start:end]

    if page_rows:
        lines = [
            f"{start + idx + 1}. {_format_platform(platform)} | {_format_username(username)} ({user_id}) - {count} msgs"
            for idx, (((platform, user_id, username), count)) in enumerate(page_rows)
        ]
        body = "\n".join(lines)
    else:
        body = "No users found."

    text = (
        "👤 **All Users**\n"
        f"{page_label}\n"
        f"Showing {start + 1 if total_items else 0}-{min(end, total_items)} of {total_items}\n\n"
        f"{body}"
    )
    keyboard = _subview_keyboard("users", page, total_items)
    return text, keyboard


def _render_chats_view(analytics, requested_page: int):
    chat_label_by_id = analytics["chat_label_by_id"]
    rows = sorted(
        analytics["chat_counts"].items(),
        key=lambda item: (-item[1], chat_label_by_id.get(item[0], f"Chat {item[0][1]}"), item[0][1], item[0][0]),
    )
    total_items = len(rows)
    max_page = 0 if total_items == 0 else (total_items - 1) // PAGE_SIZE
    page = min(max(requested_page, 0), max_page)
    page_label = f"Page {page + 1}/{max_page + 1}"

    start = page * PAGE_SIZE
    end = start + PAGE_SIZE
    page_rows = rows[start:end]

    if page_rows:
        lines = [
            f"{start + idx + 1}. {_format_platform(platform)} | {chat_label_by_id.get((platform, chat_id), f'Chat {chat_id}')} ({chat_id}) - {count} intrx"
            for idx, ((platform, chat_id), count) in enumerate(page_rows)
        ]
        body = "\n".join(lines)
    else:
        body = "No chats/groups found."

    text = (
        "💬 **All Chats/Groups**\n"
        f"{page_label}\n"
        f"Showing {start + 1 if total_items else 0}-{min(end, total_items)} of {total_items}\n\n"
        f"{body}"
    )
    keyboard = _subview_keyboard("chats", page, total_items)
    return text, keyboard


def _render_activity_view(analytics, requested_page: int):
    recent_activities = analytics["recent_activities"][:MAX_RECENT_ACTIVITY]
    total_items = len(recent_activities)
    max_page = 0 if total_items == 0 else (total_items - 1) // PAGE_SIZE
    page = min(max(requested_page, 0), max_page)
    page_label = f"Page {page + 1}/{max_page + 1}"

    start = page * PAGE_SIZE
    end = start + PAGE_SIZE
    page_rows = recent_activities[start:end]

    chat_label_by_id = analytics["chat_label_by_id"]
    if page_rows:
        lines = []
        for idx, meta in enumerate(page_rows):
            platform = _format_platform(_safe_str(meta.get("platform"), "Unknown"))
            username = _format_username(_safe_str(meta.get("username"), "Unknown"))
            chat_id = _safe_str(meta.get("chat_id"), "Unknown")
            chat_label = chat_label_by_id.get((platform, chat_id), _safe_str(meta.get("chat_title"), "Unknown"))
            date_string = _safe_str(meta.get("date_string"), "Unknown")
            lines.append(f"{start + idx + 1}. {platform} | {username} in {chat_label} at {date_string}")
        body = "\n".join(lines)
    else:
        body = "No activity found."

    text = (
        "🕒 **Last 100 Recent Activity**\n"
        f"{page_label}\n"
        f"Showing {start + 1 if total_items else 0}-{min(end, total_items)} of {total_items}\n\n"
        f"{body}"
    )
    keyboard = _subview_keyboard("activity", page, total_items)
    return text, keyboard


def _render_view(view: str, page: int):
    analytics = _load_analytics_data()

    if analytics["total_interactions"] == 0:
        return "No interaction data found in database.", _main_keyboard()

    if view == "main":
        return _main_view_text(analytics), _main_keyboard()
    if view == "users":
        return _render_users_view(analytics, page)
    if view == "chats":
        return _render_chats_view(analytics, page)
    if view == "activity":
        return _render_activity_view(analytics, page)

    return _main_view_text(analytics), _main_keyboard()

@app.on_message(filters.command("shinai_analytics"))
async def show_analytics(client: Client, msg: Message):
    if not msg.from_user:
        return

    if msg.from_user.id != ADMIN_USER_ID:
        return

    try:
        report, keyboard = _render_view("main", 0)
        await msg.reply(report, reply_markup=keyboard)
    except Exception as e:
        logger.error(f"Error generating analytics: {e}", exc_info=True)
        await msg.reply("An error occurred while generating analytics.")


@app.on_callback_query(filters.regex(r"^analytics:"))
async def analytics_callback(client: Client, callback: CallbackQuery):
    if not callback.from_user:
        return

    if callback.from_user.id != ADMIN_USER_ID:
        return

    data = callback.data or ""
    parts = data.split(":")

    view = "main"
    page = 0

    if len(parts) == 3:
        view = parts[1]
        page = _safe_int(parts[2], 0)

    if view not in {"main", "users", "chats", "activity"}:
        view = "main"
        page = 0

    try:
        if view in {"users", "chats", "activity"}:
            analytics = _load_analytics_data()
            total_items = _view_item_count(analytics, view)
            max_page = 0 if total_items == 0 else (total_items - 1) // PAGE_SIZE
            if page < 0:
                page = 0
                await callback.answer("You are already on the first page.")
            elif page > max_page:
                page = max_page
                await callback.answer("No more results to show.")

        text, keyboard = _render_view(view, page)
        if not callback.message:
            await callback.answer("Unable to update this analytics message.", show_alert=True)
            return
        await callback.message.edit_text(text, reply_markup=keyboard)
        await callback.answer()
    except Exception as e:
        logger.error(f"Error processing analytics callback: {e}", exc_info=True)
        await callback.answer("Failed to load analytics view.", show_alert=True)
