from typing import Optional, List
import asyncio
import httpx
from pyrogram import Client, enums
from pyrogram.types import Message, ChatPermissions

from shin_ai.platforms.models import UnifiedMessage, UnifiedUser, UnifiedChat, UnifiedMedia, UnifiedMessageEntity
from shin_ai.platforms.base import PlatformAdapter
from shin_ai.config import TELEGRAM_BOT_TOKEN
from shin_ai.utils.logger_config import logger

class TelegramPlatform(PlatformAdapter):
    def __init__(self, client: Client):
        self.client = client

    @property
    def platform_name(self) -> str:
        return "telegram"
        
    @property
    def supports_stickers(self) -> bool:
        return True

    async def get_bot_user(self) -> UnifiedUser:
        me = await self.client.get_me()
        return UnifiedUser(
            id=me.id,
            username=me.username,
            first_name=me.first_name or "",
            is_self=True
        )

    async def start(self) -> None:
        logger.info("Starting Telegram client...")
        await self.client.start()

        me = await self.client.get_me()

        if not getattr(me, "is_bot", False):
            raise RuntimeError(
                "Telegram session is authenticated as a user account, not a bot. "
                "Delete shin_ai_bot.session and restart to re-authenticate with TELEGRAM_BOT_TOKEN."
            )

        expected_bot_id = None
        token = (TELEGRAM_BOT_TOKEN or "").strip()
        if ":" in token:
            token_prefix = token.split(":", 1)[0]
            if token_prefix.isdigit():
                expected_bot_id = int(token_prefix)

        if expected_bot_id is not None and me.id != expected_bot_id:
            raise RuntimeError(
                "Telegram session does not match TELEGRAM_BOT_TOKEN "
                f"(expected bot id {expected_bot_id}, got {me.id}). "
                "Delete shin_ai_bot.session and restart."
            )

        # Ensure polling works even if a webhook was previously configured.
        try:
            if hasattr(self.client, "delete_webhook"):
                await self.client.delete_webhook(drop_pending_updates=False)
                logger.info("Telegram webhook cleared for long polling (pyrogram API).")
            elif TELEGRAM_BOT_TOKEN:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(
                        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/deleteWebhook",
                        data={"drop_pending_updates": "false"},
                    )
                if resp.is_success:
                    logger.info("Telegram webhook cleared for long polling (Bot API fallback).")
                else:
                    logger.warning(f"Telegram Bot API deleteWebhook failed: HTTP {resp.status_code}")
            else:
                logger.warning("Skipping webhook clear because TELEGRAM_BOT_TOKEN is empty.")
        except Exception as e:
            logger.warning(f"Unable to clear Telegram webhook: {e}")

        logger.info(f"Telegram Platform started as @{me.username or 'unknown'} ({me.id}).")

    async def stop(self) -> None:
        await self.client.stop()
        logger.info("Telegram Platform stopped.")

    async def send_message(self, chat_id: int | str, text: str, reply_to_message_id: Optional[int | str] = None) -> int | str:
        if reply_to_message_id:
            msg = await self.client.send_message(int(chat_id), text, reply_to_message_id=int(reply_to_message_id))
        else:
            msg = await self.client.send_message(int(chat_id), text)
        return msg.id

    async def send_sticker(self, chat_id: int | str, sticker_id: str, reply_to_message_id: Optional[int | str] = None) -> int | str:
        if reply_to_message_id:
            msg = await self.client.send_sticker(int(chat_id), sticker_id, reply_to_message_id=int(reply_to_message_id))
        else:
            msg = await self.client.send_sticker(int(chat_id), sticker_id)
        return msg.id

    async def react(self, chat_id: int | str, message_id: int | str, reaction: str) -> None:
        try:
            await self.client.send_reaction(
                chat_id=int(chat_id),
                message_id=int(message_id),
                emoji=reaction,
            )
        except AttributeError:
            # Using msg.react directly if possible on the native msg
            msg = await self.client.get_messages(int(chat_id), int(message_id))
            if msg:
                await msg.react(reaction)

    async def send_chat_action(self, chat_id: int | str, action: str) -> None:
        tg_action = getattr(enums.ChatAction, action.upper(), None)
        if tg_action:
            await self.client.send_chat_action(int(chat_id), tg_action)

    async def download_media(self, media: UnifiedMedia) -> bytes:
        if media.native_obj:
            file_stream = await self.client.download_media(media.native_obj, in_memory=True)
            return file_stream.getvalue()
        return b""

    async def get_message(self, chat_id: int | str, message_id: int | str) -> Optional[UnifiedMessage]:
        try:
            msg = await self.client.get_messages(int(chat_id), int(message_id))
            if msg:
                return self.to_unified_message(msg)
        except Exception as e:
            logger.error(f"Error getting message on Telegram: {e}")
        return None

    async def get_user_by_username(self, username: str) -> Optional[UnifiedUser]:
        try:
            user = await self.client.get_users(username)
            if user:
                return UnifiedUser(
                    id=user.id,
                    username=user.username,
                    first_name=user.first_name or "",
                    is_self=user.is_self
                )
        except Exception:
            return None
        return None

    async def get_chat_member_status(self, chat_id: int | str, user_id: int | str) -> str:
        try:
            mem = await self.client.get_chat_member(int(chat_id), int(user_id))
            return str(mem.status).replace("ChatMemberStatus.", "")
        except Exception:
            return "Unknown"

    async def ban_chat_member(self, chat_id: int | str, user_id: int | str) -> None:
        await self.client.ban_chat_member(int(chat_id), int(user_id))

    async def kick_chat_member(self, chat_id: int | str, user_id: int | str) -> None:
        # Telegram has no native kick - ban then immediately unban
        await self.client.ban_chat_member(int(chat_id), int(user_id))
        await self.client.unban_chat_member(int(chat_id), int(user_id))

    async def unban_chat_member(self, chat_id: int | str, user_id: int | str) -> None:
        await self.client.unban_chat_member(int(chat_id), int(user_id))

    async def restrict_chat_member(self, chat_id: int | str, user_id: int | str, can_send_messages: bool) -> None:
        if can_send_messages:
            permissions = ChatPermissions(
                can_send_messages=True,
                can_send_media_messages=True,
                can_add_web_page_previews=True,
            )
        else:
            permissions = ChatPermissions()
        await self.client.restrict_chat_member(int(chat_id), int(user_id), permissions)

    async def create_chat_invite_link(self, chat_id: int | str) -> str:
        link = await self.client.create_chat_invite_link(int(chat_id), member_limit=1)
        return link.invite_link

    def to_unified_message(self, msg: Message) -> UnifiedMessage:
        chat_type = str(msg.chat.type).replace("ChatType.", "")
        
        chat = UnifiedChat(
            id=msg.chat.id,
            title=msg.chat.title,
            type=chat_type
        )
        
        from_user = None
        if msg.from_user:
            from_user = UnifiedUser(
                id=msg.from_user.id,
                username=msg.from_user.username,
                first_name=msg.from_user.first_name or "",
                is_self=msg.from_user.is_self
            )
            
        unified_msg = UnifiedMessage(
            platform=self.platform_name,
            id=msg.id,
            chat=chat,
            from_user=from_user,
            text=msg.text,
            caption=msg.caption,
            date=msg.date.timestamp() if msg.date else 0.0,
            native_msg=msg
        )
        
        if getattr(msg, "mentioned", False):
            unified_msg.mentioned = True
            
        if msg.reply_to_message_id:
            unified_msg.reply_to_message_id = msg.reply_to_message_id
            if msg.reply_to_message:
                unified_msg.reply_to_message = self.to_unified_message(msg.reply_to_message)
                
        # Handle media
        if msg.photo:
            unified_msg.photo = UnifiedMedia(type="PHOTO", id=msg.photo.file_id, native_obj=msg.photo)
        if msg.sticker:
            unified_msg.sticker = UnifiedMedia(
                type="STICKER", 
                id=msg.sticker.file_id, 
                emoji=msg.sticker.emoji, 
                is_animated=msg.sticker.is_animated,
                is_video=msg.sticker.is_video,
                native_obj=msg.sticker
            )
        if msg.video:
            unified_msg.video = UnifiedMedia(type="VIDEO", id=msg.video.file_id, native_obj=msg.video)
        if msg.animation:
            unified_msg.animation = UnifiedMedia(type="ANIMATION", id=msg.animation.file_id, native_obj=msg.animation)
        if msg.voice:
            unified_msg.voice = UnifiedMedia(type="VOICE", id=msg.voice.file_id, native_obj=msg.voice)
        if msg.audio:
            unified_msg.audio = UnifiedMedia(type="AUDIO", id=msg.audio.file_id, native_obj=msg.audio)
        if msg.document:
            unified_msg.document = UnifiedMedia(type="DOCUMENT", id=msg.document.file_id, native_obj=msg.document)
            
        # Handle entities
        def convert_entities(entities, source):
            res = []
            for e in entities:
                ent = UnifiedMessageEntity(
                    type=str(e.type).replace("MessageEntityType.", ""),
                    offset=e.offset,
                    length=e.length
                )
                if e.user:
                    ent.user = UnifiedUser(id=e.user.id, username=e.user.username, first_name=e.user.first_name or "", is_self=e.user.is_self)
                res.append(ent)
            return res

        if msg.entities:
            unified_msg.entities = convert_entities(msg.entities, msg.text)
        if msg.caption_entities:
            unified_msg.caption_entities = convert_entities(msg.caption_entities, msg.caption)
            
        return unified_msg
