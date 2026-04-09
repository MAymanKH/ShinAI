from __future__ import annotations

import asyncio
from collections import OrderedDict
from threading import RLock
from typing import Optional

from neonize import NewClient
from neonize.proto import Neonize_pb2 as neonize_proto
from neonize.proto.Neonize_pb2 import JID, Message as MessageEvent
from neonize.proto.waE2E.WAWebProtobufsE2E_pb2 import ContextInfo, Message as WaMessage
from neonize.utils import Jid2String, build_jid
from neonize.utils.enum import ChatPresence, ChatPresenceMedia, ParticipantChange

from shin_ai.platforms.base import PlatformAdapter
from shin_ai.platforms.models import (
    UnifiedChat,
    UnifiedMedia,
    UnifiedMessage,
    UnifiedMessageEntity,
    UnifiedUser,
)
from shin_ai.utils.logger_config import logger


class WhatsAppPlatform(PlatformAdapter):
    def __init__(self, session_name: str):
        self.client = NewClient(session_name)
        self._session_name = session_name
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._connect_task: Optional[asyncio.Task] = None
        self._bot_user_cache: Optional[UnifiedUser] = None
        self._cache_lock = RLock()
        self._raw_message_cache: OrderedDict[tuple[str, str], MessageEvent] = OrderedDict()
        self._unified_message_cache: OrderedDict[tuple[str, str], UnifiedMessage] = OrderedDict()
        self._cache_limit = 2000

    @property
    def platform_name(self) -> str:
        return "whatsapp"

    @property
    def supports_stickers(self) -> bool:
        # Stickers in this bot rely on Telegram sticker IDs, so we intentionally disable
        # native sticker actions for WhatsApp to avoid invalid media sends.
        return False

    @property
    def event_loop(self) -> Optional[asyncio.AbstractEventLoop]:
        return self._loop

    async def _run_sync(self, func, *args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)

    def _trim_cache_if_needed(self) -> None:
        with self._cache_lock:
            while len(self._raw_message_cache) > self._cache_limit:
                self._raw_message_cache.popitem(last=False)
            while len(self._unified_message_cache) > self._cache_limit:
                self._unified_message_cache.popitem(last=False)

    def _jid_to_user_id(self, jid: JID) -> str:
        if jid.User:
            return jid.User
        return Jid2String(jid)

    def _jid_to_username(self, jid: JID) -> str:
        if jid.User:
            return jid.User
        return Jid2String(jid)

    def _chat_id_to_jid(self, chat_id: int | str) -> JID:
        chat = str(chat_id)
        if "@" in chat:
            user, server = chat.split("@", 1)
            return build_jid(user, server)
        if "-" in chat:
            return build_jid(chat, "g.us")
        return build_jid(chat, "s.whatsapp.net")

    def _user_id_to_jid(self, user_id: int | str) -> JID:
        user = str(user_id)
        if "@" in user:
            raw_user, server = user.split("@", 1)
            return build_jid(raw_user, server)
        return build_jid(user, "s.whatsapp.net")

    def _unwrap_message(self, message: WaMessage) -> WaMessage:
        current = message

        while True:
            advanced = False

            if current.deviceSentMessage.ListFields() and current.deviceSentMessage.message.ListFields():
                current = current.deviceSentMessage.message
                advanced = True

            for wrapper_name in (
                "ephemeralMessage",
                "viewOnceMessage",
                "viewOnceMessageV2",
                "viewOnceMessageV2Extension",
                "editedMessage",
            ):
                wrapper = getattr(current, wrapper_name, None)
                if wrapper and wrapper.ListFields() and getattr(wrapper, "message", None):
                    inner = wrapper.message
                    if inner and inner.ListFields():
                        current = inner
                        advanced = True
                        break

            if not advanced:
                break

        return current

    def _extract_context_info(self, message: WaMessage) -> Optional[ContextInfo]:
        for field_name in (
            "extendedTextMessage",
            "imageMessage",
            "videoMessage",
            "audioMessage",
            "documentMessage",
            "stickerMessage",
        ):
            payload = getattr(message, field_name, None)
            if payload and payload.ListFields() and hasattr(payload, "contextInfo"):
                context_info = payload.contextInfo
                if context_info and context_info.ListFields():
                    return context_info
        return None

    def _extract_text_and_caption(self, message: WaMessage) -> tuple[Optional[str], Optional[str]]:
        text: Optional[str] = None
        caption: Optional[str] = None

        if message.conversation:
            text = message.conversation
        elif message.extendedTextMessage.ListFields():
            text = message.extendedTextMessage.text or None

        if message.imageMessage.ListFields():
            caption = message.imageMessage.caption or None
        elif message.videoMessage.ListFields():
            caption = message.videoMessage.caption or None
        elif message.documentMessage.ListFields():
            caption = message.documentMessage.caption or None

        return text, caption

    def _apply_media(self, unified_msg: UnifiedMessage, message: WaMessage) -> None:
        native_payload = {"wa_message": message}

        if message.imageMessage.ListFields():
            unified_msg.photo = UnifiedMedia(type="PHOTO", id="wa-image", native_obj=native_payload)
        if message.stickerMessage.ListFields():
            unified_msg.sticker = UnifiedMedia(
                type="STICKER",
                id="wa-sticker",
                is_animated=bool(message.stickerMessage.isAnimated),
                native_obj=native_payload,
            )
        if message.videoMessage.ListFields():
            unified_msg.video = UnifiedMedia(type="VIDEO", id="wa-video", native_obj=native_payload)
        if message.audioMessage.ListFields():
            if bool(message.audioMessage.PTT):
                unified_msg.voice = UnifiedMedia(type="VOICE", id="wa-voice", native_obj=native_payload)
            else:
                unified_msg.audio = UnifiedMedia(type="AUDIO", id="wa-audio", native_obj=native_payload)
        if message.documentMessage.ListFields():
            unified_msg.document = UnifiedMedia(type="DOCUMENT", id="wa-document", native_obj=native_payload)

    def _build_quoted_message(self, context_info: ContextInfo, chat: UnifiedChat) -> Optional[UnifiedMessage]:
        if not context_info.stanzaID:
            return None
        if not context_info.quotedMessage.ListFields():
            return None

        participant = context_info.participant or ""
        participant_user = participant.split("@", 1)[0] if participant else "unknown"

        quoted_user = UnifiedUser(
            id=participant_user,
            username=participant_user if participant_user != "unknown" else None,
            first_name=participant_user,
            is_self=False,
        )

        quoted_body = self._unwrap_message(context_info.quotedMessage)
        quoted_text, quoted_caption = self._extract_text_and_caption(quoted_body)

        quoted = UnifiedMessage(
            platform=self.platform_name,
            id=context_info.stanzaID,
            chat=chat,
            from_user=quoted_user,
            text=quoted_text,
            caption=quoted_caption,
            date=0.0,
            native_msg=context_info.quotedMessage,
        )
        self._apply_media(quoted, quoted_body)
        return quoted

    def _build_entities_from_context(self, context_info: Optional[ContextInfo], source_text: str) -> list[UnifiedMessageEntity]:
        entities: list[UnifiedMessageEntity] = []
        if not context_info:
            return entities

        for mentioned_jid in context_info.mentionedJID:
            user_id = mentioned_jid.split("@", 1)[0]
            token = f"@{user_id}"
            offset = source_text.find(token)
            length = len(token) if offset >= 0 else 0
            entities.append(
                UnifiedMessageEntity(
                    type="MENTION",
                    offset=max(offset, 0),
                    length=length,
                    user=UnifiedUser(
                        id=user_id,
                        username=user_id,
                        first_name=user_id,
                        is_self=False,
                    ),
                )
            )
        return entities

    def _cache_message(self, unified: UnifiedMessage, event_msg: MessageEvent) -> None:
        cache_key = (str(unified.chat.id), str(unified.id))
        with self._cache_lock:
            self._raw_message_cache[cache_key] = event_msg
            self._unified_message_cache[cache_key] = unified
            self._raw_message_cache.move_to_end(cache_key)
            self._unified_message_cache.move_to_end(cache_key)
        self._trim_cache_if_needed()

    def get_cached_raw_message(self, chat_id: int | str, message_id: int | str) -> Optional[MessageEvent]:
        cache_key = (str(chat_id), str(message_id))
        with self._cache_lock:
            return self._raw_message_cache.get(cache_key)

    def ingest_event_message(self, event_msg: MessageEvent) -> UnifiedMessage:
        unified = self.to_unified_message(event_msg)
        self._cache_message(unified, event_msg)
        return unified

    def to_unified_message(self, event_msg: MessageEvent) -> UnifiedMessage:
        source = event_msg.Info.MessageSource
        chat_jid = source.Chat
        sender_jid = source.Sender
        if source.SenderAlt.ListFields():
            sender_jid = source.SenderAlt
        if source.IsFromMe and self.client.me and self.client.me.JID.ListFields():
            sender_jid = self.client.me.JID

        chat_id = Jid2String(chat_jid)
        chat_type = "GROUP" if bool(source.IsGroup) else "PRIVATE"

        chat = UnifiedChat(
            id=chat_id,
            title=None,
            type=chat_type,
        )

        sender_id = self._jid_to_user_id(sender_jid)
        sender_username = self._jid_to_username(sender_jid)
        sender_name = event_msg.Info.Pushname or sender_username or sender_id

        from_user = UnifiedUser(
            id=sender_id,
            username=sender_username,
            first_name=sender_name,
            is_self=bool(source.IsFromMe),
        )

        body = self._unwrap_message(event_msg.Message)
        text, caption = self._extract_text_and_caption(body)

        unified_msg = UnifiedMessage(
            platform=self.platform_name,
            id=event_msg.Info.ID,
            chat=chat,
            from_user=from_user,
            text=text,
            caption=caption,
            date=float(event_msg.Info.Timestamp or 0),
            native_msg=event_msg,
        )

        self._apply_media(unified_msg, body)

        context_info = self._extract_context_info(body)
        source_text = (text or caption or "")

        if context_info and context_info.stanzaID:
            unified_msg.reply_to_message_id = context_info.stanzaID
            unified_msg.reply_to_message = self._build_quoted_message(context_info, chat)

        if context_info:
            unified_msg.entities = self._build_entities_from_context(context_info, source_text)
            if self.client.me and self.client.me.JID.ListFields():
                my_jid = Jid2String(self.client.me.JID)
                if any(jid == my_jid for jid in context_info.mentionedJID):
                    unified_msg.mentioned = True

        return unified_msg

    async def get_bot_user(self) -> UnifiedUser:
        if self._bot_user_cache:
            return self._bot_user_cache

        me = self.client.me
        if me and me.JID.ListFields():
            user_id = self._jid_to_user_id(me.JID)
            username = self._jid_to_username(me.JID)
            first_name = me.PushName or username or user_id
            self._bot_user_cache = UnifiedUser(
                id=user_id,
                username=username,
                first_name=first_name,
                is_self=True,
            )
            return self._bot_user_cache

        # The WhatsApp session may still be initializing; return a safe fallback.
        return UnifiedUser(id="self", username="self", first_name="ShinAI", is_self=True)

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._connect_task = asyncio.create_task(asyncio.to_thread(self.client.connect))
        await asyncio.sleep(0.1)

        if self._connect_task.done() and self._connect_task.exception():
            raise self._connect_task.exception()

        logger.info(f"WhatsApp Platform connect task started (session={self._session_name}).")

    async def stop(self) -> None:
        try:
            await self._run_sync(self.client.disconnect)
        except Exception as e:
            logger.error(f"Error while disconnecting WhatsApp client: {e}")

        if self._connect_task and not self._connect_task.done():
            self._connect_task.cancel()

        logger.info("WhatsApp Platform stopped.")

    async def _cache_outgoing_message(self, chat_jid: JID, send_response: neonize_proto.SendResponse) -> None:
        if not send_response.ID:
            return

        outgoing = MessageEvent()
        outgoing.Info.ID = send_response.ID
        outgoing.Info.Timestamp = int(send_response.Timestamp or 0)
        outgoing.Info.MessageSource.Chat.CopyFrom(chat_jid)
        outgoing.Info.MessageSource.IsFromMe = True
        outgoing.Info.MessageSource.IsGroup = chat_jid.Server == "g.us"

        if self.client.me and self.client.me.JID.ListFields():
            outgoing.Info.MessageSource.Sender.CopyFrom(self.client.me.JID)
            outgoing.Info.Pushname = self.client.me.PushName

        if send_response.Message.ListFields():
            outgoing.Message.CopyFrom(send_response.Message)

        unified = self.to_unified_message(outgoing)
        self._cache_message(unified, outgoing)

    async def send_message(self, chat_id: int | str, text: str, reply_to_message_id: Optional[int | str] = None) -> int | str:
        chat_jid = self._chat_id_to_jid(chat_id)
        raw_quoted = self.get_cached_raw_message(Jid2String(chat_jid), str(reply_to_message_id)) if reply_to_message_id else None

        if raw_quoted:
            response = await self._run_sync(self.client.reply_message, text, raw_quoted, chat_jid)
        else:
            response = await self._run_sync(self.client.send_message, chat_jid, text)

        await self._cache_outgoing_message(chat_jid, response)
        return response.ID

    async def send_sticker(self, chat_id: int | str, sticker_id: str, reply_to_message_id: Optional[int | str] = None) -> int | str:
        logger.info("Sticker action ignored on WhatsApp because stickers are disabled for cross-platform compatibility.")
        return 0

    async def react(self, chat_id: int | str, message_id: int | str, reaction: str) -> None:
        chat_jid = self._chat_id_to_jid(chat_id)
        raw_target = self.get_cached_raw_message(Jid2String(chat_jid), message_id)
        if not raw_target:
            logger.warning(f"Cannot react on WhatsApp: missing cached target message {message_id} in chat {chat_id}")
            return

        sender_jid = raw_target.Info.MessageSource.Sender
        reaction_message = await self._run_sync(
            self.client.build_reaction,
            chat_jid,
            sender_jid,
            str(message_id),
            reaction,
        )
        await self._run_sync(self.client.send_message, chat_jid, reaction_message)

    async def send_chat_action(self, chat_id: int | str, action: str) -> None:
        chat_jid = self._chat_id_to_jid(chat_id)
        normalized = action.lower()

        if normalized == "typing":
            await self._run_sync(
                self.client.send_chat_presence,
                chat_jid,
                ChatPresence.CHAT_PRESENCE_COMPOSING,
                ChatPresenceMedia.CHAT_PRESENCE_MEDIA_TEXT,
            )
        elif normalized in {"cancel", "stop", "paused"}:
            await self._run_sync(
                self.client.send_chat_presence,
                chat_jid,
                ChatPresence.CHAT_PRESENCE_PAUSED,
                ChatPresenceMedia.CHAT_PRESENCE_MEDIA_TEXT,
            )

    async def download_media(self, media: UnifiedMedia) -> bytes:
        payload = media.native_obj
        wa_message = None

        if isinstance(payload, dict):
            wa_message = payload.get("wa_message")
        elif isinstance(payload, WaMessage):
            wa_message = payload

        if wa_message is None:
            return b""

        data = await self._run_sync(self.client.download_any, wa_message)
        return data or b""

    async def get_message(self, chat_id: int | str, message_id: int | str) -> Optional[UnifiedMessage]:
        cache_key = (str(chat_id), str(message_id))
        with self._cache_lock:
            return self._unified_message_cache.get(cache_key)

    async def get_user_by_username(self, username: str) -> Optional[UnifiedUser]:
        # WhatsApp doesn't expose a public @username. We interpret this as phone number.
        clean = username.strip().lstrip("@").split("@", 1)[0]
        if not clean.isdigit():
            return None

        jid = build_jid(clean, "s.whatsapp.net")
        try:
            infos = await self._run_sync(self.client.get_user_info, jid)
            if infos:
                info = infos[0]
                first_name = clean
                if info.UserInfo.Status:
                    first_name = info.UserInfo.Status
                return UnifiedUser(
                    id=clean,
                    username=clean,
                    first_name=first_name,
                    is_self=False,
                )
        except Exception:
            return None

        return None

    async def get_chat_member_status(self, chat_id: int | str, user_id: int | str) -> str:
        chat_jid = self._chat_id_to_jid(chat_id)
        if chat_jid.Server != "g.us":
            return "MEMBER"

        target = str(user_id)
        try:
            group_info = await self._run_sync(self.client.get_group_info, chat_jid)
            for participant in group_info.Participants:
                candidate_ids = {participant.JID.User, Jid2String(participant.JID)}
                if target in candidate_ids:
                    if participant.IsSuperAdmin:
                        return "OWNER"
                    if participant.IsAdmin:
                        return "ADMINISTRATOR"
                    return "MEMBER"
        except Exception:
            return "Unknown"

        return "Unknown"

    async def ban_chat_member(self, chat_id: int | str, user_id: int | str) -> None:
        chat_jid = self._chat_id_to_jid(chat_id)
        user_jid = self._user_id_to_jid(user_id)
        await self._run_sync(
            self.client.update_group_participants,
            chat_jid,
            [user_jid],
            ParticipantChange.REMOVE,
        )

    async def kick_chat_member(self, chat_id: int | str, user_id: int | str) -> None:
        await self.ban_chat_member(chat_id, user_id)

    async def unban_chat_member(self, chat_id: int | str, user_id: int | str) -> None:
        chat_jid = self._chat_id_to_jid(chat_id)
        user_jid = self._user_id_to_jid(user_id)
        await self._run_sync(
            self.client.update_group_participants,
            chat_jid,
            [user_jid],
            ParticipantChange.ADD,
        )

    async def restrict_chat_member(self, chat_id: int | str, user_id: int | str, can_send_messages: bool) -> None:
        raise NotImplementedError("WhatsApp does not expose per-user mute/unmute in this adapter.")

    async def create_chat_invite_link(self, chat_id: int | str) -> str:
        chat_jid = self._chat_id_to_jid(chat_id)
        if chat_jid.Server != "g.us":
            return ""
        return await self._run_sync(self.client.get_group_invite_link, chat_jid, False)
