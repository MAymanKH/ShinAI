from __future__ import annotations

import asyncio
from collections import OrderedDict
from pathlib import Path
from threading import RLock
from typing import Any, Optional, TYPE_CHECKING

def _load_neonize_symbols():
    restore_validate = None
    runtime_version = None

    try:
        import google.protobuf
        from google.protobuf import runtime_version

        major_version = int(str(google.protobuf.__version__).split(".", 1)[0])
        if major_version < 7:
            restore_validate = runtime_version.ValidateProtobufRuntimeVersion
            runtime_version.ValidateProtobufRuntimeVersion = lambda *args, **kwargs: None
    except Exception:
        restore_validate = None

    try:
        from neonize import NewClient
        from neonize.proto import Neonize_pb2 as neonize_proto
        from neonize.proto.Neonize_pb2 import JID, Message as MessageEvent
        from neonize.proto.waE2E.WAWebProtobufsE2E_pb2 import ContextInfo, Message as WaMessage
        from neonize.utils import Jid2String, build_jid
        from neonize.utils.enum import ChatPresence, ChatPresenceMedia, ParticipantChange

        return (
            NewClient,
            neonize_proto,
            JID,
            MessageEvent,
            ContextInfo,
            WaMessage,
            Jid2String,
            build_jid,
            ChatPresence,
            ChatPresenceMedia,
            ParticipantChange,
        )
    finally:
        if restore_validate is not None and runtime_version is not None:
            runtime_version.ValidateProtobufRuntimeVersion = restore_validate


(
    NewClient,
    neonize_proto,
    JID,
    MessageEvent,
    ContextInfo,
    WaMessage,
    Jid2String,
    build_jid,
    ChatPresence,
    ChatPresenceMedia,
    ParticipantChange,
) = _load_neonize_symbols()

if TYPE_CHECKING:
    from neonize.proto.Neonize_pb2 import JID as JIDType, Message as MessageEventType, SendResponse as SendResponseType
    from neonize.proto.waE2E.WAWebProtobufsE2E_pb2 import ContextInfo as ContextInfoType, Message as WaMessageType
else:
    # Runtime aliases must be concrete classes (not typing.Any), because Neonize
    # uses these in decorators and isinstance-style checks.
    JIDType = JID
    MessageEventType = MessageEvent
    SendResponseType = Any
    ContextInfoType = ContextInfo
    WaMessageType = WaMessage

from shin_ai.platforms.base import PlatformAdapter
from shin_ai.platforms.models import (
    UnifiedChat,
    UnifiedMedia,
    UnifiedMessage,
    UnifiedMessageEntity,
    UnifiedUser,
)
from shin_ai.data.loader import DATA_DIR
from shin_ai.utils.logger_config import logger
WHATSAPP_STICKERS_DIR = DATA_DIR / "whatsapp_stickers"
WHATSAPP_STICKERS_DIR.mkdir(parents=True, exist_ok=True)

class WhatsAppPlatform(PlatformAdapter):
    def __init__(self, session_name: str):
        self.client = NewClient(session_name)
        self._session_name = session_name
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._connect_task: Optional[asyncio.Task] = None
        self._bot_user_cache: Optional[UnifiedUser] = None
        self._cache_lock = RLock()
        self._raw_message_cache: OrderedDict[tuple[str, str], MessageEventType] = OrderedDict()
        self._unified_message_cache: OrderedDict[tuple[str, str], UnifiedMessage] = OrderedDict()
        self._cache_limit = 2000

    @property
    def platform_name(self) -> str:
        return "whatsapp"

    @property
    def supports_stickers(self) -> bool:
        # WhatsApp stickers are supported through Neonize when the sticker source is
        # a valid URL/path (optionally prefixed with `wa:`).
        return True

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

    def _jid_to_user_id(self, jid: JIDType) -> str:
        raw_jid = Jid2String(jid)
        normalized = self._normalize_jid_identity(raw_jid)
        if normalized:
            return normalized
        if jid.User:
            return str(jid.User).split(":", 1)[0]
        return raw_jid

    def _jid_to_username(self, jid: JIDType) -> str:
        raw_jid = Jid2String(jid)
        local_user = self._extract_local_user_id(raw_jid)
        if local_user:
            return local_user
        if jid.User:
            return str(jid.User).split(":", 1)[0]
        return raw_jid

    def _normalize_message_timestamp(self, raw_timestamp: int | float | None) -> float:
        if not raw_timestamp:
            return 0.0

        ts = float(raw_timestamp)
        # Neonize timestamps can be emitted in milliseconds depending on source wrappers.
        if ts > 1e12:
            ts = ts / 1000.0
        return ts

    def _media_id(self, message_id: int | str, media_type: str) -> str:
        return f"{message_id}:{media_type}"

    def _normalize_jid_identity(self, jid_value: str) -> str:
        raw = (jid_value or "").strip().lower()
        if not raw:
            return ""

        # Normalize device-scoped IDs like 201234567890:12@s.whatsapp.net to
        # the stable user identity 201234567890@s.whatsapp.net.
        if "@" in raw:
            user, server = raw.split("@", 1)
            user = user.split(":", 1)[0]
            return f"{user}@{server}"

        return raw.split(":", 1)[0]

    def _extract_local_user_id(self, jid_value: str) -> str:
        normalized = self._normalize_jid_identity(jid_value)
        if not normalized:
            return ""
        if "@" in normalized:
            return normalized.split("@", 1)[0]
        return normalized

    def _chat_id_to_jid(self, chat_id: int | str) -> JIDType:
        chat = str(chat_id)
        if "@" in chat:
            user, server = chat.split("@", 1)
            return build_jid(user, server)
        if "-" in chat:
            return build_jid(chat, "g.us")
        return build_jid(chat, "s.whatsapp.net")

    def _user_id_to_jid(self, user_id: int | str) -> JIDType:
        user = str(user_id)
        if "@" in user:
            raw_user, server = user.split("@", 1)
            return build_jid(raw_user, server)
        return build_jid(user, "s.whatsapp.net")

    def _unwrap_message(self, message: WaMessageType) -> WaMessageType:
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

    def _extract_context_info(self, message: WaMessageType) -> Optional[ContextInfoType]:
        # Scan every populated sub-message field for a contextInfo child.
        # This is intentionally exhaustive: instead of hardcoding a list of
        # known message types, we iterate over ALL fields the protobuf reports
        # as set, and check whether they contain a contextInfo with data.
        for field_descriptor, value in message.ListFields():
            if not hasattr(value, "contextInfo"):
                continue
            try:
                ctx = value.contextInfo
                if ctx and ctx.ListFields():
                    return ctx
            except Exception:
                continue

        # Also check top-level contextInfo (some message types place it there).
        top_ctx = getattr(message, "contextInfo", None)
        if top_ctx:
            try:
                if top_ctx.ListFields():
                    return top_ctx
            except Exception:
                pass

        return None

    def _extract_text_and_caption(self, message: WaMessageType) -> tuple[Optional[str], Optional[str]]:
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

    def _apply_media(
        self,
        unified_msg: UnifiedMessage,
        message: WaMessageType,
        download_message: WaMessageType = None,
        message_id: int | str | None = None,
    ) -> None:
        # Use the original (non-unwrapped) message for downloads so that
        # Neonize's download_any() can locate the media URL and encryption
        # keys that may live in outer wrapper layers (ephemeral, view-once, etc.).
        native_payload = {"wa_message": download_message or message}

        media_msg_id = message_id or unified_msg.id

        if message.imageMessage.ListFields():
            mime = message.imageMessage.mimetype or "image/jpeg"
            unified_msg.photo = UnifiedMedia(
                type="PHOTO",
                id=self._media_id(media_msg_id, "photo"),
                mime_type=mime,
                native_obj=native_payload,
            )
        if message.stickerMessage.ListFields():
            mime = message.stickerMessage.mimetype or "image/webp"
            unified_msg.sticker = UnifiedMedia(
                type="STICKER",
                id=self._media_id(media_msg_id, "sticker"),
                is_animated=bool(message.stickerMessage.isAnimated),
                mime_type=mime,
                native_obj=native_payload,
            )
        if message.videoMessage.ListFields():
            mime = message.videoMessage.mimetype or "video/mp4"
            unified_msg.video = UnifiedMedia(
                type="VIDEO",
                id=self._media_id(media_msg_id, "video"),
                mime_type=mime,
                native_obj=native_payload,
            )
        if message.audioMessage.ListFields():
            mime = message.audioMessage.mimetype or "audio/ogg"
            if bool(message.audioMessage.PTT):
                unified_msg.voice = UnifiedMedia(
                    type="VOICE",
                    id=self._media_id(media_msg_id, "voice"),
                    mime_type=mime,
                    native_obj=native_payload,
                )
            else:
                unified_msg.audio = UnifiedMedia(
                    type="AUDIO",
                    id=self._media_id(media_msg_id, "audio"),
                    mime_type=mime,
                    native_obj=native_payload,
                )
        if message.documentMessage.ListFields():
            mime = message.documentMessage.mimetype or "application/octet-stream"
            unified_msg.document = UnifiedMedia(
                type="DOCUMENT",
                id=self._media_id(media_msg_id, "document"),
                mime_type=mime,
                native_obj=native_payload,
            )

    def _build_quoted_message(self, context_info: ContextInfoType, chat: UnifiedChat) -> Optional[UnifiedMessage]:
        if not context_info.stanzaID:
            return None
        if not context_info.quotedMessage.ListFields():
            return None

        participant = context_info.participant or ""
        participant_id = self._normalize_jid_identity(participant)
        participant_user = self._extract_local_user_id(participant)
        display_name = participant_user or participant_id or "unknown"

        bot_tokens = self._collect_bot_identity_tokens()
        participant_tokens = self._collect_mentioned_identity_tokens([participant]) if participant else set()

        quoted_user = UnifiedUser(
            id=participant_id or display_name,
            username=participant_user if participant_user != "unknown" else None,
            first_name=display_name,
            is_self=bool(bot_tokens & participant_tokens),
        )

        quoted_body = self._unwrap_message(context_info.quotedMessage)
        quoted_text, quoted_caption = self._extract_text_and_caption(quoted_body)

        quoted = UnifiedMessage(
            platform=self.platform_name,
            id=str(context_info.stanzaID),
            chat=chat,
            from_user=quoted_user,
            text=quoted_text,
            caption=quoted_caption,
            date=0.0,
            native_msg=context_info.quotedMessage,
        )
        self._apply_media(
            quoted,
            quoted_body,
            context_info.quotedMessage,
            message_id=str(context_info.stanzaID),
        )
        return quoted

    def _build_entities_from_context(
        self,
        context_info: Optional[ContextInfoType],
        source_text: str,
    ) -> list[UnifiedMessageEntity]:
        entities: list[UnifiedMessageEntity] = []
        if not context_info:
            return entities

        for mentioned_jid in context_info.mentionedJID:
            normalized_id = self._normalize_jid_identity(mentioned_jid)
            user_name = self._extract_local_user_id(mentioned_jid)
            mention_token = user_name or normalized_id
            token = f"@{mention_token}" if mention_token else ""
            offset = source_text.find(token)
            length = len(token) if offset >= 0 else 0
            entities.append(
                UnifiedMessageEntity(
                    type="MENTION",
                    offset=max(offset, 0),
                    length=length,
                    user=UnifiedUser(
                        id=normalized_id or user_name or mentioned_jid,
                        username=user_name or None,
                        first_name=user_name or normalized_id or mentioned_jid,
                        is_self=False,
                    ),
                )
            )
        return entities

    def _build_text_caption_entities(
        self,
        context_info: Optional[ContextInfoType],
        text: Optional[str],
        caption: Optional[str],
    ) -> tuple[list[UnifiedMessageEntity], list[UnifiedMessageEntity]]:
        text_entities = self._build_entities_from_context(context_info, text or "")
        caption_entities = self._build_entities_from_context(context_info, caption or "")

        # Keep Telegram-like separation: entities map to text, caption_entities map to captions.
        if text:
            caption_entities = []
        elif caption:
            text_entities = []

        return text_entities, caption_entities

    def _cache_message(self, unified: UnifiedMessage, event_msg: MessageEventType) -> None:
        cache_key = (str(unified.chat.id), str(unified.id))
        with self._cache_lock:
            self._raw_message_cache[cache_key] = event_msg
            self._unified_message_cache[cache_key] = unified
            self._raw_message_cache.move_to_end(cache_key)
            self._unified_message_cache.move_to_end(cache_key)
        self._trim_cache_if_needed()

    def _is_same_chat_identity(self, first_chat_id: str, second_chat_id: str) -> bool:
        if first_chat_id == second_chat_id:
            return True

        normalized_first = self._normalize_jid_identity(first_chat_id)
        normalized_second = self._normalize_jid_identity(second_chat_id)
        if normalized_first and normalized_first == normalized_second:
            return True

        first_local = self._extract_local_user_id(first_chat_id)
        second_local = self._extract_local_user_id(second_chat_id)
        return bool(first_local and first_local == second_local)

    def _find_cache_key(
        self,
        cache_map: OrderedDict[tuple[str, str], Any],
        chat_id: int | str,
        message_id: int | str,
    ) -> Optional[tuple[str, str]]:
        requested_chat = str(chat_id)
        requested_msg = str(message_id)
        exact_key = (requested_chat, requested_msg)

        if exact_key in cache_map:
            return exact_key

        # WhatsApp may emit the same chat identity in slightly different JID
        # forms (e.g., device-scoped IDs). Match by normalized identity too.
        for candidate_key in reversed(cache_map):
            candidate_chat, candidate_msg = candidate_key
            if candidate_msg != requested_msg:
                continue
            if self._is_same_chat_identity(candidate_chat, requested_chat):
                return candidate_key

        return None

    def _get_cached_unified_message(self, chat_id: int | str, message_id: int | str) -> Optional[UnifiedMessage]:
        with self._cache_lock:
            cache_key = self._find_cache_key(self._unified_message_cache, chat_id, message_id)
            if not cache_key:
                return None
            self._unified_message_cache.move_to_end(cache_key)
            return self._unified_message_cache.get(cache_key)

    def get_cached_raw_message(self, chat_id: int | str, message_id: int | str) -> Optional[MessageEventType]:
        with self._cache_lock:
            cache_key = self._find_cache_key(self._raw_message_cache, chat_id, message_id)
            if not cache_key:
                return None
            self._raw_message_cache.move_to_end(cache_key)
            return self._raw_message_cache.get(cache_key)

    def ingest_event_message(self, event_msg: MessageEventType) -> UnifiedMessage:
        unified = self.to_unified_message(event_msg)
        self._cache_message(unified, event_msg)
        return unified

    def _collect_bot_identity_tokens(self) -> set[str]:
        """Build a set of ALL possible identity strings for the bot.

        This includes full JID, normalized JID, local user part, LID (Linked
        Identity), and alternate representations.  We use this set to test
        against mentionedJID entries with a single set-intersection.
        """
        tokens: set[str] = set()
        me = self.client.me
        if not me:
            return tokens

        # --- Phone-based JID (e.g. 201234567890@s.whatsapp.net) ---
        if me.JID.ListFields():
            jid = me.JID

            if jid.User:
                tokens.add(jid.User.lower())

            full_jid = Jid2String(jid)
            if full_jid:
                tokens.add(full_jid.lower())

            normalized = self._normalize_jid_identity(full_jid)
            if normalized:
                tokens.add(normalized.lower())

            local = self._extract_local_user_id(full_jid)
            if local:
                tokens.add(local.lower())

            raw_user = str(jid.User).split(":", 1)[0] if jid.User else ""
            if raw_user:
                tokens.add(raw_user.lower())

        # --- LID (Linked Identity, e.g. 45776516415716@lid) ---
        # WhatsApp now uses LIDs in group mentionedJID instead of phone JIDs.
        try:
            lid = me.LID
            if lid and lid.ListFields():
                if lid.User:
                    tokens.add(lid.User.lower())

                lid_full = Jid2String(lid)
                if lid_full:
                    tokens.add(lid_full.lower())

                lid_normalized = self._normalize_jid_identity(lid_full)
                if lid_normalized:
                    tokens.add(lid_normalized.lower())

                lid_local = self._extract_local_user_id(lid_full)
                if lid_local:
                    tokens.add(lid_local.lower())
        except Exception:
            pass

        return tokens

    def _collect_mentioned_identity_tokens(self, mentioned_jid_list: list[str]) -> set[str]:
        """Build a set of ALL identity strings from a mentionedJID list.

        For every JID string in the list, we add the raw string, the
        normalised form, and the extracted local user part.
        """
        tokens: set[str] = set()
        for jid_str in mentioned_jid_list:
            raw = (jid_str or "").strip()
            if not raw:
                continue
            tokens.add(raw.lower())

            normalized = self._normalize_jid_identity(raw)
            if normalized:
                tokens.add(normalized.lower())

            local = self._extract_local_user_id(raw)
            if local:
                tokens.add(local.lower())

            # Also try stripping device suffix from the user part
            user_part = raw.split("@", 1)[0] if "@" in raw else raw
            user_part = user_part.split(":", 1)[0]
            if user_part:
                tokens.add(user_part.lower())

        return tokens

    def to_unified_message(self, event_msg: MessageEventType) -> UnifiedMessage:
        source = event_msg.Info.MessageSource
        chat_jid = source.Chat
        sender_jid = source.Sender
        if source.SenderAlt.ListFields():
            sender_jid = source.SenderAlt
        if source.IsFromMe and self.client.me and self.client.me.JID.ListFields():
            sender_jid = self.client.me.JID

        raw_chat_id = Jid2String(chat_jid)
        chat_id = self._normalize_jid_identity(raw_chat_id) or raw_chat_id
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
            date=self._normalize_message_timestamp(event_msg.Info.Timestamp),
            native_msg=event_msg,
        )

        self._apply_media(
            unified_msg,
            body,
            event_msg.Message,
            message_id=str(event_msg.Info.ID),
        )

        context_info = self._extract_context_info(body)
        source_text = (text or caption or "")

        if context_info and context_info.stanzaID:
            unified_msg.reply_to_message_id = str(context_info.stanzaID)

            cached_parent = self._get_cached_unified_message(chat.id, unified_msg.reply_to_message_id)
            if cached_parent:
                unified_msg.reply_to_message = cached_parent
            else:
                unified_msg.reply_to_message = self._build_quoted_message(context_info, chat)

        # --- Mention detection (rewritten for robustness) ---
        if context_info:
            (
                unified_msg.entities,
                unified_msg.caption_entities,
            ) = self._build_text_caption_entities(context_info, text, caption)

            raw_mentioned = list(context_info.mentionedJID)

            if raw_mentioned:
                bot_tokens = self._collect_bot_identity_tokens()
                mention_tokens = self._collect_mentioned_identity_tokens(raw_mentioned)
                overlap = bot_tokens & mention_tokens
                if overlap:
                    unified_msg.mentioned = True

        # Text-based mention fallback: scan the message text for @<bot_id>.
        if not unified_msg.mentioned and source_text and self.client.me and self.client.me.JID.User:
            bot_tokens = self._collect_bot_identity_tokens()
            for token in bot_tokens:
                if token and f"@{token}" in source_text.lower():
                    unified_msg.mentioned = True
                    break

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

    async def _cache_outgoing_message(self, chat_jid: JIDType, send_response: SendResponseType) -> None:
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

    def _resolve_sticker_source(self, sticker_id: str) -> Optional[str]:
        """
        Resolve WhatsApp sticker source from AI sticker ID.

        Accepted formats:
        - `sticker:wa:https://...`
        - `sticker:wa:/absolute/or/relative/path.webp`
        - `sticker:https://...` (without wa: prefix)
        - `sticker:/absolute/or/relative/path.webp` (without wa: prefix)
        - A Telegram sticker ID that is mapped in WHATSAPP_STICKERS
        """
        raw = (sticker_id or "").strip()
        if not raw:
            return None

        source = raw[3:].strip() if raw.lower().startswith("wa:") else raw
        if not source:
            return None

        if source.startswith(("http://", "https://")):
            return source

        local_path = Path(source).expanduser()
        if local_path.is_file():
            return str(local_path)

        data_dir_path = WHATSAPP_STICKERS_DIR / source
        if data_dir_path.is_file():
            return str(data_dir_path)

        return None

    async def send_sticker(self, chat_id: int | str, sticker_id: str, reply_to_message_id: Optional[int | str] = None) -> int | str:
        chat_jid = self._chat_id_to_jid(chat_id)
        sticker_source = self._resolve_sticker_source(sticker_id)

        if not sticker_source:
            logger.warning(
                "Invalid WhatsApp sticker source '%s'. Use sticker:wa:<https-url-or-local-path>.",
                sticker_id,
            )
            return 0

        raw_quoted = self.get_cached_raw_message(Jid2String(chat_jid), str(reply_to_message_id)) if reply_to_message_id else None

        try:
            if raw_quoted:
                response = await self._run_sync(self.client.send_sticker, chat_jid, sticker_source, raw_quoted, passthrough=True)
            else:
                response = await self._run_sync(self.client.send_sticker, chat_jid, sticker_source, passthrough=True)
        except Exception as e:
            logger.warning("Passthrough sticker failed (%s), retrying with conversion...", e)
            if raw_quoted:
                response = await self._run_sync(self.client.send_sticker, chat_jid, sticker_source, raw_quoted, passthrough=False)
            else:
                response = await self._run_sync(self.client.send_sticker, chat_jid, sticker_source, passthrough=False)

        await self._cache_outgoing_message(chat_jid, response)
        return response.ID

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
        return self._get_cached_unified_message(chat_id, message_id)

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
