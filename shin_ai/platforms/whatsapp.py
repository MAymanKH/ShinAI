from __future__ import annotations

import asyncio
from collections import OrderedDict
from pathlib import Path
from threading import RLock
from typing import Any, Optional

from shin_ai.platforms.base import PlatformAdapter
from shin_ai.platforms.models import (
    UnifiedChat,
    UnifiedMedia,
    UnifiedMessage,
    UnifiedUser,
)
from shin_ai.platforms.whatsapp_helpers import (
    collect_mentioned_identity_tokens,
    extract_local_user_id,
    find_cache_key,
    jid_to_user_id,
    jid_to_username,
    normalize_jid_identity,
    normalize_message_timestamp,
)
from shin_ai.platforms.whatsapp_message import (
    apply_media,
    build_text_caption_entities,
    extract_context_info,
    extract_text_and_caption,
    unwrap_message,
)
from shin_ai.platforms.whatsapp_runtime import (
    ChatPresence,
    ChatPresenceMedia,
    ContextInfoType,
    JIDType,
    Jid2String,
    MessageEvent,
    MessageEventType,
    NewClient,
    ParticipantChange,
    SendResponseType,
    WaMessage,
    WaMessageType,
    build_jid,
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
        self._group_title_cache: dict[str, str] = {}

    @property
    def platform_name(self) -> str:
        return "whatsapp"

    @property
    def supports_stickers(self) -> bool:
        # WhatsApp stickers are supported through Neonize when the sticker source is
        # a valid URL/path (optionally prefixed with `wa:`).
        return True

    @property
    def supports_member_restrictions(self) -> bool:
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

    def _jid_to_user_id(self, jid: JIDType) -> str:
        return jid_to_user_id(jid)

    def _jid_to_username(self, jid: JIDType) -> str:
        return jid_to_username(jid)

    def _normalize_message_timestamp(self, raw_timestamp: int | float | None) -> float:
        return normalize_message_timestamp(raw_timestamp)

    def _media_id(self, message_id: int | str, media_type: str) -> str:
        from shin_ai.platforms.whatsapp_helpers import media_id

        return media_id(message_id, media_type)

    def _normalize_jid_identity(self, jid_value: str) -> str:
        return normalize_jid_identity(jid_value)

    def _extract_local_user_id(self, jid_value: str) -> str:
        return extract_local_user_id(jid_value)

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
        return unwrap_message(message)

    def _extract_context_info(self, message: WaMessageType) -> Optional[ContextInfoType]:
        return extract_context_info(message)

    def _extract_text_and_caption(self, message: WaMessageType) -> tuple[Optional[str], Optional[str]]:
        return extract_text_and_caption(message)

    def _apply_media(
        self,
        unified_msg: UnifiedMessage,
        message: WaMessageType,
        download_message: WaMessageType = None,
        message_id: int | str | None = None,
    ) -> None:
        apply_media(unified_msg, message, download_message, message_id)

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
    ):
        from shin_ai.platforms.whatsapp_message import build_entities_from_context

        return build_entities_from_context(context_info, source_text)

    def _build_text_caption_entities(
        self,
        context_info: Optional[ContextInfoType],
        text: Optional[str],
        caption: Optional[str],
    ):
        return build_text_caption_entities(context_info, text, caption)

    def _cache_message(self, unified: UnifiedMessage, event_msg: MessageEventType) -> None:
        cache_key = (str(unified.chat.id), str(unified.id))
        with self._cache_lock:
            self._raw_message_cache[cache_key] = event_msg
            self._unified_message_cache[cache_key] = unified
            self._raw_message_cache.move_to_end(cache_key)
            self._unified_message_cache.move_to_end(cache_key)
        self._trim_cache_if_needed()

    def _is_same_chat_identity(self, first_chat_id: str, second_chat_id: str) -> bool:
        from shin_ai.platforms.whatsapp_helpers import is_same_chat_identity

        return is_same_chat_identity(first_chat_id, second_chat_id)

    def _find_cache_key(
        self,
        cache_map: OrderedDict[tuple[str, str], Any],
        chat_id: int | str,
        message_id: int | str,
    ) -> Optional[tuple[str, str]]:
        return find_cache_key(cache_map, chat_id, message_id)

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

    async def ingest_event_message(self, event_msg: MessageEventType) -> UnifiedMessage:
        unified = await self.to_unified_message(event_msg)
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
        return collect_mentioned_identity_tokens(mentioned_jid_list)

    async def to_unified_message(self, event_msg: MessageEventType) -> UnifiedMessage:
        source = event_msg.Info.MessageSource
        chat_jid = source.Chat
        sender_jid = source.Sender
        if source.SenderAlt.ListFields():
            sender_jid = source.SenderAlt
        if source.IsFromMe and self.client.me and self.client.me.JID.ListFields():
            sender_jid = self.client.me.JID

        raw_chat_id = Jid2String(chat_jid)
        chat_id = self._normalize_jid_identity(raw_chat_id) or raw_chat_id
        chat_type = "GROUP" if (chat_jid.Server == "g.us" or bool(source.IsGroup)) else "PRIVATE"

        chat_title = None
        if chat_type == "GROUP":
            if chat_id in self._group_title_cache:
                chat_title = self._group_title_cache[chat_id]
            else:
                try:
                    group_info = await self._run_sync(self.client.get_group_info, chat_jid)
                    if group_info and group_info.GroupName and group_info.GroupName.Name:
                        chat_title = group_info.GroupName.Name
                        self._group_title_cache[chat_id] = chat_title
                except Exception as e:
                    logger.debug(f"Failed to fetch group info for {chat_id}: {e}")

        chat = UnifiedChat(
            id=chat_id,
            title=chat_title,
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

        unified = await self.to_unified_message(outgoing)
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
            # Best effort: try resolving the message first so we can populate cache.
            await self.get_message(chat_id, message_id)
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
        cached = self._get_cached_unified_message(chat_id, message_id)
        if cached:
            return cached

        # Best effort fallback for clients exposing message retrieval APIs.
        chat_jid = self._chat_id_to_jid(chat_id)
        for method_name in ("get_message", "get_messages", "get_message_by_id"):
            fetch_fn = getattr(self.client, method_name, None)
            if not callable(fetch_fn):
                continue

            try:
                if method_name == "get_messages":
                    fetched = await self._run_sync(fetch_fn, chat_jid, [str(message_id)])
                else:
                    fetched = await self._run_sync(fetch_fn, chat_jid, str(message_id))

                if isinstance(fetched, (list, tuple)):
                    fetched = fetched[0] if fetched else None

                if fetched and hasattr(fetched, "Info") and hasattr(fetched, "Message"):
                    return await self.ingest_event_message(fetched)
            except Exception:
                continue

        return None

    async def get_user_by_username(self, username: str) -> Optional[UnifiedUser]:
        # WhatsApp doesn't expose a public @username. We interpret this as phone number.
        clean_input = username.strip().lstrip("@").split("@", 1)[0]
        clean = "".join(ch for ch in clean_input if ch.isdigit())
        if not clean:
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
        raise NotImplementedError("WhatsApp adapter does not support per-user mute/unmute.")

    async def create_chat_invite_link(self, chat_id: int | str) -> str:
        chat_jid = self._chat_id_to_jid(chat_id)
        if chat_jid.Server != "g.us":
            return ""
        return await self._run_sync(self.client.get_group_invite_link, chat_jid, False)
