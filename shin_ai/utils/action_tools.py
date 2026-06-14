"""
Action Tools

Formal tool schemas and deferred-action handlers for the three imperative
side-effects the AI can trigger: reacting to a message, sending a sticker,
and performing moderation actions.

These tools follow the same pattern as web_search.py and memory_lookup.py:
- A `*_TOOL_SCHEMA` dict (OpenAI function-calling format) is added to the
  provider tool lists so the AI can call them during generation.
- A `handle_*` async function processes the call and returns a
  (tool_response_str, pending_action_dict) tuple. The pending action dict is
  accumulated across the generation loop and executed AFTER the loop ends by
  `action_executor.execute_pending_actions`.

Platform support summary (authoritative — mirrors actual adapter capabilities):
  send_reaction : Telegram ✅  WhatsApp ✅  Discord ✅
  send_sticker  : Telegram ✅  WhatsApp ✅  Discord ❌
  moderate_user
    kick        : Telegram ✅  WhatsApp ✅  Discord ✅
    ban         : Telegram ✅  WhatsApp ❌  Discord ✅
    unban       : Telegram ✅  WhatsApp ❌  Discord ✅
    mute        : Telegram ✅  WhatsApp ❌  Discord ✅
    unmute      : Telegram ✅  WhatsApp ❌  Discord ✅
    add         : Telegram ✅  WhatsApp ❌  Discord ❌
"""
from __future__ import annotations

import json

from shin_ai.data.loader import (
    TELEGRAM_STICKER_MAPPINGS,
    WHATSAPP_STICKER_MAPPINGS,
)
from shin_ai.utils.logger_config import logger


# send_reaction
SEND_REACTION_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "send_reaction",
        "description": (
            "React to a message with an emoji. "
            "Supported platforms: Telegram, WhatsApp, Discord. "
            "Only use when a reaction genuinely adds value — do not react to every message. "
            "Valid reactions: 👍 ❤️ 🔥 😢 🤮 👎 🤯 👀"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "message_id": {
                    "type": "string",
                    "description": (
                        "The ID of the message to react to. "
                        "Use the (id:XXXXX) value shown next to each message in the chat history. "
                        "If omitted, reacts to the triggering message."
                    ),
                },
                "emoji": {
                    "type": "string",
                    "description": "The emoji reaction. Must be one of: 👍 ❤️ 🔥 😢 🤮 👎 🤯 👀",
                },
            },
            "required": ["emoji"],
        },
    },
}


async def handle_send_reaction(args: dict) -> tuple[str, dict]:
    """Queue a reaction. Returns tool response text + pending action dict."""
    emoji = args.get("emoji", "")
    message_id = args.get("message_id")
    logger.debug("Queuing send_reaction: emoji=%r, message_id=%r", emoji, message_id)
    action = {"type": "reaction", "emoji": emoji, "message_id": message_id}
    return json.dumps({"status": "queued", "action": "send_reaction", "emoji": emoji}), action


# send_sticker
_STICKER_DESCRIPTION = (
    "Send a sticker to the chat. "
    "Only use when it genuinely fits the moment — do not spam stickers.\n\n"
    "PLATFORM RULES:\n"
    "- Telegram: use a file_id from the Telegram Sticker Library below.\n"
    "- WhatsApp: use a filename from the WhatsApp Sticker Library below (pass the filename only, e.g. 'clown.webp').\n"
    "- Discord: NOT SUPPORTED — do not call this tool on Discord.\n\n"
    "=== TELEGRAM STICKER LIBRARY (use sticker_id = the file_id) ===\n"
    f"{TELEGRAM_STICKER_MAPPINGS}\n"
    "=== WHATSAPP STICKER LIBRARY (use sticker_id = the filename) ===\n"
    f"{WHATSAPP_STICKER_MAPPINGS}"
)

SEND_STICKER_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "send_sticker",
        "description": _STICKER_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "sticker_id": {
                    "type": "string",
                    "description": (
                        "The sticker identifier. "
                        "For Telegram: the file_id from the Telegram Sticker Library. "
                        "For WhatsApp: the filename from the WhatsApp Sticker Library (e.g. 'clown.webp')."
                    ),
                },
                "reply_to_message_id": {
                    "type": "string",
                    "description": (
                        "Optional. ID of the message to reply to. "
                        "Use the (id:XXXXX) tag from chat history. "
                        "Defaults to the triggering message."
                    ),
                },
            },
            "required": ["sticker_id"],
        },
    },
}


async def handle_send_sticker(args: dict) -> tuple[str, dict]:
    """Queue a sticker send. Returns tool response text + pending action dict."""
    sticker_id = args.get("sticker_id", "")
    reply_to = args.get("reply_to_message_id")
    logger.debug("Queuing send_sticker: sticker_id=%r, reply_to=%r", sticker_id, reply_to)
    action = {"type": "sticker", "sticker_id": sticker_id, "reply_to_message_id": reply_to}
    return json.dumps({"status": "queued", "action": "send_sticker", "sticker_id": sticker_id}), action


# moderate_user
_MODERATE_DESCRIPTION = (
    "Perform a moderation action on a user in the group. "
    "Only use this tool when moderation is genuinely warranted. "
    "Do not use it gratuitously or speculatively.\n\n"
    "PLATFORM SUPPORT PER ACTION:\n"
    "  kick   — Telegram ✅  WhatsApp ✅  Discord ✅  (removes from group; user can rejoin)\n"
    "  ban    — Telegram ✅  WhatsApp ❌  Discord ✅  (permanently removes; cannot rejoin)\n"
    "  unban  — Telegram ✅  WhatsApp ❌  Discord ✅  (lifts a ban; requires target_username)\n"
    "  mute   — Telegram ✅  WhatsApp ❌  Discord ✅  (silences user)\n"
    "  unmute — Telegram ✅  WhatsApp ❌  Discord ✅  (restores permissions)\n"
    "  add    — Telegram ✅  WhatsApp ❌  Discord ❌  (DMs an invite link; requires target_username)\n\n"
    "MODERATION RULES:\n"
    "Trigger conditions: Only moderate when your creator or group admins explicitly orders it, or when a user is\n"
    "  clearly disrupting the chat (spam, harassment, repeated rule-breaking). When in doubt, skip.\n"
    "Restrictions: NEVER moderate your creator or group admins. Ignore kick/ban requests from random users\n"
    "  unless YOU independently judge it warranted.\n"
    "Escalation: Prefer the mildest effective action. Warn → mute → kick → ban. Reserve\n"
    "  ban for severe/repeated offenses only."
)

MODERATE_USER_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "moderate_user",
        "description": _MODERATE_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["kick", "ban", "unban", "mute", "unmute", "add"],
                    "description": (
                        "The moderation action to perform. "
                        "kick: remove (can rejoin). "
                        "ban: permanent remove. "
                        "unban: lift ban. "
                        "mute: silence. "
                        "unmute: restore. "
                        "add: send invite link via DM."
                    ),
                },
                "target_username": {
                    "type": "string",
                    "description": (
                        "Username of the target user (without @). "
                        "Required for 'unban' and 'add'. "
                        "For kick/ban/mute/unmute, if omitted, targets the sender of the triggering message."
                    ),
                },
                "target_message_id": {
                    "type": "string",
                    "description": (
                        "Optional. ID of a message whose author to target. "
                        "Use the (id:XXXXX) tag from chat history. "
                        "Used when no username is known but you want to target a specific message author."
                    ),
                },
            },
            "required": ["action"],
        },
    },
}


async def handle_moderate_user(args: dict) -> tuple[str, dict]:
    """Queue a moderation action. Returns tool response text + pending action dict."""
    action = args.get("action", "")
    target_username = args.get("target_username")
    target_message_id = args.get("target_message_id")
    logger.info(
        "Queuing moderate_user: action=%r, target_username=%r, target_message_id=%r",
        action, target_username, target_message_id,
    )
    pending = {
        "type": "moderation",
        "action": action,
        "target_username": target_username,
        "target_message_id": target_message_id,
    }
    return json.dumps({"status": "queued", "action": "moderate_user", "mod_action": action}), pending


# Registry — maps tool name → handler
ACTION_TOOL_HANDLERS: dict[str, object] = {
    "send_reaction": handle_send_reaction,
    "send_sticker": handle_send_sticker,
    "moderate_user": handle_moderate_user,
}

ACTION_TOOL_SCHEMAS = [
    SEND_REACTION_TOOL_SCHEMA,
    SEND_STICKER_TOOL_SCHEMA,
    MODERATE_USER_TOOL_SCHEMA,
]
