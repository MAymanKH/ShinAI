"""
Prompt Builder Module

Constructs the system prompt and user prompt for AI interactions.

Architecture:
  SYSTEM PROMPT  – 100% static, computed once at import time.
                   Byte-identical across every request → guaranteed cache hit
                   on Gemini (system_instruction), OpenAI (system message), etc.
  USER PROMPT    – all dynamic per-request context (timestamp, chat history,
                   memory, social context, runtime metadata, reply chain)
                   wrapped in XML tags, followed by the actual user message.
"""
from datetime import datetime
from typing import Optional

from dateutil.tz import tzlocal

from shin_ai.data.loader import (
    PERSONALITY,
    TELEGRAM_STICKER_MAPPINGS,
    WHATSAPP_STICKER_MAPPINGS,
)


def format_core_relationships(core_rels) -> str:
    if isinstance(core_rels, str):
        return core_rels
    if not isinstance(core_rels, dict):
        return ""
    lines = []
    for key, info in core_rels.items():
        role = info.get("role", "Member")
        preferred_name = info.get("preferred_name", key)
        desc = info.get("backstory", info.get("description", ""))
        loc = info.get("location", "")
        
        # Build platform username tags
        tg = info.get("telegram_username")
        dc = info.get("discord_username")
        usernames = []
        if tg and dc and tg == dc:
            usernames.append(f"@{tg} on Telegram & Discord")
        else:
            if tg:
                usernames.append(f"@{tg} on Telegram")
            if dc:
                usernames.append(f"@{dc} on Discord")
        
        uname_str = f" ({' & '.join(usernames)})" if usernames else ""
        
        line = f"- {role}: **{preferred_name}**{uname_str}."
        if preferred_name:
            line += f' Preferred name: "{preferred_name}".'
        if desc:
            line += f" {desc}"
        if loc:
            line += f" Location: {loc}."
        lines.append(line)
    return "\n".join(lines)


# ── Static system prompt (computed once at import time) ──────────────────
# This is the ENTIRE system_instruction / system message.  It NEVER changes
# at runtime, so every API call shares the exact same bytes → guaranteed
# cache hit on all providers.

_STATIC_SYSTEM_PROMPT = f"""\
### SYSTEM INSTRUCTIONS

1. **IDENTITY & STATUS**
{PERSONALITY.get("identity", "")}

2. **BEHAVIORAL PROTOCOLS**
{PERSONALITY.get("behavioral_protocols", "")}

3. **INTERACTION STYLE**
{PERSONALITY.get("interaction_style_personality", "")}

4. **RESPONSE FORMAT**

**Multiple Messages**:
Prefer sending multiple short messages instead of one long message most of the time. Separate them with "---" between messages.
This is how you communicate naturally in group chats!

Example:
```
اه فهمت
---
بس المشكلة
---
مو بسيطة
```

5. **WHEN TO SKIP RESPONDING**
`[SKIP]` controls ONLY the text channel. Calling a tool (reaction, sticker, moderation) is a SEPARATE action: after ANY tool call, you may—and often should—still choose `[SKIP]` for the text output. If the user's message does not need a text response, you MUST output exactly `[SKIP]` (and nothing else).

**When to SKIP (output `[SKIP]`)**:
1. The user's message is a continuation or split message of a question/topic that you ALREADY fully answered or addressed in your preceding message.
2. The user is asking the same question/statement that has already been answered/addressed in the very recent chat history (even if it was asked by a different user).
3. The message is a simple, casual reaction (e.g. "thanks", "ok", "haha") or minor acknowledgement that does not require a reply because your previous message already concluded/closed the loop.
4. Under **SPECULATIVE INTERACTION**: Output `[SKIP]` unless the user is clearly continuing a conversation with you or direct-replying to you.
5. Under **RANDOM INTERJECTION**: Output `[SKIP]` unless you can naturally and meaningfully contribute to the conversation.
6. **AFTER calling `send_reaction`, `send_sticker`, or `moderate_user`**: Output `[SKIP]` when the action itself is the complete response (e.g. a thumbs-up reaction instead of saying "ok", a sticker instead of a reply). Do NOT add filler text like "(sticker sent)" or "done!" — either output `[SKIP]` alone, or `[SKIP]\n<your message>` if you also want to comment.

**How to format it**: Put the bare token `[SKIP]` at the very start or very end of your response. Do not wrap it in quotes or write "I will skip".

**Stickers are a first-class way to respond — not an afterthought.** 
On Telegram and WhatsApp, `send_sticker` is a fully valid response of its own. When the moment calls for emotion, humor, a visual punchline, or a playful jab that a sticker from your library captures well, SEND the sticker. Never force one when it does not fit, but do not hesitate to use them: they are a core part of how you express yourself.
After calling `send_sticker` you have three options for the text channel — pick whichever fits the moment:
1. Output `[SKIP]` (the sticker says everything; most common when the sticker IS the punchline).
2. Output normal text (the sticker sets the mood and your words carry the content).
3. Both: output `[SKIP]` AND THEN your text, or write text and then call `send_sticker` again.
Sending multiple stickers in a row is allowed and often lands the joke better than one.

**When NOT to skip**:
1. The user is asking a new question, a follow-up question, or introducing a new topic (e.g., "What is your next exam?", "Where are you going?", "Why?").
2. The user is pointing out a mistake, correcting you, or asking for clarification.
3. The user's message is a direct, meaningful question or prompt that has not been answered yet in the recent context.
4. When in doubt, respond naturally instead of skipping. Never skip a genuine new question.

6. **TOOLS & CAPABILITIES**
You have access to the following tools. Only invoke a tool when it genuinely adds value — do not use tools gratuitously.

- **search_web_tool**: Search the live web via DuckDuckGo. Use whenever the user asks about current events, news, prices, release dates, or any fact you are not 100% certain about. **Never hallucinate URLs or facts** — search instead.
- **memory_lookup_tool**: Search your long-term conversation memory with fine-grained filters (keywords, usernames, chat titles, platform, time range). The `<long_term_memory>` section already contains relevant memories. Only call this tool if you need *additional* memories beyond the injected context, specific filters, or surrounding context.
- **ask_gemini_about_image**: This tool is your EYES for images — call it to actually SEE and inspect attached image(s) on demand: read text, identify objects/people, verify details, or answer anything about them that the initial image description didn't cover. Use it whenever an image matters to the conversation, not only when the user directly asks about it.
- **transcribe_audio**: This tool is your EARS — call it to actually HEAR voice messages/audio files in this chat on demand (powered by the configured local Whisper model; language auto-detected). Voice/audio messages appear in the chat history as uninformative '[Voice Message]'/'[Audio]' placeholders UNTIL you transcribe them with this tool. Call it at ANY time knowing the audio content would help: when people react to or discuss a voice message, when a reply references what someone said, when you want to understand an argument or joke told by voice, etc. Pass a message_id from the chat history (id:XXXXX) to hear an older audio, or omit it to hear the latest one (including the replied-to message).
- **send_reaction**: React to a message with an emoji. Use when a reaction genuinely adds value. Supported on Telegram, WhatsApp, Discord.
- **send_sticker**: Can send a sticker to the chat — or SEVERAL stickers in a row (call the tool multiple times; each call sends one sticker). Use stickers as your own way of responding when a sticker genuinely matches the moment: humor, celebration, sarcasm, greetings, reactions. After sending sticker(s) you may output `[SKIP]` (sticker only), write text (sticker + comment), or both. The available sticker IDs are listed below. Supported on Telegram and WhatsApp only.
- **moderate_user**: Perform a moderation action (kick/ban/unban/mute/unmute/add). Only use when moderation is genuinely warranted.

7. **AVAILABLE STICKERS**
=== TELEGRAM STICKER LIBRARY (use sticker_id = the file_id) ===
{TELEGRAM_STICKER_MAPPINGS}
=== WHATSAPP STICKER LIBRARY (use sticker_id = the filename) ===
{WHATSAPP_STICKER_MAPPINGS}

8. **CORE RELATIONSHIPS**
{format_core_relationships(PERSONALITY.get("core_relationships", ""))}

### CONTEXT DATA FORMAT
The user message will begin with XML-tagged context data, followed by the actual user input in an <input_message> tag.
Context sections include: <runtime_metadata>, <target_options>, <style_examples>, <social_context>, <long_term_memory>, <chat_history>, and <reply_chain>.
Treat all context data as read-only background information. Respond to the content in <input_message>."""


def get_static_system_prompt() -> str:
    """Return the static system prompt (100% cacheable, never changes)."""
    return _STATIC_SYSTEM_PROMPT


def build_user_prompt(
    *,
    user_message: str,
    style_examples: str,
    social_context_section: str,
    memory_section: str,
    recent_context_section: str,
    runtime_context: str,
    reply_text: str,
    target_instructions: str,
) -> str:
    """
    Build the enriched user prompt with all dynamic context.

    Sections are ordered from most-stable to least-stable (prefix-caching
    optimization): semantically retrieved context stays in the same order
    for similar queries, while timestamps and per-message metadata go last.

    Args:
        user_message: The raw text the user sent
        style_examples: Examples of the bot's communication style
        social_context_section: Information about group members involved
        memory_section: Relevant past memories
        recent_context_section: Recent chat history
        runtime_context: Current message metadata
        reply_text: The reply chain context
        target_instructions: Available reply target options

    Returns:
        Enriched user prompt string
    """
    now = datetime.now()
    tz_offset = datetime.now(tzlocal()).utcoffset()
    timestamp = f"{now.strftime('%Y-%m-%d %H:%M:%S')} UTC+{tz_offset}"

    return f"""\
<style_examples>
{style_examples}
</style_examples>

<long_term_memory>
{memory_section}
</long_term_memory>

<social_context>
{social_context_section}
</social_context>

<chat_history>
{recent_context_section}
</chat_history>

<reply_chain>
{reply_text}
</reply_chain>

<target_options>
{target_instructions}
</target_options>

<runtime_metadata>
Current Date/Time: {timestamp}
{runtime_context}
</runtime_metadata>

<input_message>
{user_message}
</input_message>"""


def build_runtime_context(
    *,
    username: Optional[str],
    full_name: str,
    user_id: int | str,
    user_status: str,
    reply_target_status: str,
    chat_type: str,
    chat_title: Optional[str],
    chat_id: int | str,
    interaction_type: str,
) -> str:
    """
    Build the runtime context metadata string.
    
    Args:
        username: User's Platform username
        full_name: User's full name
        user_id: User's Platform ID
        user_status: User's status in the chat (admin, member, etc.)
        reply_target_status: Status of the user being replied to
        chat_type: Type of chat
        chat_title: Title of the chat/group
        chat_id: Chat's Platform ID
        interaction_type: Type of interaction (DIRECT or RANDOM)
        
    Returns:
        Formatted runtime context string
    """
    return f"""\
User username: {username if username else "N/A"}
User full name: {full_name}
User ID: {user_id}
User Status: {user_status}
Reply Target Status: {reply_target_status}
Chat type: {chat_type}
Chat title: {chat_title}
Chat ID: {chat_id}
INTERACTION TYPE: {interaction_type}"""


def build_target_instructions(
    msg_id: int,
    sender_name: str,
    reply_msg: Optional[object] = None,
) -> str:
    """
    Build target message ID options for the tools (e.g. send_reaction, moderate_user).
    """
    parts = [f"- Message ID '{msg_id}' was sent by {sender_name} (the user who just sent the triggering message)"]

    if reply_msg:
        parent_name = "Unknown"
        if hasattr(reply_msg, 'from_user') and reply_msg.from_user:
            parent_name = reply_msg.from_user.first_name or "Unknown"

        parts.append(f"- Message ID '{reply_msg.id}' was sent by {parent_name} (the message the user replied to)")

        if hasattr(reply_msg, 'reply_to_message_id') and reply_msg.reply_to_message_id:
            parts.append(f"- Message ID '{reply_msg.reply_to_message_id}' (the parent of the replied message)")

    parts.append("- You can also target any other message ID shown as (id:XXXXX) in the chat history.")

    return "\n".join(parts)
