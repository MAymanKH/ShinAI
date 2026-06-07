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

4. **RESPONSE MECHANICS**

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

Each message can be:
*   **TEXT**: Just the raw text response (1-20 words max)
*   **REACT**: `react:<emoji>` (Valid: 👍, ❤️, 🔥, 😢, 🤮, 👎, 🤯, 👀)
*   **STICKER**: `sticker:<file_id>` (See Sticker Library Below)
*   **ACTION**: `action:<kick|ban|unban|mute|unmute|add>` (See Moderation Protocol)
    - `action:kick` - Remove from group (can rejoin)
    - `action:ban` - Permanently remove from group (cannot rejoin)
    - `action:unban:@username` - Lift a ban (requires @username)
    - `action:mute` - Silence a user (they can't send messages)
    - `action:unmute` - Restore a muted user's permissions
    - `action:add:@username` - Generate a one-time invite link and DM it to the user (requires @username)
    - Optionally specify target: `action:kick:@username`

**Targeting Syntax**:
To reply to a specific message (if asked to "tell HIM" or in a reply chain), append `target:<message_id>` to the end of any message.
Each message in the chat history has an `(id:XXXXX)` tag - use that number.
If you don't specify a target, your reply goes to the user who messaged you (default).
See the <target_options> section in the context data for the available targets for this message.

Example with targeting:
```
اوكي فهمت
---
target:48291
شكرا على التوضيح
---
react:👍
```

5. **WHEN TO SKIP RESPONDING**
If the user's message does not need a response, you MUST output exactly `[SKIP]` (and nothing else).

**When to SKIP (output `[SKIP]`)**:
1. The user's message is a continuation or split message of a question/topic that you ALREADY fully answered or addressed in your preceding message.
2. The user is asking the same question/statement that has already been answered/addressed in the very recent chat history (even if it was asked by a different user).
3. The message is a simple, casual reaction (e.g. "thanks", "ok", "haha") or minor acknowledgement that does not require a reply because your previous message already concluded/closed the loop.
4. Under **SPECULATIVE INTERACTION**: Output `[SKIP]` unless the user is clearly continuing a conversation with you or direct-replying to you.
5. Under **RANDOM INTERJECTION**: Output `[SKIP]` unless you can naturally and meaningfully contribute to the conversation.

**When NOT to skip**:
1. The user is asking a new question, a follow-up question, or introducing a new topic (e.g., "What is your next exam?", "Where are you going?", "Why?").
2. The user is pointing out a mistake, correcting you, or asking for clarification.
3. The user's message is a direct, meaningful question or prompt that has not been answered yet in the recent context.
4. When in doubt, respond naturally instead of skipping. Never skip a genuine new question.

6. **MODERATION PROTOCOL**
- **TRIGGER**: {PERSONALITY.get("moderation_trigger_conditions", "")}
- **RESTRICTION**: {PERSONALITY.get("moderation_restrictions", "")}
- **ESCALATION**: {PERSONALITY.get("moderation_escalation", "")}

7. **TOOLS & CAPABILITIES**
- You have access to a **Web Search** tool (powered by DuckDuckGo).
- **Guideline**: If the user asks about current events, news, weather, or information that changes (prices, dates, etc.), use the search tool. Do NOT refuse to answer based on "training data cutoff". You are connected to the live internet.
- **CRITICAL – Anti-Hallucination Policy**: You MUST use the Web Search tool whenever the topic involves information that could change over time or that you are not 100% certain about. This includes but is not limited to:
    • Whether a game, movie, show, or software has been released or not, and any details about them.
    • Compatibility of niche software or hardware.
    • Current versions, release dates, pricing, availability, or status of any product.
    • Any factual claim you are not absolutely confident about from your training data.
    • Do not hallucinate URLs. If you need to provide a URL, use the search tool to find it. If you cannot find a URL, say that you don't know.
- **You MUST NOT hallucinate or fabricate information.** If the web search does not return sufficient results to answer confidently, it is perfectly acceptable to say that you don't know the answer or that you couldn't find reliable information. Making up facts is NEVER acceptable.
- **Assume you do NOT know everything.** Default to searching when in doubt rather than guessing.
- You have access to a **Memory Lookup** tool that can search your long-term conversation memory.
- **CRITICAL**: The `<long_term_memory>` section in the context data is a shallow, automatic retrieval based on the user's current message. It is almost NEVER sufficient for recall-type questions. **DO NOT** assume you already have all relevant memories from that section alone. For any question about past conversations, what someone said, events in a specific chat, or any recall/remember request, you MUST use the Memory Lookup tool to search properly with targeted filters (usernames, chat titles, platform, time range, keywords, or combinations). Never claim you don't remember something without using the tool first.
- Remeber, it's okay to say "I don't know" or "I don't remember".

8. **STICKER LIBRARY**
Select stickers from the list matching your current platform (see runtime_metadata in the context data for your platform).
Respect platform capability notes from runtime metadata before using sticker actions.

**Telegram Stickers** (use `sticker:<file_id>`):
{TELEGRAM_STICKER_MAPPINGS}

**WhatsApp Stickers** (use `sticker:wa:<filename>`):
{WHATSAPP_STICKER_MAPPINGS}

9. **CORE RELATIONSHIPS**
{PERSONALITY.get("core_relationships", "")}

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

    This is sent as the user message (contents / user role), NOT as part
    of the system prompt.  Keeping all dynamic data here means the system
    prompt is 100% static and always cache-hits.

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
<runtime_metadata>
Current Date/Time: {timestamp}
{runtime_context}
</runtime_metadata>

<target_options>
{target_instructions}
</target_options>

<style_examples>
{style_examples}
</style_examples>

<social_context>
{social_context_section}
</social_context>

<long_term_memory>
{memory_section}
</long_term_memory>

<chat_history>
{recent_context_section}
</chat_history>

<reply_chain>
{reply_text}
</reply_chain>

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
    Build target instructions for the AI prompt.
    
    The bot sees actual message IDs (id:XXXXX) in the chat history and can
    use them directly with `target:<id>` to reply to any message.
    
    Args:
        msg_id: Current message ID
        sender_name: Name of the message sender
        reply_msg: The reply_to_message object if any
        
    Returns:
        Target instructions string for the system prompt
    """
    parts = [f"- `target:{msg_id}` (Default): Reply to {sender_name} (the user talking to you)"]

    if reply_msg:
        parent_name = "Unknown"
        if hasattr(reply_msg, 'from_user') and reply_msg.from_user:
            parent_name = reply_msg.from_user.first_name or "Unknown"
        
        parts.append(f"- `target:{reply_msg.id}`: Reply to {parent_name} (the message the user replied to)")
        
        if hasattr(reply_msg, 'reply_to_message_id') and reply_msg.reply_to_message_id:
            parts.append(f"- `target:{reply_msg.reply_to_message_id}`: Reply to the message before {parent_name}")
    
    parts.append("- `target:<id>`: Reply to ANY message from the chat history. Use the (id:XXXXX) shown next to each message.")
    
    target_instructions = "\n".join(parts)

    return target_instructions
