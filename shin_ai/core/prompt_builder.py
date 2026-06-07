"""
Prompt Builder Module

Constructs the system prompt for AI interactions.

The prompt is split into a STATIC PREFIX (personality, rules, mechanics,
sticker library) and a DYNAMIC SUFFIX (timestamp, context, runtime data).
This maximises API prompt-caching (Gemini implicit caching, OpenAI prefix
caching) because the static prefix is byte-identical across every request.
"""
from datetime import datetime
from typing import Optional

from dateutil.tz import tzlocal

from shin_ai.data.loader import (
    PERSONALITY,
    TELEGRAM_STICKER_MAPPINGS,
    WHATSAPP_STICKER_MAPPINGS,
)


# ── Static prefix (computed once at import time) ────────────────────────
# This block never changes at runtime, so every API call shares the exact
# same leading bytes → maximum cache hits.

_STATIC_SYSTEM_PREFIX = f"""\
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
See the <target_options> section below for the available targets for this message.

Example with targeting:
```
اوكي فهمت
---
target:48291
شكرا على التوضيح
---
react:👍
```

5. **MODERATION PROTOCOL**
- **TRIGGER**: {PERSONALITY.get("moderation_trigger_conditions", "")}
- **RESTRICTION**: {PERSONALITY.get("moderation_restrictions", "")}
- **ESCALATION**: {PERSONALITY.get("moderation_escalation", "")}

6. **TOOLS & CAPABILITIES**
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
- **CRITICAL**: The `<long_term_memory>` section below is a shallow, automatic retrieval based on the user's current message. It is almost NEVER sufficient for recall-type questions. **DO NOT** assume you already have all relevant memories from that section alone. For any question about past conversations, what someone said, events in a specific chat, or any recall/remember request, you MUST use the Memory Lookup tool to search properly with targeted filters (usernames, chat titles, platform, time range, keywords, or combinations). Never claim you don't remember something without using the tool first.
- Remeber, it's okay to say "I don't know" or "I don't remember".

7. **STICKER LIBRARY**
Select stickers from the list matching your current platform (see runtime_metadata for your platform).
Respect platform capability notes from runtime metadata before using sticker actions.

**Telegram Stickers** (use `sticker:<file_id>`):
{TELEGRAM_STICKER_MAPPINGS}

**WhatsApp Stickers** (use `sticker:wa:<filename>`):
{WHATSAPP_STICKER_MAPPINGS}

8. **CORE RELATIONSHIPS**
{PERSONALITY.get("core_relationships", "")}
"""


def build_system_prompt(
    *,
    style_examples: str,
    social_context_section: str,
    memory_section: str,
    recent_context_section: str,
    runtime_context: str,
    reply_text: str,
    target_instructions: str,
    sticker_mappings: str,
) -> str:
    """
    Build the complete system prompt for AI interaction.

    The prompt is structured as:
      STATIC PREFIX  – personality, rules, mechanics, sticker library
                       (byte-identical across requests → cached by API)
      DYNAMIC SUFFIX – timestamp, context data, runtime metadata, user input
                       (changes per request)

    Args:
        style_examples: Examples of the bot's communication style
        social_context_section: Information about group members involved
        memory_section: Relevant past memories
        recent_context_section: Recent chat history
        runtime_context: Current message metadata
        reply_text: The reply chain context
        target_instructions: Available reply target options
        sticker_mappings: (kept for interface compat, no longer used)

    Returns:
        Complete system prompt string
    """
    # Get current timestamp with timezone
    now = datetime.now()
    tz_offset = datetime.now(tzlocal()).utcoffset()
    timestamp = f"{now.strftime('%Y-%m-%d %H:%M:%S')} UTC+{tz_offset}"

    # Dynamic suffix — everything that changes per request
    dynamic_suffix = f"""
### CONTEXT DATA
The following sections contain the state of the world for THIS request. Treat them as read-only data.

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

### USER INPUT
<input_message>
{reply_text}
</input_message>
"""

    return _STATIC_SYSTEM_PREFIX + dynamic_suffix


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
