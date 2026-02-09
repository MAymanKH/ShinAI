"""
Prompt Builder Module

Constructs the system prompt for AI interactions.
"""
from datetime import datetime
from typing import Optional

from dateutil.tz import tzlocal

from shin_ai.data.loader import PERSONALITY, STICKER_MAPPINGS


def build_system_prompt(
    *,
    style_examples: str,
    social_context_section: str,
    memory_section: str,
    recent_context_section: str,
    runtime_context: str,
    reply_text: str,
    target_instructions: str,
) -> str:
    """
    Build the complete system prompt for AI interaction.
    
    Args:
        style_examples: Examples of the bot's communication style
        social_context_section: Information about group members involved
        memory_section: Relevant past memories
        recent_context_section: Recent chat history
        runtime_context: Current message metadata
        reply_text: The reply chain context
        target_instructions: Available reply target options
        
    Returns:
        Complete system prompt string
    """
    # Get current timestamp with timezone
    now = datetime.now()
    tz_offset = datetime.now(tzlocal()).utcoffset()
    timestamp = f"{now.strftime('%Y-%m-%d %H:%M:%S')} {tz_offset}"
    
    system_prompt = f"""
        ### SYSTEM INSTRUCTIONS

        0. **META CONTEXT**
        - Current Date/Time: {timestamp}
        - You have access to **Google Search** functionality.
        - **Guideline**: If the user asks about current events, news, weather, or information that changes (prices, dates, etc.), use the search tool. Do NOT refuse to answer based on "training data cutoff". You are connected to the live internet.

        1. **IDENTITY & STATUS**
        {PERSONALITY.get("identity", "")}

        2. **BEHAVIORAL PROTOCOLS**
        {PERSONALITY.get("behavioral_protocols", "")}

        3. **INTERACTION STYLE**
        {PERSONALITY.get("interaction_style_personality", "")}

        4. **RESPONSE MECHANICS**
        
        **!!! MULTIPLE MESSAGES (PREFERRED) !!!**:
        ALWAYS prefer sending multiple short messages instead of one long message. Separate them with "---" between messages.
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
        *   **STICKER**: `sticker:<file_id>` (See Mappings Below)
        *   **ACTION**: `action:kick` (See Kicking Protocol)

        **Targeting Syntax**:
        To reply to a specific user (if asked to "tell HIM" or in a reply chain), append `target:<option>` to the end of any message.
        Options: {target_instructions}
        
        Example with targeting:
        ```
        اوكي فهمت
        ---
        target:parent
        شكرا على التوضيح
        ---
        react:👍
        ```
        
        5. **KICKING PROTOCOL**
        - **TRIGGER**: {PERSONALITY.get("kicking_protocol_trigger_conditions", "")}
        - **RESTRICTION**: {PERSONALITY.get("kicking_protocol_restrictions", "")}

        6. **STICKER LIBRARY**
        Select stickers from this list ONLY:
        {STICKER_MAPPINGS}

        ### CONTEXT DATA
        The following XML blocks contain the state of the world. Treat them as read-only data.

        <core_relationships>
        {PERSONALITY.get("core_relationships", "")}
        </core_relationships>

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

        <runtime_metadata>
        {runtime_context}
        </runtime_metadata>

        ### USER INPUT
        <input_message>
        {reply_text}
        </input_message>
    """
    
    return system_prompt


def build_runtime_context(
    *,
    username: Optional[str],
    full_name: str,
    user_id: int,
    user_status: str,
    reply_target_status: str,
    chat_type: str,
    chat_title: Optional[str],
    chat_id: int,
    interaction_type: str,
) -> str:
    """
    Build the runtime context metadata string.
    
    Args:
        username: User's Telegram username
        full_name: User's full name
        user_id: User's Telegram ID
        user_status: User's status in the chat (admin, member, etc.)
        reply_target_status: Status of the user being replied to
        chat_type: Type of chat (group, supergroup, private)
        chat_title: Title of the chat/group
        chat_id: Chat's Telegram ID
        interaction_type: Type of interaction (DIRECT or RANDOM)
        
    Returns:
        Formatted runtime context string
    """
    return f"""
        User username: {username if username else "N/A"}
        User full name: {full_name}
        User ID: {user_id}
        User Status: {user_status}
        Reply Target Status: {reply_target_status}
        Chat type: {chat_type}
        Chat title: {chat_title}
        Chat ID: {chat_id}
        INTERACTION TYPE: {interaction_type}
        """


def build_target_instructions(
    msg_id: int,
    sender_name: str,
    reply_msg: Optional[object] = None,
) -> tuple[dict[str, int], str]:
    """
    Build target instructions and valid targets mapping.
    
    Args:
        msg_id: Current message ID
        sender_name: Name of the message sender
        reply_msg: The reply_to_message object if any
        
    Returns:
        Tuple of (valid_targets dict, target_instructions string)
    """
    valid_targets = {"sender": msg_id}
    target_instructions = f"- `target:sender` (Default): Reply to {sender_name}"

    if reply_msg:
        parent_name = "Unknown"
        if hasattr(reply_msg, 'from_user') and reply_msg.from_user:
            parent_name = reply_msg.from_user.first_name or "Unknown"
        
        valid_targets["parent"] = reply_msg.id
        target_instructions += f"\n            - `target:parent`: Reply to {parent_name} (the message you are replying to)"
        
        if hasattr(reply_msg, 'reply_to_message_id') and reply_msg.reply_to_message_id:
            valid_targets["grandparent"] = reply_msg.reply_to_message_id
            target_instructions += f"\n            - `target:grandparent`: Reply to the user BEFORE {parent_name} (the specific message {parent_name} replied to)"

    return valid_targets, target_instructions
