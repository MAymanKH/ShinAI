"""
Response Parser Module

Parses AI responses to extract text, reactions, stickers, actions, and targets.
"""
import re
from dataclasses import dataclass
from typing import Optional


# Valid moderation actions the bot can perform
VALID_MOD_ACTIONS = {"kick", "ban", "unban", "mute", "unmute", "add"}


@dataclass
class ParsedResponse:
    """Structured representation of a parsed AI response."""
    text_content: str = ""
    reaction: Optional[str] = None
    sticker_id: Optional[str] = None
    mod_action: Optional[str] = None  # kick, ban, unban, mute, unmute, add
    mod_target_username: Optional[str] = None
    target_id: Optional[int | str] = None  # Platform message ID to reply to
    
    @property
    def has_content(self) -> bool:
        """Check if the response has any actionable content."""
        return bool(
            self.text_content or 
            self.reaction or 
            self.sticker_id or 
            self.mod_action
        )


def parse_ai_response(answer: Optional[str]) -> list[ParsedResponse]:
    """
    Parse an AI response string to extract structured components.
    
    The AI can return multiple messages separated by "---" or "message:" markers.
    Each message can contain:
    - Plain text
    - react:<emoji>
    - sticker:<file_id>
    - action:<kick|ban|unban|mute|unmute|add> with optional :@username
    - target:<message_id>
    
    Args:
        answer: Raw response string from the AI
        
    Returns:
        List of ParsedResponse objects (one per message)
    """
    if not answer or not isinstance(answer, str):
        return [ParsedResponse()]
    
    # Split by message separators
    # Support both "---" and "message:" or "messageN:" patterns
    raw_messages = re.split(r'(?:^|\n)(?:---|message\d*:)', answer.strip(), flags=re.MULTILINE)
    raw_messages = [m.strip() for m in raw_messages if m.strip()]
    
    # If no separators found, treat as single message
    if not raw_messages:
        raw_messages = [answer.strip()]
    
    results = []
    
    for text_content in raw_messages:
        result = ParsedResponse()
        
        # Extract Moderation Action with optional target username
        # Matches "action:kick", "action:ban:@username", "action:mute", etc.
        action_match = re.search(r"action:(kick|ban|unban|mute|unmute|add)(?::(@?[\w_]+))?", text_content)
        if action_match:
            result.mod_action = action_match.group(1)
            if action_match.group(2):
                result.mod_target_username = action_match.group(2)
            text_content = text_content.replace(action_match.group(0), "").strip()

        # Extract target option. Supports numeric IDs (Telegram/Discord) and
        # string IDs (e.g., WhatsApp stanza IDs).
        target_match = re.search(r"target:([^\s]+)", text_content)
        if target_match:
            raw_target_id = target_match.group(1).strip()
            result.target_id = int(raw_target_id) if raw_target_id.isdigit() else raw_target_id
            text_content = text_content.replace(target_match.group(0), "").strip()

        # Extract React
        react_match = re.search(r"react:(\S+)", text_content)
        if react_match:
            result.reaction = react_match.group(1)
            text_content = text_content.replace(react_match.group(0), "").strip()
        
        # Extract Sticker
        sticker_match = re.search(r"sticker:(\S+)", text_content)
        if sticker_match:
            result.sticker_id = sticker_match.group(1)
            text_content = text_content.replace(sticker_match.group(0), "").strip()
            # Stickers are sent alone - clear any text
            text_content = ""

        result.text_content = text_content
        
        # Only add if there's actual content
        if result.has_content:
            results.append(result)
    
    # Return at least one empty response if nothing parsed
    return results if results else [ParsedResponse()]


def is_ai_response_valid(answer: Optional[str]) -> bool:
    """
    Check if an AI response is valid (non-empty string with content).
    
    Args:
        answer: Response from the AI
        
    Returns:
        True if the response is valid, False otherwise
    """
    return (
        answer is not None and
        isinstance(answer, str) and
        bool(answer.strip())
    )
