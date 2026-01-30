"""
Response Parser Module

Parses AI responses to extract text, reactions, stickers, actions, and targets.
"""
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ParsedResponse:
    """Structured representation of a parsed AI response."""
    text_content: str = ""
    reaction: Optional[str] = None
    sticker_id: Optional[str] = None
    kick_action: bool = False
    kick_target_username: Optional[str] = None
    target_option: str = "sender"
    
    @property
    def has_content(self) -> bool:
        """Check if the response has any actionable content."""
        return bool(
            self.text_content or 
            self.reaction or 
            self.sticker_id or 
            self.kick_action
        )


def parse_ai_response(answer: Optional[str]) -> ParsedResponse:
    """
    Parse an AI response string to extract structured components.
    
    The AI can return responses in these formats:
    - Plain text
    - react:<emoji>
    - sticker:<file_id>
    - action:kick or action:kick:@username
    - target:<sender|parent|grandparent>
    
    Args:
        answer: Raw response string from the AI
        
    Returns:
        ParsedResponse with extracted components
    """
    result = ParsedResponse()
    
    if not answer or not isinstance(answer, str):
        return result
    
    text_content = answer.strip()
    
    # Extract Action with optional target
    # Matches "action:kick" or "action:kick:@username" or "action:kick:username"
    kick_match = re.search(r"action:kick(?::(@?[\w_]+))?", text_content)
    if kick_match:
        result.kick_action = True
        if kick_match.group(1):
            result.kick_target_username = kick_match.group(1)
        text_content = text_content.replace(kick_match.group(0), "").strip()

    # Extract Target Option
    target_match = re.search(r"target:(\w+)", text_content)
    if target_match:
        result.target_option = target_match.group(1)
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
    return result


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
