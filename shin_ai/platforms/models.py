from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class UnifiedUser:
    id: int | str
    username: str | None
    first_name: str
    is_self: bool
    
    @property
    def full_name(self) -> str:
        return self.first_name

@dataclass
class UnifiedChat:
    id: int | str
    title: str | None
    type: str  # "PRIVATE", "GROUP", "SUPERGROUP", "CHANNEL"

@dataclass
class UnifiedMessageEntity:
    type: str  # "MENTION", "TEXT_MENTION", etc.
    offset: int
    length: int
    user: UnifiedUser | None = None

@dataclass
class UnifiedMedia:
    type: str  # "PHOTO", "STICKER", "VIDEO", "ANIMATION", "VOICE", "AUDIO", "DOCUMENT"
    id: str
    is_animated: bool = False
    is_video: bool = False
    emoji: str | None = None
    mime_type: str | None = None
    
    # Platform-specific native object, used only by adapters (e.g. for downloading)
    native_obj: Any = None

@dataclass
class UnifiedMessage:
    platform: str  # e.g., "telegram", "discord"
    id: int | str
    chat: UnifiedChat
    from_user: UnifiedUser | None
    
    text: str | None = None
    caption: str | None = None
    
    reply_to_message_id: int | str | None = None
    reply_to_message: Optional['UnifiedMessage'] = None
    
    date: float = 0.0
    
    # Media handles
    photo: UnifiedMedia | None = None
    sticker: UnifiedMedia | None = None
    video: UnifiedMedia | None = None
    animation: UnifiedMedia | None = None
    voice: UnifiedMedia | None = None
    audio: UnifiedMedia | None = None
    document: UnifiedMedia | None = None
    
    # Entities
    entities: list[UnifiedMessageEntity] = field(default_factory=list)
    caption_entities: list[UnifiedMessageEntity] = field(default_factory=list)
    
    # Mention flags
    mentioned: bool = False
    is_speculative_reply: bool = False
    
    # Native message object for any platform-specific functions
    native_msg: Any = None