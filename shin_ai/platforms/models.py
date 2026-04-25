from dataclasses import dataclass, field
from typing import Optional, List, Any

@dataclass
class UnifiedUser:
    id: int | str
    username: Optional[str]
    first_name: str
    is_self: bool
    
    @property
    def full_name(self) -> str:
        return self.first_name

@dataclass
class UnifiedChat:
    id: int | str
    title: Optional[str]
    type: str  # "PRIVATE", "GROUP", "SUPERGROUP", "CHANNEL"

@dataclass
class UnifiedMessageEntity:
    type: str  # "MENTION", "TEXT_MENTION", etc.
    offset: int
    length: int
    user: Optional[UnifiedUser] = None

@dataclass
class UnifiedMedia:
    type: str  # "PHOTO", "STICKER", "VIDEO", "ANIMATION", "VOICE", "AUDIO", "DOCUMENT"
    id: str
    is_animated: bool = False
    is_video: bool = False
    emoji: Optional[str] = None
    mime_type: Optional[str] = None
    
    # Platform-specific native object, used only by adapters (e.g. for downloading)
    native_obj: Any = None

@dataclass
class UnifiedMessage:
    platform: str  # e.g., "telegram", "discord"
    id: int | str
    chat: UnifiedChat
    from_user: Optional[UnifiedUser]
    
    text: Optional[str] = None
    caption: Optional[str] = None
    
    reply_to_message_id: Optional[int | str] = None
    reply_to_message: Optional['UnifiedMessage'] = None
    
    date: float = 0.0
    
    # Media handles
    photo: Optional[UnifiedMedia] = None
    sticker: Optional[UnifiedMedia] = None
    video: Optional[UnifiedMedia] = None
    animation: Optional[UnifiedMedia] = None
    voice: Optional[UnifiedMedia] = None
    audio: Optional[UnifiedMedia] = None
    document: Optional[UnifiedMedia] = None
    
    # Entities
    entities: List[UnifiedMessageEntity] = field(default_factory=list)
    caption_entities: List[UnifiedMessageEntity] = field(default_factory=list)
    
    # Mention flags
    mentioned: bool = False
    is_speculative_reply: bool = False
    
    # Native message object for any platform-specific functions
    native_msg: Any = None