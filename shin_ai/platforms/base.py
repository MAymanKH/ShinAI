from abc import ABC, abstractmethod
from typing import Optional, List, Tuple
from shin_ai.platforms.models import UnifiedMessage, UnifiedUser, UnifiedMedia

class PlatformAdapter(ABC):
    """Abstract base class for all platform adapters."""
    
    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Name of the platform (e.g. 'telegram', 'discord')."""
        pass
        
    @property
    @abstractmethod
    def supports_stickers(self) -> bool:
        """Whether the platform supports sending native stickers by ID."""
        pass

    @property
    def supports_member_restrictions(self) -> bool:
        """Whether the platform supports per-user mute/unmute operations."""
        return True

    @abstractmethod
    async def get_bot_user(self) -> UnifiedUser:
        """Returns the bot's user object."""
        pass
        
    @abstractmethod
    async def start(self) -> None:
        """Starts the platform client."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stops the platform client and cleans up resources."""
        pass

    @abstractmethod
    async def send_message(self, chat_id: int | str, text: str, reply_to_message_id: Optional[int | str] = None) -> int | str:
        """Sends a text message and returns its ID."""
        pass

    @abstractmethod
    async def send_sticker(self, chat_id: int | str, sticker_id: str, reply_to_message_id: Optional[int | str] = None) -> int | str:
        """Sends a sticker and returns its message ID. If unsupported natively, should handle degradation."""
        pass
        
    @abstractmethod
    async def react(self, chat_id: int | str, message_id: int | str, reaction: str) -> None:
        """Reacts to a message."""
        pass
        
    @abstractmethod
    async def send_chat_action(self, chat_id: int | str, action: str) -> None:
        """Sends a chat action (e.g., 'typing', 'cancel')."""
        pass
        
    @abstractmethod
    async def download_media(self, media: UnifiedMedia) -> bytes:
        """Downloads a media object into memory."""
        pass
        
    @abstractmethod
    async def get_message(self, chat_id: int | str, message_id: int | str) -> Optional[UnifiedMessage]:
        """Fetches a specific message."""
        pass
        
    @abstractmethod
    async def get_user_by_username(self, username: str) -> Optional[UnifiedUser]:
        """Resolves a username to a UnifiedUser."""
        pass
        
    @abstractmethod
    async def get_chat_member_status(self, chat_id: int | str, user_id: int | str) -> str:
        """Gets member status (e.g. 'ADMINISTRATOR', 'MEMBER', 'OWNER'). Returns 'Unknown' if unavailable."""
        pass
        
    @abstractmethod
    async def ban_chat_member(self, chat_id: int | str, user_id: int | str) -> None:
        """Bans a user from the chat."""
        pass

    @abstractmethod
    async def kick_chat_member(self, chat_id: int | str, user_id: int | str) -> None:
        """Kicks a user from the chat (they can rejoin)."""
        pass
        
    @abstractmethod
    async def unban_chat_member(self, chat_id: int | str, user_id: int | str) -> None:
        """Unbans a user."""
        pass

    @abstractmethod
    async def restrict_chat_member(self, chat_id: int | str, user_id: int | str, can_send_messages: bool) -> None:
        """Mutes or unmutes a user."""
        pass
        
    @abstractmethod
    async def create_chat_invite_link(self, chat_id: int | str) -> str:
        """Creates an invite link for the chat."""
        pass
