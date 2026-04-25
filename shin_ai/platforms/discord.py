from typing import Optional
import asyncio
import discord
from shin_ai.platforms.models import UnifiedMessage, UnifiedUser, UnifiedChat, UnifiedMedia, UnifiedMessageEntity
from shin_ai.platforms.base import PlatformAdapter
from shin_ai.utils.logger_config import logger

class DiscordPlatform(PlatformAdapter):
    def __init__(self, token: str):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        self.client = discord.Client(intents=intents)
        self.token = token
        self._bot_user = None

    @property
    def platform_name(self) -> str:
        return "discord"

    @property
    def supports_stickers(self) -> bool:
        return False

    async def get_bot_user(self) -> UnifiedUser:
        if self._bot_user:
            return self._bot_user
        me = self.client.user
        if not me:
            raise RuntimeError("Discord client not logged in")
            
        self._bot_user = UnifiedUser(
            id=me.id,
            username=me.name,
            first_name=me.display_name,
            is_self=True
        )
        return self._bot_user

    async def start(self) -> None:
        await self.client.login(self.token)
        asyncio.create_task(self.client.connect())
        await self.client.wait_until_ready()
        logger.info(f"Discord Platform started as {self.client.user}")

    async def stop(self) -> None:
        await self.client.close()
        logger.info("Discord Platform stopped.")

    async def send_message(self, chat_id: int | str, text: str, reply_to_message_id: Optional[int | str] = None) -> int | str:
        channel = self.client.get_channel(int(chat_id)) or await self.client.fetch_channel(int(chat_id))
        
        reference = None
        if reply_to_message_id:
            reference = discord.MessageReference(
                message_id=int(reply_to_message_id),
                channel_id=int(chat_id)
            )
                
        msg = await channel.send(content=text, reference=reference)
        return msg.id

    async def send_sticker(self, chat_id: int | str, sticker_id: str, reply_to_message_id: Optional[int | str] = None) -> int | str:
        # User requested to drop sticker support for discord
        logger.info("Stickers are dropped for discord, doing nothing.")
        return 0

    async def react(self, chat_id: int | str, message_id: int | str, reaction: str) -> None:
        try:
            channel = self.client.get_channel(int(chat_id)) or await self.client.fetch_channel(int(chat_id))
            msg = await channel.fetch_message(int(message_id))
            await msg.add_reaction(reaction)
        except Exception as e:
            logger.error(f"Error reacting on Discord: {e}")

    async def send_chat_action(self, chat_id: int | str, action: str) -> None:
        if action.lower() == "typing":
            channel = self.client.get_channel(int(chat_id)) or await self.client.fetch_channel(int(chat_id))
            await channel.typing()

    async def download_media(self, media: UnifiedMedia) -> bytes:
        if media.native_obj and isinstance(media.native_obj, discord.Attachment):
            return await media.native_obj.read()
        return b""

    async def get_message(self, chat_id: int | str, message_id: int | str) -> Optional[UnifiedMessage]:
        try:
            channel = self.client.get_channel(int(chat_id)) or await self.client.fetch_channel(int(chat_id))
            msg = await channel.fetch_message(int(message_id))
            return self.to_unified_message(msg)
        except discord.NotFound:
            return None
        except Exception as e:
            logger.error(f"Error getting Discord message: {e}")
            return None

    async def get_user_by_username(self, username: str) -> Optional[UnifiedUser]:
        # Discord users aren't easily searchable by just username without a guild
        # Best effort iteration over all cached members
        username = username.lstrip("@").lower()
        for guild in self.client.guilds:
            member = guild.get_member_named(username)
            if member:
                return UnifiedUser(
                    id=member.id,
                    username=member.name,
                    first_name=member.display_name,
                    is_self=(member.id == self.client.user.id)
                )

            for candidate in guild.members:
                if candidate.name.lower() == username or candidate.display_name.lower() == username:
                    return UnifiedUser(
                        id=candidate.id,
                        username=candidate.name,
                        first_name=candidate.display_name,
                        is_self=(candidate.id == self.client.user.id)
                    )
        return None

    async def get_chat_member_status(self, chat_id: int | str, user_id: int | str) -> str:
        channel = self.client.get_channel(int(chat_id))
        if not channel or not hasattr(channel, "guild"):
            return "Unknown"
        
        try:
            member = await channel.guild.fetch_member(int(user_id))
            if channel.guild.owner_id == member.id:
                return "OWNER"
            if member.guild_permissions.administrator:
                return "ADMINISTRATOR"
            return "MEMBER"
        except Exception:
            return "Unknown"

    async def ban_chat_member(self, chat_id: int | str, user_id: int | str) -> None:
        channel = self.client.get_channel(int(chat_id))
        if channel and hasattr(channel, "guild"):
            # Using discord.Object is more reliable as it doesn't require the user to be in cache
            await channel.guild.ban(discord.Object(id=int(user_id)))

    async def kick_chat_member(self, chat_id: int | str, user_id: int | str) -> None:
        channel = self.client.get_channel(int(chat_id))
        if channel and hasattr(channel, "guild"):
            member = channel.guild.get_member(int(user_id))
            if member:
                await channel.guild.kick(member)

    async def unban_chat_member(self, chat_id: int | str, user_id: int | str) -> None:
        channel = self.client.get_channel(int(chat_id))
        if channel and hasattr(channel, "guild"):
            await channel.guild.unban(discord.Object(id=int(user_id)))

    async def restrict_chat_member(self, chat_id: int | str, user_id: int | str, can_send_messages: bool) -> None:
        channel = self.client.get_channel(int(chat_id))
        if channel and hasattr(channel, "guild"):
            member = channel.guild.get_member(int(user_id))
            if member:
                # To mute, we timeout
                if not can_send_messages:
                    from datetime import timedelta
                    await member.timeout(discord.utils.utcnow() + timedelta(days=28))
                else:
                    await member.timeout(None)

    async def create_chat_invite_link(self, chat_id: int | str) -> str:
        channel = self.client.get_channel(int(chat_id))
        if channel and isinstance(channel, discord.abc.GuildChannel):
            invite = await channel.create_invite(max_uses=1)
            return invite.url
        return ""

    def to_unified_message(self, msg: discord.Message) -> UnifiedMessage:
        chat_type = "PRIVATE" if isinstance(msg.channel, discord.DMChannel) else "GROUP"
        
        chat = UnifiedChat(
            id=msg.channel.id,
            title=getattr(msg.channel, "name", None),
            type=chat_type
        )
        
        from_user = None
        if msg.author:
            from_user = UnifiedUser(
                id=msg.author.id,
                username=msg.author.name,
                first_name=msg.author.display_name,
                is_self=(msg.author.id == self.client.user.id) if self.client.user else False
            )
            
        unified_msg = UnifiedMessage(
            platform=self.platform_name,
            id=msg.id,
            chat=chat,
            from_user=from_user,
            text=msg.content,
            date=msg.created_at.timestamp(),
            native_msg=msg
        )
        
        if self.client.user and self.client.user.mentioned_in(msg):
            unified_msg.mentioned = True
            
        if msg.reference and msg.reference.message_id:
            unified_msg.reply_to_message_id = msg.reference.message_id
            
            # Note: We aren't fetching the full reply instantly here to avoid 
            # API spam, but you would normally resolve msg.reference.resolved 
            if msg.reference.resolved and isinstance(msg.reference.resolved, discord.Message):
                unified_msg.reply_to_message = self.to_unified_message(msg.reference.resolved)
                
        # Handle media by scanning attachments and keeping first hit per media slot.
        if msg.attachments:
            for att in msg.attachments:
                content_type_str = str(att.content_type or "").lower()
                filename_str = str(att.filename or "").lower()

                if "image" in content_type_str and not unified_msg.photo:
                    unified_msg.photo = UnifiedMedia(type="PHOTO", id=str(att.id), native_obj=att)
                    continue

                if "video" in content_type_str and not unified_msg.video:
                    unified_msg.video = UnifiedMedia(type="VIDEO", id=str(att.id), native_obj=att)
                    continue

                is_audio = "audio" in content_type_str or filename_str.endswith(
                    ('.ogg', '.mp3', '.wav', '.m4a', '.flac', '.opus', '.webm')
                )
                if is_audio and not unified_msg.audio and not unified_msg.voice:
                    unified_msg.audio = UnifiedMedia(
                        type="AUDIO",
                        id=str(att.id),
                        mime_type=att.content_type,
                        native_obj=att,
                    )
                    continue

                if not unified_msg.document:
                    unified_msg.document = UnifiedMedia(
                        type="DOCUMENT",
                        id=str(att.id),
                        mime_type=att.content_type,
                        native_obj=att,
                    )
                
        # Emulate entities for mentions
        entities = []
        for user in msg.mentions:
            # Just roughly creating mentions
            mention_str = f"<@{user.id}>"
            if mention_str in msg.content:
                offset = msg.content.find(mention_str)
                if offset != -1:
                    ent = UnifiedMessageEntity(
                        type="MENTION", 
                        offset=offset, 
                        length=len(mention_str),
                        user=UnifiedUser(id=user.id, username=user.name, first_name=user.display_name, is_self=False)
                    )
                    entities.append(ent)
        
        unified_msg.entities = entities

        return unified_msg
