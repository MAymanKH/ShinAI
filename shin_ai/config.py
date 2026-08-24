"""
ShinAI Configuration Module

All configuration is sourced from config.yaml via the provider registry.
"""
from pathlib import Path

from shin_ai.settings import get_settings

_cfg = get_settings()

# Path Configuration
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SHIN_AI_DATA_DIR = Path(__file__).parent / "data"

# Platform Credentials
TELEGRAM_API_ID = _cfg.platform.telegram_api_id
TELEGRAM_API_HASH = _cfg.platform.telegram_api_hash
TELEGRAM_BOT_TOKEN = _cfg.platform.telegram_bot_token
DISCORD_BOT_TOKEN = _cfg.platform.discord_bot_token

# Platform Enablement
TELEGRAM_ENABLED = _cfg.platform.telegram_enabled
DISCORD_ENABLED = _cfg.platform.discord_enabled
WHATSAPP_ENABLED = _cfg.platform.whatsapp_enabled

# Platform Readiness
TELEGRAM_CONFIGURED = bool(
    TELEGRAM_API_ID and TELEGRAM_API_HASH and TELEGRAM_BOT_TOKEN
)
DISCORD_CONFIGURED = bool(DISCORD_BOT_TOKEN)

# Admin
ADMIN_USER_ID = _cfg.admin_user_id

# General
DEBUG = _cfg.debug
MIN_REPLY_DELAY_SECONDS = _cfg.min_delay_seconds
MAX_REPLY_DELAY_SECONDS = _cfg.max_delay_seconds
RANDOM_TRIGGER_PROBABILITY = _cfg.random_trigger_probability
STYLE_GROUP_ID = _cfg.style_group_id
GROUP_MAX_RESPONSES_PER_WINDOW = _cfg.group_rate_limit_max_responses
GROUP_RATE_LIMIT_WINDOW_SECONDS = _cfg.group_rate_limit_window_seconds
WEB_SEARCH_TIMEOUT_SECONDS = _cfg.web_search_timeout_seconds

# AI Provider Operational Settings
AI_PROVIDER_TIMEOUT_SECONDS = _cfg.ai.timeout_seconds
AI_PROVIDER_MAX_RETRIES = _cfg.ai.max_retries
GLOBAL_AI_TIMEOUT_SECONDS = _cfg.ai.global_timeout_seconds

# Embeddings
EMBEDDING_MODEL = _cfg.embedding_model
EMBEDDING_MAX_CONCURRENCY = _cfg.embedding.max_concurrency
EMBEDDING_BATCH_SIZE = _cfg.embedding.batch_size

# Audio Transcription
WHISPER_MODEL = _cfg.whisper.model
WHISPER_LANGUAGE = _cfg.whisper.language
WHISPER_CPU_THREADS = _cfg.whisper.cpu_threads
WHISPER_MAX_CONCURRENT = _cfg.whisper.max_concurrent_transcriptions
WHISPER_PROCESS_ISOLATION = _cfg.whisper.process_isolation
WHISPER_IDLE_TIMEOUT_SECONDS = _cfg.whisper.idle_timeout_seconds

# Runtime admission and cache limits
MAX_CONCURRENT_INTERACTIONS = _cfg.runtime.max_concurrent_interactions
MAX_PENDING_INTERACTIONS = _cfg.runtime.max_pending_interactions
PER_CHAT_QUEUE_SIZE = _cfg.runtime.per_chat_queue_size
INTERACTION_TTL_SECONDS = _cfg.runtime.interaction_ttl_seconds
SHUTDOWN_GRACE_SECONDS = _cfg.runtime.shutdown_grace_seconds
CONTEXT_MAX_CHATS = _cfg.runtime.context_max_chats
CONTEXT_MESSAGES_PER_CHAT = _cfg.runtime.context_messages_per_chat
CONTEXT_TTL_SECONDS = _cfg.runtime.context_ttl_seconds

# Multi-instance coordination
COORDINATION_BACKEND = _cfg.coordination.backend
COORDINATION_NAMESPACE = _cfg.coordination.namespace
COORDINATION_DATABASE_PATH = _cfg.coordination.database_path
COORDINATION_LEASE_SECONDS = _cfg.coordination.lease_seconds
EVENT_DEDUP_TTL_SECONDS = _cfg.coordination.event_dedup_ttl_seconds
COORDINATION_CLEANUP_INTERVAL_SECONDS = _cfg.coordination.cleanup_interval_seconds

# Logging
LOG_FILE = _cfg.logging.file
LOG_MAX_BYTES = _cfg.logging.max_bytes
LOG_BACKUP_COUNT = _cfg.logging.backup_count
LOG_CONTENT_PREVIEW_CHARS = _cfg.logging.content_preview_chars

# Gemini Models (sourced from config.yaml ai.providers[type=gemini].models)
# Used by gemini_keys.py to populate the model list.
_gemini_cfg = next(
    (provider for provider in _cfg.ai.providers.values() if provider.type == "gemini"),
    None,
)
GEMINI_MODELS: list[str] = list(_gemini_cfg.models) if _gemini_cfg else []
