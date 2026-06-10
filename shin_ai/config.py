"""
ShinAI Configuration Module

All configuration is sourced from config.yaml via the provider registry.
"""
from pathlib import Path

from shin_ai.providers.registry import get_config

_cfg = get_config()

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

# AI Provider Operational Settings
AI_PROVIDER_TIMEOUT_SECONDS = _cfg.ai.timeout_seconds
AI_PROVIDER_MAX_RETRIES = _cfg.ai.max_retries

# Embeddings
EMBEDDING_MODEL = _cfg.embedding_model

# Audio Transcription
WHISPER_MODEL = _cfg.whisper.model
WHISPER_LANGUAGE = _cfg.whisper.language
WHISPER_CPU_THREADS = _cfg.whisper.cpu_threads

# Gemini Models (sourced from config.yaml ai.providers[type=gemini].models)
# Used by gemini_keys.py to populate the model list.
from shin_ai.providers.registry import get_first_gemini_provider as _get_gemini

_gemini_cfg = _get_gemini()
GEMINI_MODELS: list[str] = _gemini_cfg.models if _gemini_cfg else []