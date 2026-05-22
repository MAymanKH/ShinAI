"""
ShinAI Configuration Module

Centralizes all environment variable loading and configuration management.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Path Configuration
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SHIN_AI_DATA_DIR = Path(__file__).parent / "data"

# Telegram API Credentials
TELEGRAM_API_ID = os.getenv("TELEGRAM_API_ID")
TELEGRAM_API_HASH = os.getenv("TELEGRAM_API_HASH")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

# Platform Enablement
TELEGRAM_ENABLED = os.getenv("TELEGRAM_ENABLED", "true").lower() == "true"
DISCORD_ENABLED = os.getenv("DISCORD_ENABLED", "false").lower() == "true"
WHATSAPP_ENABLED = os.getenv("WHATSAPP_ENABLED", "false").lower() == "true"

# Platform Configuration
TELEGRAM_CONFIGURED = bool(TELEGRAM_API_ID and TELEGRAM_API_HASH and TELEGRAM_BOT_TOKEN)
DISCORD_CONFIGURED = bool(DISCORD_BOT_TOKEN)

# Admin Configuration
ADMIN_USER_ID = int(os.getenv("ADMIN_USER_ID", "0"))

# General Settings
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
AI_CHOICE = os.getenv("AI_CHOICE", "gemini")
MIN_REPLY_DELAY_SECONDS = float(os.getenv("MIN_REPLY_DELAY_SECONDS", "0"))
MAX_REPLY_DELAY_SECONDS = float(os.getenv("MAX_REPLY_DELAY_SECONDS", "0"))
RANDOM_TRIGGER_PROBABILITY = float(os.getenv("RANDOM_TRIGGER_PROBABILITY", "0.05"))
STYLE_GROUP_ID = os.getenv("STYLE_GROUP_ID")

# AI Provider Configuration
AI_PROVIDER_TIMEOUT_SECONDS = float(os.getenv("AI_PROVIDER_TIMEOUT_SECONDS", "60"))
AI_PROVIDER_MAX_RETRIES = int(os.getenv("AI_PROVIDER_MAX_RETRIES", "3"))
AI_FALLBACK_PROVIDERS = [
	p.strip()
	for p in os.getenv("AI_FALLBACK_PROVIDERS", "").split(",")
	if p.strip()
]

# OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL")

# Local LLM
LOCAL_MODEL = os.getenv("LOCAL_MODEL")

# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL")

# Cerebras
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")

# Audio Transcription
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3-turbo")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "auto")
WHISPER_CPU_THREADS = int(os.getenv("WHISPER_CPU_THREADS", "2"))