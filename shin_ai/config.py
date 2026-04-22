"""
ShinAI Configuration Module

Centralizes all environment variable loading and configuration management.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ===========================================
# Path Configuration
# ===========================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SHIN_AI_DATA_DIR = Path(__file__).parent / "data"

# ===========================================
# Telegram API Credentials
# ===========================================
TELEGRAM_API_ID = os.getenv("TELEGRAM_API_ID")
TELEGRAM_API_HASH = os.getenv("TELEGRAM_API_HASH")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
WHATSAPP_SESSION_NAME = "shin_ai_whatsapp"

TELEGRAM_ENABLED = os.getenv("TELEGRAM_ENABLED", "true").lower() == "true"
DISCORD_ENABLED = os.getenv("DISCORD_ENABLED", "false").lower() == "true"
WHATSAPP_ENABLED = os.getenv("WHATSAPP_ENABLED", "false").lower() == "true"

TELEGRAM_CONFIGURED = bool(TELEGRAM_API_ID and TELEGRAM_API_HASH and TELEGRAM_BOT_TOKEN)
DISCORD_CONFIGURED = bool(DISCORD_BOT_TOKEN)

# ===========================================
# Admin Configuration
# ===========================================
ADMIN_USER_ID = int(os.getenv("ADMIN_USER_ID", "0"))

# ===========================================
# General Settings
# ===========================================
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
AI_CHOICE = os.getenv("AI_CHOICE", "gemini")

# ===========================================
# AI Provider Configuration
# ===========================================

AI_PROVIDER_TIMEOUT_SECONDS = float(os.getenv("AI_PROVIDER_TIMEOUT_SECONDS", "60"))
AI_PROVIDER_MAX_RETRIES = int(os.getenv("AI_PROVIDER_MAX_RETRIES", "3"))

# Gemini
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

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

# Audio Transcription (OpenAI Whisper)
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

# ===========================================
# Style Learning Configuration
# ===========================================
STYLE_GROUP_ID = os.getenv("STYLE_GROUP_ID")  # Optional: Group ID to learn communication style from