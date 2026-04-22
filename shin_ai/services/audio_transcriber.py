"""
Audio Transcription Service

Uses OpenAI's open-source Whisper model to transcribe voice messages
and audio files locally. No paid API required.
"""
import asyncio
import io
import tempfile
import threading
from pathlib import Path
from typing import Optional

from shin_ai.config import WHISPER_MODEL
from shin_ai.utils.logger_config import logger

# ── Lazy-loaded singleton ────────────────────────────────────────────
_model = None
_model_lock = threading.Lock()


def _get_model():
    """Load the Whisper model on first use (thread-safe singleton)."""
    global _model
    if _model is not None:
        return _model

    with _model_lock:
        # Double-check after acquiring lock
        if _model is not None:
            return _model

        import whisper

        model_name = WHISPER_MODEL or "small"
        logger.info(f"Loading Whisper model '{model_name}' (first-time setup may take a moment)...")
        _model = whisper.load_model(model_name)
        logger.info(f"Whisper model '{model_name}' loaded successfully.")
        return _model


def _transcribe_sync(audio_bytes: bytes, mime_type: str) -> str:
    """Synchronous transcription — intended to run in a thread pool."""
    if not audio_bytes:
        return ""

    # Determine a suitable file extension from the MIME type so ffmpeg
    # can identify the container format when loading from the temp file.
    ext_map = {
        "audio/ogg": ".ogg",
        "audio/opus": ".opus",
        "audio/mpeg": ".mp3",
        "audio/mp4": ".m4a",
        "audio/mp3": ".mp3",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/flac": ".flac",
        "audio/aac": ".aac",
        "audio/webm": ".webm",
        "audio/x-m4a": ".m4a",
    }

    # Normalise and look up; fall back to .ogg which is the most common
    # format for voice messages across all three platforms.
    mime_lower = (mime_type or "").split(";")[0].strip().lower()
    suffix = ext_map.get(mime_lower, ".ogg")

    tmp_path: Optional[str] = None
    try:
        # Write to a temp file because Whisper expects a file path
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        model = _get_model()
        result = model.transcribe(
            tmp_path,
            # Let Whisper auto-detect the language for best multilingual support.
            # This works well for Arabic, English, and mixed-language messages.
            fp16=False,  # Use FP32 for CPU compatibility
        )

        text = (result.get("text") or "").strip()
        detected_lang = result.get("language", "unknown")
        logger.info(
            f"Whisper transcription complete: lang={detected_lang}, "
            f"length={len(text)} chars"
        )
        return text
    except Exception as e:
        logger.error(f"Whisper transcription failed: {e}")
        return ""
    finally:
        # Clean up the temp file
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass


async def transcribe_audio(audio_bytes: bytes, mime_type: str = "audio/ogg") -> str:
    """
    Transcribe audio bytes to text using Whisper.

    Runs the (blocking) inference in a thread pool so it doesn't block
    the async event loop.

    Args:
        audio_bytes: Raw audio data (any format ffmpeg can decode).
        mime_type:   MIME type of the audio (used to pick the right
                     container extension for ffmpeg).

    Returns:
        The transcribed text, or an empty string on failure.
    """
    if not audio_bytes:
        return ""

    return await asyncio.to_thread(_transcribe_sync, audio_bytes, mime_type)
