"""
Audio Transcription Service

Uses faster-whisper (CTranslate2 backend) with the large-v3-turbo model
to transcribe voice messages and audio files locally. No paid API required.
"""
import asyncio
import tempfile
import threading
from pathlib import Path
from typing import Optional

from shin_ai.config import WHISPER_MODEL, WHISPER_CPU_THREADS, WHISPER_LANGUAGE
from shin_ai.utils.logger_config import logger

# ── Lazy-loaded singleton ────────────────────────────────────────────
_model = None
_model_lock = threading.Lock()


def _get_model():
    """Load the faster-whisper model on first use (thread-safe singleton)."""
    global _model
    if _model is not None:
        return _model

    with _model_lock:
        # Double-check after acquiring lock
        if _model is not None:
            return _model

        from faster_whisper import WhisperModel

        model_name = WHISPER_MODEL or "large-v3-turbo"
        cpu_threads = WHISPER_CPU_THREADS

        logger.info(
            f"Loading faster-whisper model '{model_name}' "
            f"(device=cpu, compute_type=int8, cpu_threads={cpu_threads})..."
        )
        _model = WhisperModel(
            model_name,
            device="cpu",
            compute_type="int8",
            cpu_threads=cpu_threads,
        )
        logger.info(f"faster-whisper model '{model_name}' loaded successfully.")
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
        # Write to a temp file because faster-whisper expects a file path
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        model = _get_model()
        
        lang_param = None if WHISPER_LANGUAGE.lower() == "auto" else WHISPER_LANGUAGE

        segments, info = model.transcribe(
            tmp_path,
            language=lang_param,
            task="transcribe",
            beam_size=5,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        # faster-whisper returns a generator; materialise it into text.
        text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())

        detected_lang = getattr(info, "language", "unknown")
        lang_prob = getattr(info, "language_probability", 0.0)
        logger.debug(
            "faster-whisper done: lang=%s (prob=%.2f), %d chars",
            detected_lang, lang_prob, len(text),
        )
        return text
    except Exception as e:
        logger.error("faster-whisper transcription failed: %s", e, exc_info=True)
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
    Transcribe audio bytes to text using faster-whisper.

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
