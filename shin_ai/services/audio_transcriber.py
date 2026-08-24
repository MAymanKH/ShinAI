"""Bounded local audio transcription with optional process isolation."""

from __future__ import annotations

import asyncio
import atexit
import gc
import multiprocessing
import tempfile
import threading
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from shin_ai.config import (
    WHISPER_CPU_THREADS,
    WHISPER_IDLE_TIMEOUT_SECONDS,
    WHISPER_LANGUAGE,
    WHISPER_MAX_CONCURRENT,
    WHISPER_MAX_FILE_BYTES,
    WHISPER_MODEL,
    WHISPER_PROCESS_ISOLATION,
    WHISPER_TIMEOUT_SECONDS,
)
from shin_ai.utils.logger_config import logger

_MIME_SUFFIXES = {
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


def _file_suffix(mime_type: str) -> str:
    normalized = (mime_type or "").split(";", 1)[0].strip().lower()
    return _MIME_SUFFIXES.get(normalized, ".ogg")


def _create_model(model_name: str, cpu_threads: int):
    from faster_whisper import WhisperModel

    return WhisperModel(
        model_name,
        device="cpu",
        compute_type="int8",
        cpu_threads=cpu_threads,
    )


def _transcribe_with_model(
    model: Any,
    audio_bytes: bytes,
    mime_type: str,
    language: str,
) -> tuple[str, str, float]:
    if not audio_bytes:
        return "", "unknown", 0.0

    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=_file_suffix(mime_type), delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name

        language_parameter = None if language.lower() == "auto" else language
        segments, info = model.transcribe(
            temp_path,
            language=language_parameter,
            task="transcribe",
            beam_size=5,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
        return (
            text,
            str(getattr(info, "language", "unknown")),
            float(getattr(info, "language_probability", 0.0)),
        )
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)


def _whisper_worker(
    connection,
    model_name: str,
    cpu_threads: int,
    language: str,
    idle_timeout_seconds: float,
) -> None:
    """Child entry point. It intentionally imports/loads Whisper only here."""
    model = None
    try:
        while connection.poll(idle_timeout_seconds):
            try:
                command = connection.recv()
            except EOFError:
                break
            if not command or command[0] == "shutdown":
                break

            _, request_id, audio_bytes, mime_type = command
            try:
                if model is None:
                    model = _create_model(model_name, cpu_threads)
                text, detected_language, probability = _transcribe_with_model(
                    model,
                    audio_bytes,
                    mime_type,
                    language,
                )
                connection.send(
                    (request_id, True, text, detected_language, probability, None)
                )
            except BaseException as error:
                connection.send(
                    (request_id, False, "", "unknown", 0.0, repr(error))
                )
            finally:
                del command
                del audio_bytes
    finally:
        model = None
        gc.collect()
        connection.close()


class WhisperProcessManager:
    """Owns one serial spawned worker so native memory is fully reclaimable."""

    def __init__(
        self,
        *,
        model_name: str,
        cpu_threads: int,
        language: str,
        idle_timeout_seconds: float,
        timeout_seconds: float,
        process_context=None,
    ) -> None:
        self.model_name = model_name
        self.cpu_threads = cpu_threads
        self.language = language
        self.idle_timeout_seconds = idle_timeout_seconds
        self.timeout_seconds = timeout_seconds
        self._context = process_context or multiprocessing.get_context("spawn")
        self._lock = threading.RLock()
        self._process = None
        self._connection = None

    @property
    def running(self) -> bool:
        return bool(self._process is not None and self._process.is_alive())

    def _discard_worker(self, *, terminate: bool) -> None:
        process = self._process
        connection = self._connection
        self._process = None
        self._connection = None
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass
        if process is not None:
            if terminate and process.is_alive():
                process.terminate()
            process.join(timeout=5.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=2.0)

    def _start_worker(self) -> None:
        if self.running:
            return
        self._discard_worker(terminate=False)
        parent_connection, child_connection = self._context.Pipe(duplex=True)
        process = self._context.Process(
            target=_whisper_worker,
            args=(
                child_connection,
                self.model_name,
                self.cpu_threads,
                self.language,
                self.idle_timeout_seconds,
            ),
            name="shinai-whisper-worker",
            daemon=True,
        )
        process.start()
        child_connection.close()
        self._connection = parent_connection
        self._process = process
        logger.info(
            "Whisper worker started — pid=%s model=%s idle_timeout=%.0fs",
            process.pid,
            self.model_name,
            self.idle_timeout_seconds,
        )

    def transcribe(self, audio_bytes: bytes, mime_type: str) -> str:
        with self._lock:
            for attempt in range(2):
                self._start_worker()
                request_id = uuid.uuid4().hex
                try:
                    self._connection.send(
                        ("transcribe", request_id, audio_bytes, mime_type)
                    )
                    if not self._connection.poll(self.timeout_seconds):
                        self._discard_worker(terminate=True)
                        raise TimeoutError(
                            f"Whisper exceeded {self.timeout_seconds:.0f}s timeout"
                        )
                    response = self._connection.recv()
                except (BrokenPipeError, EOFError, OSError):
                    self._discard_worker(terminate=True)
                    if attempt == 0:
                        continue
                    raise

                response_id, success, text, language, probability, error = response
                if response_id != request_id:
                    self._discard_worker(terminate=True)
                    raise RuntimeError("Whisper worker returned a mismatched response")
                if not success:
                    raise RuntimeError(error or "Whisper worker failed")
                logger.debug(
                    "Whisper transcription complete — lang=%s probability=%.2f chars=%d",
                    language,
                    probability,
                    len(text),
                )
                return text
            return ""

    def close(self) -> None:
        with self._lock:
            if self._connection is not None and self.running:
                try:
                    self._connection.send(("shutdown",))
                except (BrokenPipeError, EOFError, OSError):
                    pass
            self._discard_worker(terminate=False)


_in_process_model = None
_in_process_model_lock = threading.Lock()
_transcription_semaphore = asyncio.Semaphore(
    1 if WHISPER_PROCESS_ISOLATION else WHISPER_MAX_CONCURRENT
)
_process_manager = WhisperProcessManager(
    model_name=WHISPER_MODEL or "large-v3-turbo",
    cpu_threads=WHISPER_CPU_THREADS,
    language=WHISPER_LANGUAGE,
    idle_timeout_seconds=WHISPER_IDLE_TIMEOUT_SECONDS,
    timeout_seconds=WHISPER_TIMEOUT_SECONDS,
)


def _get_in_process_model():
    global _in_process_model
    if _in_process_model is not None:
        return _in_process_model
    with _in_process_model_lock:
        if _in_process_model is None:
            logger.info("Loading in-process Whisper model '%s'...", WHISPER_MODEL)
            _in_process_model = _create_model(WHISPER_MODEL, WHISPER_CPU_THREADS)
    return _in_process_model


def _transcribe_in_process(audio_bytes: bytes, mime_type: str) -> str:
    text, language, probability = _transcribe_with_model(
        _get_in_process_model(),
        audio_bytes,
        mime_type,
        WHISPER_LANGUAGE,
    )
    logger.debug(
        "Whisper transcription complete — lang=%s probability=%.2f chars=%d",
        language,
        probability,
        len(text),
    )
    return text


async def transcribe_audio_source(
    loader: Callable[[], Awaitable[bytes]],
    mime_type: str = "audio/ogg",
) -> str:
    """Download and transcribe inside one slot to bound retained audio bytes."""
    async with _transcription_semaphore:
        try:
            audio_bytes = await loader()
            if not audio_bytes:
                logger.warning("Audio download returned empty data")
                return ""
            if len(audio_bytes) > WHISPER_MAX_FILE_BYTES:
                logger.warning(
                    "Audio rejected — bytes=%d limit=%d",
                    len(audio_bytes),
                    WHISPER_MAX_FILE_BYTES,
                )
                return ""
            if WHISPER_PROCESS_ISOLATION:
                return await asyncio.to_thread(
                    _process_manager.transcribe,
                    audio_bytes,
                    mime_type,
                )
            return await asyncio.to_thread(_transcribe_in_process, audio_bytes, mime_type)
        except Exception as error:
            logger.error("Whisper transcription failed: %s", error, exc_info=True)
            return ""


async def transcribe_audio(audio_bytes: bytes, mime_type: str = "audio/ogg") -> str:
    async def load_bytes() -> bytes:
        return audio_bytes

    return await transcribe_audio_source(load_bytes, mime_type)


async def close_audio_transcriber() -> None:
    global _in_process_model
    await asyncio.to_thread(_process_manager.close)
    with _in_process_model_lock:
        _in_process_model = None
    gc.collect()


atexit.register(_process_manager.close)
