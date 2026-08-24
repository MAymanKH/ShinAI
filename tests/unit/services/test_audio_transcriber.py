import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from shin_ai.services import audio_transcriber
from shin_ai.services.audio_transcriber import (
    WhisperProcessManager,
    _file_suffix,
    _transcribe_with_model,
    _TranscriptionCancelled,
)
from shin_ai.services.native_work import NativeWorkLimiter


class _Segment:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeModel:
    def __init__(self) -> None:
        self.path: str | None = None
        self.language = "missing"

    def transcribe(self, path: str, **kwargs):
        self.path = path
        self.language = kwargs["language"]
        assert Path(path).read_bytes() == b"audio"
        return iter([_Segment(" hello "), _Segment(""), _Segment("world")]), SimpleNamespace(
            language="en",
            language_probability=0.98,
        )


def test_file_suffix_normalizes_mime_parameters() -> None:
    assert _file_suffix("audio/webm; codecs=opus") == ".webm"
    assert _file_suffix("unknown/type") == ".ogg"


def test_transcribe_with_model_materializes_segments_and_removes_temp_file() -> None:
    model = _FakeModel()

    text, language, probability = _transcribe_with_model(
        model,
        b"audio",
        "audio/webm",
        "auto",
    )

    assert text == "hello world"
    assert language == "en"
    assert probability == 0.98
    assert model.language is None
    assert model.path is not None
    assert not Path(model.path).exists()


def test_process_manager_terminates_worker_after_cancellation(monkeypatch) -> None:
    class FakeConnection:
        def send(self, _command) -> None:
            return None

        def poll(self, _timeout: float) -> bool:
            return False

    manager = WhisperProcessManager(
        model_name="fake",
        cpu_threads=1,
        language="auto",
        idle_timeout_seconds=60,
        timeout_seconds=60,
    )
    manager._connection = FakeConnection()
    cancel_event = threading.Event()
    cancel_event.set()
    discarded: list[bool] = []
    monkeypatch.setattr(manager, "_start_worker", lambda: None)
    monkeypatch.setattr(
        manager,
        "_discard_worker",
        lambda *, terminate: discarded.append(terminate),
    )

    with pytest.raises(_TranscriptionCancelled):
        manager.transcribe(b"audio", "audio/ogg", cancel_event)

    assert discarded == [True]


def test_audio_downloads_are_bounded_before_inference(monkeypatch, override_settings) -> None:
    async def scenario() -> None:
        active = 0
        peak = 0

        async def oversized_loader() -> bytes:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0)
            active -= 1
            return b"1234"

        override_settings(audio_transcriber, "whisper", max_file_bytes=3)
        # One shared limiter: the point of the test is that both downloads
        # contend for the same single slot.
        limiter = NativeWorkLimiter(1, task_name="test-transcription")
        monkeypatch.setattr(audio_transcriber, "_get_limiter", lambda: limiter)
        results = await asyncio.gather(
            audio_transcriber.transcribe_audio_source(oversized_loader),
            audio_transcriber.transcribe_audio_source(oversized_loader),
        )

        assert results == ["", ""]
        assert peak == 1

    asyncio.run(scenario())


def test_cancelled_process_transcription_signals_worker_and_holds_slot(
    monkeypatch, override_settings
) -> None:
    async def scenario() -> None:
        native_started = asyncio.Event()
        cancellation_seen = asyncio.Event()
        calls = 0

        async def fake_to_thread(_function, _audio, _mime, cancel_event):
            nonlocal calls
            calls += 1
            if calls == 1:
                native_started.set()
                while not cancel_event.is_set():  # noqa: ASYNC110 - models a native thread
                    await asyncio.sleep(0)
                cancellation_seen.set()
            return "transcribed"

        override_settings(audio_transcriber, "whisper", process_isolation=True)
        monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
        limiter = NativeWorkLimiter(1, task_name="test-transcription")
        monkeypatch.setattr(audio_transcriber, "_get_limiter", lambda: limiter)

        first = asyncio.create_task(audio_transcriber.transcribe_audio(b"first"))
        await native_started.wait()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first

        second = asyncio.create_task(audio_transcriber.transcribe_audio(b"second"))
        await cancellation_seen.wait()
        assert limiter.active_count == 1
        assert await second == "transcribed"
        assert calls == 2
        await limiter.close()

    asyncio.run(scenario())
