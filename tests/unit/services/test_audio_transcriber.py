import asyncio
from pathlib import Path
from types import SimpleNamespace

from shin_ai.services.audio_transcriber import _file_suffix, _transcribe_with_model
from shin_ai.services import audio_transcriber


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


def test_audio_downloads_are_bounded_before_inference(monkeypatch) -> None:
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

        monkeypatch.setattr(audio_transcriber, "WHISPER_MAX_FILE_BYTES", 3)
        results = await asyncio.gather(
            audio_transcriber.transcribe_audio_source(oversized_loader),
            audio_transcriber.transcribe_audio_source(oversized_loader),
        )

        assert results == ["", ""]
        assert peak == 1

    asyncio.run(scenario())
