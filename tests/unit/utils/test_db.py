from pathlib import Path

from shin_ai.settings import ChromaSettings
from shin_ai.utils import db


def _settings(mode: str) -> ChromaSettings:
    return ChromaSettings(
        mode=mode,
        path=Path("/tmp/test-chroma"),
        host="chroma.internal",
        port=8123,
        ssl=True,
        tenant="tenant-a",
        database="database-a",
    )


def test_create_embedded_chroma_client_uses_local_path(monkeypatch) -> None:
    calls = []
    sentinel = object()
    monkeypatch.setattr(
        db.chromadb,
        "PersistentClient",
        lambda **kwargs: calls.append(kwargs) or sentinel,
    )

    assert db.create_chroma_client(_settings("embedded")) is sentinel
    assert calls == [
        {
            "path": "/tmp/test-chroma",
            "tenant": "tenant-a",
            "database": "database-a",
        }
    ]


def test_create_server_chroma_client_uses_shared_connection(monkeypatch) -> None:
    calls = []
    sentinel = object()
    monkeypatch.setattr(
        db.chromadb,
        "HttpClient",
        lambda **kwargs: calls.append(kwargs) or sentinel,
    )

    assert db.create_chroma_client(_settings("server")) is sentinel
    assert calls == [
        {
            "host": "chroma.internal",
            "port": 8123,
            "ssl": True,
            "tenant": "tenant-a",
            "database": "database-a",
        }
    ]
