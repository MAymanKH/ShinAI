"""Lazy Chroma client construction for embedded and shared-server modes."""

from __future__ import annotations

import asyncio
import threading

import chromadb

from shin_ai.config import (
    CHROMA_DATABASE,
    CHROMA_HOST,
    CHROMA_MODE,
    CHROMA_PATH,
    CHROMA_PORT,
    CHROMA_SSL,
    CHROMA_TENANT,
)
from shin_ai.settings import ChromaSettings

_client = None
_client_lock = threading.Lock()


def create_chroma_client(settings: ChromaSettings):
    if settings.mode == "server":
        return chromadb.HttpClient(
            host=settings.host,
            port=settings.port,
            ssl=settings.ssl,
            tenant=settings.tenant,
            database=settings.database,
        )
    return chromadb.PersistentClient(
        path=str(settings.path),
        tenant=settings.tenant,
        database=settings.database,
    )


def get_chroma_client():
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                settings = ChromaSettings(
                    mode=CHROMA_MODE,
                    path=CHROMA_PATH,
                    host=CHROMA_HOST,
                    port=CHROMA_PORT,
                    ssl=CHROMA_SSL,
                    tenant=CHROMA_TENANT,
                    database=CHROMA_DATABASE,
                )
                _client = create_chroma_client(settings)
    return _client


async def close_chroma_client() -> None:
    global _client
    with _client_lock:
        current = _client
        _client = None
    if current is not None:
        await asyncio.to_thread(current.close)
