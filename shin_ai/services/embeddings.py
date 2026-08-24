"""One lazy, bounded embedding model shared by all semantic features."""

from __future__ import annotations

import asyncio
import gc
import os
import threading
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from shin_ai.config import EMBEDDING_BATCH_SIZE, EMBEDDING_MAX_CONCURRENCY, EMBEDDING_MODEL
from shin_ai.services.native_work import NativeWorkLimiter
from shin_ai.utils.logger_config import logger

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

ModelFactory = Callable[[str], Any]
Offload = Callable[..., Awaitable[Any]]


def _default_model_factory(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


async def _offload_to_thread(function, *args):
    return await asyncio.to_thread(function, *args)


class EmbeddingService:
    """Loads one model and bounds native inference workspace concurrency."""

    def __init__(
        self,
        model_name: str,
        *,
        max_concurrency: int = 1,
        batch_size: int = 16,
        model_factory: ModelFactory = _default_model_factory,
        offload: Offload = _offload_to_thread,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model_factory = model_factory
        self._offload = offload
        self._model = None
        self._model_lock = threading.Lock()
        self._limiter = NativeWorkLimiter(
            max_concurrency,
            task_name="shinai-embedding-inference",
        )

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _get_model(self):
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is None:
                logger.info(
                    "Loading embedding model '%s'...",
                    self.model_name,
                    extra={"event_name": "model.loading"},
                )
                self._model = self._model_factory(self.model_name)
                logger.info(
                    "Embedding model '%s' loaded.",
                    self.model_name,
                    extra={"event_name": "model.ready"},
                )
        return self._model

    def _encode_sync(self, texts: str | Sequence[str]):
        return self._get_model().encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

    async def encode(self, texts: str | Sequence[str]):
        async def run(commit):
            commit()
            return await self._offload(self._encode_sync, texts)

        return await self._limiter.run(run)

    async def close(self) -> None:
        await self._limiter.close()
        with self._model_lock:
            self._model = None
        gc.collect()


_service: EmbeddingService | None = None
_service_lock = threading.Lock()


def get_embedding_service() -> EmbeddingService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = EmbeddingService(
                    EMBEDDING_MODEL,
                    max_concurrency=EMBEDDING_MAX_CONCURRENCY,
                    batch_size=EMBEDDING_BATCH_SIZE,
                )
    return _service


async def close_embedding_service() -> None:
    global _service
    service = _service
    _service = None
    if service is not None:
        await service.close()
