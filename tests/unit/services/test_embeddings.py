import asyncio

import pytest

from shin_ai.services.embeddings import EmbeddingService


class _FakeModel:
    def __init__(self) -> None:
        self.active = 0
        self.peak = 0
        self.calls = 0

    def encode(self, texts, **_kwargs):
        self.calls += 1
        return texts


def test_embedding_service_loads_once_and_bounds_inference() -> None:
    async def scenario() -> None:
        model = _FakeModel()
        factory_calls = 0

        def factory(_name: str):
            nonlocal factory_calls
            factory_calls += 1
            return model

        async def offload(function, *args):
            model.active += 1
            model.peak = max(model.peak, model.active)
            await asyncio.sleep(0.01)
            result = function(*args)
            model.active -= 1
            return result

        service = EmbeddingService(
            "fake",
            max_concurrency=2,
            batch_size=4,
            model_factory=factory,
            offload=offload,
        )
        results = await asyncio.gather(*(service.encode(f"text-{i}") for i in range(8)))

        assert results == [f"text-{i}" for i in range(8)]
        assert factory_calls == 1
        assert model.calls == 8
        assert model.peak == 2
        assert service.loaded

        await service.close()
        assert not service.loaded

    asyncio.run(scenario())


def test_cancelled_inference_keeps_its_slot_until_native_work_finishes() -> None:
    async def scenario() -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        calls = 0

        async def offload(function, *args):
            nonlocal calls
            calls += 1
            if calls == 1:
                started.set()
                await release.wait()
            return function(*args)

        service = EmbeddingService(
            "fake",
            max_concurrency=1,
            model_factory=lambda _name: _FakeModel(),
            offload=offload,
        )
        first = asyncio.create_task(service.encode("first"))
        await started.wait()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first

        second = asyncio.create_task(service.encode("second"))
        await asyncio.sleep(0)
        assert calls == 1
        assert service._limiter.active_count == 1

        release.set()
        assert await second == "second"
        await service.close()

    asyncio.run(scenario())
