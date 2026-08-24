import asyncio
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
