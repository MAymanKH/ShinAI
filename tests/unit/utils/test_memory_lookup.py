import asyncio
import importlib
import sys
import types


def test_semantic_lookup_converts_candidate_embeddings(monkeypatch) -> None:
    fake_memory = types.ModuleType("shin_ai.utils.memory")
    fake_memory._get_memory_collection = lambda: None
    monkeypatch.setitem(sys.modules, "shin_ai.utils.memory", fake_memory)

    fake_numpy = types.ModuleType("numpy")
    fake_numpy.array = lambda value: ("array", value)
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)

    sys.modules.pop("shin_ai.utils.memory_lookup", None)
    memory_lookup = importlib.import_module("shin_ai.utils.memory_lookup")

    class FakeEmbedding:
        @staticmethod
        def tolist():
            return [1.0, 0.0]

    class FakeEmbeddingService:
        @staticmethod
        async def encode(_text):
            return FakeEmbedding()

    class FakeCollection:
        @staticmethod
        def query(**_kwargs):
            return {
                "documents": [["first", "second"]],
                "distances": [[0.1, 0.2]],
                "embeddings": [[[1.0, 0.0], [0.0, 1.0]]],
                "metadatas": [[{"timestamp": 2}, {"timestamp": 1}]],
            }

    async def call_direct(function, *args, **kwargs):
        return function(*args, **kwargs)

    def select_first(query, candidates, limit):
        assert query == [1.0, 0.0]
        assert candidates == ("array", ([1.0, 0.0], [0.0, 1.0]))
        assert limit == 1
        return [0]

    monkeypatch.setattr(memory_lookup, "get_embedding_service", lambda: FakeEmbeddingService())
    monkeypatch.setattr(memory_lookup, "_get_memory_collection", lambda: FakeCollection())
    monkeypatch.setattr(memory_lookup.asyncio, "to_thread", call_direct)
    monkeypatch.setattr(memory_lookup, "_mmr_indices", select_first)

    result = asyncio.run(memory_lookup._lookup_with_keywords("query", None, 1))

    assert len(result) == 1
    assert result[0]["text"] == "first"
    assert result[0]["timestamp_epoch"] == 2
