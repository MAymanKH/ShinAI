import gc
import tracemalloc

import pytest

from shin_ai.utils.context_manager import ContextBuffer


@pytest.mark.memory
def test_oversized_chat_messages_are_not_retained_at_full_size() -> None:
    buffer = ContextBuffer(
        max_chats=128,
        messages_per_chat=4,
        ttl_seconds=60,
        max_text_chars=1_000,
    )
    oversized = "x" * 100_000
    tracemalloc.start()
    baseline, _ = tracemalloc.get_traced_memory()

    for index in range(1_000):
        buffer.append(f"chat-{index}", {"text": f"{oversized}{index}"})

    gc.collect()
    retained, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert len(buffer) == 128
    assert retained - baseline < 2_000_000
    assert peak - baseline < 4_000_000
