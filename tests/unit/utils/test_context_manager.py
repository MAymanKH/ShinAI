from shin_ai.utils.context_manager import ContextBuffer


def _entry(value: int) -> dict:
    return {"value": value}


def test_context_buffer_bounds_messages_and_chat_count() -> None:
    buffer = ContextBuffer(max_chats=2, messages_per_chat=2, ttl_seconds=60)

    buffer.append("a", _entry(1))
    buffer.append("a", _entry(2))
    buffer.append("a", _entry(3))
    buffer.append("b", _entry(4))
    buffer.append("c", _entry(5))

    assert buffer.snapshot("a") == []
    assert buffer.snapshot("b") == [_entry(4)]
    assert buffer.snapshot("c") == [_entry(5)]
    assert len(buffer) == 2


def test_context_buffer_refreshes_lru_on_read() -> None:
    buffer = ContextBuffer(max_chats=2, messages_per_chat=2, ttl_seconds=60)
    buffer.append("a", _entry(1))
    buffer.append("b", _entry(2))

    assert buffer.snapshot("a") == [_entry(1)]
    buffer.append("c", _entry(3))

    assert buffer.snapshot("a") == [_entry(1)]
    assert buffer.snapshot("b") == []


def test_context_buffer_expires_idle_chats() -> None:
    now = [0.0]
    buffer = ContextBuffer(
        max_chats=2,
        messages_per_chat=2,
        ttl_seconds=10,
        clock=lambda: now[0],
    )
    buffer.append("a", _entry(1))
    now[0] = 11

    assert buffer.snapshot("a") == []
    assert len(buffer) == 0
