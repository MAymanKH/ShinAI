from shin_ai.core.request_context import (
    reset_request_context,
    web_search_count,
    web_search_exhausted,
    web_search_start_time,
)


def test_reset_request_context_clears_tool_state() -> None:
    web_search_count.set(3)
    web_search_start_time.set(123.0)
    web_search_exhausted.set(True)

    reset_request_context()

    assert web_search_count.get() == 0
    assert web_search_start_time.get() == 0.0
    assert web_search_exhausted.get() is False
