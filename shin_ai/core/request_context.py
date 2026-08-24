"""Request-scoped state shared by provider tools."""

import contextvars


web_search_count = contextvars.ContextVar("web_search_count", default=0)
web_search_start_time = contextvars.ContextVar("web_search_start_time", default=0.0)
web_search_exhausted = contextvars.ContextVar("web_search_exhausted", default=False)


def reset_request_context() -> None:
    web_search_count.set(0)
    web_search_start_time.set(0.0)
    web_search_exhausted.set(False)
