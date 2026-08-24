import pytest

from shin_ai.providers.gemini_errors import GeminiFailureKind, classify_gemini_error


class APIError(Exception):
    def __init__(self, status_code: int, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (APIError(429, "resource exhausted"), GeminiFailureKind.RATE_LIMIT),
        (APIError(503, "service unavailable"), GeminiFailureKind.TRANSIENT),
        (APIError(401, "invalid key"), GeminiFailureKind.AUTHENTICATION),
        (APIError(400, "bad request"), GeminiFailureKind.INVALID_REQUEST),
        (TimeoutError(), GeminiFailureKind.TIMEOUT),
        (RuntimeError("unknown transport problem"), GeminiFailureKind.UNKNOWN),
    ],
)
def test_classify_gemini_error(error, kind) -> None:
    assert classify_gemini_error(error).kind == kind


def test_classifier_uses_retry_after() -> None:
    failure = classify_gemini_error(APIError(429, "quota", retry_after=17.5))

    assert failure.retry_after_seconds == 17.5
