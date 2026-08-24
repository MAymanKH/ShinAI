"""Provider retry, deadline, failover, and media-fallback routing."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from shin_ai.platforms.base import PlatformAdapter
from shin_ai.platforms.models import UnifiedMessage
from shin_ai.providers.registry import ProviderSettings, get_first_gemini_provider, get_provider_chain
from shin_ai.settings import get_settings
from shin_ai.utils.logger_config import logger

ProviderExecutor = Callable[..., Awaitable[tuple[str, list[dict]]]]
MediaDescriber = Callable[[str, list[dict]], Awaitable[str]]


async def call_ai_provider(
    *,
    msg: UnifiedMessage,
    system_prompt: str,
    prompt: str,
    media_list: list[dict],
    original_prompt: str | None = None,
    platform: PlatformAdapter | None = None,
    provider_chain: Sequence[ProviderSettings] | None = None,
    executor: ProviderExecutor | None = None,
    media_describer: MediaDescriber | None = None,
    # Resolved in the body, not here: a default argument is evaluated when the
    # module is imported, which would make importing this module require a
    # readable config.yaml.
    max_retries: int | None = None,
    attempt_timeout_seconds: float | None = None,
    global_timeout_seconds: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[str | None, list[dict]]:
    """Try configured providers under one total deadline."""
    ai = get_settings().ai
    max_retries = ai.max_retries if max_retries is None else max_retries
    attempt_timeout_seconds = (
        ai.timeout_seconds if attempt_timeout_seconds is None else attempt_timeout_seconds
    )
    global_timeout_seconds = (
        ai.global_timeout_seconds if global_timeout_seconds is None else global_timeout_seconds
    )

    chain = list(provider_chain if provider_chain is not None else get_provider_chain())
    execute = executor or execute_provider_once
    describe_media = media_describer or describe_media_with_gemini
    last_error: BaseException | None = None
    media_context: str | None = None
    chain_started = clock()

    for provider in chain:
        if clock() - chain_started > global_timeout_seconds:
            logger.error(
                "Global AI timeout exhausted — budget=%.0fs",
                global_timeout_seconds,
                extra={"event_name": "provider.deadline"},
            )
            break

        provider_prompt = prompt
        provider_media = media_list if media_list else []
        if media_list and provider.type != "gemini":
            if media_context is None:
                media_context = await describe_media(original_prompt or prompt, media_list)
            if media_context:
                provider_prompt = append_media_context(prompt, media_context)
            else:
                logger.warning(
                    "Provider cannot use Gemini media description — provider=%s",
                    provider.name,
                )

        retry_prompt = provider_prompt
        attempts = max(1, max_retries)
        for attempt in range(1, attempts + 1):
            if clock() - chain_started >= global_timeout_seconds:
                logger.error(
                    "Global AI timeout exhausted during retry",
                    extra={"event_name": "provider.deadline"},
                )
                break
            try:
                # No wall-clock wrapper here: a provider that rotates keys and
                # models does that work itself, and cutting the rotation short
                # left later models untried. Each provider applies the attempt
                # timeout to a single model call instead.
                answer, pending_actions = await execute(
                    provider,
                    msg,
                    system_prompt,
                    retry_prompt,
                    provider_media,
                    platform,
                    attempt_timeout_seconds=attempt_timeout_seconds,
                )
                if not isinstance(answer, str) or (not answer.strip() and not pending_actions):
                    raise RuntimeError("AI provider returned empty response")
                logger.info(
                    "AI provider succeeded — provider=%s attempt=%s",
                    provider.name,
                    attempt,
                    extra={"event_name": "provider.succeeded"},
                )
                return answer, pending_actions
            except TimeoutError as error:
                last_error = error
                retry_prompt = provider_prompt
            except Exception as error:
                last_error = error
                error_text = str(error).strip()[:400]
                retry_prompt = (
                    f"{provider_prompt}\n\n[INTERNAL RETRY CONTEXT - NOT A USER MESSAGE]\n"
                    f"Previous attempt failed with {type(error).__name__}: {error_text}\n"
                    "If needed, adapt your response approach to avoid the same failure."
                )
            if attempt >= attempts:
                break

        if len(chain) > 1:
            logger.warning(
                "Provider failed; falling back — provider=%s attempts=%s",
                provider.name,
                attempts,
                extra={"event_name": "provider.fallback"},
            )

    if last_error:
        # The type matters: a bare TimeoutError stringifies to "", which made
        # the old message read "Last error: " and hid what had gone wrong.
        logger.error(
            "All providers failed. Last error: %s: %s",
            type(last_error).__name__,
            last_error,
        )
    return None, []


async def describe_media_with_gemini(user_message: str, media_list: list[dict]) -> str:
    """Describe media once for providers that may not support raw image input."""
    if get_first_gemini_provider() is None:
        logger.warning("No Gemini provider configured for media description")
        return ""

    from shin_ai.providers.gemini import gemini_api

    system_prompt = (
        "You describe attached media for another AI model. Return a concise, "
        "factual description of what is visible and any readable text. "
        "If the user asks about something specific, identify and answer it directly."
    )
    summary_prompt = (
        "Describe the attached media for another AI provider that cannot see images. "
        "Include visually relevant details, readable text, people, objects, actions, and layout.\n\n"
        f"User message/context:\n{user_message}"
    )
    try:
        media_context, _ = await gemini_api(
            system_prompt,
            summary_prompt,
            media_list=media_list,
        )
    except Exception as error:
        logger.error("Gemini media fallback failed: %s", error)
        return ""

    result = media_context.strip() if isinstance(media_context, str) else ""
    if result:
        logger.info("Gemini media description ready — chars=%d", len(result))
    return result


def append_media_context(prompt: str, media_context: str) -> str:
    return (
        f"{prompt}\n\n"
        "[INTERNAL MEDIA CONTEXT - generated by Gemini from attached media, not a user message]\n"
        f"{media_context}"
    )


async def execute_provider_once(
    provider: ProviderSettings,
    msg: UnifiedMessage,
    system_prompt: str,
    prompt: str,
    media_list: list[dict],
    platform: PlatformAdapter | None = None,
    *,
    attempt_timeout_seconds: float | None = None,
) -> tuple[str, list[dict]]:
    tool_context: Any = (platform, msg) if platform is not None else None

    if provider.type == "gemini":
        from shin_ai.providers.gemini import gemini_api

        return await gemini_api(
            system_prompt,
            prompt,
            media_list=media_list,
            tool_context=tool_context,
            attempt_timeout_seconds=attempt_timeout_seconds,
        )
    if provider.type == "openai":
        from shin_ai.providers.openai_compatible import openai_provider

        return await openai_provider(
            provider,
            system_prompt,
            prompt,
            media_list=media_list,
            tool_context=tool_context,
            attempt_timeout_seconds=attempt_timeout_seconds,
        )

    logger.error("Unknown provider type '%s' for provider '%s'", provider.type, provider.name)
    return "", []
