"""
Generic OpenAI-Compatible Provider

Handles any provider that exposes an OpenAI-compatible chat completions API
(OpenRouter, Groq, Cerebras, DeepSeek, Together AI, vLLM, Ollama, etc.).
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
from typing import TYPE_CHECKING, Any

from openai import AsyncOpenAI

from shin_ai.providers.tool_loop import run_tool_calling_chat
from shin_ai.utils.logger_config import logger

if TYPE_CHECKING:
    from shin_ai.providers.registry import ProviderConfig


# Per-provider semaphore cache (keyed by provider name)
_semaphores: dict[str, asyncio.Semaphore] = {}
_semaphore_lock = asyncio.Lock()

# Per-provider AsyncOpenAI client cache (keyed by base_url + api_key hash)
_client_cache: dict[str, AsyncOpenAI] = {}
_client_lock = asyncio.Lock()


async def _get_semaphore(cfg: ProviderConfig) -> asyncio.Semaphore | None:
    """Return a per-provider semaphore if concurrency is configured, else None."""
    if cfg.concurrency is None:
        return None
    if cfg.name not in _semaphores:
        async with _semaphore_lock:
            if cfg.name not in _semaphores:
                _semaphores[cfg.name] = asyncio.Semaphore(cfg.concurrency)
    return _semaphores[cfg.name]


async def _get_client(cfg: ProviderConfig) -> AsyncOpenAI:
    """Return a cached AsyncOpenAI client for the provider, or create one."""
    key_digest = hashlib.sha256(cfg.api_key.encode("utf-8")).hexdigest()
    cache_key = f"{cfg.base_url}:{key_digest}"
    if cache_key not in _client_cache:
        async with _client_lock:
            if cache_key not in _client_cache:
                client = AsyncOpenAI(
                    api_key=cfg.api_key,
                    base_url=cfg.base_url,
                )
                _client_cache[cache_key] = client
                logger.debug("Created new AsyncOpenAI client for provider '%s'", cfg.name)
    return _client_cache[cache_key]


async def close_openai_clients() -> None:
    """Close cached HTTP pools and reset process-local provider state."""
    async with _client_lock:
        clients = list({id(client): client for client in _client_cache.values()}.values())
        _client_cache.clear()
    async with _semaphore_lock:
        _semaphores.clear()

    async def _close(client: AsyncOpenAI) -> None:
        try:
            result = client.close()
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.exception("Failed to close an OpenAI-compatible client")

    await asyncio.gather(*(_close(client) for client in clients))


async def openai_provider(
    cfg: ProviderConfig,
    system_prompt: str,
    prompt: str,
    media_list: list[dict] | None = None,
    tool_context: Any = None,
) -> tuple[str, list[dict]]:
    """
    Call any OpenAI-compatible provider defined in config.yaml.

    Args:
        cfg:           The ProviderConfig for this provider.
        system_prompt: The static system prompt.
        prompt:        The user prompt (may include dynamic context).
        media_list:    Optional list of media dicts for vision-capable providers.
        tool_context:  Optional (platform, triggering_msg) tuple that gives
                       context-bound tools (e.g. transcribe_audio) access to
                       the current chat.

    Returns:
        (response_text, pending_actions) where pending_actions is a list of
        action dicts queued by send_reaction / send_sticker / moderate_user
        tool calls during the generation loop.

    Raises:
        Exception on any API or network error (caller handles retry/fallback).
    """
    if not cfg.api_key:
        logger.error("Provider '%s': api_key is not set in config.yaml", cfg.name)
        raise ValueError(f"Provider '{cfg.name}' has no api_key configured.")

    if not cfg.base_url:
        logger.error("Provider '%s': base_url is not set in config.yaml", cfg.name)
        raise ValueError(f"Provider '{cfg.name}' has no base_url configured.")

    if not cfg.model:
        logger.error("Provider '%s': model is not set in config.yaml", cfg.name)
        raise ValueError(f"Provider '{cfg.name}' has no model configured.")

    client = await _get_client(cfg)
    semaphore = await _get_semaphore(cfg)

    async def _call() -> tuple[str, list[dict]]:
        return await run_tool_calling_chat(
            provider_name=cfg.name,
            create_completion=client.chat.completions.create,
            system_prompt=system_prompt,
            prompt=prompt,
            model=cfg.model,
            media_list=media_list,
            tool_context=tool_context,
        )

    if semaphore is not None:
        async with semaphore:
            return await _call()

    return await _call()
