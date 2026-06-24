"""
Provider Registry

Loads config.yaml, validates it, and exposes typed ProviderConfig objects
and get_provider_chain() respecting failover / round_robin rotation.
"""
from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"

# Data Classes
@dataclass
class ProviderConfig:
    name: str
    type: str                          # "gemini" | "openai"
    base_url: str | None = None        # required for type=openai
    api_key: str | None = None         # required for type=openai
    model: str | None = None           # required for type=openai; optional for gemini
    models: list[str] = field(default_factory=list)  # gemini multi-model list
    concurrency: int | None = None     # optional per-provider semaphore limit


@dataclass
class AIConfig:
    timeout_seconds: float
    max_retries: int
    providers: dict[str, ProviderConfig]  # keyed by name
    primary: str
    fallbacks: list[str]
    rotation: str                          # "failover" | "round_robin"


@dataclass
class WhisperConfig:
    model: str
    language: str
    cpu_threads: int


@dataclass
class PlatformConfig:
    telegram_enabled: bool
    telegram_api_id: str | None
    telegram_api_hash: str | None
    telegram_bot_token: str | None
    discord_enabled: bool
    discord_bot_token: str | None
    whatsapp_enabled: bool


@dataclass
class FirecrawlConfig:
    api_key: str | None = None


@dataclass
class ShinAIConfig:
    platform: PlatformConfig
    admin_user_id: int
    debug: bool
    min_delay_seconds: float
    max_delay_seconds: float
    random_trigger_probability: float
    whisper: WhisperConfig
    embedding_model: str
    style_group_id: str | None
    ai: AIConfig
    firecrawl: FirecrawlConfig


# Internal loader
_config_cache: ShinAIConfig | None = None
_cache_lock = threading.Lock()

# Round-robin counter (atomic via lock)
_rr_counter = itertools.count()
_rr_lock = threading.Lock()


def _load_yaml() -> dict[str, Any]:
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {_CONFIG_PATH}.\n"
            "Copy config.yaml.example to config.yaml and fill in your values."
        )
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_provider(raw: dict) -> ProviderConfig:
    name = raw.get("name") or ""
    ptype = raw.get("type") or ""

    if not name:
        raise ValueError("Each provider entry must have a 'name' field.")
    if ptype not in ("gemini", "openai"):
        raise ValueError(
            f"Provider '{name}': type must be 'gemini' or 'openai', got '{ptype}'."
        )

    if ptype == "openai":
        missing = [k for k in ("base_url", "api_key", "model") if not raw.get(k)]
        if missing:
            raise ValueError(
                f"Provider '{name}' (type=openai) is missing required fields: {missing}"
            )

    concurrency = raw.get("concurrency")
    if concurrency is not None:
        concurrency = int(concurrency)

    return ProviderConfig(
        name=name,
        type=ptype,
        base_url=raw.get("base_url"),
        api_key=str(raw["api_key"]) if raw.get("api_key") is not None else None,
        model=raw.get("model"),
        models=[str(m) for m in raw.get("models", [])],
        concurrency=concurrency,
    )


def _parse_config(raw: dict) -> ShinAIConfig:
    platform_raw = raw.get("platform", {})
    tg = platform_raw.get("telegram", {})
    dc = platform_raw.get("discord", {})
    wa = platform_raw.get("whatsapp", {})

    platform = PlatformConfig(
        telegram_enabled=bool(tg.get("enabled", True)),
        telegram_api_id=str(tg["api_id"]) if tg.get("api_id") else None,
        telegram_api_hash=str(tg["api_hash"]) if tg.get("api_hash") else None,
        telegram_bot_token=str(tg["bot_token"]) if tg.get("bot_token") else None,
        discord_enabled=bool(dc.get("enabled", False)),
        discord_bot_token=str(dc["bot_token"]) if dc.get("bot_token") else None,
        whatsapp_enabled=bool(wa.get("enabled", False)),
    )

    response_raw = raw.get("response", {})
    whisper_raw = raw.get("whisper", {})

    ai_raw = raw.get("ai", {})
    providers_raw = ai_raw.get("providers", [])
    if not providers_raw:
        raise ValueError("config.yaml must define at least one provider under ai.providers.")

    providers: dict[str, ProviderConfig] = {}
    for entry in providers_raw:
        cfg = _parse_provider(entry)
        if cfg.name in providers:
            raise ValueError(f"Duplicate provider name: '{cfg.name}'")
        providers[cfg.name] = cfg

    primary = ai_raw.get("primary") or ""
    if not primary:
        raise ValueError("config.yaml must specify ai.primary.")
    if primary not in providers:
        raise ValueError(
            f"ai.primary '{primary}' is not defined in ai.providers. "
            f"Available: {list(providers.keys())}"
        )

    fallbacks_raw = ai_raw.get("fallbacks") or []
    fallbacks: list[str] = []
    for fb in fallbacks_raw:
        fb = str(fb)
        if fb not in providers:
            raise ValueError(
                f"Fallback provider '{fb}' is not defined in ai.providers. "
                f"Available: {list(providers.keys())}"
            )
        if fb != primary:
            fallbacks.append(fb)

    rotation = str(ai_raw.get("rotation", "failover")).lower()
    if rotation not in ("failover", "round_robin"):
        raise ValueError(
            f"ai.rotation must be 'failover' or 'round_robin', got '{rotation}'."
        )

    ai_cfg = AIConfig(
        timeout_seconds=float(ai_raw.get("timeout_seconds", 60)),
        max_retries=int(ai_raw.get("max_retries", 3)),
        providers=providers,
        primary=primary,
        fallbacks=fallbacks,
        rotation=rotation,
    )

    fc_raw = raw.get("firecrawl", {})
    firecrawl_cfg = FirecrawlConfig(
        api_key=str(fc_raw["api_key"]) if fc_raw.get("api_key") is not None else None
    )

    return ShinAIConfig(
        platform=platform,
        admin_user_id=int(raw.get("admin_user_id", 0)),
        debug=bool(raw.get("debug", False)),
        min_delay_seconds=float(response_raw.get("min_delay_seconds", 0.0)),
        max_delay_seconds=float(response_raw.get("max_delay_seconds", 0.0)),
        random_trigger_probability=float(
            response_raw.get("random_trigger_probability", 0.05)
        ),
        whisper=WhisperConfig(
            model=str(whisper_raw.get("model", "large-v3-turbo")),
            language=str(whisper_raw.get("language", "auto")),
            cpu_threads=int(whisper_raw.get("cpu_threads", 2)),
        ),
        embedding_model=str(raw.get("embedding_model", "intfloat/multilingual-e5-large")),
        style_group_id=str(raw["style_group_id"]) if raw.get("style_group_id") else None,
        ai=ai_cfg,
        firecrawl=firecrawl_cfg,
    )


# Public API
def get_config() -> ShinAIConfig:
    """Return the parsed ShinAIConfig, loading from disk once and caching."""
    global _config_cache
    if _config_cache is not None:
        return _config_cache
    with _cache_lock:
        if _config_cache is None:
            _config_cache = _parse_config(_load_yaml())
    return _config_cache


def reload_config() -> ShinAIConfig:
    """Force a reload from disk (useful after editing config.yaml at runtime)."""
    global _config_cache
    with _cache_lock:
        _config_cache = _parse_config(_load_yaml())
    return _config_cache


def get_primary() -> ProviderConfig:
    """Return the primary provider config."""
    cfg = get_config()
    return cfg.ai.providers[cfg.ai.primary]


def get_provider_chain() -> list[ProviderConfig]:
    """
    Return an ordered list of ProviderConfig to try for a single request.

    - failover:    always [primary, fallback1, fallback2, ...]
    - round_robin: rotate the starting position per-call, cycling through
                   all defined providers (primary + fallbacks).
    """
    cfg = get_config()
    ai = cfg.ai
    all_names = [ai.primary] + ai.fallbacks

    if ai.rotation == "round_robin":
        with _rr_lock:
            idx = next(_rr_counter) % len(all_names)
        # Rotate the list so each call starts at a different provider
        rotated = all_names[idx:] + all_names[:idx]
        return [ai.providers[n] for n in rotated]

    # failover (default)
    return [ai.providers[n] for n in all_names]


def get_first_gemini_provider() -> ProviderConfig | None:
    """Return the first provider with type='gemini', or None."""
    cfg = get_config()
    for p in cfg.ai.providers.values():
        if p.type == "gemini":
            return p
    return None
