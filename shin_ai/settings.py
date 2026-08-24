"""Typed, side-effect-free application settings.

The YAML parser lives outside the provider registry so application code and tests
can load configuration without importing provider rotation state or platform SDKs.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass(frozen=True, slots=True)
class ProviderSettings:
    name: str
    type: str
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    models: tuple[str, ...] = ()
    concurrency: int | None = None


@dataclass(frozen=True, slots=True)
class AISettings:
    timeout_seconds: float
    max_retries: int
    global_timeout_seconds: float
    providers: dict[str, ProviderSettings]
    primary: str
    fallbacks: tuple[str, ...]
    rotation: str


@dataclass(frozen=True, slots=True)
class WhisperSettings:
    model: str
    language: str
    cpu_threads: int
    max_concurrent_transcriptions: int
    process_isolation: bool
    idle_timeout_seconds: float
    timeout_seconds: float
    max_file_bytes: int


@dataclass(frozen=True, slots=True)
class EmbeddingSettings:
    model: str
    max_concurrency: int
    batch_size: int


@dataclass(frozen=True, slots=True)
class ChromaSettings:
    path: Path


@dataclass(frozen=True, slots=True)
class PlatformSettings:
    telegram_enabled: bool
    telegram_api_id: str | None
    telegram_api_hash: str | None
    telegram_bot_token: str | None
    discord_enabled: bool
    discord_bot_token: str | None
    whatsapp_enabled: bool


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    max_concurrent_interactions: int
    max_pending_interactions: int
    per_chat_queue_size: int
    interaction_ttl_seconds: float
    shutdown_grace_seconds: float
    typing_action_timeout_seconds: float
    context_max_chats: int
    context_messages_per_chat: int
    context_message_chars: int
    context_ttl_seconds: float
    platform_message_cache_size: int
    media_max_items: int
    media_max_file_bytes: int
    media_max_total_bytes: int


@dataclass(frozen=True, slots=True)
class CoordinationSettings:
    backend: str
    namespace: str
    database_path: Path
    lease_seconds: float
    event_dedup_ttl_seconds: float
    reply_state_ttl_seconds: float
    cleanup_interval_seconds: float


@dataclass(frozen=True, slots=True)
class LoggingSettings:
    debug: bool
    file: Path | None
    max_bytes: int
    backup_count: int
    content_preview_chars: int


@dataclass(frozen=True, slots=True)
class FirecrawlSettings:
    api_key: str | None = None


@dataclass(frozen=True, slots=True)
class ShinAISettings:
    platform: PlatformSettings
    admin_user_id: int
    logging: LoggingSettings
    min_delay_seconds: float
    max_delay_seconds: float
    random_trigger_probability: float
    group_rate_limit_max_responses: int
    group_rate_limit_window_seconds: float
    web_search_timeout_seconds: float
    whisper: WhisperSettings
    embedding: EmbeddingSettings
    chroma: ChromaSettings
    runtime: RuntimeSettings
    coordination: CoordinationSettings
    style_group_id: str | None
    ai: AISettings
    firecrawl: FirecrawlSettings

    @property
    def debug(self) -> bool:
        return self.logging.debug

    @property
    def embedding_model(self) -> str:
        return self.embedding.model


_settings_cache: ShinAISettings | None = None
_settings_lock = threading.Lock()


def _positive_int(value: Any, *, name: str, default: int) -> int:
    parsed = int(default if value is None else value)
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return parsed


def _positive_float(value: Any, *, name: str, default: float) -> float:
    parsed = float(default if value is None else value)
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return parsed


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_provider(raw: dict[str, Any]) -> ProviderSettings:
    name = str(raw.get("name") or "").strip()
    provider_type = str(raw.get("type") or "").strip().lower()
    if not name:
        raise ValueError("Each provider entry must have a 'name' field.")
    if provider_type not in {"gemini", "openai"}:
        raise ValueError(
            f"Provider '{name}': type must be 'gemini' or 'openai', got '{provider_type}'."
        )
    if provider_type == "openai":
        missing = [key for key in ("base_url", "api_key", "model") if not raw.get(key)]
        if missing:
            raise ValueError(
                f"Provider '{name}' (type=openai) is missing required fields: {missing}"
            )

    models = tuple(str(model).strip() for model in raw.get("models", ()) if str(model).strip())
    if provider_type == "gemini" and not models:
        raise ValueError(f"Provider '{name}' (type=gemini) must define at least one model.")

    concurrency = raw.get("concurrency")
    if concurrency is not None:
        concurrency = _positive_int(
            concurrency,
            name=f"ai.providers[{name}].concurrency",
            default=1,
        )
    return ProviderSettings(
        name=name,
        type=provider_type,
        base_url=_optional_string(raw.get("base_url")),
        api_key=_optional_string(raw.get("api_key")),
        model=_optional_string(raw.get("model")),
        models=models,
        concurrency=concurrency,
    )


def parse_settings(raw: dict[str, Any], *, project_root: Path = PROJECT_ROOT) -> ShinAISettings:
    """Validate a decoded YAML mapping and return immutable settings."""
    platform_raw = raw.get("platform") or {}
    telegram = platform_raw.get("telegram") or {}
    discord = platform_raw.get("discord") or {}
    whatsapp = platform_raw.get("whatsapp") or {}
    platform = PlatformSettings(
        telegram_enabled=bool(telegram.get("enabled", True)),
        telegram_api_id=_optional_string(telegram.get("api_id")),
        telegram_api_hash=_optional_string(telegram.get("api_hash")),
        telegram_bot_token=_optional_string(telegram.get("bot_token")),
        discord_enabled=bool(discord.get("enabled", False)),
        discord_bot_token=_optional_string(discord.get("bot_token")),
        whatsapp_enabled=bool(whatsapp.get("enabled", False)),
    )

    response = raw.get("response") or {}
    min_delay = float(response.get("min_delay_seconds", 0.0))
    max_delay = float(response.get("max_delay_seconds", 0.0))
    if min_delay < 0 or max_delay < min_delay:
        raise ValueError(
            "response delays must satisfy 0 <= min_delay_seconds <= max_delay_seconds."
        )
    probability = float(response.get("random_trigger_probability", 0.05))
    if not 0.0 <= probability <= 1.0:
        raise ValueError("response.random_trigger_probability must be between 0 and 1.")

    logging_raw = raw.get("logging") or {}
    log_file_value = logging_raw.get("file", "shinai_bot.log")
    log_file = None if log_file_value in (None, "", False) else Path(str(log_file_value))
    if log_file is not None and not log_file.is_absolute():
        log_file = project_root / log_file
    logging_settings = LoggingSettings(
        debug=bool(logging_raw.get("debug", raw.get("debug", False))),
        file=log_file,
        max_bytes=_positive_int(
            logging_raw.get("max_bytes"), name="logging.max_bytes", default=25_000_000
        ),
        backup_count=_positive_int(
            logging_raw.get("backup_count"), name="logging.backup_count", default=5
        ),
        content_preview_chars=max(0, int(logging_raw.get("content_preview_chars", 120))),
    )

    runtime_raw = raw.get("runtime") or {}
    context_raw = runtime_raw.get("context") or {}
    runtime = RuntimeSettings(
        max_concurrent_interactions=_positive_int(
            runtime_raw.get("max_concurrent_interactions"),
            name="runtime.max_concurrent_interactions",
            default=24,
        ),
        max_pending_interactions=_positive_int(
            runtime_raw.get("max_pending_interactions"),
            name="runtime.max_pending_interactions",
            default=256,
        ),
        per_chat_queue_size=_positive_int(
            runtime_raw.get("per_chat_queue_size"),
            name="runtime.per_chat_queue_size",
            default=20,
        ),
        interaction_ttl_seconds=_positive_float(
            runtime_raw.get("interaction_ttl_seconds"),
            name="runtime.interaction_ttl_seconds",
            default=300.0,
        ),
        shutdown_grace_seconds=_positive_float(
            runtime_raw.get("shutdown_grace_seconds"),
            name="runtime.shutdown_grace_seconds",
            default=30.0,
        ),
        typing_action_timeout_seconds=_positive_float(
            runtime_raw.get("typing_action_timeout_seconds"),
            name="runtime.typing_action_timeout_seconds",
            default=2.0,
        ),
        context_max_chats=_positive_int(
            context_raw.get("max_chats"), name="runtime.context.max_chats", default=2_000
        ),
        context_messages_per_chat=_positive_int(
            context_raw.get("messages_per_chat"),
            name="runtime.context.messages_per_chat",
            default=50,
        ),
        context_message_chars=_positive_int(
            context_raw.get("message_chars"),
            name="runtime.context.message_chars",
            default=4_000,
        ),
        context_ttl_seconds=_positive_float(
            context_raw.get("ttl_seconds"),
            name="runtime.context.ttl_seconds",
            default=7_200.0,
        ),
        platform_message_cache_size=_positive_int(
            runtime_raw.get("platform_message_cache_size"),
            name="runtime.platform_message_cache_size",
            default=500,
        ),
        media_max_items=_positive_int(
            runtime_raw.get("media_max_items"),
            name="runtime.media_max_items",
            default=5,
        ),
        media_max_file_bytes=_positive_int(
            runtime_raw.get("media_max_file_bytes"),
            name="runtime.media_max_file_bytes",
            default=10_000_000,
        ),
        media_max_total_bytes=_positive_int(
            runtime_raw.get("media_max_total_bytes"),
            name="runtime.media_max_total_bytes",
            default=20_000_000,
        ),
    )
    if runtime.media_max_total_bytes < runtime.media_max_file_bytes:
        raise ValueError(
            "runtime.media_max_total_bytes must be greater than or equal to "
            "runtime.media_max_file_bytes."
        )

    coordination_raw = raw.get("coordination") or {}
    coordination_backend = str(coordination_raw.get("backend", "sqlite")).strip().lower()
    if coordination_backend not in {"sqlite", "memory"}:
        raise ValueError("coordination.backend must be 'sqlite' or 'memory'.")
    namespace = str(coordination_raw.get("namespace", "shinai")).strip()
    if not namespace:
        raise ValueError("coordination.namespace cannot be empty.")
    database_path = Path(
        str(coordination_raw.get("database_path", "data/coordination.sqlite3"))
    )
    if not database_path.is_absolute():
        database_path = project_root / database_path
    coordination = CoordinationSettings(
        backend=coordination_backend,
        namespace=namespace,
        database_path=database_path,
        lease_seconds=_positive_float(
            coordination_raw.get("lease_seconds"),
            name="coordination.lease_seconds",
            default=240.0,
        ),
        event_dedup_ttl_seconds=_positive_float(
            coordination_raw.get("event_dedup_ttl_seconds"),
            name="coordination.event_dedup_ttl_seconds",
            default=86_400.0,
        ),
        reply_state_ttl_seconds=_positive_float(
            coordination_raw.get("reply_state_ttl_seconds"),
            name="coordination.reply_state_ttl_seconds",
            default=86_400.0,
        ),
        cleanup_interval_seconds=_positive_float(
            coordination_raw.get("cleanup_interval_seconds"),
            name="coordination.cleanup_interval_seconds",
            default=300.0,
        ),
    )

    embedding_raw = raw.get("embedding") or {}
    embedding = EmbeddingSettings(
        model=str(
            embedding_raw.get(
                "model", raw.get("embedding_model", "intfloat/multilingual-e5-large")
            )
        ),
        max_concurrency=_positive_int(
            embedding_raw.get("max_concurrency"),
            name="embedding.max_concurrency",
            default=1,
        ),
        batch_size=_positive_int(
            embedding_raw.get("batch_size"), name="embedding.batch_size", default=16
        ),
    )

    chroma_raw = raw.get("chroma") or {}
    chroma_path = Path(str(chroma_raw.get("path", "chroma_db")))
    if not chroma_path.is_absolute():
        chroma_path = project_root / chroma_path
    chroma = ChromaSettings(path=chroma_path)

    whisper_raw = raw.get("whisper") or {}
    whisper = WhisperSettings(
        model=str(whisper_raw.get("model", "large-v3-turbo")),
        language=str(whisper_raw.get("language", "auto")),
        cpu_threads=_positive_int(
            whisper_raw.get("cpu_threads"), name="whisper.cpu_threads", default=2
        ),
        max_concurrent_transcriptions=_positive_int(
            whisper_raw.get("max_concurrent_transcriptions"),
            name="whisper.max_concurrent_transcriptions",
            default=1,
        ),
        process_isolation=bool(whisper_raw.get("process_isolation", True)),
        idle_timeout_seconds=_positive_float(
            whisper_raw.get("idle_timeout_seconds"),
            name="whisper.idle_timeout_seconds",
            default=600.0,
        ),
        timeout_seconds=_positive_float(
            whisper_raw.get("timeout_seconds"),
            name="whisper.timeout_seconds",
            default=180.0,
        ),
        max_file_bytes=_positive_int(
            whisper_raw.get("max_file_bytes"),
            name="whisper.max_file_bytes",
            default=25_000_000,
        ),
    )

    ai_raw = raw.get("ai") or {}
    providers_raw = ai_raw.get("providers") or []
    if not providers_raw:
        raise ValueError("config.yaml must define at least one provider under ai.providers.")
    providers: dict[str, ProviderSettings] = {}
    for entry in providers_raw:
        provider = _parse_provider(entry)
        if provider.name in providers:
            raise ValueError(f"Duplicate provider name: '{provider.name}'")
        providers[provider.name] = provider

    primary = str(ai_raw.get("primary") or "").strip()
    if not primary:
        raise ValueError("config.yaml must specify ai.primary.")
    if primary not in providers:
        raise ValueError(
            f"ai.primary '{primary}' is not defined in ai.providers. Available: {list(providers)}"
        )
    fallbacks: list[str] = []
    for fallback_value in ai_raw.get("fallbacks") or []:
        fallback = str(fallback_value)
        if fallback not in providers:
            raise ValueError(
                f"Fallback provider '{fallback}' is not defined in ai.providers. "
                f"Available: {list(providers)}"
            )
        if fallback != primary and fallback not in fallbacks:
            fallbacks.append(fallback)
    rotation = str(ai_raw.get("rotation", "failover")).lower()
    if rotation not in {"failover", "round_robin"}:
        raise ValueError(
            f"ai.rotation must be 'failover' or 'round_robin', got '{rotation}'."
        )
    ai = AISettings(
        timeout_seconds=_positive_float(
            ai_raw.get("timeout_seconds"), name="ai.timeout_seconds", default=60.0
        ),
        max_retries=_positive_int(
            ai_raw.get("max_retries"), name="ai.max_retries", default=3
        ),
        global_timeout_seconds=_positive_float(
            ai_raw.get("global_timeout_seconds"),
            name="ai.global_timeout_seconds",
            default=180.0,
        ),
        providers=providers,
        primary=primary,
        fallbacks=tuple(fallbacks),
        rotation=rotation,
    )

    web_search_raw = raw.get("web_search") or {}
    firecrawl_raw = raw.get("firecrawl") or {}
    return ShinAISettings(
        platform=platform,
        admin_user_id=int(raw.get("admin_user_id", 0)),
        logging=logging_settings,
        min_delay_seconds=min_delay,
        max_delay_seconds=max_delay,
        random_trigger_probability=probability,
        group_rate_limit_max_responses=_positive_int(
            response.get("group_max_responses"),
            name="response.group_max_responses",
            default=3,
        ),
        group_rate_limit_window_seconds=_positive_float(
            response.get("group_rate_limit_window_seconds"),
            name="response.group_rate_limit_window_seconds",
            default=10.0,
        ),
        web_search_timeout_seconds=_positive_float(
            web_search_raw.get("timeout_seconds"),
            name="web_search.timeout_seconds",
            default=30.0,
        ),
        whisper=whisper,
        embedding=embedding,
        chroma=chroma,
        runtime=runtime,
        coordination=coordination,
        style_group_id=_optional_string(raw.get("style_group_id")),
        ai=ai,
        firecrawl=FirecrawlSettings(api_key=_optional_string(firecrawl_raw.get("api_key"))),
    )


def load_settings(path: Path | None = None) -> ShinAISettings:
    path = path or DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {path}. Copy config.yaml.example to config.yaml and edit it."
        )
    with path.open("r", encoding="utf-8") as config_file:
        decoded = yaml.safe_load(config_file) or {}
    if not isinstance(decoded, dict):
        raise ValueError("config.yaml must contain a YAML mapping at its top level.")
    return parse_settings(decoded, project_root=path.resolve().parent)


def get_settings() -> ShinAISettings:
    global _settings_cache
    if _settings_cache is None:
        with _settings_lock:
            if _settings_cache is None:
                _settings_cache = load_settings()
    return _settings_cache


def reload_settings() -> ShinAISettings:
    global _settings_cache
    with _settings_lock:
        _settings_cache = load_settings()
    return _settings_cache
