"""Typed, side-effect-free application settings.

The YAML parser lives outside the provider registry so application code and tests
can load configuration without importing provider rotation state or platform SDKs.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from shin_ai.billing.money import decimal_amount, usd_to_units
from shin_ai.billing.types import ModelPrice
from shin_ai.utils.chat_identity import normalize_chat_id

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
    api_keys: dict[str, str] = field(default_factory=dict)
    chats: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def is_available_for(self, platform: str | None, chat_id: int | str | None) -> bool:
        if not self.chats:
            return True
        if platform is None or chat_id is None:
            return False
        return normalize_chat_id(platform, chat_id) in self.chats.get(platform, ())


@dataclass(frozen=True, slots=True)
class AISettings:
    timeout_seconds: float
    max_retries: int
    global_timeout_seconds: float
    preflight_timeout_seconds: float
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
class RetrievalSettings:
    """Relevance gates for semantic lookups, expressed as cosine distance.

    Chroma collections run on squared-L2. For the unit-normalised embeddings
    this project uses, cosine_distance == squared_l2 / 2 exactly, so these
    thresholds are applied after converting the reported distance rather than
    by re-indexing the collections into cosine space.
    """

    memory_max_distance: float
    social_max_distance: float
    lookup_max_distance: float


@dataclass(frozen=True, slots=True)
class ChromaSettings:
    mode: str
    path: Path
    host: str
    port: int
    ssl: bool
    tenant: str
    database: str


@dataclass(frozen=True, slots=True)
class PlatformSettings:
    telegram_enabled: bool
    telegram_api_id: str | None
    telegram_api_hash: str | None
    telegram_bot_token: str | None
    discord_enabled: bool
    discord_bot_token: str | None
    whatsapp_enabled: bool

    @property
    def telegram_configured(self) -> bool:
        return bool(self.telegram_api_id and self.telegram_api_hash and self.telegram_bot_token)

    @property
    def discord_configured(self) -> bool:
        return bool(self.discord_bot_token)


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
class BillingWebSettings:
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8080
    public_url: str = "http://localhost:8080"
    login_ttl_seconds: int = 600
    session_ttl_seconds: int = 86400
    support_email: str = ""


@dataclass(frozen=True, slots=True)
class PaymobSettings:
    enabled: bool = False
    mode: str = "test"
    integration_ids: tuple[int, ...] = ()
    merchant_id: int = 0
    secret_key_env: str = "PAYMOB_SECRET_KEY"
    public_key_env: str = "PAYMOB_PUBLIC_KEY"
    hmac_secret_env: str = "PAYMOB_HMAC_SECRET"
    api_key_env: str = "PAYMOB_API_KEY"
    egp_per_usd: Decimal = Decimal("1")
    fee_bps: int = 0
    fixed_fee_cents: int = 0
    packages: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class BillingSettings:
    enabled: bool
    database_path: Path
    providers: tuple[str, ...]
    max_request_units: int
    max_retries: int
    attempt_timeout_seconds: float
    global_timeout_seconds: float
    max_input_chars: int = 64_000
    max_output_tokens: int = 1_024
    markup_bps: int = 2500
    pricing: tuple[ModelPrice, ...] = ()
    web: BillingWebSettings = field(default_factory=BillingWebSettings)
    paymob: PaymobSettings = field(default_factory=PaymobSettings)


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
    retrieval: RetrievalSettings
    chroma: ChromaSettings
    runtime: RuntimeSettings
    coordination: CoordinationSettings
    style_group_id: str | None
    ai: AISettings
    firecrawl: FirecrawlSettings
    billing: BillingSettings

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


def _cosine_distance(value: Any, *, name: str, default: float) -> float:
    parsed = float(default if value is None else value)
    if not 0.0 < parsed <= 2.0:
        raise ValueError(f"{name} must be a cosine distance in (0, 2].")
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
        raise ValueError(f"Provider '{name}': type must be 'gemini' or 'openai', got '{provider_type}'.")
    if provider_type == "openai":
        missing = [key for key in ("base_url", "api_key", "model") if not raw.get(key)]
        if missing:
            raise ValueError(f"Provider '{name}' (type=openai) is missing required fields: {missing}")

    models = tuple(str(model).strip() for model in raw.get("models", ()) if str(model).strip())
    if provider_type == "gemini" and not models:
        raise ValueError(f"Provider '{name}' (type=gemini) must define at least one model.")

    api_keys: dict[str, str] = {}
    if provider_type == "gemini":
        keys_raw = raw.get("api_keys")
        if not isinstance(keys_raw, dict) or not keys_raw:
            raise ValueError(
                f"Provider '{name}' (type=gemini) must define api_keys "
                "as a non-empty mapping of aliases to API keys."
            )
        for alias, key in keys_raw.items():
            if not isinstance(alias, str) or not alias.strip() or not isinstance(key, str) or not key.strip():
                raise ValueError(f"Provider '{name}': api_keys aliases and values must be non-empty strings.")
            if alias.strip() in api_keys:
                raise ValueError(f"Provider '{name}': duplicate Gemini key alias '{alias.strip()}'.")
            api_keys[alias.strip()] = key.strip()

    concurrency = raw.get("concurrency")
    if concurrency is not None:
        concurrency = _positive_int(
            concurrency,
            name=f"ai.providers[{name}].concurrency",
            default=1,
        )

    chats_raw = raw.get("chats", {})
    if not isinstance(chats_raw, dict):
        raise ValueError(f"Provider '{name}': chats must be a mapping of platforms to chat ID lists.")
    chats: dict[str, tuple[str, ...]] = {}
    for platform, chat_ids in chats_raw.items():
        if platform not in {"telegram", "discord", "whatsapp"}:
            raise ValueError(f"Provider '{name}': chats platform must be telegram, discord, or whatsapp.")
        if not isinstance(chat_ids, list) or not chat_ids:
            raise ValueError(f"Provider '{name}': chats.{platform} must be a non-empty list of chat IDs.")
        if any(
            isinstance(chat_id, bool) or not isinstance(chat_id, (int, str)) or not str(chat_id).strip()
            for chat_id in chat_ids
        ):
            raise ValueError(
                f"Provider '{name}': chats.{platform} chat IDs must be integers or non-empty strings."
            )
        chats[platform] = tuple(dict.fromkeys(normalize_chat_id(platform, chat_id) for chat_id in chat_ids))
    return ProviderSettings(
        name=name,
        type=provider_type,
        base_url=_optional_string(raw.get("base_url")),
        api_key=_optional_string(raw.get("api_key")),
        model=_optional_string(raw.get("model")),
        models=models,
        concurrency=concurrency,
        api_keys=api_keys,
        chats=chats,
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
        raise ValueError("response delays must satisfy 0 <= min_delay_seconds <= max_delay_seconds.")
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
        max_bytes=_positive_int(logging_raw.get("max_bytes"), name="logging.max_bytes", default=25_000_000),
        backup_count=_positive_int(logging_raw.get("backup_count"), name="logging.backup_count", default=5),
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
            "runtime.media_max_total_bytes must be greater than or equal to runtime.media_max_file_bytes."
        )

    coordination_raw = raw.get("coordination") or {}
    coordination_backend = str(coordination_raw.get("backend", "sqlite")).strip().lower()
    if coordination_backend not in {"sqlite", "memory"}:
        raise ValueError("coordination.backend must be 'sqlite' or 'memory'.")
    namespace = str(coordination_raw.get("namespace", "shinai")).strip()
    if not namespace:
        raise ValueError("coordination.namespace cannot be empty.")
    database_path = Path(str(coordination_raw.get("database_path", "data/coordination.sqlite3")))
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
        model=str(embedding_raw.get("model", raw.get("embedding_model", "intfloat/multilingual-e5-large"))),
        max_concurrency=_positive_int(
            embedding_raw.get("max_concurrency"),
            name="embedding.max_concurrency",
            default=1,
        ),
        batch_size=_positive_int(embedding_raw.get("batch_size"), name="embedding.batch_size", default=16),
    )

    retrieval_raw = raw.get("retrieval") or {}
    retrieval = RetrievalSettings(
        memory_max_distance=_cosine_distance(
            retrieval_raw.get("memory_max_distance"),
            name="retrieval.memory_max_distance",
            default=0.16,
        ),
        social_max_distance=_cosine_distance(
            retrieval_raw.get("social_max_distance"),
            name="retrieval.social_max_distance",
            default=0.12,
        ),
        lookup_max_distance=_cosine_distance(
            retrieval_raw.get("lookup_max_distance"),
            name="retrieval.lookup_max_distance",
            default=0.16,
        ),
    )

    chroma_raw = raw.get("chroma") or {}
    chroma_mode = str(chroma_raw.get("mode", "embedded")).strip().lower()
    if chroma_mode not in {"embedded", "server"}:
        raise ValueError("chroma.mode must be 'embedded' or 'server'.")
    chroma_path = Path(str(chroma_raw.get("path", "chroma_db")))
    if not chroma_path.is_absolute():
        chroma_path = project_root / chroma_path
    chroma_host = str(chroma_raw.get("host", "127.0.0.1")).strip()
    if not chroma_host:
        raise ValueError("chroma.host cannot be empty.")
    chroma = ChromaSettings(
        mode=chroma_mode,
        path=chroma_path,
        host=chroma_host,
        port=_positive_int(chroma_raw.get("port"), name="chroma.port", default=8000),
        ssl=bool(chroma_raw.get("ssl", False)),
        tenant=str(chroma_raw.get("tenant", "default_tenant")),
        database=str(chroma_raw.get("database", "default_database")),
    )

    whisper_raw = raw.get("whisper") or {}
    whisper = WhisperSettings(
        model=str(whisper_raw.get("model", "large-v3-turbo")),
        language=str(whisper_raw.get("language", "auto")),
        cpu_threads=_positive_int(whisper_raw.get("cpu_threads"), name="whisper.cpu_threads", default=2),
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
    gemini_keys: dict[str, str] = {}
    for entry in providers_raw:
        provider = _parse_provider(entry)
        if provider.name in providers:
            raise ValueError(f"Duplicate provider name: '{provider.name}'")
        providers[provider.name] = provider
        for alias, key in provider.api_keys.items():
            if alias in gemini_keys and gemini_keys[alias] != key:
                raise ValueError(
                    f"Conflicting Gemini key alias '{alias}' across providers; use unique aliases."
                )
            gemini_keys[alias] = key

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
                f"Fallback provider '{fallback}' is not defined in ai.providers. Available: {list(providers)}"
            )
        if fallback != primary and fallback not in fallbacks:
            fallbacks.append(fallback)
    rotation = str(ai_raw.get("rotation", "failover")).lower()
    if rotation not in {"failover", "round_robin"}:
        raise ValueError(f"ai.rotation must be 'failover' or 'round_robin', got '{rotation}'.")
    ai = AISettings(
        timeout_seconds=_positive_float(
            ai_raw.get("timeout_seconds"), name="ai.timeout_seconds", default=60.0
        ),
        max_retries=_positive_int(ai_raw.get("max_retries"), name="ai.max_retries", default=3),
        global_timeout_seconds=_positive_float(
            ai_raw.get("global_timeout_seconds"),
            name="ai.global_timeout_seconds",
            default=180.0,
        ),
        preflight_timeout_seconds=_positive_float(
            ai_raw.get("preflight_timeout_seconds"),
            name="ai.preflight_timeout_seconds",
            default=15.0,
        ),
        providers=providers,
        primary=primary,
        fallbacks=tuple(fallbacks),
        rotation=rotation,
    )

    billing_raw = raw.get("billing") or {}
    if not isinstance(billing_raw, dict):
        raise ValueError("billing must be a mapping.")
    billing_enabled = billing_raw.get("enabled", False)
    if not isinstance(billing_enabled, bool):
        raise ValueError("billing.enabled must be a boolean.")
    paid_names = billing_raw.get("providers") or []
    if not isinstance(paid_names, list) or any(not isinstance(name, str) for name in paid_names):
        raise ValueError("billing.providers must be a list of provider names.")
    paid_names = tuple(dict.fromkeys(paid_names))
    for name in paid_names:
        if name not in providers:
            raise ValueError(f"billing.providers contains unknown provider '{name}'.")
        if name in (primary, *fallbacks):
            raise ValueError(f"Paid provider '{name}' cannot appear in ai.primary or ai.fallbacks.")
    if billing_enabled and not paid_names:
        raise ValueError("billing.providers must contain at least one provider when billing is enabled.")
    if "response_cost" in billing_raw:
        raise ValueError("billing.response_cost is replaced by billing.max_request_usd and pricing")
    for name in ("max_retries", "max_input_chars", "max_output_tokens"):
        value = billing_raw.get(name)
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(f"billing.{name} must be an integer greater than zero.")
    billing_path = Path(str(billing_raw.get("database_path", "data/billing.sqlite3")))
    if not billing_path.is_absolute():
        billing_path = project_root / billing_path

    def boolean(section, key, default=False):
        value = section.get(key, default)
        if not isinstance(value, bool):
            raise ValueError(f"billing.{key} must be a boolean")
        return value

    def percent_bps(value, name):
        scaled = decimal_amount(value, name, allow_zero=True) * 100
        if scaled != scaled.to_integral_value() or scaled > 100_000:
            raise ValueError(f"{name} must have at most two decimal places and be at most 1000 percent")
        return int(scaled)

    prices_raw = billing_raw.get("pricing", [])
    if not isinstance(prices_raw, list):
        raise ValueError("billing.pricing must be a list")
    prices = []
    price_keys = set()
    for entry in prices_raw:
        if not isinstance(entry, dict):
            raise ValueError("Each billing.pricing entry must be a mapping")
        provider_name = str(entry.get("provider", ""))
        model = str(entry.get("model", "")).strip()
        adapter = str(entry.get("adapter", ""))
        version = str(entry.get("version", "")).strip()
        if provider_name not in paid_names or not model or not version:
            raise ValueError("Each price requires a paid provider, model, and version")
        if adapter not in {"gemini", "deepseek", "openrouter", "openai"}:
            raise ValueError("Price adapter must be gemini, deepseek, openrouter, or openai")
        if (providers[provider_name].type == "gemini") != (adapter == "gemini"):
            raise ValueError("Price adapter must match its provider type")
        key = (provider_name, model)
        if key in price_keys:
            raise ValueError("Duplicate provider/model price")
        price_keys.add(key)
        aliases = entry.get("aliases", [])
        if not isinstance(aliases, list) or any(
            not isinstance(alias, str) or not alias.strip() for alias in aliases
        ):
            raise ValueError("Price aliases must be a list of actual model names")
        prices.append(
            ModelPrice(
                provider_name,
                model,
                adapter,
                version,
                decimal_amount(entry.get("input_per_million_usd"), "input price", allow_zero=True),
                decimal_amount(
                    entry.get("cached_input_per_million_usd", entry.get("input_per_million_usd")),
                    "cached input price",
                    allow_zero=True,
                ),
                decimal_amount(entry.get("output_per_million_usd"), "output price", allow_zero=True),
                _positive_int(entry.get("max_input_tokens"), name="price.max_input_tokens", default=65536),
                tuple(aliases),
            )
        )
    if billing_enabled:
        for name in paid_names:
            for model in providers[name].models or (providers[name].model,):
                if (name, model) not in price_keys:
                    raise ValueError(f"Missing billing.pricing for {name}/{model}")

    web_raw = billing_raw.get("web") or {}
    paymob_raw = billing_raw.get("paymob") or {}
    if not isinstance(web_raw, dict) or not isinstance(paymob_raw, dict):
        raise ValueError("billing.web and billing.paymob must be mappings")
    web_enabled = boolean(web_raw, "enabled")
    public_url = str(web_raw.get("public_url", "http://localhost:8080")).rstrip("/")
    url = urlparse(public_url)
    if (
        url.scheme not in {"http", "https"}
        or not url.hostname
        or url.username
        or url.password
        or url.path not in {"", "/"}
        or url.query
        or url.fragment
    ):
        raise ValueError("billing.web.public_url must be an absolute origin URL")
    if url.scheme == "http" and url.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("billing.web.public_url requires HTTPS outside localhost")
    web = BillingWebSettings(
        enabled=web_enabled,
        host=str(web_raw.get("host", "127.0.0.1")),
        port=_positive_int(web_raw.get("port"), name="billing.web.port", default=8080),
        public_url=public_url,
        login_ttl_seconds=_positive_int(web_raw.get("login_ttl_seconds"), name="login TTL", default=600),
        session_ttl_seconds=_positive_int(
            web_raw.get("session_ttl_seconds"), name="session TTL", default=86400
        ),
        support_email=str(web_raw.get("support_email", "")),
    )
    payment_enabled = boolean(paymob_raw, "enabled")
    mode = str(paymob_raw.get("mode", "test"))
    integrations = paymob_raw.get("integration_ids", [])
    if (
        mode not in {"test", "live"}
        or not isinstance(integrations, list)
        or any(type(i) is not int or i <= 0 for i in integrations)
    ):
        raise ValueError("Paymob requires mode test or live and positive integration IDs")
    packages_raw = paymob_raw.get("packages_egp", [])
    if not isinstance(packages_raw, list):
        raise ValueError("Paymob packages_egp must be a list")
    merchant_id = paymob_raw.get("merchant_id", 0)
    if type(merchant_id) is not int or not 0 <= merchant_id < 2**63:
        raise ValueError("Paymob merchant_id must be a nonnegative integer")
    packages = []
    for amount in packages_raw:
        cents = decimal_amount(amount, "package amount") * 100
        if cents != cents.to_integral_value() or cents > 100_000_000:
            raise ValueError("Paymob package must be an EGP amount with at most two decimal places")
        packages.append(int(cents))
    fixed_fee = (
        decimal_amount(paymob_raw.get("fixed_fee_egp", "0"), "fixed processing allowance", allow_zero=True)
        * 100
    )
    if fixed_fee != fixed_fee.to_integral_value():
        raise ValueError("Fixed processing allowance requires at most two decimal places")
    paymob = PaymobSettings(
        enabled=payment_enabled,
        mode=mode,
        integration_ids=tuple(dict.fromkeys(integrations)),
        merchant_id=merchant_id,
        secret_key_env=str(paymob_raw.get("secret_key_env", "PAYMOB_SECRET_KEY")),
        public_key_env=str(paymob_raw.get("public_key_env", "PAYMOB_PUBLIC_KEY")),
        hmac_secret_env=str(paymob_raw.get("hmac_secret_env", "PAYMOB_HMAC_SECRET")),
        api_key_env=str(paymob_raw.get("api_key_env", "PAYMOB_API_KEY")),
        egp_per_usd=decimal_amount(paymob_raw.get("egp_per_usd", "1"), "EGP per USD"),
        fee_bps=percent_bps(paymob_raw.get("fee_percent", "0"), "fee_percent"),
        fixed_fee_cents=int(fixed_fee),
        packages=tuple(dict.fromkeys(packages)),
    )
    if web_enabled and not billing_enabled:
        raise ValueError("Enable billing before enabling its website")
    if payment_enabled and (not web_enabled or not integrations or paymob.merchant_id <= 0 or not packages):
        raise ValueError(
            "Paymob requires the billing website, merchant_id, integration_ids, and packages_egp"
        )
    if payment_enabled and mode == "live" and url.scheme != "https":
        raise ValueError("Live Paymob checkout requires an HTTPS public URL")

    billing = BillingSettings(
        enabled=billing_enabled,
        database_path=billing_path,
        providers=paid_names,
        max_request_units=usd_to_units(billing_raw.get("max_request_usd", "0.10")),
        max_retries=_positive_int(billing_raw.get("max_retries"), name="billing.max_retries", default=1),
        attempt_timeout_seconds=_positive_float(
            billing_raw.get("attempt_timeout_seconds"),
            name="billing.attempt_timeout_seconds",
            default=15,
        ),
        global_timeout_seconds=_positive_float(
            billing_raw.get("global_timeout_seconds"),
            name="billing.global_timeout_seconds",
            default=30,
        ),
        max_input_chars=_positive_int(
            billing_raw.get("max_input_chars"),
            name="billing.max_input_chars",
            default=64_000,
        ),
        max_output_tokens=_positive_int(
            billing_raw.get("max_output_tokens"),
            name="billing.max_output_tokens",
            default=1024,
        ),
        markup_bps=percent_bps(billing_raw.get("markup_percent", "25"), "markup_percent"),
        pricing=tuple(prices),
        web=web,
        paymob=paymob,
    )
    if billing.max_request_units <= 0:
        raise ValueError("billing.max_request_usd must be positive")
    if not all(
        math.isfinite(value)
        for value in (
            billing.attempt_timeout_seconds,
            billing.global_timeout_seconds,
        )
    ):
        raise ValueError("Billing timeouts must be finite.")

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
        retrieval=retrieval,
        chroma=chroma,
        runtime=runtime,
        coordination=coordination,
        style_group_id=_optional_string(raw.get("style_group_id")),
        ai=ai,
        firecrawl=FirecrawlSettings(api_key=_optional_string(firecrawl_raw.get("api_key"))),
        billing=billing,
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
