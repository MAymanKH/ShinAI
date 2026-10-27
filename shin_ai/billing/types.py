"""Values shared by prepaid accounting, provider metering, and checkout."""

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal


@dataclass(frozen=True, slots=True)
class ModelPrice:
    provider: str
    model: str
    adapter: Literal["deepseek", "gemini", "openrouter", "openai"]
    version: str
    input_per_million_usd: Decimal
    cached_input_per_million_usd: Decimal
    output_per_million_usd: Decimal
    max_input_tokens: int = 65536
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class UsageCall:
    reference: str
    attempt: str
    provider: str
    model: str
    cost_nano_usd: int | None
    usage: str
    price: str


@dataclass(frozen=True, slots=True)
class Checkout:
    order_id: str
    intention_id: str
    url: str


@dataclass(frozen=True, slots=True)
class PaymentState:
    order_id: str
    transaction_id: str
    amount_cents: int
    currency: str
    success: bool
    pending: bool
    voided: bool
    refunded_cents: int
    integration_id: int
    merchant_id: int
    is_live: bool
    special_reference: str | None = None
