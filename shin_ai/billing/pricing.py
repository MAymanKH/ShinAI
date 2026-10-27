"""Request budgets and provider usage accounting, independent of model response text."""

from __future__ import annotations

import contextvars
import json
import uuid
from dataclasses import asdict, dataclass
from decimal import ROUND_CEILING
from typing import TYPE_CHECKING, Any

from shin_ai.billing.money import decimal_amount
from shin_ai.billing.types import ModelPrice, UsageCall

if TYPE_CHECKING:
    from shin_ai.billing.store import CreditStore, Reservation


class AccountingError(RuntimeError):
    """A paid call cannot be safely priced."""


class BudgetExceeded(AccountingError):
    """The next call would exceed the authorized request budget."""


usage_meter: contextvars.ContextVar[UsageMeter | None] = contextvars.ContextVar("usage_meter", default=None)
usage_attempt: contextvars.ContextVar[str] = contextvars.ContextVar("usage_attempt", default="support")


def _data(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    dump = getattr(value, "model_dump", None)
    if dump is not None:
        return dump(exclude_none=True)
    return vars(value) if value is not None and hasattr(value, "__dict__") else {}


def _tokens(value: Any, name: str) -> int:
    if type(value) is not int or not 0 <= value <= 2**40:
        raise AccountingError(f"Missing or invalid provider token count: {name}")
    return value


def response_cost(price: ModelPrice, response: Any) -> tuple[int, dict]:
    raw = _data(getattr(response, "usage_metadata" if price.adapter == "gemini" else "usage", None))
    if not raw:
        raise AccountingError("Provider did not return usage")
    actual_model = getattr(response, "model", getattr(response, "model_version", None))
    if actual_model and price.adapter != "openrouter":
        actual_model = str(actual_model).removeprefix("models/")
        if actual_model not in (price.model, *price.aliases):
            raise AccountingError("Provider returned a model without an agreed price")
    if price.adapter == "openrouter":
        if raw.get("cost") is None:
            raise AccountingError("OpenRouter did not return its account charge")
        cost = decimal_amount(raw["cost"], "OpenRouter cost", allow_zero=True) * 1_000_000_000
    else:
        if price.adapter == "gemini":
            prompt = _tokens(raw.get("prompt_token_count"), "prompt")
            cached = _tokens(raw.get("cached_content_token_count", 0), "cached prompt")
            output = _tokens(raw.get("candidates_token_count"), "output")
            output += _tokens(raw.get("thoughts_token_count", 0), "thinking")
        else:
            prompt = _tokens(raw.get("prompt_tokens"), "prompt")
            output = _tokens(raw.get("completion_tokens"), "completion")
            if price.adapter == "deepseek":
                cached = _tokens(raw.get("prompt_cache_hit_tokens"), "cached prompt")
                missed = _tokens(raw.get("prompt_cache_miss_tokens"), "uncached prompt")
                if cached + missed != prompt:
                    raise AccountingError("DeepSeek input token totals disagree")
            else:
                cached = _tokens(
                    _data(raw.get("prompt_tokens_details")).get("cached_tokens", 0), "cached prompt"
                )
                if _data(raw.get("prompt_tokens_details")).get("cache_write_tokens", 0):
                    raise AccountingError("Separate cache-write pricing is not configured")
        if cached > prompt:
            raise AccountingError("Cached input exceeds total input")
        cost = (
            (prompt - cached) * price.input_per_million_usd
            + cached * price.cached_input_per_million_usd
            + output * price.output_per_million_usd
        ) * 1000  # USD per million tokens -> nanodollars.
    if not cost.is_finite() or cost < 0 or cost > 2**63 - 1:
        raise AccountingError("Provider cost exceeds supported accounting limits")
    raw = {
        **raw,
        "actual_model": str(getattr(response, "model", getattr(response, "model_version", price.model))),
    }
    return int(cost.to_integral_value(rounding=ROUND_CEILING)), raw


@dataclass(slots=True)
class CallTicket:
    reference: str
    attempt: str
    price: ModelPrice
    upper_nanos: int
    completed: bool = False


class UsageMeter:
    def __init__(
        self, store: CreditStore, reservation: Reservation, prices: tuple[ModelPrice, ...], markup_bps: int
    ):
        self.store = store
        self.reservation = reservation
        self.prices = {(price.provider, price.model): price for price in prices}
        self.markup_bps = markup_bps
        self.expense_nanos = 0
        self.outstanding_nanos = 0
        self.blocked = False
        self.calls = 0
        self.has_support_calls = False
        self.delivered = False

    def price_for(self, provider: str, model: str) -> ModelPrice:
        try:
            return self.prices[provider, model]
        except KeyError as error:
            raise AccountingError(f"No configured price for {provider}/{model}") from error

    def _snapshot(self, price: ModelPrice) -> str:
        return json.dumps(asdict(price), default=str, sort_keys=True)

    async def before_call(
        self, provider: str, model: str, input_tokens: int, output_tokens: int
    ) -> CallTicket:
        if self.blocked:
            raise AccountingError("An earlier provider call has unresolved accounting")
        price = self.price_for(provider, model)
        input_tokens = _tokens(input_tokens, "input bound")
        output_tokens = _tokens(output_tokens, "output bound")
        if input_tokens > price.max_input_tokens or self.calls >= 12:
            raise BudgetExceeded("Paid input or call-count limit reached")
        upper = int(
            (
                input_tokens * max(price.input_per_million_usd, price.cached_input_per_million_usd)
                + output_tokens * price.output_per_million_usd
            )
            .__mul__(1000)
            .to_integral_value(rounding=ROUND_CEILING)
        )
        projected = self.expense_nanos + self.outstanding_nanos + upper
        if projected * (10_000 + self.markup_bps) > self.reservation.credits * 1000 * 10_000:
            raise BudgetExceeded("The next model call exceeds the request spending limit")
        self.calls += 1
        if usage_attempt.get() == "support":
            self.has_support_calls = True
        ticket = CallTicket(uuid.uuid4().hex, usage_attempt.get(), price, upper)
        await self.store.record_call(
            self.reservation,
            UsageCall(
                ticket.reference,
                ticket.attempt,
                provider,
                model,
                None,
                "{}",
                self._snapshot(price),
            ),
        )
        self.outstanding_nanos += upper
        return ticket

    async def record_response(self, ticket: CallTicket, response: Any) -> None:
        if ticket.completed:
            raise AccountingError("A provider response was recorded twice")
        try:
            nanos, raw = response_cost(ticket.price, response)
            await self.store.record_call(
                self.reservation,
                UsageCall(
                    ticket.reference,
                    ticket.attempt,
                    ticket.price.provider,
                    ticket.price.model,
                    nanos,
                    json.dumps(raw, default=str, sort_keys=True),
                    self._snapshot(ticket.price),
                ),
            )
        except Exception:
            self.blocked = True
            raise
        ticket.completed = True
        self.outstanding_nanos -= ticket.upper_nanos
        self.expense_nanos += nanos
        if nanos > ticket.upper_nanos:
            # Provider overruns are an operator expense, never an unauthorized debit.
            self.blocked = True

    def failed_call(self, ticket: CallTicket) -> None:
        if not ticket.completed:
            self.blocked = True  # A timeout can still have incurred upstream expense.

    async def accept_attempt(self, attempt: str) -> None:
        await self.store.accept_attempt(self.reservation, attempt)

    async def mark_delivered(self) -> None:
        if not self.delivered:
            await self.store.mark_delivered(self.reservation)
            self.delivered = True

    async def finish(self) -> None:
        amount = await self.store.final_cost(self.reservation, self.markup_bps)
        await self.store.settle(self.reservation, amount)
