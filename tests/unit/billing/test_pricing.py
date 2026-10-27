import json
from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace

import pytest

from shin_ai.billing.pricing import AccountingError, BudgetExceeded, UsageMeter, response_cost, usage_attempt
from shin_ai.billing.store import Balance, CreditStore, Identity, Scope
from shin_ai.billing.types import ModelPrice


def price(adapter="deepseek", *, input_rate="1", cached_rate="0.1", output_rate="2"):
    return ModelPrice(
        "provider",
        "model",
        adapter,
        "rate-v1",
        Decimal(input_rate),
        Decimal(cached_rate),
        Decimal(output_rate),
    )


def completion(prompt=1000, cached=400, output=100, **extras):
    return SimpleNamespace(
        model="model",
        usage={
            "prompt_tokens": prompt,
            "prompt_cache_hit_tokens": cached,
            "prompt_cache_miss_tokens": prompt - cached,
            "completion_tokens": output,
            "completion_tokens_details": {"reasoning_tokens": 50},
            **extras,
        },
    )


def test_deepseek_cache_and_reasoning_subset_do_not_double_charge():
    nanos, raw = response_cost(price(), completion())
    assert nanos == 840_000  # 600 uncached + 400 discounted + 100 total output.
    assert raw["completion_tokens_details"]["reasoning_tokens"] == 50


def test_gemini_thinking_is_added_to_candidate_output():
    response = SimpleNamespace(
        model_version="model",
        usage_metadata={
            "prompt_token_count": 1000,
            "cached_content_token_count": 400,
            "candidates_token_count": 100,
            "thoughts_token_count": 50,
        },
    )
    assert response_cost(price("gemini"), response)[0] == 940_000


def test_openrouter_uses_reported_account_cost_instead_of_upstream_cost():
    response = SimpleNamespace(
        model="provider/actual-model",
        usage={
            "cost": "0.00000125",
            "cost_details": {"upstream_inference_cost": 999},
        },
    )
    assert response_cost(price("openrouter"), response)[0] == 1250


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(),
        SimpleNamespace(usage={}),
        completion(prompt=True),
        completion(cached=-1),
        completion(prompt_cache_miss_tokens=999),
    ],
)
def test_missing_or_inconsistent_deepseek_usage_is_not_zero_cost(response):
    with pytest.raises(AccountingError):
        response_cost(price(), response)


@pytest.mark.parametrize("output", [None, False, -1, "1"])
def test_gemini_missing_or_malformed_output_fails_closed(output):
    response = SimpleNamespace(
        model_version="model",
        usage_metadata={
            "prompt_token_count": 10,
            "candidates_token_count": output,
        },
    )
    with pytest.raises(AccountingError):
        response_cost(price("gemini"), response)


def test_returned_model_must_match_a_configured_price_or_explicit_alias():
    response = completion()
    response.model = "different-model"
    with pytest.raises(AccountingError, match="model"):
        response_cost(price(), response)
    aliased = replace(price(), aliases=("different-model",))
    assert response_cost(aliased, response)[0] == 840_000


def test_unsupported_cache_write_price_fails_closed():
    response = SimpleNamespace(
        model="model",
        usage={
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "prompt_tokens_details": {"cached_tokens": 10, "cache_write_tokens": 50},
        },
    )
    with pytest.raises(AccountingError, match="cache-write"):
        response_cost(price("openai"), response)


async def funded_meter(tmp_path, *, rate=None, cap=10_000, markup=0):
    store = CreditStore(tmp_path / "meter.sqlite3")
    owner = Identity("discord", "payer")
    chat = Scope("discord", "chat", "friends")
    await store.grant(owner, 100_000, "test-funds")
    await store.configure(owner, chat, "only_me")
    reservation = await store.reserve("request", owner, (chat,), cap)
    await store.configure_reservation(reservation, markup)
    return store, owner, reservation, UsageMeter(store, reservation, (rate or price(),), markup)


@pytest.mark.asyncio
async def test_multiple_calls_round_once_and_keep_price_snapshot(tmp_path):
    tiny = price(input_rate="0.01", cached_rate="0.01", output_rate="0.01")
    store, owner, _, meter = await funded_meter(tmp_path, rate=tiny, markup=2500)
    token = usage_attempt.set("accepted")
    try:
        for _ in range(2):
            ticket = await meter.before_call("provider", "model", 1, 0)
            await meter.record_response(ticket, completion(prompt=1, cached=0, output=0))
        await meter.accept_attempt("accepted")
        await meter.mark_delivered()
        await meter.finish()
        await meter.finish()
    finally:
        usage_attempt.reset(token)
    assert await store.balance(owner) == Balance(99_999, 0, 0)
    history = await store.history(owner)
    assert history["usage"][0]["customer_charge"] == 1
    assert history["usage"][0]["cost_nano_usd"] == 20
    records = await store._run(
        lambda connection: [dict(row) for row in connection.execute("SELECT * FROM call_usage")]
    )
    assert len(records) == 2
    assert all(json.loads(record["price"])["version"] == "rate-v1" for record in records)


@pytest.mark.asyncio
async def test_failed_attempt_expense_is_excluded_from_customer_charge(tmp_path):
    store, owner, _, meter = await funded_meter(tmp_path)
    for attempt in ("failed", "accepted"):
        token = usage_attempt.set(attempt)
        try:
            ticket = await meter.before_call("provider", "model", 1000, 100)
            await meter.record_response(ticket, completion())
        finally:
            usage_attempt.reset(token)
    await meter.accept_attempt("accepted")
    await meter.mark_delivered()
    await meter.finish()
    assert meter.expense_nanos == 1_680_000
    assert await store.balance(owner) == Balance(99_160, 0, 0)
    assert (await store.history(owner))["usage"][0]["cost_nano_usd"] == 840_000


@pytest.mark.asyncio
async def test_undelivered_interaction_refunds_all_reserved_funds(tmp_path):
    store, owner, _, meter = await funded_meter(tmp_path)
    ticket = await meter.before_call("provider", "model", 1000, 100)
    await meter.record_response(ticket, completion())
    await meter.accept_attempt(ticket.attempt)
    await meter.finish()
    assert await store.balance(owner) == Balance(100_000, 0, 0)


@pytest.mark.asyncio
async def test_missing_usage_blocks_further_spend_and_cannot_be_accepted(tmp_path):
    store, owner, _, meter = await funded_meter(tmp_path)
    ticket = await meter.before_call("provider", "model", 1000, 100)
    with pytest.raises(AccountingError):
        await meter.record_response(ticket, SimpleNamespace())
    with pytest.raises(AccountingError, match="unresolved"):
        await meter.before_call("provider", "model", 1, 1)
    with pytest.raises(ValueError, match="incomplete"):
        await meter.accept_attempt(ticket.attempt)
    await meter.mark_delivered()
    await meter.finish()
    assert await store.balance(owner) == Balance(100_000, 0, 0)


@pytest.mark.asyncio
async def test_next_call_requires_budget_including_markup(tmp_path):
    store, owner, _, meter = await funded_meter(tmp_path, cap=10, markup=2500)
    with pytest.raises(BudgetExceeded):
        await meter.before_call("provider", "model", 1000, 100)
    assert meter.calls == 0
    await meter.finish()
    assert await store.balance(owner) == Balance(100_000, 0, 0)


@pytest.mark.asyncio
async def test_provider_overrun_never_debits_more_than_reserved(tmp_path):
    store, owner, _, meter = await funded_meter(tmp_path, cap=2)
    ticket = await meter.before_call("provider", "model", 1, 0)
    await meter.record_response(ticket, completion(prompt=1000, cached=0, output=0))
    assert meter.blocked
    await meter.accept_attempt(ticket.attempt)
    await meter.mark_delivered()
    await meter.finish()
    assert await store.balance(owner) == Balance(99_998, 0, 0)


@pytest.mark.asyncio
async def test_restart_settles_known_delivered_usage_with_saved_markup(tmp_path):
    store, owner, _, meter = await funded_meter(tmp_path, markup=2500)
    ticket = await meter.before_call("provider", "model", 1000, 100)
    await meter.record_response(ticket, completion())
    await meter.accept_attempt(ticket.attempt)
    await meter.mark_delivered()
    await CreditStore(store.path).recover()
    assert await store.balance(owner) == Balance(98_950, 0, 0)
