"""Paid admission, delivery, and fallback using the real durable ledger."""

import uuid
from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from shin_ai.billing.pricing import usage_attempt, usage_meter
from shin_ai.billing.store import Balance, CreditStore, Identity, Scope
from shin_ai.billing.types import ModelPrice
from shin_ai.core import handler
from shin_ai.core.action_executor import ActionExecutionResult
from shin_ai.core.request_context import delivery_callback, paid_route, record_delivery
from shin_ai.platforms.models import UnifiedChat, UnifiedMessage, UnifiedUser
from shin_ai.settings import BillingSettings, ProviderSettings, get_settings

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest.fixture
def scenario(tmp_path, monkeypatch):
    store = CreditStore(tmp_path / "usage.sqlite3")
    original = get_settings()
    paid_provider = ProviderSettings("paid-test", "openai", model="test", api_key="unused")
    settings = replace(
        original,
        ai=replace(original.ai, providers={**original.ai.providers, paid_provider.name: paid_provider}),
        billing=BillingSettings(
            True,
            store.path,
            (paid_provider.name,),
            2,
            1,
            5,
            10,
            markup_bps=0,
            pricing=(
                ModelPrice("paid-test", "test", "openai", "fixture", Decimal(0), Decimal(0), Decimal(2)),
            ),
        ),
    )
    monkeypatch.setattr(handler, "get_settings", lambda: settings)
    monkeypatch.setattr(handler, "get_credit_store", lambda: store)
    monkeypatch.setattr(handler, "is_trivial_message", lambda _msg: False)
    monkeypatch.setattr(handler, "_get_recent_context", lambda *_args: "")
    monkeypatch.setattr(handler, "_build_target_instructions", lambda *_args: "")
    monkeypatch.setattr(handler, "prepare_prompt_and_media", AsyncMock(return_value=("hello", [])))
    for name in (
        "_get_reply_chain_text",
        "_get_style_examples",
        "get_social_context",
        "_get_memory_section",
        "_build_runtime_context",
    ):
        monkeypatch.setattr(handler, name, AsyncMock(return_value=""))
    monkeypatch.setattr(handler, "_passes_speculative_preflight", AsyncMock(return_value=True))
    monkeypatch.setattr(handler, "start_typing", AsyncMock(return_value=object()))
    monkeypatch.setattr(handler, "stop_typing", AsyncMock())
    monkeypatch.setattr(
        handler, "execute_pending_actions", AsyncMock(return_value=ActionExecutionResult([], []))
    )
    memory = AsyncMock()
    send = AsyncMock(return_value=["hello back"])
    group = AsyncMock(return_value=True)
    cooldown = AsyncMock(return_value=True)
    monkeypatch.setattr(handler, "save_interaction_memory", memory)
    monkeypatch.setattr(handler, "execute_text_messages", send)
    monkeypatch.setattr(handler, "check_group_rate_limit_shared", group)
    monkeypatch.setattr(handler, "check_rate_limit_shared", cooldown)
    calls = []

    async def provider(**_kwargs):
        calls.append("paid" if paid_route.get() else "free")
        await record_mock_usage()
        return "hello back", []

    monkeypatch.setattr(handler, "call_ai_provider", provider)
    msg = UnifiedMessage(
        "telegram",
        10,
        UnifiedChat(20, "Friends", "GROUP"),
        UnifiedUser(30, "user", "User", False),
        text="hello",
        mentioned=True,
    )
    platform = SimpleNamespace(platform_name="telegram", coordination_scope="telegram:billing-test")
    payload = handler._AdmittedInteraction(platform, msg, "test-log-id")
    return SimpleNamespace(
        store=store,
        owner=Identity("telegram", "30"),
        msg=msg,
        payload=payload,
        calls=calls,
        group=group,
        cooldown=cooldown,
        memory=memory,
        send=send,
    )


async def record_mock_usage():
    meter = usage_meter.get()
    if meter is None:
        return
    attempt = uuid.uuid4().hex
    token = usage_attempt.set(attempt)
    try:
        ticket = await meter.before_call("paid-test", "test", 0, 1)
        await meter.record_response(
            ticket, SimpleNamespace(model="test", usage={"prompt_tokens": 0, "completion_tokens": 1})
        )
        await meter.accept_attempt(attempt)
    finally:
        usage_attempt.reset(token)


async def fund(scenario, *, owner=None, mode="only_me"):
    owner = owner or scenario.owner
    await scenario.store.grant(owner, 10, "grant-test")
    await scenario.store.configure(owner, Scope("telegram", "chat", "20"), mode)
    return owner


async def test_personal_paid_delivery_bypasses_free_quota(scenario):
    await fund(scenario)
    scenario.group.return_value = False
    await handler._process_admitted_interaction(scenario.payload)
    assert scenario.calls == ["paid"]
    scenario.group.assert_not_awaited()
    scenario.cooldown.assert_awaited_once()
    assert await scenario.store.balance(scenario.owner) == Balance(8, 0)
    assert paid_route.get() is None
    assert delivery_callback.get() is None


async def test_shared_sponsor_pays_for_other_member(scenario):
    sponsor = await fund(scenario, owner=Identity("telegram", "999"), mode="everyone")
    await handler._process_admitted_interaction(scenario.payload)
    assert await scenario.store.balance(sponsor) == Balance(8, 0)
    assert await scenario.store.balance(scenario.owner) == Balance(0, 0)
    assert scenario.calls == ["paid"]


@pytest.mark.parametrize("free_allowed", [True, False])
async def test_paid_exhaustion_refunds_before_free_admission(scenario, monkeypatch, free_allowed):
    await fund(scenario)
    events = []

    async def provider(**_kwargs):
        if paid_route.get():
            events.append("paid")
            assert await scenario.store.balance(scenario.owner) == Balance(8, 2)
            return None, []
        events.append("free")
        return "free response", []

    async def group(*_args, **_kwargs):
        events.append("quota")
        assert paid_route.get() is None
        assert await scenario.store.balance(scenario.owner) == Balance(10, 0)
        return free_allowed

    monkeypatch.setattr(handler, "call_ai_provider", provider)
    monkeypatch.setattr(handler, "check_group_rate_limit_shared", group)
    await handler._process_admitted_interaction(scenario.payload)
    assert events == (["paid", "quota", "free"] if free_allowed else ["paid", "quota"])
    assert await scenario.store.balance(scenario.owner) == Balance(10, 0)
    assert scenario.send.await_count == int(free_allowed)


async def test_empty_balance_uses_free_route(scenario):
    await scenario.store.configure(scenario.owner, Scope("telegram", "chat", "20"), "only_me")
    await handler._process_admitted_interaction(scenario.payload)
    assert scenario.calls == ["free"]
    scenario.group.assert_awaited_once()
    assert await scenario.store.balance(scenario.owner) == Balance(0, 0)


@pytest.mark.parametrize("answer,delivery", [("[SKIP]", ["unused"]), ("hello", [])])
async def test_no_visible_result_releases_paid_reservation(scenario, monkeypatch, answer, delivery):
    await fund(scenario)
    monkeypatch.setattr(handler, "call_ai_provider", AsyncMock(return_value=(answer, [])))
    scenario.send.return_value = delivery
    await handler._process_admitted_interaction(scenario.payload)
    assert await scenario.store.balance(scenario.owner) == Balance(10, 0)
    scenario.group.assert_not_awaited()
    if answer == "[SKIP]":
        scenario.send.assert_not_awaited()


async def test_memory_error_after_delivery_keeps_charge_and_resets_context(scenario):
    await fund(scenario)

    async def save_memory(**_kwargs):
        assert await scenario.store.balance(scenario.owner) == Balance(8, 0)
        raise RuntimeError("memory unavailable")

    scenario.memory.side_effect = save_memory
    with pytest.raises(RuntimeError, match="memory unavailable"):
        await handler._process_admitted_interaction(scenario.payload)
    assert await scenario.store.balance(scenario.owner) == Balance(8, 0)
    assert scenario.calls == ["paid"]
    assert paid_route.get() is None
    assert delivery_callback.get() is None


async def test_replayed_paid_event_does_not_fall_back_to_free(scenario):
    await fund(scenario)
    await handler._process_admitted_interaction(scenario.payload)
    await handler._process_admitted_interaction(scenario.payload)
    assert scenario.calls == ["paid"]
    assert scenario.send.await_count == 1
    scenario.group.assert_not_awaited()
    assert await scenario.store.balance(scenario.owner) == Balance(8, 0)


async def test_random_interjection_does_not_spend_shared_credits(scenario):
    sponsor = await fund(scenario, owner=Identity("telegram", "999"), mode="everyone")
    scenario.msg.mentioned = False
    await handler._process_admitted_interaction(scenario.payload)
    assert scenario.calls == ["free"]
    scenario.group.assert_awaited_once()
    assert await scenario.store.balance(sponsor) == Balance(10, 0)


async def test_paid_requests_still_obey_user_cooldown(scenario):
    await fund(scenario)
    scenario.cooldown.return_value = False
    await handler._process_admitted_interaction(scenario.payload)
    assert scenario.calls == []
    assert await scenario.store.balance(scenario.owner) == Balance(10, 0)


async def test_delivery_exception_refunds_without_free_replay(scenario):
    await fund(scenario)
    scenario.send.side_effect = RuntimeError("delivery failed")
    with pytest.raises(RuntimeError, match="delivery failed"):
        await handler._process_admitted_interaction(scenario.payload)
    assert await scenario.store.balance(scenario.owner) == Balance(10, 0)
    assert scenario.calls == ["paid"]
    scenario.group.assert_not_awaited()
    assert paid_route.get() is None


async def test_partial_delivery_then_exception_keeps_charge(scenario):
    await fund(scenario)

    async def send(**_kwargs):
        await record_delivery()
        raise RuntimeError("second message failed")

    scenario.send.side_effect = send
    with pytest.raises(RuntimeError, match="second message failed"):
        await handler._process_admitted_interaction(scenario.payload)
    assert await scenario.store.balance(scenario.owner) == Balance(8, 0)
    assert scenario.calls == ["paid"]
    scenario.group.assert_not_awaited()
    assert paid_route.get() is None
    assert delivery_callback.get() is None


async def test_successful_action_without_text_is_billable(scenario, monkeypatch):
    await fund(scenario)

    async def action_provider(**_kwargs):
        await record_mock_usage()
        return "[SKIP]", [{"type": "reaction"}]

    monkeypatch.setattr(handler, "call_ai_provider", action_provider)
    monkeypatch.setattr(
        handler,
        "execute_pending_actions",
        AsyncMock(return_value=ActionExecutionResult([], [{"type": "reaction"}])),
    )
    await handler._process_admitted_interaction(scenario.payload)
    assert await scenario.store.balance(scenario.owner) == Balance(8, 0)
    scenario.send.assert_not_awaited()


async def test_replayed_released_event_is_not_delivered_free(scenario, monkeypatch):
    await fund(scenario)
    provider = AsyncMock(return_value=("[SKIP]", []))
    monkeypatch.setattr(handler, "call_ai_provider", provider)
    await handler._process_admitted_interaction(scenario.payload)
    await handler._process_admitted_interaction(scenario.payload)
    provider.assert_awaited_once()
    scenario.group.assert_not_awaited()
    assert await scenario.store.balance(scenario.owner) == Balance(10, 0)
