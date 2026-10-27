"""Account HTTP flow against the real monetary store and an injected gateway."""

import dataclasses
import uuid
from decimal import Decimal
from types import SimpleNamespace

import pytest
import pytest_asyncio
from aiohttp import CookieJar
from aiohttp.test_utils import TestClient, TestServer

from shin_ai.billing import web as account_web
from shin_ai.billing.money import purchase_quote
from shin_ai.billing.store import CreditStore, Identity
from shin_ai.billing.types import Checkout, PaymentState
from shin_ai.settings import BillingSettings, BillingWebSettings, PaymobSettings

ORIGIN = "http://localhost:8080"
OWNER = Identity("discord", "100")
OTHER = Identity("discord", "200")


class Gateway:
    def __init__(self):
        self.created = []
        self.transactions = {}
        self.fail_checkout = False
        self.closed = False

    async def create_checkout(self, purchase, billing_data):
        self.created.append((purchase, billing_data))
        if self.fail_checkout:
            raise TimeoutError("Remote result unknown")
        return Checkout(
            str(len(self.created) + 100),
            f"intention-{len(self.created)}",
            "https://accept.paymob.com/unifiedcheckout/?test=yes",
        )

    def verify_callback(self, body, signature):
        if signature != "verified":
            raise ValueError("Invalid signature")
        return str(body["transaction_id"])

    async def fetch_transaction(self, transaction_id):
        return self.transactions[transaction_id]

    async def close(self):
        self.closed = True


def settings():
    return BillingSettings(
        enabled=True,
        database_path=None,
        providers=(),
        max_request_units=100_000,
        max_retries=1,
        attempt_timeout_seconds=1,
        global_timeout_seconds=3,
        web=BillingWebSettings(enabled=True, public_url=ORIGIN, support_email="help@example.test"),
        paymob=PaymobSettings(
            enabled=True,
            mode="test",
            merchant_id=42,
            integration_ids=(17,),
            egp_per_usd=Decimal("50"),
            fee_bps=250,
            fixed_fee_cents=300,
            packages=(10_000, 20_000),
        ),
    )


@pytest_asyncio.fixture
async def site(tmp_path):
    clock = [1000.0]
    store = CreditStore(tmp_path / "billing.sqlite3", clock=lambda: clock[0])
    gateway = Gateway()
    config = settings()
    client = TestClient(
        TestServer(account_web.create_web_app(store, config, gateway)), cookie_jar=CookieJar(unsafe=True)
    )
    await client.start_server()
    try:
        yield SimpleNamespace(client=client, store=store, gateway=gateway, settings=config, clock=clock)
    finally:
        await client.close()


async def sign_in(site, owner=OWNER):
    token = await site.store.issue_login(owner, 60)
    response = await site.client.post("/auth/exchange", json={"token": token}, headers={"Origin": ORIGIN})
    assert response.status == 200
    cookie = response.cookies[account_web.COOKIE]
    session = await site.store.get_session(cookie.value)
    assert session.owner == owner
    return session


def checkout_data(session, *, reference=None):
    return {
        "csrf": session.csrf,
        "reference": reference or str(uuid.uuid4()),
        "amount_cents": "10000",
        "first_name": "Maya",
        "last_name": "Reader",
        "email": "maya@example.test",
        "phone_number": "+201001234567",
    }


def state(purchase, order="101", **changes):
    base = PaymentState(
        order_id=order,
        transaction_id="9001",
        amount_cents=purchase.amount_cents,
        currency="EGP",
        success=True,
        pending=False,
        voided=False,
        refunded_cents=0,
        integration_id=17,
        merchant_id=42,
        is_live=False,
        special_reference=purchase.reference,
    )
    return dataclasses.replace(base, **changes)


@pytest.mark.asyncio
async def test_login_is_explicit_single_use_and_does_not_leak_token(site):
    token = await site.store.issue_login(OWNER, 60)
    page = await site.client.get("/login")
    assert page.status == 200
    assert token not in await page.text()
    assert "no-store" in page.headers["Cache-Control"]
    assert "frame-ancestors 'none'" in page.headers["Content-Security-Policy"]
    invalid_origin = await site.client.post(
        "/auth/exchange", json={"token": token}, headers={"Origin": "https://attacker.test"}
    )
    assert invalid_origin.status == 403
    valid = await site.client.post("/auth/exchange", json={"token": token}, headers={"Origin": ORIGIN})
    assert valid.status == 200
    cookie = valid.cookies[account_web.COOKIE]
    assert cookie["httponly"] and cookie["samesite"] == "Lax"
    replay = await site.client.post("/auth/exchange", json={"token": token}, headers={"Origin": ORIGIN})
    assert replay.status == 401


@pytest.mark.asyncio
async def test_expired_login_and_session_are_rejected(site):
    token = await site.store.issue_login(OWNER, 1)
    site.clock[0] += 2
    response = await site.client.post("/auth/exchange", json={"token": token}, headers={"Origin": ORIGIN})
    assert response.status == 401
    session = await sign_in(site)
    site.clock[0] = session.expires_at + 1
    page = await site.client.get("/")
    assert "get a sign-in link" in await page.text()
    response = await site.client.post("/checkout", data=checkout_data(session), headers={"Origin": ORIGIN})
    assert response.status == 401


@pytest.mark.asyncio
async def test_account_and_checkout_use_session_owner_and_real_payment_confirmation(site):
    await site.store.grant(OTHER, 9_000_000, "other-money")
    session = await sign_in(site)
    page = await site.client.get("/?user=200")
    html = await page.text()
    assert "$9.00" not in html
    assert "Discord · 100" in html
    assert "Adds $1.89" in html
    data = checkout_data(session)
    data.update({"owner": "200", "credited_units": "999999999"})
    checkout = await site.client.post(
        "/checkout", data=data, headers={"Origin": ORIGIN}, allow_redirects=False
    )
    assert checkout.status == 303
    purchase = site.gateway.created[0][0]
    assert purchase.owner == OWNER and purchase.credited_units == 1_890_000
    assert (await site.store.balance(OWNER)).available == 0
    # Browser success parameters never fund the balance.
    returned = await site.client.get("/payment-return?success=true&amount_cents=10000")
    assert returned.status == 200
    assert (await site.store.balance(OWNER)).available == 0
    site.gateway.transactions["9001"] = state(purchase)
    for _ in range(2):
        response = await site.client.post("/webhooks/paymob?hmac=verified", json={"transaction_id": "9001"})
        assert response.status == 200
    assert (await site.store.balance(OWNER)).available == 1_890_000
    assert (await site.store.balance(OTHER)).available == 9_000_000
    html = await (await site.client.get("/")).text()
    assert purchase.reference in html and "Paid" in html
    logout = await site.client.post(
        "/logout", data={"csrf": session.csrf}, headers={"Origin": ORIGIN}, allow_redirects=False
    )
    assert logout.status == 303
    assert (
        await site.store.get_session(
            site.client.session.cookie_jar.filter_cookies(site.client.make_url("/")).get(
                account_web.COOKIE, ""
            )
        )
        is None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["csrf", "origin", "package", "contact", "reference"])
async def test_checkout_rejects_forgery_and_invalid_form(site, change):
    session = await sign_in(site)
    data = checkout_data(session)
    origin = ORIGIN
    if change == "csrf":
        data["csrf"] = "wrong"
    elif change == "origin":
        origin = "https://attacker.test"
    elif change == "package":
        data["amount_cents"] = "1"
    elif change == "contact":
        data["email"] = "not-an-email"
    else:
        data["reference"] = "invalid"
    response = await site.client.post(
        "/checkout", data=data, headers={"Origin": origin}, allow_redirects=False
    )
    assert response.status in {400, 403}
    assert not site.gateway.created


@pytest.mark.asyncio
async def test_checkout_reference_cannot_be_taken_from_another_owner(site):
    reference = str(uuid.uuid4())
    units, quote = purchase_quote(site.settings.paymob, 10_000)
    await site.store.create_purchase(OTHER, reference, 10_000, "EGP", units, quote)
    session = await sign_in(site)
    response = await site.client.post(
        "/checkout",
        data=checkout_data(session, reference=reference),
        headers={"Origin": ORIGIN},
        allow_redirects=False,
    )
    assert response.status == 400 and not site.gateway.created


@pytest.mark.asyncio
async def test_checkout_retry_and_ambiguous_timeout_never_create_twice(site):
    session = await sign_in(site)
    data = checkout_data(session)
    site.gateway.fail_checkout = True
    first = await site.client.post("/checkout", data=data, headers={"Origin": ORIGIN}, allow_redirects=False)
    second = await site.client.post("/checkout", data=data, headers={"Origin": ORIGIN}, allow_redirects=False)
    assert first.status == 503 and second.status == 409
    assert len(site.gateway.created) == 1
    assert (await site.store.balance(OWNER)).available == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changed",
    [
        {"amount_cents": 5000},
        {"currency": "USD"},
        {"merchant_id": 99},
        {"integration_id": 99},
        {"is_live": True},
    ],
)
async def test_mismatched_canonical_payments_never_fund(site, changed):
    session = await sign_in(site)
    await site.client.post(
        "/checkout", data=checkout_data(session), headers={"Origin": ORIGIN}, allow_redirects=False
    )
    purchase = site.gateway.created[0][0]
    site.gateway.transactions["9001"] = state(purchase, **changed)
    response = await site.client.post("/webhooks/paymob?hmac=verified", json={"transaction_id": "9001"})
    assert response.status == 400
    assert (await site.store.balance(OWNER)).available == 0


@pytest.mark.asyncio
async def test_pending_and_forged_notifications_do_not_fund(site):
    session = await sign_in(site)
    await site.client.post(
        "/checkout", data=checkout_data(session), headers={"Origin": ORIGIN}, allow_redirects=False
    )
    purchase = site.gateway.created[0][0]
    site.gateway.transactions["9001"] = state(purchase, pending=True)
    forged = await site.client.post("/webhooks/paymob?hmac=forged", json={"transaction_id": "9001"})
    assert forged.status == 400
    pending = await site.client.post("/webhooks/paymob?hmac=verified", json={"transaction_id": "9001"})
    assert pending.status == 200
    assert (await site.store.balance(OWNER)).available == 0


@pytest.mark.asyncio
async def test_telegram_accounts_have_no_external_checkout(site):
    session = await sign_in(site, Identity("telegram", "100"))
    html = await (await site.client.get("/")).text()
    assert "Telegram Stars" in html and '<form method="post" action="/checkout">' not in html
    response = await site.client.post(
        "/checkout", data=checkout_data(session), headers={"Origin": ORIGIN}, allow_redirects=False
    )
    assert response.status == 403 and not site.gateway.created


@pytest.mark.asyncio
async def test_sensitive_identifiers_are_escaped_and_assets_are_same_origin(site):
    await sign_in(site, Identity("discord", '<script>alert("x")</script>'))
    html = await (await site.client.get("/")).text()
    assert '<script>alert("x")</script>' not in html
    assert "&lt;script&gt;" in html
    js = await site.client.get("/assets/account.js")
    assert js.status == 200
    source = await js.text()
    assert "replaceState" in source and "localStorage" not in source
    missing = await site.client.get("/assets/secrets.py")
    assert missing.status == 404


def test_remote_http_public_url_is_rejected(tmp_path):
    config = settings()
    config = dataclasses.replace(
        config, web=dataclasses.replace(config.web, public_url="http://account.example.test")
    )
    with pytest.raises(ValueError, match="HTTPS"):
        account_web.create_web_app(CreditStore(tmp_path / "bill.sqlite3"), config, Gateway())


@pytest.mark.asyncio
async def test_linking_requires_private_proof_and_combines_balances_once(site):
    session = await sign_in(site)
    source = Identity("whatsapp", "201001234567@s.whatsapp.net")
    await site.store.grant(OWNER, 2_000_000, "primary")
    await site.store.grant(source, 3_000_000, "source")
    proof = await site.store.issue_login(source, 60)
    page = await site.client.get("/login")
    assert 'id="link-account"' in await page.text()
    forged = await site.client.post(
        "/auth/link",
        json={"token": proof, "csrf": "wrong"},
        headers={"Origin": ORIGIN},
    )
    assert forged.status == 403
    assert (await site.store.balance(OWNER)).available == 2_000_000
    linked = await site.client.post(
        "/auth/link",
        json={"token": proof, "csrf": session.csrf, "owner": "attacker"},
        headers={"Origin": ORIGIN},
    )
    assert linked.status == 200
    assert (await site.store.balance(OWNER)).available == 5_000_000
    assert (await site.store.balance(source)).available == 5_000_000
    replay = await site.client.post(
        "/auth/link",
        json={"token": proof, "csrf": session.csrf},
        headers={"Origin": ORIGIN},
    )
    assert replay.status == 409
    assert (await site.store.balance(OWNER)).available == 5_000_000


@pytest.mark.asyncio
async def test_expired_link_proof_does_not_merge_accounts(site):
    session = await sign_in(site)
    proof = await site.store.issue_login(OTHER, 1)
    site.clock[0] += 2
    response = await site.client.post(
        "/auth/link",
        json={"token": proof, "csrf": session.csrf},
        headers={"Origin": ORIGIN},
    )
    assert response.status == 409


@pytest.mark.asyncio
async def test_telegram_signin_cannot_bypass_checkout_after_linking_to_discord(site):
    session = await sign_in(site)
    telegram = Identity("telegram", "300")
    proof = await site.store.issue_login(telegram, 60)
    linked = await site.client.post(
        "/auth/link",
        json={"token": proof, "csrf": session.csrf},
        headers={"Origin": ORIGIN},
    )
    assert linked.status == 200
    token = await site.store.issue_login(telegram, 60)
    response = await site.client.post("/auth/exchange", json={"token": token}, headers={"Origin": ORIGIN})
    assert response.status == 200
    session = await site.store.get_session(response.cookies[account_web.COOKIE].value)
    assert session.owner == OWNER and session.login_platform == "telegram"
    html = await (await site.client.get("/")).text()
    assert "Telegram Stars" in html and '<form method="post" action="/checkout">' not in html
    refused = await site.client.post(
        "/checkout", data=checkout_data(session), headers={"Origin": ORIGIN}, allow_redirects=False
    )
    assert refused.status == 403 and not site.gateway.created


@pytest.mark.asyncio
async def test_canonical_callback_restores_checkout_lost_to_timeout(site):
    session = await sign_in(site)
    site.gateway.fail_checkout = True
    response = await site.client.post(
        "/checkout", data=checkout_data(session), headers={"Origin": ORIGIN}, allow_redirects=False
    )
    assert response.status == 503
    purchase = site.gateway.created[0][0]
    site.gateway.transactions["9001"] = state(purchase)
    response = await site.client.post("/webhooks/paymob?hmac=verified", json={"transaction_id": "9001"})
    assert response.status == 200
    assert (await site.store.balance(OWNER)).available == purchase.credited_units


@pytest.mark.asyncio
async def test_production_cookie_is_secure_and_body_size_is_bounded(tmp_path):
    store = CreditStore(tmp_path / "billing.sqlite3")
    config = settings()
    config = dataclasses.replace(
        config, web=dataclasses.replace(config.web, public_url="https://account.example.test")
    )
    async with TestClient(TestServer(account_web.create_web_app(store, config, Gateway()))) as client:
        token = await store.issue_login(OWNER, 60)
        response = await client.post(
            "/auth/exchange", json={"token": token}, headers={"Origin": "https://account.example.test"}
        )
        assert response.status == 200 and response.cookies[account_web.COOKIE]["secure"]
        oversized = await client.post(
            "/auth/exchange", data="x" * (33 * 1024), headers={"Origin": "https://account.example.test"}
        )
        assert oversized.status == 413


def test_signin_script_waits_for_click_and_clears_fragment_before_exchange():
    import shutil
    import subprocess
    from pathlib import Path

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is unavailable for the standalone browser script check")
    script = Path(account_web.__file__).with_name("static") / "account.js"
    result = subprocess.run(
        [
            node,
            "-e",
            r"""
const fs = require('fs');
const vm = require('vm');
const calls = [];
const buttons = {};
for (const id of ['sign-in', 'link-account']) {
  buttons[id] = { hidden: true, disabled: false, dataset: {csrf: 'proof'},
    addEventListener(name, handler) { this[name] = handler; } };
}
const elements = {...buttons, 'login-status': {}, 'link-choice': {hidden: true}};
const context = {
  URLSearchParams,
  document: {getElementById(id) { return elements[id] || null; }},
  window: {location: {hash: '#token=private-proof', pathname: '/login',
    replace(path) {calls.push(['navigate', path]);}},
    history: {replaceState(_state, _title, path) {calls.push(['clear', path]);}}},
  fetch: async (path, options) => {
    calls.push(['fetch', path, JSON.parse(options.body), options.credentials]);
    return {ok: true, json: async () => ({ok: true})};
  }
};
vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8'), context);
if (calls.length !== 1 || calls[0][0] !== 'clear') throw Error('Link was redeemed before a click');
if (elements['link-choice'].hidden) throw Error('Linking choice was not exposed');
Promise.resolve(buttons['link-account'].click()).then(() => {
  const exchange = calls[1];
  if (exchange[1] !== '/auth/link' || exchange[2].token !== 'private-proof' ||
      exchange[2].csrf !== 'proof' || exchange[3] !== 'same-origin') throw Error('Wrong exchange');
  if (calls[2][0] !== 'navigate' || calls[2][1] !== '/') throw Error('Missing account navigation');
}).catch(error => {console.error(error); process.exitCode = 1;});
""",
            str(script),
        ],
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.asyncio
async def test_refund_child_notification_applies_verified_parent_state(site):
    session = await sign_in(site)
    await site.client.post("/checkout", data=checkout_data(session), headers={"Origin": ORIGIN})
    purchase = site.gateway.created[0][0]
    site.gateway.transactions["9001"] = state(purchase)
    await site.client.post("/webhooks/paymob?hmac=verified", json={"transaction_id": "9001"})
    # PaymobClient verifies the child's ID, then returns the authenticated original payment.
    site.gateway.transactions["9002"] = state(purchase, refunded_cents=purchase.amount_cents)
    response = await site.client.post("/webhooks/paymob?hmac=verified", json={"transaction_id": "9002"})
    assert response.status == 200
    assert (await site.store.balance(OWNER)).available == 0
    assert (await site.store.get_purchase(purchase.reference)).status == "refunded"


@pytest.mark.asyncio
async def test_purchase_to_metered_reply_receipt_and_spent_refund(site):
    from shin_ai.billing.pricing import UsageMeter
    from shin_ai.billing.store import Scope
    from shin_ai.billing.types import ModelPrice

    session = await sign_in(site)
    await site.client.post("/checkout", data=checkout_data(session), headers={"Origin": ORIGIN})
    purchase = site.gateway.created[0][0]
    site.gateway.transactions["9001"] = state(purchase)
    await site.client.post("/webhooks/paymob?hmac=verified", json={"transaction_id": "9001"})
    scope = Scope("discord", "chat", "friends")
    await site.store.configure(OWNER, scope, "everyone")
    reservation = await site.store.reserve("reply-1", OTHER, (scope,), 100_000)
    assert reservation.owner == OWNER
    await site.store.configure_reservation(reservation, 2500)
    price = ModelPrice("paid", "test", "deepseek", "test-v1", Decimal("1"), Decimal("0.1"), Decimal("2"))
    meter = UsageMeter(site.store, reservation, (price,), 2500)
    ticket = await meter.before_call("paid", "test", 1000, 100)
    await meter.record_response(
        ticket,
        SimpleNamespace(
            model="test",
            usage={
                "prompt_tokens": 1000,
                "prompt_cache_hit_tokens": 200,
                "prompt_cache_miss_tokens": 800,
                "completion_tokens": 100,
            },
        ),
    )
    await meter.accept_attempt(ticket.attempt)
    await meter.mark_delivered()
    await meter.finish()
    # $0.00102 upstream + 25% markup = $0.001275 customer charge.
    assert (await site.store.balance(OWNER)).available == purchase.credited_units - 1275
    html = await (await site.client.get("/")).text()
    assert "$0.001275" in html and "reply-1" in html
    site.gateway.transactions["9001"] = state(purchase, refunded_cents=purchase.amount_cents)
    await site.client.post("/webhooks/paymob?hmac=verified", json={"transaction_id": "9001"})
    balance = await site.store.balance(OWNER)
    assert balance.available == 0 and balance.debt == 1275
    assert await site.store.reserve("blocked-reply", OTHER, (scope,), 100_000) is None
