import copy
import hashlib
import hmac
from types import SimpleNamespace

import httpx
import pytest

from shin_ai.billing.paymob import GatewayError, InvalidPayment, PaymobClient
from shin_ai.settings import PaymobSettings

pytestmark = pytest.mark.asyncio


@pytest.fixture
def configured(monkeypatch):
    for key, value in {
        "PAYMOB_SECRET_KEY": "sk_test_secret",
        "PAYMOB_PUBLIC_KEY": "pk_test_public",
        "PAYMOB_HMAC_SECRET": "merchant-hmac",
        "PAYMOB_API_KEY": "legacy-api-key",
    }.items():
        monkeypatch.setenv(key, value)
    return PaymobSettings(enabled=True, integration_ids=(6741,), merchant_id=992246)


@pytest.fixture
def transaction():
    return {
        "id": 2556706,
        "amount_cents": 100,
        "currency": "EGP",
        "created_at": "2020-03-25T18:39:44.719228",
        "integration_id": 6741,
        "error_occured": False,
        "has_parent_transaction": False,
        "is_3d_secure": True,
        "is_auth": False,
        "is_capture": False,
        "is_refunded": False,
        "is_standalone_payment": True,
        "is_voided": False,
        "owner": 4705,
        "pending": False,
        "success": True,
        "is_live": False,
        "refunded_amount_cents": 0,
        "source_data": {"pan": "2346", "sub_type": "MasterCard", "type": "card"},
        "order": {
            "id": 4778239,
            "currency": "EGP",
            "merchant_order_id": "purchase-1",
            "merchant": {"id": 992246},
        },
    }


def signature():
    # Official Paymob worked example, with a test-only merchant secret.
    message = (
        "1002020-03-25T18:39:44.719228EGPfalsefalse25567066741truefalsefalsefalse"
        "truefalse47782394705false2346MasterCardcardtrue"
    )
    return hmac.new(b"merchant-hmac", message.encode(), hashlib.sha512).hexdigest()


async def test_official_hmac_example_and_unsigned_refund_fields(configured, transaction):
    gateway = PaymobClient(configured, "https://shin.example")
    assert gateway.verify_callback({"obj": transaction}, signature()) == "2556706"
    transaction["refunded_amount_cents"] = 100
    transaction["order"]["merchant_order_id"] = "untrusted-other-owner"
    # Returned ID is all the caller may use; it must then perform authenticated inquiry.
    assert gateway.verify_callback({"obj": transaction}, signature()) == "2556706"


@pytest.mark.parametrize(
    "field,value",
    [
        ("amount_cents", 1000),
        ("success", "true"),
        ("id", {"id": 2556706}),
        ("is_auth", 0),
        ("pending", None),
    ],
)
async def test_hmac_rejects_tampering_and_noncanonical_types(configured, transaction, field, value):
    gateway = PaymobClient(configured, "https://shin.example")
    transaction[field] = value
    with pytest.raises(InvalidPayment):
        gateway.verify_callback({"obj": transaction}, signature())


@pytest.mark.parametrize("received", ["", "x" * 128, "0" * 127, None])
async def test_hmac_rejects_missing_or_malformed_signature(configured, transaction, received):
    with pytest.raises(InvalidPayment):
        PaymobClient(configured, "https://shin.example").verify_callback({"obj": transaction}, received)


async def test_create_checkout_uses_intention_contract(configured):
    requests = []

    async def respond(request):
        requests.append(request)
        assert str(request.url) == "https://accept.paymob.com/v1/intention/"
        assert request.headers["Authorization"] == "Token sk_test_secret"
        import json

        body = json.loads(request.content)
        assert body["amount"] == 100
        assert body["special_reference"] == "purchase-1"
        assert body["payment_methods"] == [6741]
        assert body["notification_url"] == "https://shin.example/webhooks/paymob"
        assert body["redirection_url"] == "https://shin.example/payment-return"
        assert body["billing_data"]["first_name"] == "Ayman"
        assert "owner" not in body
        return httpx.Response(
            201,
            json={
                "intention_order_id": 4778239,
                "id": "pi_test_example",
                "client_secret": "egy_csk_test_example",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        gateway = PaymobClient(configured, "https://shin.example/", client=client)
        checkout = await gateway.create_checkout(
            SimpleNamespace(reference="purchase-1", amount_cents=100, currency="EGP"),
            {
                "first_name": "Ayman",
                "last_name": "K",
                "email": "a@example.com",
                "phone_number": "+201234567890",
            },
        )
    assert checkout.order_id == "4778239"
    assert checkout.intention_id == "pi_test_example"
    assert checkout.url == (
        "https://accept.paymob.com/unifiedcheckout/?publicKey=pk_test_public&clientSecret=egy_csk_test_example"
    )
    assert len(requests) == 1


async def test_inquiry_uses_api_key_and_canonical_merchant_id(configured, transaction):
    calls = []

    def respond(request):
        calls.append(request)
        if request.url.path == "/api/auth/tokens":
            assert request.method == "POST"
            assert request.content == b'{"api_key":"legacy-api-key"}'
            return httpx.Response(200, json={"token": "inquiry-token"})
        assert request.url.path == "/api/acceptance/transactions/2556706"
        assert request.headers["Authorization"] == "Bearer inquiry-token"
        assert not request.url.query
        return httpx.Response(200, json=transaction)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        state = await PaymobClient(configured, "https://shin.example", client=client).fetch_transaction(
            "2556706"
        )
    assert state.success and not state.pending
    assert state.merchant_id == 992246  # The callback owner user ID is a different concept.
    assert state.special_reference == "purchase-1"
    assert len(calls) == 2


@pytest.mark.parametrize(
    "change",
    [
        {"integration_id": 123},
        {"is_live": True},
        {"success": "true"},
        {"refunded_amount_cents": -1},
        {"refunded_amount_cents": 101},
        {"id": 2556707},
        {"order": {"merchant": {"id": 999}}},
    ],
)
async def test_inquiry_rejects_mismatched_or_malformed_payment(configured, transaction, change):
    transaction.update(change)

    def respond(request):
        return httpx.Response(
            200, json={"token": "token"} if request.url.path == "/api/auth/tokens" else transaction
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(InvalidPayment):
            await PaymobClient(configured, "https://shin.example", client=client).fetch_transaction("2556706")


async def test_reference_lookup_uses_documented_endpoint(configured, transaction):
    paths = []

    def respond(request):
        paths.append(request.url.path)
        if request.url.path == "/api/auth/tokens":
            return httpx.Response(200, json={"token": "token"})
        if request.url.path == "/api/ecommerce/orders/transaction_inquiry":
            assert request.method == "POST"
            assert request.content == b'{"auth_token":"token","merchant_order_id":"purchase-1"}'
        return httpx.Response(200, json=transaction)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        state = await PaymobClient(configured, "https://shin.example", client=client).lookup_purchase(
            "purchase-1"
        )
    assert state.transaction_id == "2556706"
    assert "/api/ecommerce/orders/transaction_inquiry" in paths


async def test_refund_requeries_original_payment(configured, transaction):
    refunded = False

    def respond(request):
        nonlocal refunded
        if request.url.path == "/api/auth/tokens":
            return httpx.Response(200, json={"token": "token"})
        if request.url.path == "/api/acceptance/void_refund/refund":
            assert request.headers["Authorization"] == "Token sk_test_secret"
            assert request.content == b'{"transaction_id":2556706,"amount_cents":40}'
            refunded = True
            # The refund response itself is deliberately unsuitable as a source of truth.
            return httpx.Response(201, json={"success": True})
        result = copy.deepcopy(transaction)
        result["refunded_amount_cents"] = 40 if refunded else 0
        return httpx.Response(200, json=result)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        state = await PaymobClient(configured, "https://shin.example", client=client).refund("2556706", 40)
    assert state.refunded_cents == 40


async def test_refund_child_resolves_parent(configured, transaction):
    child = copy.deepcopy(transaction)
    child.update(id=2556707, has_parent_transaction=True, parent_transaction="2556706", amount_cents=40)
    transaction["refunded_amount_cents"] = 40

    def respond(request):
        if request.url.path == "/api/auth/tokens":
            return httpx.Response(200, json={"token": "token"})
        return httpx.Response(200, json=child if request.url.path.endswith("2556707") else transaction)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        state = await PaymobClient(configured, "https://shin.example", client=client).fetch_transaction(
            "2556707"
        )
    assert state.transaction_id == "2556706"
    assert state.refunded_cents == 40


async def test_unconfigured_credentials_never_make_request(configured, monkeypatch):
    monkeypatch.delenv("PAYMOB_API_KEY")
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: pytest.fail("network"))
    ) as client:
        with pytest.raises(GatewayError, match="credential"):
            await PaymobClient(configured, "https://shin.example", client=client).fetch_transaction("2556706")


async def test_network_failure_is_unknown_and_does_not_leak_secret(configured):
    def respond(request):
        raise httpx.ReadTimeout("request has legacy-api-key", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(GatewayError, match="unknown") as error:
            await PaymobClient(configured, "https://shin.example", client=client).fetch_transaction("2556706")
    assert "legacy-api-key" not in str(error.value)


async def test_test_live_key_mismatch_is_rejected(configured, monkeypatch):
    monkeypatch.setenv("PAYMOB_SECRET_KEY", "sk_live_secret")
    with pytest.raises(GatewayError, match="test/live"):
        await PaymobClient(configured, "https://shin.example").create_checkout(None, {})
