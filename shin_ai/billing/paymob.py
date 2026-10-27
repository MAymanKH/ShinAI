"""Paymob Egypt hosted checkout and authenticated payment-state verification."""

from __future__ import annotations

import hashlib
import hmac as hmac_module
import os
import re
from typing import Any
from urllib.parse import urlencode, urlsplit

import httpx

from shin_ai.billing.types import Checkout, PaymentState

BASE_URL = "https://accept.paymob.com"
HMAC_FIELDS = (
    "amount_cents",
    "created_at",
    "currency",
    "error_occured",
    "has_parent_transaction",
    "id",
    "integration_id",
    "is_3d_secure",
    "is_auth",
    "is_capture",
    "is_refunded",
    "is_standalone_payment",
    "is_voided",
    "order.id",
    "owner",
    "pending",
    "source_data.pan",
    "source_data.sub_type",
    "source_data.type",
    "success",
)
_BOOL_FIELDS = frozenset(
    {
        "error_occured",
        "has_parent_transaction",
        "is_3d_secure",
        "is_auth",
        "is_capture",
        "is_refunded",
        "is_standalone_payment",
        "is_voided",
        "pending",
        "success",
    }
)


class GatewayError(RuntimeError):
    """A gateway operation failed or its remote outcome is not known."""


class InvalidPayment(ValueError):
    """A callback or canonical transaction cannot safely fund this installation."""


def _identifier(value: Any) -> str:
    if type(value) is int and 0 < value < 2**63:
        return str(value)
    if isinstance(value, str) and re.fullmatch(r"[0-9]{1,19}", value) and 0 < int(value) < 2**63:
        return value
    raise InvalidPayment("Invalid gateway identifier")


def _integer(value: Any, *, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value < 2**63:
        raise InvalidPayment("Invalid gateway amount or identifier")
    return value


def _boolean(data: dict, key: str) -> bool:
    value = data.get(key)
    if type(value) is not bool:
        raise InvalidPayment("Invalid gateway status")
    return value


class PaymobClient:
    def __init__(self, paymob_settings, public_url: str, *, client: httpx.AsyncClient | None = None):
        self.settings = paymob_settings
        self.public_url = public_url.rstrip("/")
        self._client = client
        self._owns_client = client is None

    def _secret(self, setting: str, *, kind: str | None = None) -> str:
        if not self.settings.enabled or self.settings.mode not in {"test", "live"}:
            raise GatewayError("Paymob is disabled")
        value = os.environ.get(getattr(self.settings, setting), "")
        if not value or value != value.strip() or len(value) > 16_384:
            raise GatewayError("Required Paymob credential is unavailable")
        if kind is not None and not re.fullmatch(
            rf"(?:[a-z]{{3}}_)?{kind}_{self.settings.mode}_[A-Za-z0-9_-]+",
            value,
        ):
            raise GatewayError("Paymob key does not match the configured test/live mode")
        return value

    async def _request(self, method: str, path: str, **kwargs) -> dict:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=20, follow_redirects=False)
        try:
            response = await self._client.request(
                method,
                BASE_URL + path,
                timeout=20,
                follow_redirects=False,
                **kwargs,
            )
        except httpx.HTTPError:
            raise GatewayError("Paymob request outcome is unknown; reconcile before retrying") from None
        if response.status_code >= 400 or response.is_redirect:
            raise GatewayError(f"Paymob request failed with HTTP {response.status_code}")
        try:
            data = response.json()
        except ValueError:
            raise GatewayError("Paymob returned an invalid response") from None
        if not isinstance(data, dict):
            raise GatewayError("Paymob returned an invalid response")
        return data

    async def create_checkout(self, purchase, billing_data: dict) -> Checkout:
        secret = self._secret("secret_key_env", kind="sk")
        public = self._secret("public_key_env", kind="pk")
        if not self.settings.integration_ids:
            raise GatewayError("No Paymob payment integrations are configured")
        parsed = urlsplit(self.public_url)
        if parsed.scheme != "https" and not (
            self.settings.mode == "test" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        ):
            raise GatewayError("Paymob checkout requires an HTTPS public URL")
        customer = {}
        for key in ("first_name", "last_name", "email", "phone_number"):
            value = billing_data.get(key)
            if not isinstance(value, str) or not value.strip() or len(value) > 254:
                raise InvalidPayment("Provide name, email, and phone number for checkout")
            customer[key] = value.strip()
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", customer["email"]):
            raise InvalidPayment("Provide a valid checkout email")
        if not re.fullmatch(r"[+0-9 ()-]{6,40}", customer["phone_number"]):
            raise InvalidPayment("Provide a valid checkout phone number")
        customer.update({key: "NA" for key in ("apartment", "floor", "street", "building", "city", "state")})
        amount = _integer(purchase.amount_cents, minimum=1)
        data = await self._request(
            "POST",
            "/v1/intention/",
            headers={"Authorization": f"Token {secret}"},
            json={
                "amount": amount,
                "currency": purchase.currency,
                "payment_methods": list(self.settings.integration_ids),
                "items": [{"name": "ShinAI usage balance", "amount": amount, "quantity": 1}],
                "billing_data": customer,
                "special_reference": purchase.reference,
                "notification_url": self.public_url + "/webhooks/paymob",
                "redirection_url": self.public_url + "/payment-return",
            },
        )
        order_id = _identifier(data.get("intention_order_id"))
        intention_id, client_secret = data.get("id"), data.get("client_secret")
        if not all(
            isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,512}", value)
            for value in (intention_id, client_secret)
        ):
            raise GatewayError("Paymob checkout identity is missing; reconcile this purchase")
        return Checkout(
            order_id,
            intention_id,
            BASE_URL
            + "/unifiedcheckout/?"
            + urlencode(
                {
                    "publicKey": public,
                    "clientSecret": client_secret,
                }
            ),
        )

    def verify_callback(self, body: dict, hmac: str) -> str:
        secret = self._secret("hmac_secret_env")
        if not isinstance(body, dict) or not isinstance(body.get("obj"), dict):
            raise InvalidPayment("Invalid payment callback")
        if not isinstance(hmac, str) or not re.fullmatch(r"[0-9a-f]{128}", hmac):
            raise InvalidPayment("Invalid payment signature")
        transaction = body["obj"]
        values = []
        for field in HMAC_FIELDS:
            value: Any = transaction
            for part in field.split("."):
                if not isinstance(value, dict) or part not in value:
                    raise InvalidPayment("Incomplete signed payment callback")
                value = value[part]
            if field in _BOOL_FIELDS:
                if type(value) is not bool:
                    raise InvalidPayment("Invalid signed payment status")
                values.append("true" if value else "false")
            elif type(value) is int or isinstance(value, str):
                if len(str(value)) > 1024:
                    raise InvalidPayment("Invalid signed payment value")
                values.append(str(value))
            else:
                # Paymob does not document null canonicalization for all methods.
                raise InvalidPayment("Unsupported signed payment value")
        expected = hmac_module.new(secret.encode(), "".join(values).encode(), hashlib.sha512).hexdigest()
        if not hmac_module.compare_digest(expected, hmac):
            raise InvalidPayment("Invalid payment signature")
        if _integer(transaction.get("integration_id"), minimum=1) not in self.settings.integration_ids:
            raise InvalidPayment("Payment belongs to another integration")
        _identifier(transaction["order"]["id"])
        _integer(transaction["amount_cents"], minimum=1)
        return _identifier(transaction.get("id"))

    async def _auth_token(self) -> str:
        api_key = self._secret("api_key_env")
        data = await self._request("POST", "/api/auth/tokens", json={"api_key": api_key})
        token = data.get("token")
        if not isinstance(token, str) or not token or len(token) > 16_384:
            raise GatewayError("Paymob inquiry authentication failed")
        return token

    def _payment(self, data: dict) -> PaymentState:
        try:
            order = data["order"]
            merchant_id = _integer(order["merchant"]["id"], minimum=1)
            integration = _integer(data["integration_id"], minimum=1)
            if merchant_id != self.settings.merchant_id or integration not in self.settings.integration_ids:
                raise InvalidPayment("Payment belongs to another merchant or integration")
            live = _boolean(data, "is_live")
            if live != (self.settings.mode == "live"):
                raise InvalidPayment("Payment test/live mode does not match")
            amount = _integer(data["amount_cents"], minimum=1)
            refunded = _integer(data["refunded_amount_cents"])
            if refunded > amount or order.get("currency") != data["currency"]:
                raise InvalidPayment("Invalid payment currency or refund amount")
            if _boolean(data, "is_refunded") and refunded == 0:
                raise InvalidPayment("Refunded payment is missing its cumulative refund amount")
            if not isinstance(data["currency"], str) or not re.fullmatch(r"[A-Z]{3}", data["currency"]):
                raise InvalidPayment("Invalid payment currency")
            special_reference = order.get("merchant_order_id")
            if special_reference is not None and (
                not isinstance(special_reference, str) or len(special_reference) > 256
            ):
                raise InvalidPayment("Invalid purchase reference")
            return PaymentState(
                _identifier(order["id"]),
                _identifier(data["id"]),
                amount,
                data["currency"],
                _boolean(data, "success")
                and not _boolean(data, "error_occured")
                and not _boolean(data, "is_auth"),
                _boolean(data, "pending"),
                _boolean(data, "is_voided"),
                refunded,
                integration,
                merchant_id,
                live,
                special_reference,
            )
        except (KeyError, TypeError):
            raise InvalidPayment("Incomplete canonical payment response") from None

    async def fetch_transaction(self, transaction_id: str) -> PaymentState:
        transaction_id = _identifier(transaction_id)
        token = await self._auth_token()
        data = await self._request(
            "GET",
            f"/api/acceptance/transactions/{transaction_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        if _identifier(data.get("id")) != transaction_id:
            raise InvalidPayment("Payment inquiry returned a different transaction")
        has_parent = _boolean(data, "has_parent_transaction")
        if (data.get("is_refund") is True or data.get("is_void") is True) and not has_parent:
            raise InvalidPayment("Refund or void transaction is missing its original payment")
        if has_parent:
            # Refund callbacks may identify the child rather than the original payment.
            parent = _identifier(data.get("parent_transaction"))
            if parent == transaction_id:
                raise InvalidPayment("Invalid payment parent")
            original = await self._request(
                "GET",
                f"/api/acceptance/transactions/{parent}",
                headers={"Authorization": f"Bearer {token}"},
            )
            if _identifier(original.get("id")) != parent or _boolean(original, "has_parent_transaction"):
                raise InvalidPayment("Invalid payment parent")
            child, payment = self._payment(data), self._payment(original)
            if child.order_id != payment.order_id:
                raise InvalidPayment("Refund belongs to another order")
            return payment
        return self._payment(data)

    async def lookup_purchase(
        self, reference: str | None = None, *, order_id: str | None = None
    ) -> PaymentState:
        if bool(reference) == bool(order_id):
            raise InvalidPayment("Supply one purchase reference or gateway order ID")
        payload = {"auth_token": await self._auth_token()}
        if order_id is not None:
            payload["order_id"] = _identifier(order_id)
        else:
            if not isinstance(reference, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", reference):
                raise InvalidPayment("Invalid purchase reference")
            payload["merchant_order_id"] = reference
        data = await self._request("POST", "/api/ecommerce/orders/transaction_inquiry", json=payload)
        payment = await self.fetch_transaction(_identifier(data.get("id")))
        if (order_id is not None and payment.order_id != str(order_id)) or (
            reference is not None and payment.special_reference != reference
        ):
            raise InvalidPayment("Payment inquiry returned another purchase")
        return payment

    async def refund(self, transaction_id: str, amount_cents: int) -> PaymentState:
        transaction_id = _identifier(transaction_id)
        amount = _integer(amount_cents, minimum=1)
        current = await self.fetch_transaction(transaction_id)
        if (
            current.pending
            or not current.success
            or current.voided
            or amount > current.amount_cents - current.refunded_cents
        ):
            raise InvalidPayment("Payment cannot be refunded by this amount")
        await self._request(
            "POST",
            "/api/acceptance/void_refund/refund",
            headers={"Authorization": f"Token {self._secret('secret_key_env', kind='sk')}"},
            json={"transaction_id": int(current.transaction_id), "amount_cents": amount},
        )
        # Never apply wallet reversals from the command response or unsigned callback fields.
        return await self.fetch_transaction(current.transaction_id)

    async def close(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
        self._client = None
