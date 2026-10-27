from copy import deepcopy

import pytest

from shin_ai.settings import parse_settings


def configuration():
    return {
        "ai": {
            "primary": "free",
            "providers": [
                {"name": "free", "type": "gemini", "models": ["free-model"], "api_keys": {"f": "test-free"}},
                {"name": "paid", "type": "gemini", "models": ["paid-model"], "api_keys": {"p": "test-paid"}},
            ],
        },
        "billing": {
            "enabled": True,
            "providers": ["paid"],
            "max_request_usd": "0.10",
            "markup_percent": "25",
            "pricing": [
                {
                    "provider": "paid",
                    "model": "paid-model",
                    "adapter": "gemini",
                    "version": "test-v1",
                    "input_per_million_usd": "0.1",
                    "output_per_million_usd": "0.4",
                    "aliases": ["paid-model-version-1"],
                }
            ],
            "web": {"enabled": True, "public_url": "https://account.example.test"},
            "paymob": {
                "enabled": True,
                "merchant_id": 42,
                "integration_ids": [17],
                "egp_per_usd": "50",
                "packages_egp": ["100.00"],
            },
        },
    }


def test_monetary_configuration_parses_without_live_credentials(monkeypatch):
    for name in ("PAYMOB_SECRET_KEY", "PAYMOB_API_KEY", "PAYMOB_HMAC_SECRET", "PAYMOB_PUBLIC_KEY"):
        monkeypatch.delenv(name, raising=False)
    billing = parse_settings(configuration()).billing
    assert billing.max_request_units == 100_000 and billing.markup_bps == 2500
    assert billing.paymob.packages == (10_000,)
    assert billing.pricing[0].aliases == ("paid-model-version-1",)


@pytest.mark.parametrize(
    "section,key,value,match",
    [
        ("paymob", "merchant_id", True, "merchant_id"),
        ("paymob", "packages_egp", "100", "list"),
        ("paymob", "packages_egp", ["1.001"], "two decimal"),
        ("paymob", "integration_ids", [True], "integration"),
        ("paymob", "egp_per_usd", "NaN", "finite"),
        ("web", "public_url", "http://account.example.test", "HTTPS"),
        ("web", "public_url", "https://account.example.test/path", "origin"),
        ("web", "enabled", False, "website"),
    ],
)
def test_invalid_payment_configuration_fails_before_startup(section, key, value, match):
    raw = deepcopy(configuration())
    raw["billing"][section][key] = value
    with pytest.raises(ValueError, match=match):
        parse_settings(raw)


def test_missing_paid_price_is_rejected():
    raw = configuration()
    raw["billing"]["pricing"] = []
    with pytest.raises(ValueError, match=r"Missing billing\.pricing"):
        parse_settings(raw)
