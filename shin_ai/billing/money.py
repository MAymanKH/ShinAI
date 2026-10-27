"""Exact money conversions and purchase quotes. Wallet units are millionths of USD."""

from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from typing import Any

MAX_UNITS = 2**63 - 1


def decimal_amount(value: Any, name: str, *, allow_zero: bool = False) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a decimal amount")
    try:
        result = Decimal(str(value))
    except Exception as error:
        raise ValueError(f"{name} must be a decimal amount") from error
    if not result.is_finite() or result < 0 or (not allow_zero and result == 0):
        raise ValueError(f"{name} must be finite and positive")
    return result


def usd_to_units(value: Any) -> int:
    scaled = decimal_amount(value, "USD amount", allow_zero=True) * 1_000_000
    if scaled != scaled.to_integral_value() or scaled > MAX_UNITS:
        raise ValueError("USD amounts require at most six decimal places and must fit the wallet")
    return int(scaled)


def format_usd(units: int) -> str:
    value = format(Decimal(units) / 1_000_000, ".6f").rstrip("0").rstrip(".")
    if "." not in value:
        value += ".00"
    elif len(value.rsplit(".", 1)[1]) < 2:
        value += "0"
    return f"${value}"


def purchase_quote(settings, amount_cents: int) -> tuple[int, dict]:
    if amount_cents not in settings.packages:
        raise ValueError("Choose an available top-up package")
    fee = int((Decimal(amount_cents) * settings.fee_bps / 10_000).to_integral_value(rounding=ROUND_CEILING))
    fee += settings.fixed_fee_cents
    if fee >= amount_cents:
        raise ValueError("The processing allowance exceeds the package amount")
    units = int(
        (Decimal(amount_cents - fee) * 10_000 / settings.egp_per_usd).to_integral_value(rounding=ROUND_FLOOR)
    )
    if not 0 < units <= MAX_UNITS:
        raise ValueError("The credited balance is outside supported limits")
    return units, {
        "currency": "EGP",
        "gross_cents": amount_cents,
        "fee_cents": fee,
        "egp_per_usd": str(settings.egp_per_usd),
        "fee_bps": settings.fee_bps,
        "fixed_fee_cents": settings.fixed_fee_cents,
        "credited_units": units,
        "integration_ids": list(settings.integration_ids),
        "merchant_id": settings.merchant_id,
        "is_live": settings.mode == "live",
        "version": "egp-usd-v1",
    }
