"""Optional account pages and verified payment notifications for hosted ShinAI."""

from __future__ import annotations

import hmac
import json
import re
import time
import uuid
from collections import OrderedDict, deque
from html import escape
from pathlib import Path
from urllib.parse import urlsplit

from aiohttp import web

from shin_ai.billing.money import format_usd, purchase_quote
from shin_ai.utils.logger_config import logger

COOKIE = "shinai_session"
_STORE = web.AppKey("billing_store", object)
_SETTINGS = web.AppKey("billing_settings", object)
_GATEWAY = web.AppKey("payment_gateway", object)
_LIMITS = web.AppKey("request_limits", object)
_STATIC = Path(__file__).with_name("static")


def _origin(settings) -> str:
    parsed = urlsplit(settings.web.public_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username:
        raise ValueError("billing.web.public_url must be an HTTP(S) origin")
    if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("The account website requires HTTPS outside localhost")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("billing.web.public_url must not contain a path, query, or fragment")
    return f"{parsed.scheme}://{parsed.netloc}"


def _page(title: str, body: str) -> web.Response:
    return web.Response(
        text=f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)} · ShinAI</title><link rel="stylesheet" href="/assets/account.css">
<script defer src="/assets/account.js"></script></head>
<body><header><a class="brand" href="/">ShinAI<span>Account</span></a>
<a href="/support">Support</a></header><main>{body}</main>
<footer>Prepaid bot usage. No subscription.</footer></body></html>""",
        content_type="text/html",
    )


def _redirect(location: str) -> web.Response:
    return web.Response(status=303, headers={"Location": location})


def _error(message: str, status: int = 400) -> web.Response:
    response = _page(
        "Request not completed",
        f'<section class="narrow"><h1>Request not completed</h1><p>{escape(message)}</p><a href="/">Back to your account</a></section>',
    )
    response.set_status(status)
    return response


def _require_origin(request: web.Request) -> None:
    expected = _origin(request.app[_SETTINGS])
    if request.headers.get("Origin") != expected:
        raise web.HTTPForbidden(text="Open this page from the account website and try again.")


def _rate_limit(request: web.Request, key: str, maximum: int) -> None:
    now = time.monotonic()
    buckets: OrderedDict[str, deque[float]] = request.app[_LIMITS]
    bucket = buckets.setdefault(key, deque())
    while bucket and bucket[0] < now - 60:
        bucket.popleft()
    if len(bucket) >= maximum:
        raise web.HTTPTooManyRequests(text="Too many attempts. Try again in a minute.")
    bucket.append(now)
    buckets.move_to_end(key)
    while len(buckets) > 4096:
        buckets.popitem(last=False)


async def _session(request: web.Request):
    token = request.cookies.get(COOKIE, "")
    if not token or len(token) > 256:
        return None
    return await request.app[_STORE].get_session(token)


async def _authorized_post(request: web.Request):
    _require_origin(request)
    session = await _session(request)
    if session is None:
        raise web.HTTPUnauthorized(text="Your sign-in has expired. Request a new link from the bot.")
    form = await request.post()
    csrf = form.get("csrf", "")
    if not isinstance(csrf, str) or not hmac.compare_digest(csrf.encode(), session.csrf.encode()):
        raise web.HTTPForbidden(text="Refresh the account page and try again.")
    return session, form


def _history_rows(records: list[dict], *, purchases: bool) -> str:
    rows = []
    for record in records[:30]:
        reference = str(record["reference"])
        status = str(record["status"]).replace("_", " ")
        if purchases:
            amount = f"{record['amount_cents'] / 100:.2f} {escape(record['currency'])}"
            label = "Top-up"
        else:
            amount = format_usd(record["customer_charge"])
            label = record.get("model") or "Bot usage"
        rows.append(
            f"<li><div><strong>{escape(label)}</strong><small>{escape(reference[:64])}</small></div>"
            f'<div class="align-end">{amount}<small>{escape(status.capitalize())}</small></div></li>'
        )
    if rows:
        return '<ul class="history">' + "".join(rows) + "</ul>"
    return (
        '<p class="empty">No purchases yet.</p>' if purchases else '<p class="empty">No paid usage yet.</p>'
    )


async def account(request: web.Request) -> web.Response:
    session = await _session(request)
    if session is None:
        return _page(
            "Sign in",
            '<section class="narrow"><h1>Your bot, your balance.</h1><p>Send <code>/account</code> to ShinAI in a private chat to get a sign-in link.</p></section>',
        )
    store = request.app[_STORE]
    settings = request.app[_SETTINGS]
    balance = await store.balance(session.owner)
    history = await store.history(session.owner, limit=30)
    csrf = escape(session.csrf, quote=True)
    identity = f"{session.owner.platform.capitalize()} · {session.owner.user_id}"
    debt = getattr(balance, "debt", 0)
    debt_html = (
        f'<p class="notice">A reversed payment left {format_usd(debt)} to recover. Future top-ups cover this amount first.</p>'
        if debt
        else ""
    )
    package_html = ""
    if session.login_platform == "telegram":
        package_html = "<p>Purchases inside Telegram use Telegram Stars. Telegram checkout is not available yet. You can still view your existing balance and history here.</p>"
    elif request.app[_GATEWAY] is not None:
        choices = []
        for index, cents in enumerate(settings.paymob.packages):
            units, quote = purchase_quote(settings.paymob, cents)
            choices.append(
                f'<label class="package"><input type="radio" name="amount_cents" value="{cents}" {"checked" if index == 0 else ""}><span><strong>{cents / 100:.2f} EGP</strong><small>Adds {format_usd(units)} after {quote["fee_cents"] / 100:.2f} EGP processing allowance</small></span></label>'
            )
        package_html = f'''<form method="post" action="/checkout"><input type="hidden" name="csrf" value="{csrf}"><input type="hidden" name="reference" value="{uuid.uuid4()}">
<fieldset><legend>Choose a top-up</legend>{"".join(choices)}</fieldset>
<div class="fields"><label>First name<input name="first_name" autocomplete="given-name" maxlength="80" required></label><label>Last name<input name="last_name" autocomplete="family-name" maxlength="80" required></label></div>
<label>Email<input name="email" type="email" autocomplete="email" maxlength="254" required></label>
<label>Phone number<input name="phone_number" type="tel" autocomplete="tel" maxlength="30" required></label>
<p class="help">Your bank may apply a currency conversion fee. Your balance is held in USD.</p><button type="submit">Continue to payment</button></form>'''
    else:
        package_html = "<p>Online top-ups are not available yet.</p>"
    returned = (
        '<p class="notice" role="status">Payment submitted. Your balance updates after payment confirmation. <a href="/">Refresh your account</a>.</p>'
        if request.query.get("payment") == "returned"
        else ""
    )
    return _page(
        "Your account",
        f'''<div class="account-heading"><div><h1>Your bot balance</h1><p>{escape(identity)}</p></div><form action="/logout" method="post"><input type="hidden" name="csrf" value="{csrf}"><button class="quiet" type="submit">Sign out</button></form></div>
{returned}{debt_html}<div class="account-grid"><section class="balance"><h2>Available</h2><p class="amount">{format_usd(balance.available)}</p><p>{format_usd(balance.reserved)} reserved for ongoing replies</p><p class="help">Paid replies use provider cost plus a {settings.markup_bps / 100:g}% service markup.</p><p class="help">Choose where your balance can be used with <code>/paid_usage</code> in the bot.</p></section>
<section class="topup"><h2>Add to your balance</h2>{package_html}</section></div>
<div class="history-grid"><section><h2>Purchases</h2>{_history_rows(history.get("purchases", []), purchases=True)}</section><section><h2>Usage</h2>{_history_rows(history.get("usage", []), purchases=False)}</section></div>''',
    )


async def login_page(request: web.Request) -> web.Response:
    session = await _session(request)
    link = ""
    if session is not None:
        link = (
            '<div id="link-choice" hidden><h2>Use one balance across accounts</h2>'
            "<p>Linking combines both balances and paid-usage settings. "
            "Your current account stays signed in.</p>"
            f'<button id="link-account" type="button" data-csrf="{escape(session.csrf, quote=True)}">Link accounts</button></div>'
        )
    return _page(
        "Sign in",
        '<section class="narrow"><h1>Sign in to ShinAI</h1><p id="login-status" role="status">Use the private link sent by the bot.</p>'
        '<button id="sign-in" type="button" hidden>Sign in with this link</button>'
        + link
        + "<noscript>Enable JavaScript to use your private sign-in link.</noscript></section>",
    )


async def exchange_login(request: web.Request) -> web.Response:
    _require_origin(request)
    _rate_limit(request, f"login:{request.remote}", 20)
    try:
        body = await request.json()
        token = body.get("token") if isinstance(body, dict) else None
        if not isinstance(token, str) or not 16 <= len(token) <= 256:
            raise ValueError("Invalid sign-in token")
        settings = request.app[_SETTINGS]
        redeemed = await request.app[_STORE].redeem_login(token, settings.web.session_ttl_seconds)
        if redeemed is None:
            raise ValueError("Invalid sign-in token")
        session_token, _ = redeemed
    except (ValueError, LookupError, json.JSONDecodeError):
        return web.json_response(
            {"error": "This link has expired or was already used. Request a new link from the bot."},
            status=401,
        )
    response = web.json_response({"ok": True})
    response.set_cookie(
        COOKIE,
        session_token,
        max_age=settings.web.session_ttl_seconds,
        httponly=True,
        secure=urlsplit(settings.web.public_url).scheme == "https",
        samesite="Lax",
        path="/",
    )
    return response


async def link_accounts(request: web.Request) -> web.Response:
    _require_origin(request)
    session = await _session(request)
    if session is None:
        raise web.HTTPUnauthorized(text="Sign in before linking accounts.")
    _rate_limit(request, f"link:{session.owner.platform}:{session.owner.user_id}", 10)
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError("Invalid account link")
        csrf, token = body.get("csrf", ""), body.get("token", "")
        if not isinstance(csrf, str) or not hmac.compare_digest(csrf.encode(), session.csrf.encode()):
            raise web.HTTPForbidden(text="Refresh the sign-in page and try again.")
        if not isinstance(token, str) or not 16 <= len(token) <= 256:
            raise ValueError("Invalid account link")
        await request.app[_STORE].link_identity(session.owner, token)
    except (ValueError, LookupError, json.JSONDecodeError):
        return web.json_response(
            {
                "error": "Accounts could not be linked. Request a new private link, and wait for ongoing replies or refunds to finish."
            },
            status=409,
        )
    return web.json_response({"ok": True})


async def logout(request: web.Request) -> web.Response:
    await _authorized_post(request)
    await request.app[_STORE].revoke_session(request.cookies[COOKIE])
    response = _redirect("/")
    response.del_cookie(COOKIE, path="/")
    return response


def _billing_data(form) -> dict[str, str]:
    result = {}
    for name in ("first_name", "last_name", "email", "phone_number"):
        value = form.get(name, "")
        limit = 254 if name == "email" else 80 if name.endswith("name") else 30
        if (
            not isinstance(value, str)
            or not value.strip()
            or len(value) > limit
            or any(ord(c) < 32 for c in value)
        ):
            raise ValueError("Enter your name, email, and phone number to continue.")
        result[name] = value.strip()
    if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", result["email"]):
        raise ValueError("Enter a valid email address.")
    if not re.fullmatch(r"\+?[0-9][0-9 ()-]{6,28}", result["phone_number"]):
        raise ValueError("Enter a valid phone number, including its country code.")
    return result


async def checkout(request: web.Request) -> web.Response:
    session, form = await _authorized_post(request)
    _rate_limit(request, f"checkout:{session.owner.platform}:{session.owner.user_id}", 10)
    gateway = request.app[_GATEWAY]
    if gateway is None:
        return _error("Online top-ups are not available yet.", 503)
    if session.login_platform == "telegram":
        return _error("Telegram purchases require Telegram Stars. Checkout is not available yet.", 403)
    settings = request.app[_SETTINGS]
    store = request.app[_STORE]
    try:
        reference = str(uuid.UUID(form.get("reference", "")))
        if not re.fullmatch(r"[0-9]+", str(form.get("amount_cents", ""))):
            raise ValueError("Choose an available top-up package.")
        amount = int(form["amount_cents"])
        units, quote = purchase_quote(settings.paymob, amount)
        billing_data = _billing_data(form)
        purchase = await store.create_purchase(session.owner, reference, amount, "EGP", units, quote)
        if purchase.status in {"paid", "partially_refunded", "refunded", "failed"}:
            return _redirect("/")
        if purchase.checkout_url:
            return _redirect(purchase.checkout_url)
        if not await store.claim_checkout(reference):
            return _error(
                "This payment is being prepared or confirmed. Refresh your account before trying another top-up.",
                409,
            )
        try:
            checkout_result = await gateway.create_checkout(purchase, billing_data)
            await store.bind_checkout(reference, checkout_result)
        except Exception:
            logger.exception("Checkout creation needs reconciliation")
            return _error(
                "Payment could not be confirmed. This top-up has been saved for review; do not repeat the payment.",
                503,
            )
        return _redirect(checkout_result.url)
    except (ValueError, LookupError, TypeError, AttributeError):
        return _error("Check the package and contact details, then try again.")


def _matches_quote(purchase, state) -> bool:
    quote = purchase.quote
    return (
        state.amount_cents == purchase.amount_cents
        and state.currency == purchase.currency
        and state.merchant_id == quote.get("merchant_id")
        and state.integration_id in quote.get("integration_ids", [])
        and state.is_live is quote.get("is_live")
    )


async def payment_notification(request: web.Request) -> web.Response:
    gateway = request.app[_GATEWAY]
    if gateway is None:
        raise web.HTTPServiceUnavailable(text="Payment processing is disabled.")
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError("Invalid notification")
        transaction_id = gateway.verify_callback(body, request.query.get("hmac", ""))
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        raise web.HTTPBadRequest(text="Invalid payment notification.") from None
    try:
        state = await gateway.fetch_transaction(transaction_id)
        # The gateway verifies the requested ID and resolves refund children to their parent.
        store = request.app[_STORE]
        purchase = await store.get_purchase_by_order(state.order_id)
        if purchase is None and state.special_reference:
            purchase = await store.get_purchase(state.special_reference)
            if purchase is not None:
                if not _matches_quote(purchase, state):
                    raise ValueError("Payment does not match purchase")
                await store.restore_order(purchase.reference, state)
        if purchase is None:
            raise web.HTTPServiceUnavailable(text="Payment association is pending.")
        await store.apply_payment(state)
    except web.HTTPException:
        raise
    except (ValueError, LookupError):
        raise web.HTTPBadRequest(text="Payment does not match a purchase.") from None
    except Exception:
        logger.exception("Payment confirmation needs reconciliation")
        raise web.HTTPServiceUnavailable(text="Payment confirmation is pending.") from None
    return web.json_response({"received": True})


async def payment_return(_request: web.Request) -> web.Response:
    return _redirect("/?payment=returned")


async def support(request: web.Request) -> web.Response:
    address = request.app[_SETTINGS].web.support_email
    contact = (
        f'<p>Contact <a href="mailto:{escape(address, quote=True)}">{escape(address)}</a> about payments or your balance.</p>'
        if address
        else "<p>Contact the bot operator about payments or your balance.</p>"
    )
    return _page(
        "Support",
        f'<section class="narrow"><h1>Support</h1>{contact}<h2>Payments and refunds</h2><p>Include the purchase reference shown in your account. Refunds are reviewed by the operator and returned through the original payment method.</p><h2>Your information</h2><p>This service stores your chat account identifier, balance, usage costs, sign-in sessions, and purchase records. Payment contact details are sent to the payment provider. Card details are entered on its checkout page.</p><p>Ask the operator about access to or deletion of your information. Payment records may need to be retained.</p></section>',
    )


async def asset(request: web.Request) -> web.Response:
    name = request.match_info["name"]
    if name not in {"account.css", "account.js"}:
        raise web.HTTPNotFound()
    return web.Response(
        body=(_STATIC / name).read_bytes(),
        content_type="text/css" if name.endswith("css") else "application/javascript",
    )


@web.middleware
async def security_headers(request: web.Request, handler):
    try:
        response = await handler(request)
    except web.HTTPException as error:
        response = web.Response(body=error.body, status=error.status, headers=error.headers)
    except Exception:
        logger.exception("Account request failed")
        response = _error("The account service could not complete this request. Try again later.", 500)
    response.headers.update(
        {
            "Cache-Control": "no-store",
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self'; form-action 'self' https://accept.paymob.com; base-uri 'none'; frame-ancestors 'none'",
        }
    )
    return response


def create_web_app(store, billing_settings, gateway=None) -> web.Application:
    _origin(billing_settings)
    if gateway is None and billing_settings.paymob.enabled:
        from shin_ai.billing.paymob import PaymobClient

        gateway = PaymobClient(billing_settings.paymob, billing_settings.web.public_url)
    app = web.Application(middlewares=[security_headers], client_max_size=32 * 1024)
    app[_STORE] = store
    app[_SETTINGS] = billing_settings
    app[_GATEWAY] = gateway
    app[_LIMITS] = OrderedDict()
    app.add_routes(
        [
            web.get("/", account),
            web.get("/login", login_page),
            web.post("/auth/exchange", exchange_login),
            web.post("/auth/link", link_accounts),
            web.post("/logout", logout),
            web.post("/checkout", checkout),
            web.post("/webhooks/paymob", payment_notification),
            web.get("/payment-return", payment_return),
            web.get("/support", support),
            web.get("/assets/{name}", asset),
        ]
    )

    async def close_gateway(_app):
        if gateway is not None:
            await gateway.close()

    app.on_cleanup.append(close_gateway)
    return app


async def start_web(store, billing_settings) -> web.AppRunner | None:
    if not billing_settings.web.enabled:
        return None
    runner = web.AppRunner(create_web_app(store, billing_settings), access_log=None)
    try:
        await runner.setup()
        await web.TCPSite(runner, billing_settings.web.host, billing_settings.web.port).start()
    except BaseException:
        await runner.cleanup()
        raise
    return runner
