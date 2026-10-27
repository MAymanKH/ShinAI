# Optional prepaid usage

ShinAI can run a small account website and Paymob checkout in the bot process.
Billing is disabled by default. Self-hosters do not need a payment account or the
website dependency. There are no subscriptions or renewals.

The implementation stays in this public repository. Keep production configuration,
credentials, the SQLite database, backups, customer records, and logs outside Git.
The main README does not need to advertise the hosted offering.

## Balance and request accounting

A payer owns one USD balance across their enabled chats. Wallet amounts are stored
as integer millionths of USD. Provider costs are recorded in billionths of USD,
then the service markup is applied and rounded up once per interaction.

Eligible direct requests try the dedicated paid provider pool first. The bot
reserves up to `billing.max_request_usd` before calling a model. Each model call
must fit the remaining request budget, including markup. A wallet needs enough
available balance for that reservation, even if the final reply costs less.
Unused reserved funds return to the wallet after settlement.

The customer charge is the cost of accepted model attempts that produced a
confirmed delivery, plus `billing.markup_percent`. Tool-loop generations and
accepted supporting model calls are included. Failed attempts remain operator
expenses. Missing usage metadata prevents an attempt from being billed. An
upstream overrun cannot debit more than the authorized reservation.

Partial delivery counts as delivery, including a confirmed sticker, reaction, or
moderation action. A skipped reply or an attempt without confirmed delivery costs
nothing. Paid usage does not consume or reset the free allowance. If paid
execution fails before delivery, the bot releases the reservation and tries the
existing free route with its existing limits. General anti-spam rules still apply.
Random bot interjections use free usage.

Payment processing is accounted for once when quoting a top-up:

```text
processing allowance = ceil(payment × fee_percent) + fixed_fee_egp
credited USD = floor_to_microUSD((payment − allowance) / egp_per_usd)
request charge = ceil_to_microUSD(accepted provider cost × (1 + markup_percent / 100))
```

`fee_percent` and the fixed fee are configured allowances, not a live gateway fee
lookup. The checkout displays the exact credited amount before payment. Changes
to fees, exchange rate, model rates, or markup do not change earlier snapshots.
Update the configuration when your commercial rates change. Taxes and separately
billed provider tools are not calculated automatically.

## Configure paid providers

1. Add a dedicated provider under `ai.providers`. Keep the free provider in
   `ai.primary`, and leave paid providers out of `ai.fallbacks`.
2. List its name in `billing.providers`. Those names are excluded from free
   routing even while billing is disabled.
3. Add a `billing.pricing` entry for every configured paid provider and model.
   Copy the exact current rates from your provider account. Set a price `version`
   that identifies the rate sheet you used.
4. Choose a request spending cap, markup, input limit, output limit, and bounded
   retry deadlines. Enable billing in your private configuration.

See `config.yaml.example` for the complete settings. Pricing adapters support:

| Adapter | Cost source |
| --- | --- |
| `deepseek` | Direct API cache-hit input, cache-miss input, and completion tokens |
| `gemini` | Uncached input, cached input, output, and thinking tokens |
| `openrouter` | The actual account charge in `usage.cost` |
| `openai` | Compatible input, cached input, and completion token metadata |

DeepSeek reasoning tokens are already part of completion tokens and are not
charged twice. Gemini thinking tokens are included in output cost. Generic
OpenAI-compatible providers must expose the supported token metadata and use the
configured pricing categories. Separate cache-write pricing is unsupported.

For providers other than OpenRouter, a returned model version must match the
configured model or an explicit `aliases` entry. Add only aliases with the same
rates. Unsupported usage fields, missing required token counts, or an unexpected
model version stop paid accounting rather than inventing a charge.

Paid calls do not rotate through free keys. Supporting image calls use eligible
paid Gemini providers only. The generation deadline includes supporting calls,
but queueing, local preparation, and delivery can add time.

## Try a manually funded balance

Grant a small test balance with a unique reference:

```bash
python -m shin_ai.billing --config config.yaml grant \
	--platform discord --user 123456789012345678 \
	--amount-usd 2.00 --reference test-grant-001
python -m shin_ai.billing --config config.yaml balance \
	--platform discord --user 123456789012345678
```

Repeating the same reference and amount is safe. A conflicting reference is
rejected. In a private bot chat, `/balance` shows available funds, reserved funds,
and any amount due after a reversed payment.

Use these commands in the chat where you want to spend your balance:

- `/paid_usage only_me` covers your own direct requests.
- `/paid_usage everyone` allows every member to spend your balance.
- `/paid_usage off` removes your permission for that chat.

On Discord, append `server` to configure the entire server. Your own eligible
balance takes priority, then a channel sponsor, then a server sponsor. Only one
shared sponsor can enable `everyone` for an exact scope. Turning your permission
off does not change another payer's setting.

## Host the account website

1. Install the optional dependency with `pip install '.[billing]'`.
2. Set `billing.web.enabled: true`. Bind to `127.0.0.1` and an unused port, such
   as `8080`. Set `public_url` to the HTTPS origin of your account domain, without
   a path. Set `support_email` to a monitored address.
3. Point the domain to your VPS and reverse proxy it to the configured port.
   For example, a Caddy site can use:

   ```caddyfile
   account.example.com {
       reverse_proxy 127.0.0.1:8080
   }
   ```

4. Restart the bot. In a private chat, send `/account` and open the link. Select
   **Sign in with this link** to redeem it. Reading the link alone does not use it.
5. To combine two platform accounts, stay signed in, request `/account` from the
   second account, open that new link in the same browser, and select **Link
   accounts**. This combines balances and permissions after proof of both
   identities. Wait for ongoing paid requests and pending refunds to finish first.

The website provides balance, purchases, usage receipts, and checkout. Chat
spending permissions stay in the bot commands. Sign-in tokens are short lived,
one time, and stored as hashes. Session cookies use HttpOnly, SameSite, and Secure
on HTTPS. Tokens travel in URL fragments and are removed before redemption.
Keep the links private and avoid recording them in proxy or application logs.

HTTP is allowed only on localhost for development. The website runs inside the
existing bot process and shares its database, so a separate repository or frontend
build is unnecessary.

## Enable Paymob

Obtain approval for the actual product and account type from Paymob before taking
live payments. Merchant onboarding, available payment methods, foreign card
acceptance, settlement, and fees depend on your agreement. Configuring integration
IDs exposes the methods Paymob has enabled for that merchant; the code cannot
enable additional methods by itself.

1. Start in `billing.paymob.mode: test`. Configure the merchant ID and payment
   integration IDs from your Paymob account.
2. Supply `PAYMOB_SECRET_KEY`, `PAYMOB_PUBLIC_KEY`, `PAYMOB_HMAC_SECRET`, and
   `PAYMOB_API_KEY` through your service environment. The configuration contains
   only their environment variable names. Never put secret values in the repo.
3. Set `egp_per_usd`, `fee_percent`, `fixed_fee_egp`, and `packages_egp`. These are
   operator quotes; example defaults are not production rates.
4. Enable `billing.paymob.enabled`. The integration creates hosted checkout with
   notification URL `/webhooks/paymob` and return URL `/payment-return` on your
   public origin. Make the notification endpoint reachable from Paymob.
5. Test each enabled payment method, pending and failed payments, repeated
   notifications, missing notifications, and partial and full refunds. Confirm the
   credited amount and usage charge in the account history.
6. After approval and those checks, configure the live merchant integrations and
   live credentials, change the mode to `live`, and restart.

The browser return page never funds a wallet. A signed callback triggers an
authenticated transaction inquiry. Funding requires the canonical merchant,
integration, amount, currency, mode, and bound order to match the saved quote.
Duplicate notifications do not add money twice. A checkout timeout keeps its
purchase for reconciliation instead of creating another remote intention.

The HMAC parser rejects unsupported null signed fields. Validate real test-mode
callbacks for every enabled method before launch. If a method returns unsupported
fields, reconcile its payment through authenticated inquiry and add a verified
fixture before enabling that method for customers.

Telegram-issued website sessions can view balances and history but cannot use
Paymob checkout. Telegram Stars purchases are not implemented. Linking accounts
does not remove this restriction from a Telegram-issued session. Check applicable
platform commerce requirements before distributing the hosted offering elsewhere.

## Reconcile and refund payments

Use the purchase reference from the account page to reconcile a missed callback
or an uncertain checkout result:

```bash
python -m shin_ai.billing --config config.yaml reconcile --reference PURCHASE_REFERENCE
```

Request a refund in EGP through the local administrator command:

```bash
python -m shin_ai.billing --config config.yaml refund \
	--reference PURCHASE_REFERENCE --amount-egp 25.00
```

The command freezes further spending before contacting Paymob. An uncertain
refund remains frozen until canonical inquiry confirms the requested cumulative
refund. Reconcile it before issuing another refund. Refund requests are never
retried automatically. If Paymob definitively rejects a refund, resolve its
status with the operator before changing the freeze in a future maintenance flow.

Refunds reverse a proportional amount of the purchase's original credited USD.
They use the original quote, not today's exchange rate. A reversal after spending
creates an amount due, blocks new paid reservations, and consumes future credits
or released holds until repaid. Partial refunds can be followed by further refunds.
The CLI does not recover live requests and can run alongside the bot.

## Recovery and migration

Run one billing-enabled bot process per database on a Unix host. A local file lock
prevents competing bot processes. Keep the database and ownership file on the
same local filesystem. Back up the SQLite database with SQLite's backup mechanism
or while the bot is stopped, and protect backups as customer data.

On restart, recovery settles persisted, known, accepted usage with confirmed
delivery and its saved markup. It releases other holds. External message delivery
and SQLite cannot be atomic. A crash before durable delivery evidence can produce
a free reply rather than an unsupported customer charge.

Nonempty databases from the old fixed-response-credit implementation require an
explicit conversion. Stop the bot, back up the database, decide how many microUSD
each old credit represents, then run:

```bash
python -m shin_ai.billing --config config.yaml migrate-legacy \
	--micro-usd-per-credit 10000
```

The number above is an example conversion of one old credit to $0.01, not an
automatic commercial rate. The command requires exclusive ownership, rejects
unresolved old holds, archives the old accounting tables, preserves permissions,
and records converted grants. Replace `response_cost` in old configuration with
`max_request_usd` and the explicit model prices before starting the bot again.
