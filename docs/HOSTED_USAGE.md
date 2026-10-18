# Optional prepaid usage

ShinAI supports optional prepaid response credits funded through a local
administrator command. The website and payment gateway are not implemented yet.

## Repository and deployment

Keep the bot, optional billing code, payment integration, and website in this
public repository. Keep payment documentation here rather than promoting the
offering throughout the main README.

Billing is disabled by default. Self-hosters can run ShinAI with their own
providers without a payment account, billing service, or website. Billing must
not add startup requirements when disabled.

Production credentials, provider keys, customer identities, balances, payment
records, and deployment settings stay outside version control. Public source
code does not grant access to the hosted service or its data.

## Prepaid usage

Keep the current free provider route and its usage limits. Add prepaid credits
owned by the person who buys them. There are no subscriptions or renewals.

A person has one balance shared across their selected chats. Each enabled chat
has a spending mode:

- `only_me`: cover requests from the balance owner.
- `everyone`: cover requests from any member of that chat.

Support Discord servers as a scope distinct from individual channels.
Purchasing credits does not grant chat administration rights or access to
private conversations.

## Request handling

For an eligible request, reserve available credits atomically and call the paid
provider pool directly. Use bounded retries so this route does not inherit the
free provider chain's latency. Apply the paid routing policy to supporting model
calls as well as the main response.

Settle the reservation when the billable operation succeeds. Release it on
failure before attempting free usage. Requests without eligible credits use the
existing free route and its limits. Paid usage does not consume or reset the
free allowance. General anti-spam protection applies to both routes.

Persist balances, reservations, and an accounting history. Use stable request
and payment references to prevent duplicate charges or credit grants. Recover
outstanding reservations after restarts.

## Website and payments

Host a small optional website and payment backend alongside the bot on the
existing VPS, using HTTPS. The site provides account access, top-up packages,
balances, purchase history, and chat spending settings.

Paymob is the initial gateway candidate. Live integration depends on approval
for an Egyptian individual selling prepaid AI usage, including confirmation of
available payment methods and international card acceptance.

Create a purchase before opening hosted checkout. Grant credits only after
verified payment confirmation, never from the browser return page. Process
duplicate notifications safely, reconcile missed notifications, and record
refunds and reversals. Additional gateways fund the same balance.

Account for platform-specific purchase requirements, including Telegram Stars
for digital sales inside Telegram and applicable Discord Premium Apps rules.

## Try administrator-funded credits

Add a dedicated provider to `ai.providers`. List its name in `billing.providers`
and set `billing.enabled: true` in your local configuration. Keep the free
provider in `ai.primary` and leave paid providers out of `ai.fallbacks`.
`config.yaml.example` lists all billing settings. Names in `billing.providers`
are excluded from the free route even when billing is disabled.

Use the local CLI to grant test credits. Replace the example Discord user ID
with the payer's numeric platform ID. Use a distinct reference for each grant.
Repeating the same reference and payload does not add credits twice. A reference
with a different payload is rejected.

```bash
python -m shin_ai.billing --config config.yaml grant \
	--platform discord --user 123456789012345678 \
	--credits 20 --reference test-grant-001
python -m shin_ai.billing --config config.yaml balance \
	--platform discord --user 123456789012345678
```

In a private chat with the bot, use `/balance` to view available and reserved
credits. Balance commands in groups redirect you to a private chat.

Use `/paid_usage only_me` in each chat where your credits can cover your own
direct requests. Use `/paid_usage everyone` to let other members spend your
balance on direct requests. Use `/paid_usage off` to remove your permission.
In Discord, append `server` to any of these commands to configure the whole
server instead of just the current channel.

## Spending and recovery rules

The default cost is one credit per interaction with at least one confirmed text
message, sticker, reaction, or moderation action. Multiple outputs cost one
interaction. Partial delivery counts as success. Skipped responses, capacity
notices, and requests with no confirmed delivery do not consume credits.
Settlement occurs at first confirmed delivery, before context and memory
persistence. Random and speculative interactions always use free usage.

Your own enabled balance takes priority over shared sponsorship. Otherwise, a
channel sponsor takes priority over a server sponsor. Empty balances are skipped.
Only one shared sponsor can enable `everyone` for an exact chat or server scope.
Turning your setting off does not disable another person's sponsorship.

Accounts are identified by platform and immutable user ID. Cross-platform account
linking is not implemented yet. A balance is shared across that platform user's
enabled chats. Use separate grants for identities on different platforms.

The paid pool has a hard generation deadline, configurable retries, a text input
limit, and an output token limit. Supporting image calls use paid Gemini providers
only. If none is eligible, image helpers cannot use the free Gemini pool during a
paid request. Existing queue delays, local preparation, delivery delays, media
size limits, and tool limits still apply. A model-call deadline is not an
end-to-end response-time guarantee.

Billing currently requires a Unix host and one billing-enabled bot process per
database. An exclusive file lock prevents competing bot processes. Administrator
CLI commands can run while the bot is active and never recover outstanding work.
At startup, the bot refunds unresolved reservations and preserves their request
IDs so they cannot be charged again. Repeated initialization does not recover
live work. Store the database and its ownership file on the same local filesystem.

External delivery and database settlement cannot be atomic. A process crash
between them can leave an ambiguous delivery, which startup recovery refunds.
This policy favors the user over charging for an unconfirmed response.

The implementation combines a transactional store with request-scoped provider
routing. A lifecycle ownership lock keeps recovery simpler than renewable
reservation leases. Settlement callbacks preserve partial-delivery evidence
without making every caller coordinate the accounting steps.

Next, add authenticated account access and purchase records, then test payment
confirmation, refunds, and reconciliation before accepting real payments.
