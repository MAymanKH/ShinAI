# Optional prepaid usage

This document records the scope for prepaid usage on the hosted ShinAI bot.
The feature is planned and is not implemented yet.

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

## Implementation order and remaining choices

First implement durable credits, administrator-added test credits, paid routing,
and chat spending controls. Verify concurrency, failures, duplicate requests,
and restart recovery. Then add the website and test checkout before enabling
real purchases.

Before implementation, define the credit unit and billable success condition,
including partial delivery and tool actions. Also define overlapping sponsor
selection and whether automatic bot interjections can spend credits. These
choices must be explicit before enabling shared spending.
