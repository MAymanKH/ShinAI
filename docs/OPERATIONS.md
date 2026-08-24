# Production operations

## Two local instances

SQLite is the coordination layer for the current deployment shape: two Python
processes on one server. Give both instances the same absolute database path and
namespace when they should coordinate:

```yaml
coordination:
  backend: sqlite
  namespace: shinai-production
  database_path: /srv/shinai/shared/coordination.sqlite3
  lease_seconds: 240
  event_dedup_ttl_seconds: 86400
  reply_state_ttl_seconds: 86400
  cleanup_interval_seconds: 300
```

The directory must be writable by both service users. SQLite creates `-wal` and
`-shm` files next to the database, so sharing only the main file is insufficient.
Keep it on the server's local filesystem rather than a network filesystem.

Give the embedded vector store a different path in each instance's config:

```yaml
# Instance A
chroma:
  path: /srv/shinai/instance-a/chroma_db

# Instance B uses /srv/shinai/instance-b/chroma_db instead.
```

These embedded stores are independent, so long-term memories are not replicated
between instances. Do not point them at one shared directory. A future need for
one shared memory corpus should use Chroma's server/client mode rather than
concurrent embedded writers.

Credential handling is automatic:

- Same Telegram/Discord/WhatsApp credential: events, reply triggers, and rate
  limits coordinate across both processes.
- Different platform credentials: state stays isolated even with the same SQLite
  file and namespace.
- Same Gemini key under different JSON labels: it shares one lease and health
  record, preventing simultaneous overuse.
- Different Gemini keys under the same label: they remain independent.
- Partially overlapping Gemini pools: only the shared credentials coordinate.

Use a different namespace for staging, destructive experiments, or instances
that must be completely isolated.

## Why SQLite instead of Redis

For two processes on one machine, Redis does not materially improve the bot's
current coordination workload. SQLite already provides atomic claims, counters,
leases, TTL cleanup, and WAL concurrency without another daemon, port, password,
backup policy, or network failure mode. It also does not consume a separate
service's memory.

Redis becomes worthwhile when instances move to different machines, the worker
count grows substantially, SQLite write contention becomes measurable, or the
design needs distributed queues, pub/sub, or centralized live metrics. Redis
would not reduce the largest RAM costs here: every Python process still owns its
embedding model, platform SDK state, HTTP pools, and optional Whisper worker.

## Memory expectations and tuning

The embedding model is intentionally retained after first use, so resident memory
will not return to the pre-model baseline. Each bot process loads its own copy;
two instances can therefore approach twice the embedding baseline.

Whisper is different. With `process_isolation: true`, it loads in a child process,
handles one bounded audio download at a time, and exits after
`idle_timeout_seconds`, allowing the OS to reclaim its native allocations. A
busy server may temporarily have one Whisper worker per bot instance.

If memory still grows continuously after traffic stops, check these first:

1. Confirm `whisper.process_isolation` is `true` and observe the worker exit log.
2. Lower `runtime.max_pending_interactions` if queued messages retain too much
   platform metadata during extreme bursts.
3. Lower `runtime.context.max_chats`, `messages_per_chat`, or
   `platform_message_cache_size` when the active-chat population is smaller.
4. Give each process a different `chroma.path`; embedded Chroma is not safe for
   concurrent writers sharing one directory.
5. Compare each process separately; SQLite coordination does not combine their
   RSS values or model allocations.

## Logs

With `logging.debug: false`, logs retain the main operational trail: lifecycle,
trigger, queue, provider success/fallback, tool use, sent response/action, rate
limit, and errors. With debug enabled, filter decisions, timing/context details,
tool arguments, and useful SDK activity are added.

Every admitted interaction carries fields such as:

```text
rid=7f4ab229d1 platform=telegram chat=-100123 msg=456 user=789
```

Use that request ID to reconstruct one interaction across modules:

```bash
rg 'rid=7f4ab229d1' shinai_bot.log*
rg 'ERROR|WARNING' shinai_bot.log*
rg 'tool.requested|provider.fallback|response.sent' shinai_bot.log*
```

The rotating file format always includes `module:function:line`. Console warnings
and errors include a source location too. Set `logging.file: null` when systemd,
Docker, or another process manager already rotates stdout. Set
`content_preview_chars: 0` to hide message contents from normal logs.

## Safe rollout

1. Copy the production config and credentials into the new environment; do not
   copy them into Git.
2. Back up application data before upgrading Chroma or its pinned version.
3. Stop one old instance, start one refactored instance, and verify
   `lifecycle.ready`, trigger, provider, tool, and response events.
4. Confirm the configured SQLite path is absolute and both service users can
   create files in its directory.
5. Confirm the two instances use different embedded `chroma.path` values.
6. Start the second instance and verify shared-key `/gstats` health and that one
   incoming platform event produces one response.
7. Watch each process's RSS through at least one Whisper load/idle-exit cycle.
8. Keep the old environment available for rollback until the observation window
   is complete.
