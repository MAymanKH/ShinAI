# Architecture

ShinAI is split around one rule: platform SDK objects stay at the edges, while
the expensive and failure-prone work is owned by small services with explicit
bounds and shutdown behavior.

## Interaction flow

1. A platform handler converts the native event into `UnifiedMessage`.
2. The common trigger policy rejects unsupported, self, command, or unrelated
   events before model work.
3. Credential-scoped SQLite state deduplicates the event across local instances.
4. `InteractionScheduler` admits the event into a globally bounded,
   per-chat-ordered queue. Only the first event in a chat burst receives the
   human-like delay.
5. `media` builds bounded prompt attachments. Context, memory, style, social,
   reply-chain, and runtime sections are added by the core handler.
6. `provider_router` applies one total deadline across retries and fallbacks.
   Gemini delegates key/model-pair selection to `GeminiScheduler`.
7. `response_policy` strips model control syntax and rejects action
   meta-commentary.
8. `action_executor` sends text/actions and records only outcomes that actually
   succeeded. Long-term memory is written once per interaction.
9. `lifecycle` drains interactions, stops platform ingress, flushes caches, and
   closes model workers, HTTP clients, and SQLite handles.

## Module ownership

| Area | Owner | Important invariant |
|---|---|---|
| Admission and chat ordering | `core/interaction_scheduler.py` | Global and per-chat queue sizes never exceed config |
| Provider deadlines/fallback | `core/provider_router.py` | One hard deadline covers the full provider chain |
| Gemini rotation | `providers/gemini_scheduler.py` | Health is per credential/model pair; a model is available while any key works |
| Model response parsing | `core/response_policy.py` | Control tags never leak into user-visible text |
| Media preparation | `services/media.py` | Downloads and retained attachments are bounded |
| Embeddings | `services/embeddings.py` | One lazy model per process with bounded inference concurrency |
| Audio | `services/audio_transcriber.py` | Downloads are bounded; Whisper can run in a reclaimable child process |
| Short context | `utils/context_manager.py` | LRU, messages-per-chat, and idle TTL are all bounded |
| Cross-instance state | `coordination/` | Atomic, namespaced, expiring state; no platform secret is stored |
| Logging | `utils/logger_config.py` | Every admitted interaction has stable correlation fields |

## State boundaries

Process-local state includes interaction queues, HTTP clients, platform object
caches, short chat context, the embedding model, and the optional Whisper child.
It is deliberately bounded and released during shutdown.

Shared SQLite state includes event claims, rate counters, reply markers,
Gemini leases, rotation cursors, and pair health. Platform state is scoped by a
fingerprint of the bot credential. Gemini state is scoped by a fingerprint of
the actual API key, so different key labels do not change coordination behavior.
The key material itself is never written to SQLite.

Chroma long-term memory is separate from the coordination database. Embedded
mode uses a process-local `PersistentClient`; never point two processes at the
same embedded directory. Server mode uses `HttpClient`, allowing any number of
bot processes to share one memory corpus safely. SQLite continues to coordinate
events, rate limits, replies, and Gemini health in either Chroma mode.

## Adding functionality

- Put new platform-specific behavior behind `PlatformAdapter`.
- Put provider selection/retry behavior in `provider_router`, and provider wire
  formats in `providers/`.
- Put reusable stateful work in `services/` with a bounded constructor and a
  close function registered by `core/lifecycle.py`.
- Keep response decisions pure when possible and unit-test them without SDKs.
- Any collection keyed by chats, users, messages, requests, or clients must have
  an explicit size limit or TTL.
- Any created task, subprocess, HTTP client, or database connection must have one
  clear owner and shutdown path.
