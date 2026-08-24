# Testing

The unit suite is intentionally hermetic: it loads `config.yaml.example`, does
not contact chat platforms or model APIs, and does not load Chroma, transformer,
or Whisper models. This keeps verification fast for developers and coding agents.

## Commands

Install development dependencies in a normal project environment:

```bash
python -m pip install -e ".[test]"
```

Run the complete fast suite:

```bash
python -m pytest -q
```

Useful targeted suites:

```bash
python -m pytest -q tests/unit/providers/test_gemini_scheduler.py
python -m pytest -q tests/unit/core/test_interaction_scheduler.py
python -m pytest -q tests/unit/services/test_audio_transcriber.py
python -m pytest -q tests/unit/coordination/test_store.py
```

Before committing:

```bash
python -m compileall -q shin_ai tests main.py
python -m pytest -q
ruff check shin_ai tests main.py
```

CI uses a lightweight install that includes only the package metadata, PyYAML,
and test tools. This is deliberate: importing a new module from a unit test must
not accidentally require a platform SDK or multi-gigabyte model dependency.

## What is covered

- SQLite atomic claims, expiry, namespaces, counters, large batched reads, and
  two-connection behavior.
- Gemini pair cooldowns, all-key model availability, authentication failures,
  fair concurrent leases, exact/partial credential sharing semantics, and key
  deduplication.
- Global/per-chat scheduler order, concurrency, TTL/overflow behavior, shutdown,
  and a 5,000-event burst bound.
- Context LRU/TTL bounds under 10,000 unique chats.
- Embedding singleton loading and inference concurrency.
- Whisper temporary-file cleanup, audio-size limits, and download concurrency.
- Provider retry/fallback/deadline decisions, response control parsing, media
  preparation, correlated logging, rate limits, lifecycle order, and delivered
  outcome persistence.

## Test design rules

- Inject clocks, stores, provider executors, model factories, and platform fakes.
- Add a regression test before or with every production bug fix.
- Prefer behavior/invariant assertions over implementation snapshots.
- Use temporary SQLite paths for multi-instance tests.
- Keep real provider/platform smoke tests separate and opt-in; never make them a
  prerequisite for verifying ordinary code changes.
