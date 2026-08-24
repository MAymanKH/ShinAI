# Testing

The unit suite is intentionally hermetic: it loads `config.yaml.example`, does
not contact chat platforms or model APIs, and does not load Chroma, transformer,
or Whisper models. This keeps verification fast for developers and coding agents.

No module reads configuration at import time, so any module can be imported in a
test without a `config.yaml` on disk. Tests that need different settings use the
`override_settings` fixture, which swaps a module's `get_settings` for one
returning a modified copy — never by rebinding module constants.

## Commands

Install development dependencies in a normal project environment:

```bash
python -m pip install -r requirements.txt
python -m pip install -e ".[test]"
```

`pyproject.toml` is the canonical runtime dependency list. `requirements.txt`
carries only what package metadata cannot express: PyTorch's CPU-only wheel
index, plus `-e .`. Installing through `requirements.txt` therefore installs
exactly what the Docker image installs.

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
ruff check .
ruff format --check .
python -m pytest -q
```

CI runs those as two jobs. `lint` installs only ruff, so style problems report
in seconds. `tests` installs the same dependency set as the Docker image, so a
dependency that resolves in CI resolves in the image.

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
- The full trigger matrix in `handlers/common.py`: private chats, mentions, the
  `يالبوت` keyword, reply chains, speculative replies, media without text, and
  WhatsApp status broadcasts.
- Cosine-distance conversion and the MMR diversity re-ranker.
- Which platforms the composition root registers, and that one platform failing
  to register or start does not take out the others.
- That every adapter implements the full `PlatformAdapter` surface and declares
  each capability the action executor branches on.

## Test design rules

- Inject clocks, stores, provider executors, model factories, and platform fakes.
- Add a regression test before or with every production bug fix.
- Prefer behavior/invariant assertions over implementation snapshots.
- Use temporary SQLite paths for multi-instance tests.
- Keep real provider/platform smoke tests separate and opt-in; never make them a
  prerequisite for verifying ordinary code changes.
