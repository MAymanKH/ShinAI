"""Gemini key loading and passive health reporting.

The key file is generated from YAML at startup. Runtime rotation never rewrites secrets;
pair health is maintained by the shared coordination backend.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

from shin_ai.paths import DATA_DIR
from shin_ai.settings import AISettings, get_settings
from shin_ai.utils.logger_config import logger

GEMINI_KEYS_FILE = DATA_DIR / "gemini_keys.json"


def initialize_keys(settings: AISettings) -> None:
    """Replace the shared key file from all configured Gemini providers."""
    global _api_keys
    keys = {
        alias: key
        for provider in settings.providers.values()
        if provider.type == "gemini"
        for alias, key in provider.api_keys.items()
    }
    GEMINI_KEYS_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=GEMINI_KEYS_FILE.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(keys, temporary, indent=2)
            temporary.write("\n")
        os.replace(temporary_path, GEMINI_KEYS_FILE)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    _api_keys = None


def load_keys() -> dict[str, str]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not GEMINI_KEYS_FILE.exists():
        logger.warning(
            "Gemini key file %s is missing; start the application to generate it from config.yaml.",
            GEMINI_KEYS_FILE,
        )
        return {}
    try:
        decoded = json.loads(GEMINI_KEYS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        logger.error("Failed to load Gemini keys from %s: %s", GEMINI_KEYS_FILE, error)
        return {}
    if not isinstance(decoded, dict):
        logger.error("Gemini key file %s must contain a JSON object.", GEMINI_KEYS_FILE)
        return {}
    return {
        str(name): str(value) for name, value in decoded.items() if str(name).strip() and str(value).strip()
    }


async def get_gemini_stats_message(detailed: bool = False) -> str:
    """Render shared passive health without firing quota-consuming probe calls."""
    from shin_ai.providers.gemini import get_gemini_scheduler

    blocks = ["**Gemini Key/Model Health (shared runtime state)**"]
    for provider in get_settings().ai.providers.values():
        if provider.type != "gemini":
            continue
        snapshot = await get_gemini_scheduler(provider).health_snapshot()
        health = "\n\n".join(_format_model_health(snapshot, detailed))
        blocks.append(f"**Provider: {provider.name}**\n```\n{health}\n```")
    return "\n\n".join(blocks)


def _format_model_health(snapshot: dict, detailed: bool) -> list[str]:
    lines = []
    for model, model_data in snapshot["models"].items():
        total = model_data["total_keys"]
        eligible = model_data["eligible_keys"]
        lines.append(
            f"Model: {model}\n"
            f"Health: {'Available' if model_data['available'] else 'Unavailable'}\n"
            f"✅ Eligible keys: {eligible}/{total}"
        )
        if detailed:
            issues = []
            for pair in model_data["pairs"]:
                if pair["eligible"]:
                    continue
                cooldown = pair["cooldown_until"]
                until = (
                    datetime.fromtimestamp(cooldown).strftime("%Y-%m-%d %H:%M:%S")
                    if cooldown
                    else "manual recovery"
                )
                issue = f"• {pair['key']}: {pair['status']} until {until}"
                if pair["last_error"]:
                    issue += f" — {pair['last_error'][:80]}"
                issues.append(issue)
            if issues:
                lines.append("Issues:\n" + "\n".join(issues))
    return lines


_api_keys: dict[str, str] | None = None


def get_api_keys() -> dict[str, str]:
    """Read the key file once, on first use rather than at import."""
    global _api_keys
    if _api_keys is None:
        _api_keys = load_keys()
    return _api_keys
