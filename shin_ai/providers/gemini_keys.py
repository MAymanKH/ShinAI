"""Gemini key loading and passive health reporting.

The key file is static configuration. Runtime rotation never rewrites secrets;
pair health is maintained by the shared coordination backend.
"""

from __future__ import annotations

import json
from datetime import datetime

from shin_ai.config import DATA_DIR, GEMINI_MODELS
from shin_ai.utils.logger_config import logger


GEMINI_KEYS_FILE = DATA_DIR / "gemini_keys.json"
MODELS_LIST = tuple(GEMINI_MODELS)


def load_keys() -> dict[str, str]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not GEMINI_KEYS_FILE.exists():
        logger.warning(
            "Gemini key file %s is missing; create a JSON object of key aliases to API keys.",
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
        str(name): str(value)
        for name, value in decoded.items()
        if str(name).strip() and str(value).strip()
    }


async def get_gemini_stats_message(detailed: bool = False) -> str:
    """Render shared passive health without firing quota-consuming probe calls."""
    from shin_ai.providers.gemini import get_gemini_scheduler

    snapshot = await get_gemini_scheduler().health_snapshot()
    lines = ["**Gemini Key/Model Health (shared runtime state)**"]
    for model, model_data in snapshot["models"].items():
        total = model_data["total_keys"]
        eligible = model_data["eligible_keys"]
        lines.append(
            f"\n**Model: {model}**\n"
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
    return "\n".join(lines)


API_KEYS_MAP = load_keys()
