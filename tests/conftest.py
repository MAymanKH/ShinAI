"""Hermetic test configuration bootstrap."""

from shin_ai import settings


if not settings.DEFAULT_CONFIG_PATH.exists():
    settings.DEFAULT_CONFIG_PATH = settings.PROJECT_ROOT / "config.yaml.example"
