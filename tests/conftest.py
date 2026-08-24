"""Hermetic test configuration bootstrap."""

from shin_ai import settings


settings.DEFAULT_CONFIG_PATH = settings.PROJECT_ROOT / "config.yaml.example"
