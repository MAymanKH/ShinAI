"""Filesystem locations derived from the package layout.

These are structural facts about where the project lives, not configuration,
so they carry no dependency on settings and are safe to import anywhere.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Runtime state written by the bot: key files, reply history, coordination db.
DATA_DIR = PROJECT_ROOT / "data"

# Content shipped with the package: personality, sticker definitions, assets.
PACKAGE_DATA_DIR = Path(__file__).resolve().parent / "data"
