"""
Data Loader Module

Handles loading of personality, stickers, and members data with fallback to templates.
Loads data once at module initialization for efficiency.
"""
import importlib.util
from pathlib import Path
from typing import Any

from shin_ai.utils.logger_config import logger

# Get the data directory path
DATA_DIR = Path(__file__).parent


def _load_module_with_fallback(module_name: str) -> Any:
    """
    Load a Python module from the data directory with fallback to template.
    
    Args:
        module_name: Name of the module (without .py extension)
        
    Returns:
        The loaded module object
    """
    primary_path = DATA_DIR / f"{module_name}.py"
    template_path = DATA_DIR / f"{module_name}_template.py"
    
    # Determine which file to load
    if primary_path.exists():
        target_path = primary_path
        logger.debug(f"Loading {module_name} from primary file")
    elif template_path.exists():
        target_path = template_path
        logger.info(f"Loading {module_name} from template (primary not found)")
    else:
        raise FileNotFoundError(
            f"Neither {primary_path} nor {template_path} exists. "
            f"Please create {module_name}.py or {module_name}_template.py"
        )
    
    # Dynamic import
    spec = importlib.util.spec_from_file_location(f"{module_name}_module", target_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create module spec for {target_path}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ===========================================
# Load Personality Configuration
# ===========================================
_personality_module = _load_module_with_fallback("personality")
PERSONALITY: dict[str, str] = _personality_module.personality

# ===========================================
# Load Sticker Configuration  
# ===========================================
_stickers_module = _load_module_with_fallback("stickers")
TELEGRAM_STICKERS: dict[str, str] = getattr(_stickers_module, "TELEGRAM_STICKERS", {})
WHATSAPP_STICKERS: dict[str, str] = getattr(_stickers_module, "WHATSAPP_STICKERS", {})
TELEGRAM_STICKER_MAPPINGS: str = getattr(_stickers_module, "TELEGRAM_STICKER_MAPPINGS", "")
WHATSAPP_STICKER_MAPPINGS: str = getattr(_stickers_module, "WHATSAPP_STICKER_MAPPINGS", "")
TELEGRAM_STICKER_TO_DESCRIPTION: dict[str, str] = getattr(_stickers_module, "TELEGRAM_STICKER_TO_DESCRIPTION", {})
WHATSAPP_STICKER_TO_DESCRIPTION: dict[str, str] = getattr(_stickers_module, "WHATSAPP_STICKER_TO_DESCRIPTION", {})

# ===========================================
# Load Members Configuration
# ===========================================
MEMBERS: dict[str, dict] = PERSONALITY.get("core_relationships", {})

logger.info("Data loader initialized successfully")
