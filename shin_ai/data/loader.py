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
STICKER_MAPPINGS: str = _stickers_module.STICKER_MAPPINGS
STICKER_TO_DESCRIPTION: dict[str, str] = _stickers_module.STICKER_TO_DESCRIPTION

# Individual sticker IDs (for compatibility)
CLOWN_STICKER_FILE_ID = getattr(_stickers_module, "CLOWN_STICKER_FILE_ID", "")
LAUGHING_STICKER_FILE_ID = getattr(_stickers_module, "LAUGHING_STICKER_FILE_ID", "")
TIRED_STICKER_FILE_ID = getattr(_stickers_module, "TIRED_STICKER_FILE_ID", "")
IDK_WHAT_TO_DO_STICKER_FILE_ID = getattr(_stickers_module, "IDK_WHAT_TO_DO_STICKER_FILE_ID", "")
IDK_WHAT_TO_DO_2_STICKER_FILE_ID = getattr(_stickers_module, "IDK_WHAT_TO_DO_2_STICKER_FILE_ID", "")
EXTREME_SHOCK_STICKER_FILE_ID = getattr(_stickers_module, "EXTREME_SHOCK_STICKER_FILE_ID", "")
SASSY_WHILE_DRINKING_MIX_CHOCHOLATE_STICKER_FILE_ID = getattr(_stickers_module, "SASSY_WHILE_DRINKING_MIX_CHOCHOLATE_STICKER_FILE_ID", "")
SCARED_STICKER_FILE_ID = getattr(_stickers_module, "SCARED_STICKER_FILE_ID", "")
YOU_GOT_A_DEAL_BRO_STICKER_FILE_ID = getattr(_stickers_module, "YOU_GOT_A_DEAL_BRO_STICKER_FILE_ID", "")
HAPPY_WHILE_DRINKING_TEE_STICKER_FILE_ID = getattr(_stickers_module, "HAPPY_WHILE_DRINKING_TEE_STICKER_FILE_ID", "")
HAPPY_WHILE_DRINKING_TEE_WTIH_TEE_STICKER_FILE_ID = getattr(_stickers_module, "HAPPY_WHILE_DRINKING_TEE_WTIH_TEE_STICKER_FILE_ID", "")
NOT_HAPPY_STICKER_FILE_ID = getattr(_stickers_module, "NOT_HAPPY_STICKER_FILE_ID", "")
NOT_HAPPY_2_STICKER_FILE_ID = getattr(_stickers_module, "NOT_HAPPY_2_STICKER_FILE_ID", "")
NOT_HAPPY_3_STICKER_FILE_ID = getattr(_stickers_module, "NOT_HAPPY_3_STICKER_FILE_ID", "")

# ===========================================
# Load Members Configuration
# ===========================================
_members_module = _load_module_with_fallback("members")
MEMBERS: dict[str, dict] = _members_module.MEMBERS

logger.info("Data loader initialized successfully")
