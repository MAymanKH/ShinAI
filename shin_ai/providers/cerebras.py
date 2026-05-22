from cerebras.cloud.sdk import Cerebras

from shin_ai.config import CEREBRAS_API_KEY, CEREBRAS_MODEL
from shin_ai.providers.tool_loop import run_tool_calling_chat
from shin_ai.utils.logger_config import logger


async def cerebras_api(system_prompt, prompt) -> str:
    api_key = CEREBRAS_API_KEY
    model = CEREBRAS_MODEL

    if not api_key:
        logger.error("CEREBRAS_API_KEY not found")
        return ""

    if not model:
        logger.error("CEREBRAS_MODEL not configured")
        return ""

    try:
        client = Cerebras(api_key=api_key)
        return await run_tool_calling_chat(
            provider_name="Cerebras",
            create_completion=client.chat.completions.create,
            system_prompt=system_prompt,
            prompt=prompt,
            model=model,
        )
    except Exception as e:
        logger.error(f"Error with Cerebras API (model: {model}): {e}")
        return ""
