from openai import OpenAI

from shin_ai.config import OPENROUTER_API_KEY, OPENROUTER_MODEL
from shin_ai.providers.tool_loop import run_tool_calling_chat
from shin_ai.utils.logger_config import logger


async def openrouter_api(system_prompt, prompt) -> str:
    api_key = OPENROUTER_API_KEY
    model = OPENROUTER_MODEL

    if not api_key:
        logger.error("OPENROUTER_API_KEY not found")
        return ""

    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        return await run_tool_calling_chat(
            provider_name="OpenRouter",
            create_completion=client.chat.completions.create,
            system_prompt=system_prompt,
            prompt=prompt,
            model=model,
        )
    except Exception as e:
        logger.error(f"Error with OpenRouter API (model: {model}): {e}")
        return ""
