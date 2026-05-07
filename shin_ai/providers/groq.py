from groq import Groq

from shin_ai.config import GROQ_API_KEY, GROQ_MODEL
from shin_ai.providers.tool_loop import run_tool_calling_chat
from shin_ai.utils.logger_config import logger


async def groq_api(system_prompt, prompt) -> str:
    api_key = GROQ_API_KEY
    model = GROQ_MODEL

    if not api_key:
        logger.error("GROQ_API_KEY not found")
        return ""

    try:
        client = Groq(api_key=api_key)
        return await run_tool_calling_chat(
            provider_name="Groq",
            create_completion=client.chat.completions.create,
            system_prompt=system_prompt,
            prompt=prompt,
            model=model,
        )
    except Exception as e:
        logger.error(f"Error with Groq API (model: {model}): {e}")
        return ""
