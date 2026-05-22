from openai import OpenAI
from shin_ai.config import OPENAI_COMPAT_API_KEY, OPENAI_COMPAT_BASE_URL, OPENAI_COMPAT_MODEL
from shin_ai.providers.tool_loop import run_tool_calling_chat
from shin_ai.utils.logger_config import logger


async def openai_compatible_api(system_prompt, prompt) -> str:
    api_key = OPENAI_COMPAT_API_KEY
    base_url = OPENAI_COMPAT_BASE_URL
    model = OPENAI_COMPAT_MODEL

    if not api_key:
        logger.error("OPENAI_COMPAT_API_KEY not found")
        return ""

    if not base_url:
        logger.error("OPENAI_COMPAT_BASE_URL not found")
        return ""

    if not model:
        logger.error("OPENAI_COMPAT_MODEL not configured")
        return ""

    try:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        return await run_tool_calling_chat(
            provider_name="OpenAI-Compatible",
            create_completion=client.chat.completions.create,
            system_prompt=system_prompt,
            prompt=prompt,
            model=model,
        )
    except Exception as e:
        logger.error(f"Error with OpenAI-Compatible API (base_url: {base_url}, model: {model}): {e}")
        return ""
