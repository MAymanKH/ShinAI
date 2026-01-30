from openai import OpenAI
from shin_ai.config import OPENROUTER_API_KEY, OPENROUTER_MODEL
from shin_ai.utils.logger_config import logger
import asyncio

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

        response = await asyncio.to_thread(
            client.chat.completions.create,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            model=model,
        )

        logger.info(f"DeepSeek API call successful (model: {model})")

        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error with DeepSeek API (model: {model}): {e}")
        return ""
