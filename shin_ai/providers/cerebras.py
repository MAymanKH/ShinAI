from cerebras.cloud.sdk import Cerebras
from shin_ai.config import CEREBRAS_API_KEY
from shin_ai.utils.logger_config import logger
import asyncio

async def cerebras_api(system_prompt, prompt) -> str:
    api_key = CEREBRAS_API_KEY
    model = "gpt-oss-120b"

    if not api_key:
        logger.error("CEREBRAS_API_KEY not found")
        return ""

    try:
        client = Cerebras(api_key=api_key)

        response = await asyncio.to_thread(
            client.chat.completions.create,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            model=model,
        )

        logger.info(f"Cerebras API call successful (model: {model})")

        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error with Cerebras API (model: {model}): {e}")
        return ""
