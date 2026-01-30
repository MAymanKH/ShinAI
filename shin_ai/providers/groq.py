from groq import Groq
from shin_ai.config import GROQ_API_KEY, GROQ_MODEL
from shin_ai.utils.logger_config import logger
import asyncio

async def groq_api(system_prompt, prompt) -> str:
    api_key = GROQ_API_KEY
    model = GROQ_MODEL

    if not api_key:
        logger.error("GROQ_API_KEY not found")
        return ""

    try:
        client = Groq(api_key=api_key)

        response = await asyncio.to_thread(
            client.chat.completions.create,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            model=model,
        )

        logger.info(f"Groq API call successful (model: {model})")

        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error with Groq API (model: {model}): {e}")
        return ""
