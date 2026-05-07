import asyncio

from openai import OpenAI

from shin_ai.config import LOCAL_MODEL
from shin_ai.providers.tool_loop import run_tool_calling_chat
from shin_ai.utils.logger_config import logger


# Limit concurrent LLM calls (VERY IMPORTANT on 2-core CPU)
_llm_semaphore = asyncio.Semaphore(1)

async def local_llm(system_prompt, prompt) -> str:
    async with _llm_semaphore:
        try:
            # We use the OpenAI compatible API provided by Ollama locally
            client = OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama"  # API key is required by the SDK but ignored by Ollama
            )

            return await run_tool_calling_chat(
                provider_name="Local Ollama",
                create_completion=client.chat.completions.create,
                system_prompt=system_prompt,
                prompt=prompt,
                model=LOCAL_MODEL,
            )
                
        except Exception as e:
            logger.error(f"Ollama Error (via API): {e}")
            return ""
