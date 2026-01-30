import asyncio
from shin_ai.config import LOCAL_MODEL
from shin_ai.utils.logger_config import logger

# Limit concurrent LLM calls (VERY IMPORTANT on 2-core CPU)
_llm_semaphore = asyncio.Semaphore(1)

async def local_llm(system_prompt, prompt) -> str:
    prompt = f"""
        {system_prompt}\n\n

        Now reply to:
        User: {prompt}
    """
    async with _llm_semaphore:
        # Changed 'generate' to 'run' and removed '-p' flag
        process = await asyncio.create_subprocess_exec(
            "ollama", "run", LOCAL_MODEL, prompt,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error_msg = stderr.decode().strip()
            logger.error(f"Ollama Error: {error_msg}")
            return ""

        return stdout.decode().strip()
