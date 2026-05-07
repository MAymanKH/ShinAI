import asyncio
import json
from collections.abc import Callable
from typing import Any

from shin_ai.utils.logger_config import logger
from shin_ai.utils.memory_lookup import MEMORY_LOOKUP_TOOL_SCHEMA, memory_lookup_tool
from shin_ai.utils.web_search import WEB_SEARCH_TOOL_SCHEMA, search_web_tool


TOOLS = [WEB_SEARCH_TOOL_SCHEMA, MEMORY_LOOKUP_TOOL_SCHEMA]


async def run_tool_calling_chat(
    *,
    provider_name: str,
    create_completion: Callable[..., Any],
    system_prompt: str,
    prompt: str,
    model: str | None,
    max_turns: int = 3,
    **completion_kwargs: Any,
) -> str:
    """Run an OpenAI-compatible chat completion loop with supported tools."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    response = None
    for _ in range(max_turns):
        response = await asyncio.to_thread(
            create_completion,
            messages=messages,
            model=model,
            tools=TOOLS,
            tool_choice="auto",
            **completion_kwargs,
        )

        response_message = response.choices[0].message
        tool_calls = getattr(response_message, "tool_calls", None)
        if not tool_calls:
            logger.info(f"{provider_name} API call successful (model: {model})")
            return response_message.content or ""

        messages.append(response_message.model_dump(exclude_unset=True))
        for tool_call in tool_calls:
            tool_result = await _execute_tool_call(provider_name, tool_call)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": tool_result,
                }
            )

    if response is None:
        return ""
    return response.choices[0].message.content or ""


async def _execute_tool_call(provider_name: str, tool_call: Any) -> str:
    tool_name = tool_call.function.name

    try:
        args = json.loads(tool_call.function.arguments)
    except (TypeError, json.JSONDecodeError):
        args = {}

    if tool_name == "search_web_tool":
        query = args.get("query", "")
        logger.info(f"{provider_name} requested web search for: '{query}'")
        return await search_web_tool(query)

    if tool_name == "memory_lookup_tool":
        logger.info(f"{provider_name} requested memory lookup with args: {args}")
        return await memory_lookup_tool(**args)

    logger.warning(f"{provider_name} requested unknown tool: {tool_name}")
    return json.dumps({"error": f"Unknown tool: {tool_name}"}, ensure_ascii=False)
