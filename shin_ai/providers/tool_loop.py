import asyncio
import base64
import inspect
import json
from collections.abc import Callable
from typing import Any

from shin_ai.utils.action_tools import ACTION_TOOL_HANDLERS, ACTION_TOOL_SCHEMAS
from shin_ai.utils.logger_config import logger
from shin_ai.utils.memory_lookup import MEMORY_LOOKUP_TOOL_SCHEMA, memory_lookup_tool
from shin_ai.utils.web_search import WEB_SEARCH_TOOL_SCHEMA, search_web_tool


TOOLS = [WEB_SEARCH_TOOL_SCHEMA, MEMORY_LOOKUP_TOOL_SCHEMA, *ACTION_TOOL_SCHEMAS]


async def run_tool_calling_chat(
    *,
    provider_name: str,
    create_completion: Callable[..., Any],
    system_prompt: str,
    prompt: str,
    model: str | None,
    media_list: list[dict] | None = None,
    include_raw_images: bool = False,
    max_turns: int = 3,
    **completion_kwargs: Any,
) -> tuple[str, list[dict]]:
    """Run an OpenAI-compatible chat completion loop with supported tools.

    Returns:
        (response_text, pending_actions) where pending_actions is a list of
        action dicts queued by send_reaction / send_sticker / moderate_user
        tool calls during the generation loop.
    """
    if media_list and include_raw_images:
        content: Any = [{"type": "text", "text": prompt}]
        for idx, media_info in enumerate(media_list, 1):
            image_bytes = media_info['bytes']
            mime_type = media_info['mime_type']
            sender = media_info['sender']
            position = media_info['position']
            media_type = media_info['media_type']

            label = f"\n[Image {idx}/{len(media_list)}: {media_type} from {sender}, {position}]"
            content.append({"type": "text", "text": label})

            b64_str = base64.b64encode(image_bytes).decode('utf-8')
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{b64_str}"
                }
            })
    else:
        content = prompt

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]

    active_tools = list(TOOLS)
    if media_list:
        active_tools.append({
            "type": "function",
            "function": {
                "name": "ask_gemini_about_image",
                "description": (
                    "Ask the Gemini vision model a specific question about the attached image(s) "
                    "to get detailed visual information, read text, identify objects, or verify specific details. "
                    "Use this tool when the user's message/question asks about something specific in the image "
                    "and the initial media description is not detailed enough."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The specific question to ask Gemini about the image(s)."
                        }
                    },
                    "required": ["question"]
                }
            }
        })

    response = None
    pending_actions: list[dict] = []

    for _ in range(max_turns):
        if inspect.iscoroutinefunction(create_completion):
            response = await create_completion(
                messages=messages,
                model=model,
                tools=active_tools,
                tool_choice="auto",
                **completion_kwargs,
            )
        else:
            response = await asyncio.to_thread(
                create_completion,
                messages=messages,
                model=model,
                tools=active_tools,
                tool_choice="auto",
                **completion_kwargs,
            )

        if inspect.isawaitable(response):
            response = await response

        response_message = response.choices[0].message
        tool_calls = getattr(response_message, "tool_calls", None)
        if not tool_calls:
            return response_message.content or "", pending_actions

        messages.append(response_message.model_dump(exclude_unset=True))
        for tool_call in tool_calls:
            tool_result, pending_action = await _execute_tool_call(provider_name, tool_call, media_list)
            if pending_action is not None:
                pending_actions.append(pending_action)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": tool_result,
                }
            )

    if response is None:
        return "", pending_actions
    return response.choices[0].message.content or "", pending_actions


async def _execute_tool_call(
    provider_name: str,
    tool_call: Any,
    media_list: list[dict] | None = None,
) -> tuple[str, dict | None]:
    """Execute a tool call and return (result_str, pending_action_or_None)."""
    tool_name = tool_call.function.name

    try:
        args = json.loads(tool_call.function.arguments)
    except (TypeError, json.JSONDecodeError):
        args = {}

    if tool_name == "search_web_tool":
        query = args.get("query", "")
        logger.info(f"{provider_name} requested web search for: '{query}'")
        result = await search_web_tool(query)
        return result, None

    if tool_name == "memory_lookup_tool":
        logger.info(f"{provider_name} requested memory lookup with args: {args}")
        result = await memory_lookup_tool(**args)
        return result, None

    if tool_name == "ask_gemini_about_image":
        question = args.get("question", "")
        logger.info(f"{provider_name} requested Gemini image info for question: '{question}'")
        if not media_list:
            return "No image is currently attached to this conversation context.", None
        from shin_ai.providers.gemini import gemini_api
        try:
            system_prompt = (
                "You are an assistant that answers specific questions about attached media/images. "
                "Answer the user's question accurately, concisely, and factually based on the visual content."
            )
            answer, _ = await gemini_api(
                system_prompt=system_prompt,
                prompt=question,
                media_list=media_list
            )
            return answer, None
        except Exception as e:
            logger.error(f"Failed to ask Gemini about image: {e}")
            return f"Error querying Gemini about the image: {str(e)}", None

    handler = ACTION_TOOL_HANDLERS.get(tool_name)
    if handler:
        logger.info(f"{provider_name} requested action tool '{tool_name}' with args: {args}")
        result_str, pending_action = await handler(args)
        return result_str, pending_action

    logger.warning(f"{provider_name} requested unknown tool: {tool_name}")
    return json.dumps({"error": f"Unknown tool: {tool_name}"}, ensure_ascii=False), None
