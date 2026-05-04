from openai import OpenAI
from shin_ai.config import OPENROUTER_API_KEY, OPENROUTER_MODEL
from shin_ai.utils.logger_config import logger
from shin_ai.utils.web_search import WEB_SEARCH_TOOL_SCHEMA, search_web_tool
from shin_ai.utils.memory_lookup import MEMORY_LOOKUP_TOOL_SCHEMA, memory_lookup_tool
import asyncio
import json

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

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        tools = [WEB_SEARCH_TOOL_SCHEMA, MEMORY_LOOKUP_TOOL_SCHEMA]
        
        max_turns = 3
        current_turn = 0
        
        while current_turn < max_turns:
            response = await asyncio.to_thread(
                client.chat.completions.create,
                messages=messages,
                model=model,
                tools=tools,
                tool_choice="auto",
            )
            
            response_message = response.choices[0].message
            
            if hasattr(response_message, "tool_calls") and response_message.tool_calls:
                # Add the assistant's request to the messages
                messages.append(response_message.model_dump(exclude_unset=True))
                
                for tool_call in response_message.tool_calls:
                    if tool_call.function.name == "search_web_tool":
                        try:
                            args = json.loads(tool_call.function.arguments)
                            query = args.get("query", "")
                        except:
                            query = ""
                            
                        logger.info(f"OpenRouter requested web search for: '{query}'")
                        tool_result = await search_web_tool(query)
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.function.name,
                            "content": tool_result
                        })
                    elif tool_call.function.name == "memory_lookup_tool":
                        try:
                            args = json.loads(tool_call.function.arguments)
                        except:
                            args = {}
                        
                        logger.info(f"OpenRouter requested memory lookup with args: {args}")
                        tool_result = await memory_lookup_tool(**args)
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.function.name,
                            "content": tool_result
                        })
                current_turn += 1
                continue
            else:
                logger.info(f"OpenRouter API call successful (model: {model})")
                return response_message.content
                
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error with DeepSeek API (model: {model}): {e}")
        return ""
