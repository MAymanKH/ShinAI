import asyncio
import json
from openai import OpenAI
from shin_ai.config import LOCAL_MODEL, AI_PROVIDER_TIMEOUT_SECONDS
from shin_ai.utils.logger_config import logger
from shin_ai.utils.web_search import WEB_SEARCH_TOOL_SCHEMA, search_web_tool
from shin_ai.utils.memory_lookup import MEMORY_LOOKUP_TOOL_SCHEMA, memory_lookup_tool

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
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            tools = [WEB_SEARCH_TOOL_SCHEMA, MEMORY_LOOKUP_TOOL_SCHEMA]
            
            max_turns = 3
            current_turn = 0
            
            while current_turn < max_turns:
                # Wrap the synchronous OpenAI client call in asyncio.to_thread
                response = await asyncio.to_thread(
                    client.chat.completions.create,
                    model=LOCAL_MODEL,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                )
                
                response_message = response.choices[0].message
                
                # Check if the local model decided to call a tool
                if hasattr(response_message, "tool_calls") and response_message.tool_calls:
                    messages.append(response_message.model_dump(exclude_unset=True))
                    
                    for tool_call in response_message.tool_calls:
                        if tool_call.function.name == "search_web_tool":
                            try:
                                args = json.loads(tool_call.function.arguments)
                                query = args.get("query", "")
                            except:
                                query = ""
                                
                            logger.info(f"Local Ollama requested web search for: '{query}'")
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
                            
                            logger.info(f"Local Ollama requested memory lookup with args: {args}")
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
                    logger.info(f"Local Ollama API call successful (model: {LOCAL_MODEL})")
                    return response_message.content

            return response.choices[0].message.content
                
        except Exception as e:
            logger.error(f"Ollama Error (via API): {e}")
            return ""
