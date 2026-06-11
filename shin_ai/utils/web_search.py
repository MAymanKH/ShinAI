import asyncio
import contextvars
import json
import re
import time
import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS
from shin_ai.utils.logger_config import logger

# Request-scoped search counter. Since each user request runs in its own asyncio Task,
# a ContextVar naturally tracks requests independently.
web_search_count = contextvars.ContextVar("web_search_count", default=0)

# Simple thread-safe in-memory cache for search results
# Key: lowered stripped query string
# Value: (timestamp: float, results_json: str)
_search_cache: dict[str, tuple[float, str]] = {}
_CACHE_TTL = 300.0  # 5 minutes
_MAX_CACHE_SIZE = 100

async def _fetch_url_content(client: httpx.AsyncClient, url: str) -> str:
    """Fetch and extract text from a single URL."""
    try:
        response = await client.get(url, timeout=3.0, follow_redirects=True)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.extract()
            
        text = soup.get_text(separator=' ', strip=True)
        # Compress whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Limit the text to avoid context window explosion
        return text[:2500] + ("..." if len(text) > 2500 else "")
    except Exception as e:
        logger.warning("Failed to fetch or parse %s: %s", url, e)
        return ""

async def search_web_tool(query: str) -> str:
    """
    Search the web for the given query and fetch contents from the top results.
    Returns a JSON string representing the search results and their contents.
    """
    # Increment and check the web search limit for this request
    current_count = web_search_count.get() + 1
    web_search_count.set(current_count)
    if current_count > 5:
        logger.warning(f"Web search limit reached (count: {current_count}) for query: '{query}'")
        return json.dumps({
            "error": "Web search limit reached for this request. Please construct your final response using the search results already provided."
        })

    logger.info(f"Executing web search tool for query: '{query}' (Request count: {current_count})")
    
    clean_query = query.strip().lower()
    now = time.time()
    
    # Check cache
    if clean_query in _search_cache:
        cached_time, cached_res = _search_cache[clean_query]
        if now - cached_time < _CACHE_TTL:
            logger.info(f"Returning cached web search results for query: '{query}'")
            return cached_res
            
    try:
        results_list = await asyncio.to_thread(lambda q: list(DDGS().text(q, max_results=3)), query)
        
        if not results_list:
            return json.dumps({"error": f"No results found for the query: '{query}'."})
            
        final_results = []
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        async with httpx.AsyncClient(verify=False, headers=headers) as client:
            tasks = []
            for res in results_list:
                url = res.get('href')
                if url:
                    tasks.append(_fetch_url_content(client, url))
                    
            contents = await asyncio.gather(*tasks, return_exceptions=True)
            
            for index, res in enumerate(results_list):
                content = contents[index] if index < len(contents) and not isinstance(contents[index], Exception) else ""
                final_results.append({
                    "title": res.get("title", ""),
                    "url": res.get("href", ""),
                    "snippet": res.get("body", "") or res.get("snippet", ""),
                    "content": content
                })
        
        output_json = json.dumps({"query": query, "results": final_results}, ensure_ascii=False)
        
        # Cache successful search results
        if clean_query:
            if len(_search_cache) >= _MAX_CACHE_SIZE:
                # Evict oldest entry
                oldest_key = min(_search_cache.keys(), key=lambda k: _search_cache[k][0])
                _search_cache.pop(oldest_key, None)
            _search_cache[clean_query] = (now, output_json)
            
        return output_json
    except Exception as e:
        logger.error("Web search tool failed: %s", e, exc_info=True)
        return json.dumps({"error": f"Search failed: {str(e)}"})

# Definition schema to be used for LLM Tool bindings (OpenAI format)
WEB_SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_web_tool",
        "description": "Searches the web using DuckDuckGo to find real-time information, news, or factual data and fetches the text content of the top 3 results. Returns a JSON string containing titles, snippets, and scraped page text.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up on the web."
                }
            },
            "required": ["query"]
        }
    }
}
