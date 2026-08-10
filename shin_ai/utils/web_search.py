import asyncio
import contextvars
import json
import re
import time
from collections import OrderedDict

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS
from shin_ai.config import WEB_SEARCH_TIMEOUT_SECONDS
from shin_ai.utils.logger_config import logger

# Request-scoped search counter. Since each user request runs in its own asyncio Task,
# a ContextVar naturally tracks requests independently.
web_search_count = contextvars.ContextVar("web_search_count", default=0)
web_search_start_time = contextvars.ContextVar("web_search_start_time", default=0.0)
web_search_exhausted = contextvars.ContextVar("web_search_exhausted", default=False)


def is_web_search_exhausted() -> bool:
    """Check if web search has been exhausted for the current request context."""
    return web_search_exhausted.get()

# Thread-safe in-memory LRU cache for search results
# Key: lowered stripped query string
# Value: (timestamp: float, results_json: str)
_search_cache: OrderedDict[str, tuple[float, str]] = OrderedDict()
_CACHE_TTL = 300.0  # 5 minutes
_MAX_CACHE_SIZE = 100

async def _fetch_url_content(client: httpx.AsyncClient, url: str) -> str:
    """Fetch and extract text from a single URL with a hard total deadline."""
    try:
        response = await asyncio.wait_for(
            client.get(url, follow_redirects=True),
            timeout=8.0,  # Total wall-clock timeout: connect + read
        )
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

def _format_error_as_result(query: str, err_msg: str) -> str:
    """Format an error message as a standard successful search result JSON string."""
    return json.dumps({
        "query": query,
        "results": [
            {
                "title": "Search Status",
                "url": "",
                "snippet": f"The search could not be completed. Details: {err_msg}",
                "content": f"Search execution details: {err_msg}"
            }
        ]
    }, ensure_ascii=False)

async def _do_firecrawl_search(query: str, api_key: str, timeout: float) -> str:
    """
    Perform the search using Firecrawl /v2/search endpoint.
    Returns the JSON-formatted string on success.
    Raises an exception on any failure.
    """
    url = "https://api.firecrawl.dev/v2/search"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "query": query,
        "limit": 3,
        "scrapeOptions": {
            "formats": ["markdown"]
        }
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        if not data.get("success"):
            raise ValueError(f"Firecrawl returned success=False: {data}")
        
        data_content = data.get("data") or {}
        if isinstance(data_content, dict):
            results = data_content.get("web", [])
        elif isinstance(data_content, list):
            results = data_content
        else:
            results = []
        final_results = []
        for item in results:
            metadata = item.get("metadata") or {}
            title = metadata.get("title") or item.get("title") or ""
            snippet = metadata.get("description") or item.get("description") or ""
            content = item.get("markdown") or ""
            if len(content) > 2500:
                content = content[:2500] + "..."
            final_results.append({
                "title": title,
                "url": item.get("url") or "",
                "snippet": snippet,
                "content": content
            })
            
        return json.dumps({"query": query, "results": final_results}, ensure_ascii=False)


async def search_web_tool(query: str) -> str:
    """
    Search the web for the given query and fetch contents from the top results.
    Returns a JSON string representing the search results and their contents.
    """
    # If the search limit was already hit, immediately return the exhaustion
    # error without doing any work.  This prevents the LLM from looping.
    if web_search_exhausted.get():
        logger.warning(f"Web search already exhausted, rejecting query: '{query}'")
        return _format_error_as_result(
            query,
            "STOP: You have already exhausted all available web searches for this request. "
            "Do NOT call search_web_tool again. You MUST respond to the user now using "
            "the search results you have already gathered."
        )

    # Track overall time budget for a single user request
    now = time.time()
    start_time = web_search_start_time.get()
    if start_time == 0.0:
        web_search_start_time.set(now)
        start_time = now
        
    remaining_time = WEB_SEARCH_TIMEOUT_SECONDS - (now - start_time)
    if remaining_time <= 0:
        web_search_exhausted.set(True)
        logger.warning(f"Web search time limit exceeded before executing query: '{query}'")
        return _format_error_as_result(
            query,
            "STOP: Web search time limit exceeded. Do NOT call search_web_tool again. Respond using search results already gathered."
        )

    # Increment and check the web search limit for this request
    current_count = web_search_count.get() + 1
    web_search_count.set(current_count)
    if current_count > 3:
        web_search_exhausted.set(True)
        logger.warning(f"Web search limit reached (count: {current_count}) for query: '{query}'")
        return _format_error_as_result(
            query,
            "STOP: Web search limit reached (max 3). Do NOT call search_web_tool again. Respond using search results already gathered."
        )

    logger.info(f"Executing web search tool for query: '{query}' (Request count: {current_count}, remaining time: {remaining_time:.1f}s)")
    
    clean_query = query.strip().lower()
    
    # Check cache
    if clean_query in _search_cache:
        cached_time, cached_res = _search_cache[clean_query]
        if now - cached_time < _CACHE_TTL:
            logger.info(f"Returning cached web search results for query: '{query}'")
            return cached_res
        else:
            # Expired — remove stale entry to keep cache clean
            _search_cache.pop(clean_query, None)
            
    # Try Firecrawl search if configured
    firecrawl_key = None
    try:
        from shin_ai.providers.registry import get_config
        cfg = get_config()
        if cfg.firecrawl and cfg.firecrawl.api_key:
            firecrawl_key = cfg.firecrawl.api_key
    except Exception as e:
        logger.warning(f"Could not load Firecrawl configuration: {e}")

    if firecrawl_key:
        now_time = time.time()
        rem_time = WEB_SEARCH_TIMEOUT_SECONDS - (now_time - start_time)
        if rem_time > 0:
            try:
                logger.info(f"Attempting Firecrawl search for query: '{query}' (remaining time: {rem_time:.1f}s)")
                output_json = await asyncio.wait_for(
                    _do_firecrawl_search(query, firecrawl_key, rem_time),
                    timeout=rem_time
                )
                # Cache successful search results
                if clean_query:
                    if len(_search_cache) >= _MAX_CACHE_SIZE:
                        _search_cache.popitem(last=False)
                    _search_cache[clean_query] = (time.time(), output_json)
                logger.info(f"Web search for query '{query}' completed successfully using Firecrawl.")
                return output_json
            except Exception as e:
                logger.warning(f"Firecrawl search failed, falling back to DuckDuckGo: {e}")

    async def _do_search(deadline: float):
        results_list = await asyncio.to_thread(lambda q: list(DDGS().text(q, max_results=3)), query)

        if not results_list:
            return _format_error_as_result(query, f"No results found for the query: '{query}'.")

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
                    remaining = time.time() - deadline
                    if remaining < 2.0:
                        logger.warning("Search scraping aborted — less than 2s remaining in time budget.")
                        break
                    tasks.append(_fetch_url_content(client, url))

            if tasks:
                contents = await asyncio.gather(*tasks, return_exceptions=True)
            else:
                contents = []

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
                _search_cache.popitem(last=False)
            _search_cache[clean_query] = (time.time(), output_json)

        return output_json

    now = time.time()
    remaining_time = WEB_SEARCH_TIMEOUT_SECONDS - (now - start_time)
    if remaining_time <= 0:
        web_search_exhausted.set(True)
        logger.warning(f"Web search time limit exceeded before executing DuckDuckGo query: '{query}'")
        return _format_error_as_result(
            query,
            "STOP: Web search time limit exceeded. Do NOT call search_web_tool again. Respond using search results already gathered."
        )

    search_deadline = now + remaining_time - 1.0  # 1s safety margin

    try:
        logger.info(f"Executing web search query '{query}' using DuckDuckGo (remaining time: {remaining_time:.1f}s).")
        res = await asyncio.wait_for(_do_search(search_deadline), timeout=remaining_time)
        logger.info(f"Web search for query '{query}' completed successfully using DuckDuckGo.")
        return res
    except asyncio.TimeoutError:
        logger.warning(f"Web search timed out (overall limit: 30s) for query: '{query}'")
        return _format_error_as_result(
            query,
            f"Web search overall time limit of {WEB_SEARCH_TIMEOUT_SECONDS:.0f} seconds exceeded for this request. Please construct your final response using the search results already provided."
        )
    except Exception as e:
        err_msg = str(e)
        if "ddgsexception" in type(e).__name__.lower() or "no results found" in err_msg.lower():
            logger.warning("DuckDuckGo search returned no results or failed: %s", e)
            return _format_error_as_result(query, f"DuckDuckGo search failed or returned no results: {err_msg}")
        logger.error("Web search tool failed: %s", e, exc_info=True)
        return _format_error_as_result(query, f"Web search failed: {err_msg}")

# Definition schema to be used for LLM Tool bindings (OpenAI format)
WEB_SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_web_tool",
        "description": "Searches the web to find real-time information, news, or factual data and fetches the text content of the top 3 results. Returns a JSON string containing titles, snippets, and scraped page text.",
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
