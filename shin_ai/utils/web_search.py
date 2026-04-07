import asyncio
import httpx
from bs4 import BeautifulSoup
from duckduckgo_search import AsyncDDGS
from shin_ai.utils.logger_config import logger
import json

async def _fetch_url_content(client: httpx.AsyncClient, url: str) -> str:
    """Fetch and extract text from a single URL."""
    try:
        response = await client.get(url, timeout=5.0, follow_redirects=True)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.extract()
            
        text = soup.get_text(separator=' ', strip=True)
        # Compress whitespace
        import re
        text = re.sub(r'\s+', ' ', text)
        
        # Limit the text to avoid context window explosion
        return text[:2500] + ("..." if len(text) > 2500 else "")
    except Exception as e:
        logger.warning(f"Failed to fetch or parse {url}: {e}")
        return ""

async def search_web_tool(query: str) -> str:
    """
    Search the web for the given query and fetch contents from the top results.
    Returns a JSON string representing the search results and their contents.
    """
    logger.info(f"Executing web search tool for query: '{query}'")
    try:
        results = await AsyncDDGS().text(query, max_results=3)
        if not results:
            return json.dumps({"error": "No results found for the query."})
            
        final_results = []
        async with httpx.AsyncClient(verify=False) as client:
            tasks = []
            for res in results:
                url = res.get('href')
                if url:
                    tasks.append(_fetch_url_content(client, url))
                    
            contents = await asyncio.gather(*tasks, return_exceptions=True)
            
            for index, res in enumerate(results):
                content = contents[index] if index < len(contents) and not isinstance(contents[index], Exception) else ""
                final_results.append({
                    "title": res.get("title", ""),
                    "url": res.get("href", ""),
                    "snippet": res.get("body", ""),
                    "content": content
                })
                
        return json.dumps({"query": query, "results": final_results}, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Web search tool failed: {e}")
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
