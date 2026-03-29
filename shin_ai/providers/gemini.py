"""
Gemini AI Provider

Handles API calls to Google's Gemini models with key rotation and statistics.
"""
from google import genai
from shin_ai.config import GEMINI_MODEL, DATA_DIR
from shin_ai.utils.logger_config import logger
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import os
import time
import asyncio
import re
from datetime import datetime

# File paths for key management
GEMINI_KEYS_FILE = DATA_DIR / "gemini_keys.json"
STATS_FILE = DATA_DIR / "gemini_stats.json"


def load_keys() -> dict[str, str]:
    """Load API keys from JSON file or environment variables."""
    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # If the JSON file exists, load keys from it (including order)
    if GEMINI_KEYS_FILE.exists():
        try:
            with open(GEMINI_KEYS_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load keys from {GEMINI_KEYS_FILE}: {e}")
    
    # Fallback/Migration: Load from environment variables if JSON doesn't exist
    logger.info("Initializing gemini_keys.json from environment variables...")
    keys_map = {}
    for i in range(1, 40):  # Check up to 40 just in case
        key_name = f"GEMINI_API_KEY{i}"
        api_key = os.getenv(key_name)
        if api_key:
            keys_map[key_name] = api_key
    
    # Save the initialized keys to the file
    if keys_map:
        save_keys(keys_map)
        
    return keys_map


def save_keys(current_map: dict[str, str]) -> None:
    """Save API keys to JSON file."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(GEMINI_KEYS_FILE, "w") as f:
            json.dump(current_map, f, indent=4)
    except Exception as e:
        logger.error(f"Failed to save keys to {GEMINI_KEYS_FILE}: {e}")


def load_stats() -> dict:
    """Load key statistics from JSON file."""
    if STATS_FILE.exists():
        try:
            with open(STATS_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_stats(stats: dict) -> None:
    """Save key statistics to JSON file."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(STATS_FILE, "w") as f:
            json.dump(stats, f, indent=4)
    except Exception as e:
        logger.error(f"Failed to save stats to {STATS_FILE}: {e}")

def update_key_status(key_name, status, model=None, error_msg=None):
    if not model: return
    stats = load_stats()
    
    # Initialize key dict if missing
    if key_name not in stats:
        stats[key_name] = {}
        
    # Check for legacy format (flat dict with "status") and wipe/migrate if found
    if "status" in stats[key_name]:
        # Migration: preserve old data under its model if it exists
        old_data = stats[key_name]
        stats[key_name] = {}
        if old_data.get("model"):
            stats[key_name][old_data["model"]] = old_data

    stats[key_name][model] = {
        "status": status,
        "last_updated": time.time(),
        "last_updated_str": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "error": str(error_msg) if error_msg else None
    }
    save_stats(stats)

async def check_single_key_status(key_name, api_key, model):
    try:
        client = genai.Client(api_key=api_key)
        # Use a minimal generation prompt to test real availability (models.get doesn't catch all quota issues)
        await client.aio.models.generate_content(
            model=model, 
            contents="a", # Minimal prompt
            config=genai.types.GenerateContentConfig(max_output_tokens=1)
        )
        logger.info(f"Key status check [{model}]: {key_name} is ACTIVE")
        return {"key": key_name, "model": model, "status": "active", "error": None}
    except Exception as e:
        error_msg = str(e)
        status = "error"
        if "quota" in error_msg.lower() or "429" in error_msg:
            status = "exhausted"
        logger.warning(f"Key status check [{model}]: {key_name} is {status.upper()} - {error_msg}")
        return {"key": key_name, "model": model, "status": status, "error": error_msg}

async def get_gemini_stats_message(detailed=False):
    keys = API_KEYS_MAP # Current keys
    total_keys = len(keys)
    current_models = list(MODELS_LIST)
    
    # Run checks in parallel
    tasks = []
    for model in current_models:
        for key_name in sorted(keys.keys()):
            api_key = keys.get(key_name)
            if api_key:
                tasks.append(check_single_key_status(key_name, api_key, model))
    
    # Wait for all checks to complete
    results = await asyncio.gather(*tasks)
    
    # Group results by model
    model_results = {model: [] for model in current_models}
    for res in results:
        if res["model"] in model_results:
            model_results[res["model"]].append(res)
    
    report_lines = ["**Gemini Key Statistics (Live Check)**"]
    
    for model in current_models:
        active = 0
        exhausted = 0
        error = 0
        details = []
        
        # Iterating over the results we just collected
        for res in model_results[model]:
            key_name = res["key"]
            status = res["status"]
            
            if status == "active":
                active += 1
            elif status == "exhausted":
                exhausted += 1
                details.append(f"• {key_name}: ❌ Exhausted")
            else:
                error += 1
                err_msg = res.get("error", "Unknown error")
                details.append(f"• {key_name}: ⚠️ Error: {err_msg[:20]}...")

        available_count = active
        percentage_left = (available_count / total_keys) * 100 if total_keys > 0 else 0.0
        
        section = f"""
**Model: {model}**
Health: {percentage_left:.1f}% Available
✅ Active: {active}
❌ Exhausted: {exhausted} | ⚠️ Errors: {error}"""
        
        if detailed and details:
            section += "\nIssues:\n" + "\n".join(details)
            
        report_lines.append(section)

    return "\n".join(report_lines)

API_KEYS_MAP = load_keys()


MODELS_LIST = [
    "gemini-3-flash-preview",
    "gemini-2.5-flash"
]

# Initialize embedder for semantic query detection
logger.info("Loading semantic model for query classification...")
embedder = SentenceTransformer("intfloat/multilingual-e5-large")

# Example queries that require Google Search (real-time information)
SEARCH_NEEDED_EXAMPLES = [
    "What's the latest news about AI?",
    "Current weather in Tokyo",
    "What's happening in the stock market today?",
    "Who won the game yesterday?",
    "Latest updates on the election",
    "What's the price of Bitcoin now?",
    "Recent developments in technology",
    "Today's breaking news",
    "What time is it in New York?",
    "Current temperature in London",
    "Latest iPhone release date",
    "What's trending on Twitter today?",
    "Recent sports scores",
    "Today's exchange rate",
    "What's new in 2026?",
]

# Example queries that DON'T need search (reasoning/knowledge-based)
NO_SEARCH_EXAMPLES = [
    "Explain quantum physics to me",
    "Write a poem about love",
    "Help me debug this code",
    "What's the meaning of life?",
    "How do I solve this math problem?",
    "Tell me a joke",
    "Analyze this image",
    "Translate this to Spanish",
    "Summarize this document",
    "Give me coding advice",
]

# Pre-compute embeddings for examples (using E5 query prefix)
logger.info("Pre-computing example embeddings...")
search_needed_embeddings = embedder.encode([f"query: {q}" for q in SEARCH_NEEDED_EXAMPLES])
no_search_embeddings = embedder.encode([f"query: {q}" for q in NO_SEARCH_EXAMPLES])

REALTIME_KEYWORDS = {
    "latest", "recent", "today", "now", "current", "currently", "breaking", "live",
    "this week", "this month", "this year", "yesterday", "tomorrow", "update", "updates",
    "news", "weather", "temperature", "forecast", "score", "scores", "match", "game",
    "price", "prices", "stock", "stocks", "market", "exchange rate", "rate", "rates",
    "trending", "release date", "launch date", "earnings", "headline", "headlines"
}

WEB_LOOKUP_PHRASES = {
    "search web", "search the web", "search online", "look up", "find online",
    "check online", "use google", "browse", "web search", "internet"
}

NO_SEARCH_KEYWORDS = {
    "explain", "tutorial", "example code", "debug", "refactor", "translate",
    "summarize this", "poem", "story", "joke", "brainstorm", "idea", "meaning"
}


def _extract_focus_text(prompt: str) -> str:
    """Use the most recent user-like segment for intent classification."""
    normalized = (prompt or "").strip()
    if not normalized:
        return ""

    # Prefer the last non-empty line to avoid embedding full conversation/system context.
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    if lines:
        last_line = lines[-1]
        if len(last_line) >= 12:
            return last_line

    # Fall back to the tail section where the newest request usually appears.
    return normalized[-700:]


def _contains_any_phrase(text: str, phrases: set[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def _contains_year_like_reference(text: str) -> bool:
    years = re.findall(r"\b(20\d{2})\b", text)
    if not years:
        return False

    current_year = datetime.now().year
    # Asking about a specific modern year often implies recency-sensitive information.
    return any(abs(int(year) - current_year) <= 2 for year in years)


def needs_google_search(prompt: str, threshold: float = 0.58) -> bool:
    """
    Semantically detect if a query needs real-time web information using embeddings.
    Returns True if Google Search would be beneficial.
    """
    focus_text = _extract_focus_text(prompt)
    lowered = focus_text.lower()

    has_web_lookup_phrase = _contains_any_phrase(lowered, WEB_LOOKUP_PHRASES)
    has_realtime_keyword = _contains_any_phrase(lowered, REALTIME_KEYWORDS)
    has_year_reference = _contains_year_like_reference(lowered)
    has_no_search_keyword = _contains_any_phrase(lowered, NO_SEARCH_KEYWORDS)

    # Strong deterministic signals first.
    if has_web_lookup_phrase:
        logger.debug("Search intent: explicit web lookup phrase detected")
        return True

    if has_realtime_keyword or has_year_reference:
        logger.debug("Search intent: realtime keyword/year reference detected")
        return True

    if has_no_search_keyword and not (has_realtime_keyword or has_web_lookup_phrase):
        logger.debug("Search intent: static/creative task keyword detected")
        return False

    # Encode the most relevant user query segment with E5 query prefix.
    query_embedding = embedder.encode(f"query: {focus_text}").reshape(1, -1)
    
    # Calculate similarity to "search needed" examples
    search_similarities = cosine_similarity(query_embedding, search_needed_embeddings)[0]
    max_search_similarity = np.max(search_similarities)
    
    # Calculate similarity to "no search needed" examples
    no_search_similarities = cosine_similarity(query_embedding, no_search_embeddings)[0]
    max_no_search_similarity = np.max(no_search_similarities)
    
    similarity_delta = max_search_similarity - max_no_search_similarity

    # Confidence rules:
    # 1) Standard positive match with reduced threshold.
    # 2) Low-confidence tie-breaker biased toward enabling search to reduce false negatives.
    needs_search = (
        (max_search_similarity >= threshold and similarity_delta >= -0.02)
        or (max_search_similarity >= 0.52 and similarity_delta >= 0.06)
        or (max_search_similarity >= 0.50 and abs(similarity_delta) <= 0.02)
    )
    
    logger.debug(
        "Query similarity - Focus: '%s' | Search: %.3f | No-Search: %.3f | Delta: %.3f | Needs search: %s",
        focus_text[:120],
        max_search_similarity,
        max_no_search_similarity,
        similarity_delta,
        needs_search,
    )
    
    return needs_search

async def gemini_api(system_prompt, prompt, media_list=None)  -> str:
    failed_keys_count = 0
    
    # Intelligently reorder models based on query requirements
    if needs_google_search(prompt):
        models_to_try = ["gemini-2.5-flash", "gemini-3-flash-preview"]  # Prioritize 2.5 for search
        logger.info("🔍 Query needs Google Search - prioritizing Gemini 2.5 Flash")
    else:
        models_to_try = list(MODELS_LIST)  # Use default order (3 first)
        logger.info("🧠 Query uses reasoning - using Gemini 3 Flash Preview")

    for model in models_to_try:
        # Create a list of items to iterate over, preserving the current order
        for key_name, api_key in list(API_KEYS_MAP.items()):
            if not api_key:
                continue

            try:
                genai_client = genai.Client(api_key=api_key)
                # Prepare contents - start with text prompt
                contents = [prompt]
                
                # Add all media from the conversation context with descriptive labels
                if media_list:
                    for idx, media_info in enumerate(media_list, 1):
                        image_bytes = media_info['bytes']
                        mime_type = media_info['mime_type']
                        sender = media_info['sender']
                        position = media_info['position']
                        media_type = media_info['media_type']
                        
                        # Add a text label before each image to provide context
                        label = f"\n[Image {idx}/{len(media_list)}: {media_type} from {sender}, {position}]"
                        contents.append(label)
                        
                        # Add the actual image
                        contents.append(
                            genai.types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
                        )
                    
                    logger.info(f"Added {len(media_list)} media items with positional context to Gemini request")

                # Conditionally enable Google Search based on model
                # Gemini 3 models may have issues with Google Search in some configurations
                if "gemini-3" in model:
                    config = genai.types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        thinking_config=genai.types.ThinkingConfig(thinking_level="high")
                    )
                    logger.info(f"Using {model} without Google Search")
                else:
                    config = genai.types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        tools=[genai.types.Tool(google_search=genai.types.GoogleSearch())]
                    )
                    logger.info(f"Using {model} with Google Search enabled")

                response = await genai_client.aio.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config
                )

                logger.info(f"Gemini API call successful (model: {model}, Key: {key_name})")
                update_key_status(key_name, "active", model)
                return response.text
            except Exception as e:
                failed_keys_count += 1
                # Move the failed key to the end of the dictionary
                if key_name in API_KEYS_MAP:
                    API_KEYS_MAP[key_name] = API_KEYS_MAP.pop(key_name)
                    save_keys(API_KEYS_MAP)

                if "you exceeded your current quota, please check your plan and billing details" in str(e).lower() or "429" in str(e):
                    logger.warning(f"Gemini API key quota exceeded for model {model} (Key: {key_name}, Failed Count: {failed_keys_count})")
                    update_key_status(key_name, "exhausted", model, e)
                else:
                    logger.error(f"Error with Gemini API (model: {model}, Key: {key_name}, Failed Count: {failed_keys_count}): {e}")
                    update_key_status(key_name, "error", model, e)
                continue

        logger.warning(f"Model {model} failed for all keys. Rotating to end of list.")
        if model in models_to_try:
            models_to_try.append(models_to_try.pop(models_to_try.index(model)))

    return ""