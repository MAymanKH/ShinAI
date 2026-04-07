import uuid
import time
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity
from shin_ai.utils.db import client
from shin_ai.stylers.style_retriever import embedder
from shin_ai.utils.logger_config import logger

# Create the collection for chat memories
memory_collection = client.get_or_create_collection("chat_memories")


# ===========================================
# Semantic Time Detection
# ===========================================

# Time buckets: each bucket has a timedelta and example phrases in multiple languages/dialects.
# The E5 multilingual model handles Arabic dialects natively — no regex needed.
# "dynamic_today" means "from midnight to now" and is computed at query time.
TIME_BUCKETS = [
    {
        "delta_hours": 0.25,  # 15 minutes
        "examples": [
            "a few minutes ago", "just now", "moments ago", "a minute ago",
            "قبل شوي", "من شوي", "قبل دقايق", "قبل شويه", "الحين", "توه", "توها",
            "هسه", "لسه", "قبل لحظات",
        ],
    },
    {
        "delta_hours": 1,
        "examples": [
            "an hour ago", "1 hour ago", "in the last hour", "within the past hour",
            "قبل ساعة", "من ساعة", "قبل ساعه", "الساعة اللي فاتت", "الساعة الماضية",
        ],
    },
    {
        "delta_hours": 3,
        "examples": [
            "a few hours ago", "2 hours ago", "3 hours ago", "couple hours ago",
            "قبل ساعتين", "من ساعتين", "قبل كم ساعة", "قبل ساعات", "من ساعات",
        ],
    },
    {
        "delta_hours": "dynamic_today",  # computed at query time
        "examples": [
            "today", "earlier today", "this morning", "this afternoon", "this evening",
            "اليوم", "اليوم الصبح", "الصبح", "هالصباح", "هاليوم", "اليوم مساء",
        ],
    },
    {
        "delta_hours": 24,
        "examples": [
            "yesterday", "a day ago", "1 day ago", "since yesterday",
            "أمس", "امس", "البارحة", "البارحه", "مبارح", "إمبارح", "امبارح",
        ],
    },
    {
        "delta_hours": 48,
        "examples": [
            "2 days ago", "the day before yesterday", "a couple days ago",
            "أول أمس", "اول امس", "قبل يومين", "من يومين", "أول مبارح",
        ],
    },
    {
        "delta_hours": 72,
        "examples": [
            "3 days ago", "a few days ago", "several days ago", "past few days",
            "قبل كم يوم", "قبل ثلاث ايام", "من كم يوم", "قبل أيام",
        ],
    },
    {
        "delta_hours": 168,  # 1 week
        "examples": [
            "last week", "a week ago", "7 days ago", "in the past week", "this week",
            "الأسبوع اللي فات", "الاسبوع الماضي", "قبل أسبوع", "من اسبوع",
            "الاسبوع اللي راح", "هالاسبوع",
        ],
    },
    {
        "delta_hours": 336,  # 2 weeks
        "examples": [
            "2 weeks ago", "two weeks ago", "a couple weeks ago", "last 2 weeks",
            "قبل أسبوعين", "من اسبوعين", "قبل اسبوعين",
        ],
    },
    {
        "delta_hours": 720,  # ~1 month
        "examples": [
            "last month", "a month ago", "in the past month", "30 days ago", "this month",
            "الشهر اللي فات", "الشهر الماضي", "قبل شهر", "من شهر", "هالشهر",
        ],
    },
    {
        "delta_hours": 504,  # 3 weeks
        "examples": [
            "3 weeks ago", "three weeks ago", "about three weeks ago",
            "قبل ثلاث أسابيع", "قبل ثلاث اسابيع", "من ثلاث اسابيع",
        ],
    },
    {
        "delta_hours": 1440,  # ~2 months
        "examples": [
            "2 months ago", "two months ago", "a couple months ago", "last 2 months",
            "قبل شهرين", "من شهرين",
        ],
    },
    {
        "delta_hours": 2160,  # ~3 months
        "examples": [
            "3 months ago", "a few months ago", "several months ago", "past few months",
            "قبل كم شهر", "قبل ثلاث شهور", "من كم شهر", "قبل أشهر", "قبل شهور",
        ],
    },
    {
        "delta_hours": 4320,  # ~6 months
        "examples": [
            "6 months ago", "half a year ago", "last 6 months",
            "قبل ست شهور", "قبل نص سنة", "من نص سنة", "قبل ٦ شهور",
        ],
    },
    {
        "delta_hours": 8760,  # ~1 year
        "examples": [
            "a year ago", "last year", "1 year ago", "12 months ago", "in the past year",
            "قبل سنة", "السنة اللي فاتت", "من سنة", "العام الماضي", "السنة الماضية",
        ],
    },
]

# Flatten all examples with their bucket index for embedding
_all_examples = []
_example_bucket_indices = []
for idx, bucket in enumerate(TIME_BUCKETS):
    for example in bucket["examples"]:
        _all_examples.append(f"query: {example}")
        _example_bucket_indices.append(idx)

# Pre-compute embeddings at import time (uses the already-loaded E5 model)
logger.info(f"Pre-computing {len(_all_examples)} time reference embeddings...")
_time_example_embeddings = embedder.encode(_all_examples)
_example_bucket_indices = np.array(_example_bucket_indices)
logger.info("Time reference embeddings ready.")

# Also pre-compute "no time reference" examples so we can distinguish
# time-related queries from non-time queries
_NO_TIME_EXAMPLES = [
    "tell me a joke", "explain this code", "what do you think about AI",
    "help me with my homework", "translate this text", "who are you",
    "how are you doing", "write a poem", "analyze this image",
    "قول نكتة", "ساعدني", "مين انت", "ايش رأيك", "شرح لي الكود",
]
_no_time_embeddings = embedder.encode([f"query: {q}" for q in _NO_TIME_EXAMPLES])

# Similarity threshold: query must be more similar to time examples than non-time examples
TIME_DETECTION_MIN_SIMILARITY = 0.55


def _detect_time_filter(query: str) -> tuple[int | None, int | None]:
    """
    Semantically detect time references in the query using the local E5 model.
    Returns (start_epoch, end_epoch) or (None, None) if no time reference found.
    
    Uses pre-computed embeddings for time buckets (English + Arabic dialects).
    Zero API calls — runs entirely on the local model already in memory.
    """
    now = datetime.now().astimezone()
    
    # Encode the query
    query_emb = embedder.encode(f"query: {query}").reshape(1, -1)
    
    # Compare against time examples
    time_similarities = cosine_similarity(query_emb, _time_example_embeddings)[0]
    max_time_sim = float(np.max(time_similarities))
    
    # Compare against non-time examples
    no_time_similarities = cosine_similarity(query_emb, _no_time_embeddings)[0]
    max_no_time_sim = float(np.max(no_time_similarities))
    
    # The query must be clearly more time-related than not
    if max_time_sim < TIME_DETECTION_MIN_SIMILARITY or max_time_sim <= max_no_time_sim:
        logger.debug(
            f"No time reference detected (time_sim={max_time_sim:.3f}, "
            f"no_time_sim={max_no_time_sim:.3f})"
        )
        return None, None
    
    # Find the best matching bucket
    best_example_idx = int(np.argmax(time_similarities))
    best_bucket_idx = _example_bucket_indices[best_example_idx]
    
    # Safety: bump up one bucket so the window is always wider than needed.
    # "5 weeks ago" matching "1 month" → use "2 months" window instead.
    # The last bucket stays as-is since there's nothing bigger.
    safe_bucket_idx = min(best_bucket_idx + 1, len(TIME_BUCKETS) - 1)
    bucket = TIME_BUCKETS[safe_bucket_idx]
    
    # Compute the time delta
    delta_hours = bucket["delta_hours"]
    if delta_hours == "dynamic_today":
        # "today" = from midnight to now
        delta = timedelta(hours=now.hour, minutes=now.minute, seconds=now.second)
    else:
        delta = timedelta(hours=delta_hours)
    
    start_epoch = int((now - delta).timestamp())
    end_epoch = int(now.timestamp())
    
    matched_example = _all_examples[best_example_idx].replace("query: ", "")
    matched_bucket_hours = TIME_BUCKETS[best_bucket_idx]["delta_hours"]
    logger.info(
        f"Time reference detected: '{matched_example}' (sim={max_time_sim:.3f}) "
        f"→ matched {matched_bucket_hours}h, using safe window {delta_hours}h "
        f"→ {start_epoch} to {end_epoch}"
    )
    
    return start_epoch, end_epoch


# ===========================================
# Memory Storage
# ===========================================

def save_memory(user_id: int, username: str, prompt: str, response: str, context: str = "", chat_id: int = 0, chat_title: str = ""):
    """
    Saves a user-bot interaction to the vector database.
    """
    try:
        if not response or not prompt:
            return
        
        # Get formatted timestamp
        now_str = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

        # Format the memory text. 
        if context:
            # If there is context (previous reply), include it so the memory stands on its own
            memory_text = f"Context: {context}\nUser ({username}) said: {prompt}\nBot replied: {response}"
        else:
            memory_text = f"User ({username}) said: {prompt}\nBot replied: {response}"
        
        # Clean up reaction responses for better reading in future
        if response.startswith("react:"):
            reaction = response.split(":")[1]
            memory_text = f"User ({username}) said: {prompt}\nBot reacted with: {reaction}"
        elif response.startswith("sticker:"):
            memory_text = f"User ({username}) said: {prompt}\nBot sent a sticker."

        # Add timestamp to the readable memory text
        memory_text = f"[{now_str}]\n{memory_text}"

        # Metadata for filtering/context
        meta = {
            "user_id": str(user_id),
            "username": username or "Unknown",
            "timestamp": int(time.time()),
            "date_string": now_str,
            "type": "conversation"
        }
        if chat_id:
            meta["chat_id"] = str(chat_id)
        if chat_title:
            meta["chat_title"] = chat_title
        
        # Unique Memory ID
        mem_id = str(uuid.uuid4())
        
        # Create embedding
        # We specifically embed the interaction itself, ignoring the previous context prefix
        # This ensures that searching for "What did I say?" matches the actual content, not the context noise.
        # E5 requires "passage: " prefix for documents to be stored
        # We include the timestamp in the passage so it's nominally searchable, though semantic match is primary.
        searchable_text = f"passage: [{now_str}] User ({username}) said: {prompt}\nBot replied: {response}"
        embedding = embedder.encode(searchable_text).tolist()
        
        memory_collection.add(
            ids=[mem_id],
            documents=[memory_text],
            embeddings=[embedding],
            metadatas=[meta]
        )
        logger.info(f"Memory saved for user {username}")
    except Exception as e:
        logger.error(f"Failed to save memory: {e}")


# ===========================================
# Memory Retrieval
# ===========================================

def retrieve_memories(query: str, limit: int = 5):
    """
    Retrieves semantically relevant past interactions.
    If the query contains a time reference (e.g. "2 days ago", "قبل ساعة"),
    results are constrained to that time window via ChromaDB metadata filtering.
    """
    try:
        # E5 requires "query: " prefix for search queries
        query_emb = embedder.encode(f"query: {query}").tolist()

        # Check for time references in the query
        start_epoch, end_epoch = _detect_time_filter(query)

        where_filter = None
        if start_epoch is not None and end_epoch is not None:
            where_filter = {
                "$and": [
                    {"timestamp": {"$gte": start_epoch}},
                    {"timestamp": {"$lte": end_epoch}},
                ]
            }
            logger.info(f"Time-filtered memory search: {start_epoch} → {end_epoch}")

        # When time-filtering, fetch more results since the user is asking
        # to recall from a specific period (could have hundreds of interactions).
        effective_limit = 15 if where_filter else limit

        results = memory_collection.query(
            query_embeddings=[query_emb],
            n_results=effective_limit,
            where=where_filter,
            include=["documents", "distances"]
        )
        
        filtered_memories = []
        if results['documents']:
            docs = results['documents'][0]
            dists = results['distances'][0]

            # Use a more lenient threshold when time-filtering,
            # since the time window already constrains results.
            threshold = 1.5 if where_filter else 1.3

            for doc, dist in zip(docs, dists):
                if dist < threshold:
                    filtered_memories.append(doc)

        # If time filter was applied but returned nothing, fall back to unfiltered
        if where_filter and not filtered_memories:
            logger.info("Time-filtered search returned no results, falling back to unfiltered")
            return _retrieve_memories_unfiltered(query_emb, limit)

        return filtered_memories
    except Exception as e:
        logger.error(f"Failed to retrieve memories: {e}")
        return []


def _retrieve_memories_unfiltered(query_emb: list, limit: int = 5):
    """
    Fallback: pure semantic retrieval without any time filter.
    Accepts a pre-computed embedding to avoid re-encoding.
    """
    try:
        results = memory_collection.query(
            query_embeddings=[query_emb],
            n_results=limit,
            include=["documents", "distances"]
        )
        filtered = []
        if results['documents']:
            for doc, dist in zip(results['documents'][0], results['distances'][0]):
                if dist < 1.3:
                    filtered.append(doc)
        return filtered
    except Exception as e:
        logger.error(f"Failed to retrieve memories (unfiltered fallback): {e}")
        return []
