import asyncio
from datetime import datetime, timedelta

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from shin_ai.stylers.style_retriever import embedder
from shin_ai.utils.logger_config import logger


# Time buckets: each bucket has a timedelta and example phrases in multiple languages/dialects.
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
        "delta_hours": "dynamic_today",
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
        "delta_hours": 168,
        "examples": [
            "last week", "a week ago", "7 days ago", "in the past week", "this week",
            "الأسبوع اللي فات", "الاسبوع الماضي", "قبل أسبوع", "من اسبوع",
            "الاسبوع اللي راح", "هالاسبوع",
        ],
    },
    {
        "delta_hours": 336,
        "examples": [
            "2 weeks ago", "two weeks ago", "a couple weeks ago", "last 2 weeks",
            "قبل أسبوعين", "من اسبوعين", "قبل اسبوعين",
        ],
    },
    {
        "delta_hours": 504,
        "examples": [
            "3 weeks ago", "three weeks ago", "about three weeks ago",
            "قبل ثلاث أسابيع", "قبل ثلاث اسابيع", "من ثلاث اسابيع",
        ],
    },
    {
        "delta_hours": 720,
        "examples": [
            "last month", "a month ago", "in the past month", "30 days ago", "this month",
            "الشهر اللي فات", "الشهر الماضي", "قبل شهر", "من شهر", "هالشهر",
        ],
    },
    {
        "delta_hours": 1440,
        "examples": [
            "2 months ago", "two months ago", "a couple months ago", "last 2 months",
            "قبل شهرين", "من شهرين",
        ],
    },
    {
        "delta_hours": 2160,
        "examples": [
            "3 months ago", "a few months ago", "several months ago", "past few months",
            "قبل كم شهر", "قبل ثلاث شهور", "من كم شهر", "قبل أشهر", "قبل شهور",
        ],
    },
    {
        "delta_hours": 4320,
        "examples": [
            "6 months ago", "half a year ago", "last 6 months",
            "قبل ست شهور", "قبل نص سنة", "من نص سنة", "قبل ٦ شهور",
        ],
    },
    {
        "delta_hours": 8760,
        "examples": [
            "a year ago", "last year", "1 year ago", "12 months ago", "in the past year",
            "قبل سنة", "السنة اللي فاتت", "من سنة", "العام الماضي", "السنة الماضية",
        ],
    },
]

_all_examples = []
_example_bucket_indices = []
for idx, bucket in enumerate(TIME_BUCKETS):
    for example in bucket["examples"]:
        _all_examples.append(f"query: {example}")
        _example_bucket_indices.append(idx)

logger.info(f"Pre-computing {len(_all_examples)} time reference embeddings...")
_time_example_embeddings = embedder.encode(_all_examples)
_example_bucket_indices = np.array(_example_bucket_indices)
logger.info("Time reference embeddings ready.")

_NO_TIME_EXAMPLES = [
    "tell me a joke", "explain this code", "what do you think about AI",
    "help me with my homework", "translate this text", "who are you",
    "how are you doing", "write a poem", "analyze this image",
    "قول نكتة", "ساعدني", "مين انت", "ايش رأيك", "شرح لي الكود",
    "what did I say", "ايش قلت", "do you remember", "تتذكر",
    "who said", "مين قال", "what happened", "ايش صار",
]
_no_time_embeddings = embedder.encode([f"query: {q}" for q in _NO_TIME_EXAMPLES])

_TEMPORAL_INDICATOR_WORDS = {
    "ago", "yesterday", "today", "last", "week", "month", "year", "hour", "hours",
    "minute", "minutes", "morning", "evening", "afternoon", "earlier", "recently",
    "days", "weeks", "months", "years", "past", "previous", "prior",
    "قبل", "أمس", "امس", "البارحة", "البارحه", "اليوم", "الصبح", "مبارح",
    "امبارح", "إمبارح", "ساعة", "ساعه", "ساعتين", "ساعات", "يومين", "أيام",
    "ايام", "اسبوع", "أسبوع", "اسبوعين", "أسبوعين", "شهر", "شهرين", "شهور",
    "سنة", "سنه", "الماضي", "الماضية", "شوي", "شويه", "الحين", "توه", "توها",
    "هسه", "لسه", "لحظات", "هالصباح", "هاليوم", "هالاسبوع", "هالشهر",
}

TIME_DETECTION_MIN_SIMILARITY = 0.62
TIME_DETECTION_MIN_GAP = 0.08


async def detect_time_filter(query: str) -> tuple[int | None, int | None]:
    """
    Semantically detect time references in the query using the local E5 model.
    Returns (start_epoch, end_epoch) or (None, None) if no time reference found.
    """
    query_words = set(query.lower().split())
    if not query_words & _TEMPORAL_INDICATOR_WORDS:
        logger.debug("No temporal keywords in query, skipping time detection")
        return None, None

    now = datetime.now().astimezone()

    query_emb_tensor = await asyncio.to_thread(embedder.encode, f"query: {query}")
    query_emb = query_emb_tensor.reshape(1, -1)

    time_similarities = cosine_similarity(query_emb, _time_example_embeddings)[0]
    max_time_sim = float(np.max(time_similarities))

    no_time_similarities = cosine_similarity(query_emb, _no_time_embeddings)[0]
    max_no_time_sim = float(np.max(no_time_similarities))

    if max_time_sim < TIME_DETECTION_MIN_SIMILARITY or max_time_sim < max_no_time_sim + TIME_DETECTION_MIN_GAP:
        logger.debug(
            f"Time detection rejected (time_sim={max_time_sim:.3f}, "
            f"no_time_sim={max_no_time_sim:.3f}, "
            f"gap={max_time_sim - max_no_time_sim:.3f}, required_gap={TIME_DETECTION_MIN_GAP})"
        )
        return None, None

    best_example_idx = int(np.argmax(time_similarities))
    best_bucket_idx = _example_bucket_indices[best_example_idx]

    safe_bucket_idx = min(best_bucket_idx + 1, len(TIME_BUCKETS) - 1)
    bucket = TIME_BUCKETS[safe_bucket_idx]

    delta_hours = bucket["delta_hours"]
    if delta_hours == "dynamic_today":
        delta = timedelta(hours=now.hour, minutes=now.minute, seconds=now.second)
    else:
        delta = timedelta(hours=delta_hours)

    start_epoch = int((now - delta).timestamp())
    end_epoch = int(now.timestamp())

    matched_example = _all_examples[best_example_idx].replace("query: ", "")
    matched_bucket_hours = TIME_BUCKETS[best_bucket_idx]["delta_hours"]
    logger.debug(
        "Time reference detected: '%s' (sim=%.3f) → matched %sh, using safe window %sh → %d–%d",
        matched_example, max_time_sim, matched_bucket_hours, delta_hours, start_epoch, end_epoch,
    )

    return start_epoch, end_epoch
