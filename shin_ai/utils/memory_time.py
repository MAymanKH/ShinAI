import asyncio
import re
from datetime import datetime, timedelta

import numpy as np

from shin_ai.services.embeddings import get_embedding_service
from shin_ai.utils.logger_config import logger
from shin_ai.utils.similarity import cosine_similarities

# Time buckets: each bucket has a timedelta and example phrases in multiple languages/dialects.
# "dynamic_today" means "from midnight to now" and is computed at query time.
TIME_BUCKETS = [
    {
        "delta_hours": 0.25,  # 15 minutes
        "examples": [
            "a few minutes ago",
            "just now",
            "moments ago",
            "a minute ago",
            "قبل شوي",
            "من شوي",
            "قبل دقايق",
            "قبل شويه",
            "الحين",
            "توه",
            "توها",
            "هسه",
            "لسه",
            "قبل لحظات",
        ],
    },
    {
        "delta_hours": 1,
        "examples": [
            "an hour ago",
            "1 hour ago",
            "in the last hour",
            "within the past hour",
            "قبل ساعة",
            "من ساعة",
            "قبل ساعه",
            "الساعة اللي فاتت",
            "الساعة الماضية",
        ],
    },
    {
        "delta_hours": 3,
        "examples": [
            "a few hours ago",
            "2 hours ago",
            "3 hours ago",
            "couple hours ago",
            "قبل ساعتين",
            "من ساعتين",
            "قبل كم ساعة",
            "قبل ساعات",
            "من ساعات",
        ],
    },
    {
        "delta_hours": "dynamic_today",
        "examples": [
            "today",
            "earlier today",
            "this morning",
            "this afternoon",
            "this evening",
            "اليوم",
            "اليوم الصبح",
            "الصبح",
            "هالصباح",
            "هاليوم",
            "اليوم مساء",
        ],
    },
    {
        "delta_hours": 24,
        "examples": [
            "yesterday",
            "a day ago",
            "1 day ago",
            "since yesterday",
            "أمس",
            "امس",
            "البارحة",
            "البارحه",
            "مبارح",
            "إمبارح",
            "امبارح",
        ],
    },
    {
        "delta_hours": 48,
        "examples": [
            "2 days ago",
            "the day before yesterday",
            "a couple days ago",
            "أول أمس",
            "اول امس",
            "قبل يومين",
            "من يومين",
            "أول مبارح",
        ],
    },
    {
        "delta_hours": 72,
        "examples": [
            "3 days ago",
            "a few days ago",
            "several days ago",
            "past few days",
            "قبل كم يوم",
            "قبل ثلاث ايام",
            "من كم يوم",
            "قبل أيام",
        ],
    },
    {
        "delta_hours": 168,
        "examples": [
            "last week",
            "a week ago",
            "7 days ago",
            "in the past week",
            "this week",
            "الأسبوع اللي فات",
            "الاسبوع الماضي",
            "قبل أسبوع",
            "من اسبوع",
            "الاسبوع اللي راح",
            "هالاسبوع",
        ],
    },
    {
        "delta_hours": 336,
        "examples": [
            "2 weeks ago",
            "two weeks ago",
            "a couple weeks ago",
            "last 2 weeks",
            "قبل أسبوعين",
            "من اسبوعين",
            "قبل اسبوعين",
        ],
    },
    {
        "delta_hours": 504,
        "examples": [
            "3 weeks ago",
            "three weeks ago",
            "about three weeks ago",
            "قبل ثلاث أسابيع",
            "قبل ثلاث اسابيع",
            "من ثلاث اسابيع",
        ],
    },
    {
        "delta_hours": 720,
        "examples": [
            "last month",
            "a month ago",
            "in the past month",
            "30 days ago",
            "this month",
            "الشهر اللي فات",
            "الشهر الماضي",
            "قبل شهر",
            "من شهر",
            "هالشهر",
        ],
    },
    {
        "delta_hours": 1440,
        "examples": [
            "2 months ago",
            "two months ago",
            "a couple months ago",
            "last 2 months",
            "قبل شهرين",
            "من شهرين",
        ],
    },
    {
        "delta_hours": 2160,
        "examples": [
            "3 months ago",
            "a few months ago",
            "several months ago",
            "past few months",
            "قبل كم شهر",
            "قبل ثلاث شهور",
            "من كم شهر",
            "قبل أشهر",
            "قبل شهور",
        ],
    },
    {
        "delta_hours": 4320,
        "examples": [
            "6 months ago",
            "half a year ago",
            "last 6 months",
            "قبل ست شهور",
            "قبل نص سنة",
            "من نص سنة",
            "قبل ٦ شهور",
        ],
    },
    {
        "delta_hours": 8760,
        "examples": [
            "a year ago",
            "last year",
            "1 year ago",
            "12 months ago",
            "in the past year",
            "قبل سنة",
            "السنة اللي فاتت",
            "من سنة",
            "العام الماضي",
            "السنة الماضية",
        ],
    },
]

# Pre-computed embeddings — initialized lazily on first call to detect_time_filter
_time_example_embeddings = None
_example_bucket_indices = None
_embedding_init_lock = asyncio.Lock()

_TEMPORAL_INDICATOR_WORDS = {
    "ago",
    "yesterday",
    "today",
    "last",
    "week",
    "month",
    "year",
    "hour",
    "hours",
    "minute",
    "minutes",
    "morning",
    "evening",
    "afternoon",
    "earlier",
    "recently",
    "days",
    "weeks",
    "months",
    "years",
    "past",
    "previous",
    "prior",
    "قبل",
    "أمس",
    "امس",
    "البارحة",
    "البارحه",
    "اليوم",
    "الصبح",
    "مبارح",
    "امبارح",
    "إمبارح",
    "ساعة",
    "ساعه",
    "ساعتين",
    "ساعات",
    "يومين",
    "أيام",
    "ايام",
    "اسبوع",
    "أسبوع",
    "اسبوعين",
    "أسبوعين",
    "شهر",
    "شهرين",
    "شهور",
    "سنة",
    "سنه",
    "الماضي",
    "الماضية",
    "شوي",
    "شويه",
    "الحين",
    "توه",
    "توها",
    "هسه",
    "لسه",
    "لحظات",
    "هالصباح",
    "هاليوم",
    "هالاسبوع",
    "هالشهر",
    "النهارده",
    "انهارده",
    "دلوقتي",
    "فات",
    "فاتت",
}


def _normalize_token(word: str) -> str:
    """Fold Arabic orthographic variation so one spelling matches all forms.

    Egyptian Arabic writes the same temporal word several ways: with or
    without the definite article ("الاسبوع" / "اسبوع"), with any of the alef
    forms, and with ة or ه. Matching raw tokens meant the idiomatic phrasings
    -- which are the common ones -- never reached time detection at all.
    """
    word = word.translate(_ARABIC_FOLDING)
    if word.startswith("ال") and len(word) > 4:
        word = word[2:]
    return word


_ARABIC_FOLDING = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ة": "ه", "ى": "ي"})
_WORD_PATTERN = re.compile(r"[^\W\d_]+", re.UNICODE)


def _query_tokens(query: str) -> set[str]:
    """Tokenise a query into normalised words, ignoring punctuation."""
    return {_normalize_token(word) for word in _WORD_PATTERN.findall(query.lower())}


_NORMALIZED_TEMPORAL_WORDS = {_normalize_token(word) for word in _TEMPORAL_INDICATOR_WORDS}

# Floor on how close the query must sit to *some* time bucket. It guards the
# degenerate case where a temporal word matches nothing in TIME_BUCKETS; it is
# not a recall/non-recall discriminator, because there is no threshold that
# separates those two classes:
#
#   "الاسبوع اللي فات قلت ايه"  (recall)      gap +0.073
#   "I'll finish it today"       (not recall)  gap +0.078
#
# The old gap test against a list of non-temporal examples tried to be that
# discriminator and
# rejected half of all genuine temporal queries to catch nothing the keyword
# gate had not already caught. The gate decides *whether* a query is temporal;
# the buckets below decide *which window*.
TIME_DETECTION_MIN_SIMILARITY = 0.62


def _bucket_delta(bucket: dict, now: datetime) -> timedelta:
    """Resolve a bucket to a concrete lookback duration."""
    delta_hours = bucket["delta_hours"]
    if delta_hours == "dynamic_today":
        return timedelta(hours=now.hour, minutes=now.minute, seconds=now.second)
    return timedelta(hours=delta_hours)


async def _init_embeddings() -> None:
    """Pre-compute time reference embeddings on first use."""
    global _time_example_embeddings, _example_bucket_indices

    if _time_example_embeddings is not None:
        return

    async with _embedding_init_lock:
        if _time_example_embeddings is not None:
            return
        all_examples = []
        bucket_indices_list = []
        for idx, bucket in enumerate(TIME_BUCKETS):
            for example in bucket["examples"]:
                all_examples.append(f"query: {example}")
                bucket_indices_list.append(idx)

        logger.info("Pre-computing %d time reference embeddings...", len(all_examples))
        service = get_embedding_service()
        _time_example_embeddings = await service.encode(all_examples)
        _example_bucket_indices = np.array(bucket_indices_list)
        logger.info("Time reference embeddings ready.")


async def detect_time_filter(query: str) -> tuple[int | None, int | None]:
    """
    Semantically detect time references in the query using the local E5 model.
    Returns (start_epoch, end_epoch) or (None, None) if no time reference found.
    """
    if not _query_tokens(query) & _NORMALIZED_TEMPORAL_WORDS:
        logger.debug("No temporal keywords in query, skipping time detection")
        return None, None

    await _init_embeddings()

    now = datetime.now().astimezone()

    query_emb_tensor = await get_embedding_service().encode(f"query: {query}")
    time_similarities = cosine_similarities(query_emb_tensor, _time_example_embeddings)
    max_time_sim = float(np.max(time_similarities))

    if max_time_sim < TIME_DETECTION_MIN_SIMILARITY:
        logger.debug(
            "Time detection rejected: no bucket within reach (time_sim=%.3f, min=%.2f)",
            max_time_sim,
            TIME_DETECTION_MIN_SIMILARITY,
        )
        return None, None

    best_example_idx = int(np.argmax(time_similarities))
    best_bucket_idx = _example_bucket_indices[best_example_idx]

    # Widen by one bucket so a slightly-off match still contains the target.
    # TIME_BUCKETS is not monotonic in real duration: "dynamic_today" spans
    # midnight-to-now, so before ~03:00 it is *shorter* than the 3h bucket that
    # precedes it. Taking the max keeps the padding from narrowing the window --
    # at 00:30, "قبل ساعتين" used to widen from 3h into a 30-minute window and
    # exclude the very memory being asked about.
    next_bucket_idx = min(best_bucket_idx + 1, len(TIME_BUCKETS) - 1)
    delta = max(
        _bucket_delta(TIME_BUCKETS[best_bucket_idx], now),
        _bucket_delta(TIME_BUCKETS[next_bucket_idx], now),
    )

    start_epoch = int((now - delta).timestamp())
    end_epoch = int(now.timestamp())

    matched_example = _all_examples_cache[best_example_idx].replace("query: ", "")
    logger.debug(
        "Time reference detected: '%s' (sim=%.3f) → matched %sh, using window %.2fh → %d–%d",
        matched_example,
        max_time_sim,
        TIME_BUCKETS[best_bucket_idx]["delta_hours"],
        delta.total_seconds() / 3600,
        start_epoch,
        end_epoch,
    )

    return start_epoch, end_epoch


_all_examples_cache = []
for _idx, bucket in enumerate(TIME_BUCKETS):
    for example in bucket["examples"]:
        _all_examples_cache.append(f"query: {example}")
