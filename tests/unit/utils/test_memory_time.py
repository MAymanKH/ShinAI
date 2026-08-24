"""The keyword gate decides whether a query gets a time filter at all.

A query that fails it never reaches time detection, so a missed spelling is a
silently dropped time window rather than a degraded result.
"""

import pytest

from shin_ai.utils import memory_time


def _gate(query: str) -> bool:
    return bool(memory_time._query_tokens(query) & memory_time._NORMALIZED_TEMPORAL_WORDS)


class TestArabicNormalization:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("الاسبوع", "اسبوع"),
            ("الشهر", "شهر"),
            ("السنة", "سنه"),
            ("أمس", "امس"),
            ("إمبارح", "امبارح"),
            ("ساعة", "ساعه"),
        ],
    )
    def test_folds_article_alef_and_ta_marbuta(self, raw, expected) -> None:
        assert memory_time._normalize_token(raw) == expected

    def test_short_words_keep_their_article(self) -> None:
        """Stripping 'ال' off a very short word would leave a meaningless stem."""
        assert memory_time._normalize_token("الي") == "الي"

    def test_non_arabic_text_is_untouched(self) -> None:
        assert memory_time._normalize_token("yesterday") == "yesterday"

    def test_tokenizer_drops_punctuation(self) -> None:
        assert memory_time._query_tokens("امبارح؟") == {"امبارح"}
        assert "yesterday" in memory_time._query_tokens("What did we say, yesterday?")


class TestGateAcceptsIdiomaticArabic:
    @pytest.mark.parametrize(
        "query",
        [
            "الاسبوع اللي فات",
            "الشهر اللي فات",
            "السنة اللي فاتت",
            "الساعة اللي فاتت",
            "اليومين اللي فاتوا",
            "النهارده",
            "دلوقتي",
            "امبارح؟",
            # forms that already worked, and must keep working
            "امبارح",
            "أمس",
            "قبل ساعتين",
        ],
    )
    def test_temporal_phrasing_reaches_detection(self, query) -> None:
        assert _gate(query) is True


class TestGateRejectsNonTemporal:
    @pytest.mark.parametrize(
        "query",
        [
            "tell me a joke",
            "explain this code",
            "who are you",
            "قول نكتة",
            "ساعدني في الكود",
            "الكلام ده مهم",
            "الصور الجديدة",
            # Recall phrasing with no time reference: handled here, which is
            # why it does not belong in _NO_TIME_EXAMPLES.
            "what did I say",
            "do you remember",
            "مين قال",
        ],
    )
    def test_non_temporal_query_is_rejected(self, query) -> None:
        assert _gate(query) is False


class TestNoTimeExamples:
    def test_recall_phrasings_are_not_used_as_negatives(self) -> None:
        """They can never reach the semantic stage, and suppressed real hits."""
        for phrase in ("what did I say", "do you remember", "who said", "what happened"):
            assert phrase not in memory_time._NO_TIME_EXAMPLES

    def test_every_negative_example_is_itself_rejected_by_the_gate(self) -> None:
        """An example that passes the gate is doing a job the gate already did."""
        reachable = [q for q in memory_time._NO_TIME_EXAMPLES if _gate(q)]
        assert reachable == []
