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
            # Recall phrasing with no time reference: the gate is what rejects
            # these, which is why no semantic negative list is needed.
            "what did I say",
            "do you remember",
            "مين قال",
        ],
    )
    def test_non_temporal_query_is_rejected(self, query) -> None:
        assert _gate(query) is False


class TestNoSemanticVeto:
    def test_the_gap_test_is_gone(self) -> None:
        """It rejected genuine temporal recall without catching anything the
        keyword gate had not already caught."""
        assert not hasattr(memory_time, "TIME_DETECTION_MIN_GAP")
        assert not hasattr(memory_time, "_NO_TIME_EXAMPLES")

    def test_similarity_floor_is_kept(self) -> None:
        """Still needed for a temporal word that matches no bucket at all."""
        assert 0.0 < memory_time.TIME_DETECTION_MIN_SIMILARITY < 1.0

