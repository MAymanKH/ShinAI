import pytest

from shin_ai.utils.chat_identity import chat_scope_key, normalize_chat_id


class TestNormalizeChatId:
    @pytest.mark.parametrize(
        ("platform", "chat_id", "expected"),
        [
            ("telegram", -1001234567890, "-1001234567890"),
            ("telegram", "  -100123  ", "-100123"),
            ("discord", 987654321, "987654321"),
            # Telegram/Discord ids are opaque: no case folding, no suffix logic.
            ("telegram", "AbC:12", "AbC:12"),
        ],
    )
    def test_non_whatsapp_ids_pass_through(self, platform, chat_id, expected) -> None:
        assert normalize_chat_id(platform, chat_id) == expected

    @pytest.mark.parametrize(
        ("chat_id", "expected"),
        [
            ("201234567890@s.whatsapp.net", "201234567890@s.whatsapp.net"),
            # Device suffixes address the same conversation.
            ("201234567890:12@s.whatsapp.net", "201234567890@s.whatsapp.net"),
            ("201234567890:3", "201234567890"),
            ("201234567890", "201234567890"),
            ("120363000000000000@G.US", "120363000000000000@g.us"),
            ("  201234567890:5@s.whatsapp.net  ", "201234567890@s.whatsapp.net"),
        ],
    )
    def test_whatsapp_ids_are_canonicalised(self, chat_id, expected) -> None:
        assert normalize_chat_id("whatsapp", chat_id) == expected

    def test_device_variants_collapse_to_one_identity(self) -> None:
        """The bug this module prevents: one chat split across two buckets."""
        variants = [
            "201234567890@s.whatsapp.net",
            "201234567890:12@s.whatsapp.net",
            "201234567890:99@S.WHATSAPP.NET",
        ]
        assert len({normalize_chat_id("whatsapp", v) for v in variants}) == 1


class TestChatScopeKey:
    def test_includes_the_scope(self) -> None:
        assert chat_scope_key("telegram:abc123", "telegram", -100) == "telegram:abc123_-100"

    def test_different_scopes_do_not_collide(self) -> None:
        first = chat_scope_key("telegram:aaa", "telegram", 5)
        second = chat_scope_key("telegram:bbb", "telegram", 5)
        assert first != second

    def test_whatsapp_device_variants_share_a_key(self) -> None:
        scope = "whatsapp:fingerprint"
        assert chat_scope_key(scope, "whatsapp", "20111:4@s.whatsapp.net") == chat_scope_key(
            scope, "whatsapp", "20111@s.whatsapp.net"
        )
