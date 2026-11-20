from datetime import UTC, datetime, timedelta

from bs4 import BeautifulSoup

from shin_ai.handlers.analytics_report import render_analytics_report


def test_report_groups_platform_ids_and_handles_legacy_metadata():
    now = datetime(2026, 11, 20, 12, tzinfo=UTC)
    recent = int((now - timedelta(days=1)).timestamp())
    earlier = int((now - timedelta(days=40)).timestamp())
    visible_name = '<img src="https://example.invalid">'
    records = [
        {
            "platform": "discord",
            "user_id": "7",
            "username": "renamed",
            "chat_id": "12",
            "chat_type": "GROUP",
            "chat_title": visible_name,
            "guild_id": "4",
            "guild_name": "Team",
            "timestamp": recent,
        },
        {"user_id": "7", "chat_id": "7", "timestamp": recent},
        {
            "platform": "discord",
            "user_id": "7",
            "username": "oldname",
            "chat_id": "12",
            "guild_id": "4",
            "timestamp": earlier,
        },
        {"platform": "discord", "user_id": "7", "chat_id": "13", "guild_id": "4", "timestamp": earlier},
        {"platform": "whatsapp", "chat_id": "123@g.us", "timestamp": "invalid"},
    ]

    with render_analytics_report(records, now) as document:
        soup = BeautifulSoup(document.read(), "html.parser")

    assert [node.get_text() for node in soup.select(".metrics dd")] == ["5", "2", "4", "3"]
    assert [node.get_text() for node in soup.select("#people p strong")] == ["2", "1"]
    assert soup.select_one('.day[title="Nov 19 UTC, 2 interactions"]')
    assert soup.select_one('.heat td[title="Thu 12:00 UTC, 2 interactions"]')
    assert visible_name in soup.get_text()
    assert soup.find("img") is None
    assert "oldname" not in soup.get_text()
    assert [cell.get_text() for cell in soup.select("#chats details:last-child tbody td")] == [
        "Team",
        "4",
        "2",
        "3",
        "1",
        "2026-11-19",
    ]
    assert len(soup.select("#activity tbody tr")) == 4


def test_report_with_no_saved_interactions():
    with render_analytics_report([], datetime(2026, 11, 20, tzinfo=UTC)) as document:
        soup = BeautifulSoup(document.read(), "html.parser")

    assert [node.get_text() for node in soup.select(".metrics dd")] == ["0"] * 4
    assert "No dated interactions in this period." in soup.get_text()
