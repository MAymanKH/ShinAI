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
            "trigger": "mention",
            "deliveries": "reaction,text",
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
    assert [node.get_text() for node in soup.select("#people p strong")] == ["2", "1", "1", "0"]
    assert soup.select_one('.day[data-tip="Nov 19 UTC, 2 saved interactions"]')
    assert soup.select_one('.heat td[data-tip="Thu 12:00 UTC, 2 saved interactions"]')
    assert visible_name in soup.get_text()
    assert soup.find("img") is None
    assert "oldname" not in soup.get_text()
    assert "Team · 1 saved in 30 days" in soup.select_one("#servers details summary").get_text()
    assert [cell.get_text() for cell in soup.select("#servers tbody tr:first-child td")] == [
        visible_name,
        "12",
        "1",
        "2",
    ]
    classification = soup.select_one("#interaction-types").get_text()
    assert "Mention or name call" in classification
    assert "Reaction" in classification
    assert "Text" in classification
    assert len(soup.select("#activity tbody tr")) == 4


def test_report_compares_complete_periods_and_ranks_recent_activity_first():
    now = datetime(2026, 11, 20, 12, tzinfo=UTC)
    records = [
        {
            "platform": "discord",
            "user_id": "3",
            "chat_id": "30",
            "username": "today",
            "timestamp": now.timestamp(),
        },
        *(
            {
                "platform": "discord",
                "user_id": "1",
                "chat_id": "10",
                "username": "recent",
                "timestamp": (now - timedelta(days=age)).timestamp(),
            }
            for age in (1, 30)
        ),
        *(
            {
                "platform": "discord",
                "user_id": "2",
                "chat_id": "20",
                "username": "historical",
                "timestamp": (now - timedelta(days=age)).timestamp(),
            }
            for age in (31, 60, 61)
        ),
    ]
    with render_analytics_report(records, now) as document:
        soup = BeautifulSoup(document.read(), "html.parser")

    assert [node.get_text() for node in soup.select("#usage p strong")] == ["2", "2", "1"]
    assert [node.get_text() for node in soup.select("#people p strong")] == ["1", "1", "0", "1"]
    assert [node.get_text() for node in soup.select("#platforms tbody td")] == [
        "Discord",
        "2",
        "2",
        "1",
        "1",
        "6",
    ]
    assert "@recent" in soup.select_one("#people details:first-of-type tbody tr").get_text()
    assert "@historical" in soup.select_one("#people details:last-of-type tbody tr").get_text()
    assert [node.get_text() for node in soup.select("#chats .grid details:first-of-type tbody td.num")] == [
        "1",
        "2",
        "2",
    ]


def test_report_with_no_saved_interactions():
    with render_analytics_report([], datetime(2026, 11, 20, tzinfo=UTC)) as document:
        soup = BeautifulSoup(document.read(), "html.parser")

    assert [node.get_text() for node in soup.select(".metrics dd")] == ["0"] * 4
    assert "No dated interactions in this period." in soup.get_text()
