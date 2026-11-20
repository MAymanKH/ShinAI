"""Build the standalone admin report from saved interaction metadata."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from html import escape
from io import BytesIO

_DAY = timedelta(days=1)
_RANK_LIMIT = 20
_CHAT_TYPES = {
    "PRIVATE": "Direct message",
    "BOT": "Direct message",
    "GROUP": "Group",
    "SUPERGROUP": "Group",
    "CHANNEL": "Channel",
    "FORUM": "Forum",
    "MONOFORUM": "Forum",
}

_STYLE = """
:root {
  --paper: #f2f6f7; --white: #fff; --ink: #142c38; --muted: #49616d;
  --edge: #ccdae0; --teal: #056b66; --mint: #a8efdb;
  font: 100%/1.5 system-ui, sans-serif; color: var(--ink); background: var(--paper);
}
* { box-sizing: border-box; }
body { margin: 0; }
.wrap { width: min(72rem, calc(100% - 2rem)); margin: auto; }
header { background: var(--ink); color: var(--white); padding: 2.5rem 0 1.75rem; }
header p { color: var(--mint); margin: .4rem 0 0; }
nav { display: flex; flex-wrap: wrap; gap: .45rem 1.25rem; margin-top: 1.15rem; }
nav a { color: inherit; text-underline-offset: .25em; }
h1, h2, h3 { letter-spacing: -.035em; line-height: 1.12; margin: 0; }
h1 { font-size: clamp(2.25rem, 6vw, 4rem); }
h2 { font-size: clamp(1.45rem, 4vw, 2rem); }
h3 { font-size: 1.1rem; }
section { margin-block: 1.8rem; scroll-margin-top: 1rem; }
.panel { padding: clamp(.85rem, 3vw, 1.6rem); background: var(--white); border: 1px solid var(--edge); }
p { margin: .6rem 0; }
.note, figcaption { color: var(--muted); font-size: .875rem; }
.metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 9rem), 1fr)); margin: 0; }
.metrics div { padding: 1rem; border-inline-start: 3px solid var(--teal); }
.metrics dt { color: var(--muted); font-size: .875rem; }
.metrics dd { font-size: clamp(1.6rem, 4vw, 2.4rem); font-weight: 700; margin: 0; }
.num, dd { font-variant-numeric: tabular-nums; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 24rem), 1fr)); gap: 1rem; }
.grid > * { min-width: 0; }
.grid > section { margin: 0; }
figure { margin: 1rem 0 0; }
.daily { display: grid; grid-template-columns: repeat(30, 1fr); gap: clamp(2px, .5vw, 6px);
  height: 9rem; border-bottom: 1px solid var(--edge);
  background: repeating-linear-gradient(to top, transparent 0 calc(25% - 1px), var(--edge) calc(25% - 1px) 25%);
}
.day { display: flex; align-items: end; }
.day i { display: block; width: 100%; height: var(--fill); background: var(--teal); border-radius: 2px 2px 0 0; }
.day:last-child i { background: #3aa88d; }
.axis { display: flex; justify-content: space-between; gap: .5rem; color: var(--muted); font-size: .8rem; }
.heat { width: 100%; table-layout: fixed; border-collapse: separate; border-spacing: 2px; margin: .8rem 0; }
.heat th { font-size: .7rem; font-weight: 500; color: var(--muted); padding: 0; text-align: left; }
.heat thead th:not(:first-child) { text-align: center; }
.heat th:first-child { width: 2rem; }
.heat td { padding: 0; height: 1.05rem; border: 0; background: rgba(5, 107, 102, var(--fill)); }
.bars { list-style: none; padding: 0; margin: .8rem 0 0; }
.bars li { margin-block: .65rem; }
.bar-label { display: flex; justify-content: space-between; gap: .75rem; overflow-wrap: anywhere; }
.bar-label .num { flex-shrink: 0; }
.track { display: block; height: .38rem; background: var(--paper); margin-top: .25rem; }
.track i { display: block; height: 100%; width: var(--fill); background: var(--teal); }
details { margin-top: 1rem; border-top: 1px solid var(--edge); padding-top: .7rem; }
summary { cursor: pointer; color: var(--teal); font-weight: 650; }
.scroll { overflow-x: auto; }
.data { width: 100%; border-collapse: collapse; font-size: .875rem; margin-top: .65rem; }
.data td, .data th { padding: .55rem .45rem; border-bottom: 1px solid var(--edge); text-align: start; }
.data thead { background: var(--paper); }
.data td.num, .data th.num { text-align: end; white-space: nowrap; }
.data td { overflow-wrap: anywhere; }
footer { padding-block: 0 2rem; }
@media (max-width: 36rem) {
  .metrics { row-gap: .65rem; }
  .daily { height: 7rem; }
}
@media print {
  :root { background: white; }
  .wrap { width: 100%; } nav { display: none; } section, figure { break-inside: avoid; }
  * { print-color-adjust: exact; }
}
"""


@dataclass(slots=True)
class _Usage:
    total: int = 0
    recent: int = 0
    first: datetime | None = None
    last: datetime | None = None
    label: str = ""
    kind: str = ""

    def record(self, when: datetime | None, start: datetime, label: str = "", kind: str = "") -> None:
        self.total += 1
        if when:
            self.recent += when >= start
            if self.first is None or when < self.first:
                self.first = when
        if self.last is None or (when and when >= self.last):
            self.label = label or self.label
            self.kind = kind or self.kind
            self.last = when
        else:
            self.label = self.label or label
            self.kind = self.kind or kind


def _value(meta: dict, key: str) -> str:
    value = meta.get(key)
    text = str(value).strip() if value is not None else ""
    return "" if text.casefold() == "unknown" else text


def _time(meta: dict, as_of: datetime) -> datetime | None:
    try:
        stamp = int(meta.get("timestamp", 0))
    except (TypeError, ValueError, OverflowError):
        return None
    return datetime.fromtimestamp(stamp, UTC) if 0 < stamp <= as_of.timestamp() else None


def _chat_type(meta, platform, chat_id, user_id, guild_id):
    kind = _CHAT_TYPES.get(_value(meta, "chat_type").upper())
    if kind:
        return kind
    if guild_id or (platform == "WhatsApp" and chat_id.endswith("@g.us")):
        return "Group"
    if platform == "Telegram" and chat_id and chat_id == user_id:
        return "Direct message"
    return "Unclassified"


def _table(headers, rows, *, numeric=()):
    if not rows:
        return '<p class="note">No records to show.</p>'

    def cells(tag, values):
        return "".join(
            f"<{tag}{' class=num' if index in numeric else ''}>{escape(str(value))}</{tag}>"
            for index, value in enumerate(values)
        )

    body = "".join(f"<tr>{cells('td', row)}</tr>" for row in rows)
    return (
        f'<div class="scroll" tabindex="0"><table class="data"><thead><tr>{cells("th", headers)}</tr>'
        f"</thead><tbody>{body}</tbody></table></div>"
    )


def _bars(rows, *, denominator=None):
    if not rows:
        return '<p class="note">No records to show.</p>'
    scale = denominator or max(count for _, count in rows)
    return (
        '<ul class="bars">'
        + "".join(
            f'<li><div class="bar-label"><span>{escape(label)}</span><span class="num">'
            f"{count:,}{f' ({count / scale:.1%})' if denominator else ''}</span></div>"
            f'<span class="track" aria-hidden="true"><i style="--fill:{100 * count / scale:.4f}%"></i></span></li>'
            for label, count in rows
        )
        + "</ul>"
    )


def _metric(label, count):
    return f'<div><dt>{escape(label)}</dt><dd class="num">{count:,}</dd></div>'


def _daily_chart(start, counts):
    days = [(start.date() + offset * _DAY, count) for offset, count in enumerate(counts)]
    peak = max(counts)
    bars = "".join(
        f'<div class="day" title="{day:%b %d} UTC, {count:,} interactions">'
        f'<i style="--fill:{100 * count / (peak or 1):.4f}%"></i></div>'
        for day, count in days
    )
    caption = (
        f"Peak of {peak:,} on {max(days, key=lambda day: day[1])[0]:%b %d}. Today is in progress."
        if peak
        else "No dated interactions in this period."
    )
    return (
        f'<figure><div class="daily" aria-hidden="true">{bars}</div>'
        f'<div class="axis"><span>{days[0][0]:%b %d}</span>'
        f"<span>{days[-1][0]:%b %d} UTC</span></div>"
        f"<figcaption>{caption}</figcaption></figure>"
        f"<details><summary>Daily counts</summary>{
            _table(('Date (UTC)', 'Interactions'), [(day, f'{count:,}') for day, count in days], numeric=(1,))
        }</details>"
    )


def _heatmap(counts):
    weekdays = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
    peak = max(counts.values(), default=0)
    if not peak:
        return '<p class="note">No dated interactions in this period.</p>'
    columns = "".join(
        f'<th scope="col" aria-label="{hour:02d}:00">{f"{hour:02d}" if hour % 6 == 0 else ""}</th>'
        for hour in range(24)
    )
    rows = []
    for day, label in enumerate(weekdays):
        cells = []
        for hour in range(24):
            count = counts[day, hour]
            strength = 0.05 + 0.95 * count / peak
            description = f"{label} {hour:02d}:00 UTC, {count:,} interactions"
            cells.append(
                f'<td style="--fill:{strength:.3f}" title="{description}" aria-label="{description}"></td>'
            )
        rows.append(f'<tr><th scope="row">{label}</th>{"".join(cells)}</tr>')
    day, hour = max(counts, key=counts.get)
    return (
        f'<figure><table class="heat" aria-label="Interactions by weekday and hour in UTC">'
        f"<thead><tr><th></th>{columns}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
        f"<figcaption>Busiest slot was {weekdays[day]} at {hour:02d}:00 UTC "
        f"with {peak:,} interactions across the last 30 dates.</figcaption></figure>"
    )


def _date(when):
    return when.strftime("%Y-%m-%d") if when else "Unknown"


def _ranked(items):
    return sorted(items.items(), key=lambda item: (-item[1].total, item[0]))[:_RANK_LIMIT]


def _user_label(usage, user_id):
    return f"@{usage.label.lstrip('@')}" if usage.label else f"Account {user_id}"


def render_analytics_report(newest_first: list[dict], as_of: datetime) -> BytesIO:
    """Return a named upload with an offline report as of the snapshot time."""
    start = as_of.replace(hour=0, minute=0, second=0, microsecond=0) - 29 * _DAY
    platforms = defaultdict(_Usage)
    users = defaultdict(_Usage)
    chats = defaultdict(_Usage)
    servers = defaultdict(_Usage)
    platform_users = defaultdict(set)
    platform_chats = defaultdict(set)
    chat_users = defaultdict(set)
    server_channels = defaultdict(set)
    daily = [0] * 30
    heat = Counter()
    kinds = Counter()
    recent_rows = []
    week = previous_week = valid_times = discord_without_server = unknown_users = unknown_chats = 0

    for meta in newest_first:
        platform = _value(meta, "platform").lower() or "telegram"
        platform = "WhatsApp" if platform == "whatsapp" else platform.title()
        user_id = _value(meta, "user_id")
        chat_id = _value(meta, "chat_id")
        username = _value(meta, "username")
        title = _value(meta, "chat_title")
        guild_id = _value(meta, "guild_id") if platform == "Discord" else ""
        chat_type = _chat_type(meta, platform, chat_id, user_id, guild_id)
        kinds[chat_type] += 1
        when = _time(meta, as_of)
        platforms[platform].record(when, start)
        if user_id:
            users[platform, user_id].record(when, start, username)
            platform_users[platform].add(user_id)
        else:
            unknown_users += 1
        if chat_id:
            chats[platform, chat_id].record(
                when, start, title, "" if chat_type == "Unclassified" else chat_type
            )
            platform_chats[platform].add(chat_id)
            if user_id:
                chat_users[platform, chat_id].add(user_id)
        else:
            unknown_chats += 1
        if guild_id:
            servers[guild_id].record(when, start, _value(meta, "guild_name"))
            if chat_id:
                server_channels[guild_id].add(chat_id)
        elif platform == "Discord" and chat_type != "Direct message":
            discord_without_server += 1
        if when:
            valid_times += 1
            age = as_of - when
            week += age < 7 * _DAY
            previous_week += 7 * _DAY <= age < 14 * _DAY
            if when >= start:
                daily[(when.date() - start.date()).days] += 1
                heat[when.weekday(), when.hour] += 1
            if len(recent_rows) < 25:
                recent_rows.append((when, platform, user_id, chat_id))

    def chat_label(platform, chat_id):
        chat = chats[platform, chat_id]
        if chat.label and chat.label != "Private Chat":
            return chat.label
        participants = chat_users[platform, chat_id]
        if chat.kind == "Direct message" and len(participants) == 1:
            user_id = next(iter(participants))
            return f"Direct chat with {_user_label(users[platform, user_id], user_id)}"
        return f"Chat {chat_id}"

    total = len(newest_first)
    active_users = sum(user.recent > 0 for user in users.values())
    returning_users = sum(bool(user.recent and user.first and user.first < start) for user in users.values())
    totals = "".join(
        (
            _metric("Saved interactions", total),
            _metric("Accounts", len(users)),
            _metric("Chats & channels", len(chats)),
            _metric("Platforms", len(platforms)),
        )
    )
    ranked_platforms = sorted(platforms.items(), key=lambda item: (-item[1].total, item[0]))
    platform_rows = [
        (
            name,
            f"{usage.total:,}",
            f"{usage.recent:,}",
            f"{len(platform_users[name]):,}",
            f"{len(platform_chats[name]):,}",
        )
        for name, usage in ranked_platforms
    ]
    top_users = _ranked(users)
    top_chats = _ranked(chats)
    user_rows = [
        (_user_label(u, uid), platform, uid, f"{u.total:,}", f"{u.recent:,}", _date(u.first), _date(u.last))
        for (platform, uid), u in top_users
    ]
    chat_rows = [
        (
            chat_label(platform, cid),
            platform,
            c.kind or "Unclassified",
            cid,
            f"{len(chat_users[platform, cid]):,}",
            f"{c.total:,}",
            f"{c.recent:,}",
            _date(c.last),
        )
        for (platform, cid), c in top_chats
    ]
    top_servers = _ranked(servers)
    server_rows = [
        (
            u.label or f"Server {sid}",
            sid,
            f"{len(server_channels[sid]):,}",
            f"{u.total:,}",
            f"{u.recent:,}",
            _date(u.last),
        )
        for sid, u in top_servers
    ]
    activity_rows = [
        (
            when.strftime("%Y-%m-%d %H:%M"),
            platform,
            _user_label(users[platform, uid], uid) if uid else "Unknown",
            chat_label(platform, cid) if cid else "Unknown",
        )
        for when, platform, uid, cid in recent_rows
    ]
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ShinAI usage report</title>
<style>{_STYLE}</style>
</head>
<body>
<header><div class="wrap">
<h1>ShinAI usage report</h1>
<p>Snapshot as of {as_of:%d %b %Y, %H:%M} UTC</p>
<nav aria-label="Report sections">
<a href="#usage">Usage</a><a href="#platforms">Platforms</a><a href="#people">Accounts</a>
<a href="#chats">Chats &amp; servers</a><a href="#activity">Recent activity</a>
</nav>
</div></header>
<main class="wrap">
<section aria-label="All-time totals"><dl class="metrics">{totals}</dl></section>
<section id="usage" class="panel">
<h2>Interaction volume</h2>
<p><strong class="num">{sum(daily):,}</strong> across the last 30 UTC dates.
The past 7 days had <strong class="num">{week:,}</strong> interactions, compared with
<strong class="num">{previous_week:,}</strong> in the preceding 7 days.</p>
{_daily_chart(start, daily)}
</section>
<div class="grid">
<section id="platforms" class="panel">
<h2>Platforms</h2><p class="note">Share of all saved interactions</p>
{_bars([(name, usage.total) for name, usage in ranked_platforms], denominator=total)}
<details><summary>Platform breakdown</summary>
{_table(("Platform", "All time", "30 dates", "Accounts", "Chats"), platform_rows, numeric=(1, 2, 3, 4))}
</details>
</section>
<section class="panel"><h2>When chats are active</h2>
<p class="note">Last 30 UTC dates. Darker cells mean more interactions.</p>
{_heatmap(heat)}
</section>
</div>
<section id="people" class="panel">
<h2>Accounts</h2>
<p><strong class="num">{active_users:,}</strong> active across the last 30 UTC dates.
<strong class="num">{returning_users:,}</strong> of them also interacted before that period.</p>
<h3>Most active accounts of all time</h3>
{_bars([(f"{_user_label(u, uid)} ({platform})", u.total) for (platform, uid), u in top_users[:5]])}
<details><summary>Account details (up to {_RANK_LIMIT})</summary>
{
        _table(
            ("Account", "Platform", "ID", "All time", "30 dates", "First seen", "Last seen"),
            user_rows,
            numeric=(3, 4),
        )
    }
</details>
</section>
<section id="chats" class="panel">
<h2>Chats &amp; servers</h2>
<div class="grid">
<div><p class="note">Share of all saved interactions by chat type</p>
{_bars(kinds.most_common(), denominator=total)}
</div>
<div><p class="note">Most active chats and channels of all time</p>
{_bars([(f"{chat_label(platform, cid)} ({platform})", c.total) for (platform, cid), c in top_chats[:5]])}
</div></div>
<details><summary>Chat and channel details (up to {_RANK_LIMIT})</summary>
{
        _table(
            ("Chat", "Platform", "Type", "ID", "Accounts", "All time", "30 dates", "Last seen"),
            chat_rows,
            numeric=(4, 5, 6),
        )
    }
</details>
<details><summary>Discord servers ({len(servers):,} recorded)</summary>
<p class="note">Channel totals count channels with saved interactions. {discord_without_server:,}
Discord interactions outside known direct messages have no saved server ID.</p>
{_bars([(u.label or f"Server {sid}", u.total) for sid, u in top_servers[:5]]) if top_servers else ""}
{_table(("Server", "ID", "Channels", "All time", "30 dates", "Last seen"), server_rows, numeric=(2, 3, 4))}
</details>
</section>
<section id="activity" class="panel">
<h2>Recent activity</h2>
<details open><summary>Last {len(activity_rows):,} dated interactions</summary>
{_table(("Time (UTC)", "Platform", "Account", "Chat"), activity_rows)}
</details>
</section>
</main>
<footer class="wrap note">
<p>Totals come from ShinAI’s saved interaction history. Each account and chat is counted by its platform ID.
The same person can have an account on more than one platform. Records without a platform use the
legacy Telegram default.</p>
<p>Coverage gaps include {total - valid_times:,} unusable timestamps, {unknown_users:,} missing account IDs
and {unknown_chats:,} missing chat IDs. Those records still count toward interaction totals.</p>
</footer>
</body>
</html>
"""
    upload = BytesIO(html.encode("utf-8"))
    upload.name = f"shinai-analytics-{as_of:%Y-%m-%d}.html"
    return upload
