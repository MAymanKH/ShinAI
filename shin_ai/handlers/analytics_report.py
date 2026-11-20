"""Build the standalone admin report from saved interaction metadata."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from html import escape
from io import BytesIO

_DAY = timedelta(days=1)
_DAYS = 30
_RANK_LIMIT = 20
_TRIGGERS = {
    "direct_message": "Direct message",
    "mention": "Mention or name call",
    "reply": "Reply to ShinAI",
    "follow_up": "Automatic follow-up",
    "automatic": "Proactive interaction",
}
_DELIVERIES = {"text": "Text", "reaction": "Reaction", "sticker": "Sticker", "moderation": "Moderation"}
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
  color-scheme: dark;
  --paper: #101923; --panel: #172430; --ink: #edf5f8; --muted: #b0c3d0;
  --edge: #364b5a; --teal: #79e4cc; --header: #0c141d;
  --heat: 121, 228, 204;
  font: 100%/1.5 system-ui, sans-serif; color: var(--ink); background: var(--paper);
}
* { box-sizing: border-box; }
body { margin: 0; }
.wrap { width: min(72rem, calc(100% - 2rem)); margin: auto; }
header { background: var(--header); border-bottom: 1px solid var(--edge); padding: 2.5rem 0 1.75rem; }
header p { color: var(--teal); margin: .4rem 0 0; }
nav { display: flex; flex-wrap: wrap; gap: .45rem 1.25rem; margin-top: 1.15rem; }
nav a { color: inherit; text-underline-offset: .25em; }
h1, h2, h3 { letter-spacing: -.035em; line-height: 1.12; margin: 0; }
h1 { font-size: clamp(2.25rem, 6vw, 4rem); }
h2 { font-size: clamp(1.45rem, 4vw, 2rem); }
h3 { font-size: 1.1rem; }
section { margin-block: 1.8rem; scroll-margin-top: 1rem; }
.panel { padding: clamp(.85rem, 3vw, 1.6rem); background: var(--panel); border: 1px solid var(--edge); }
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
.day:hover i, .day:focus-visible i { background: var(--ink); }
.axis { display: flex; justify-content: space-between; gap: .5rem; color: var(--muted); font-size: .8rem; }
.heat { width: 100%; table-layout: fixed; border-collapse: separate; border-spacing: 2px; margin: .8rem 0; }
.heat th { font-size: .7rem; font-weight: 500; color: var(--muted); padding: 0; text-align: left; }
.heat thead th:not(:first-child) { text-align: center; }
.heat th:first-child { width: 2rem; }
.heat td { padding: 0; height: 1.05rem; border: 0; background: rgba(var(--heat), var(--fill)); }
[data-tip]:hover, [data-tip]:focus-visible { outline: 1px solid var(--ink); outline-offset: 1px; }
#chart-tip {
  position: fixed; z-index: 2; pointer-events: none; max-width: calc(100vw - 16px);
  padding: .35rem .65rem; background: var(--ink); color: var(--paper);
  border-radius: .3rem; font-size: .8rem; font-weight: 650; white-space: nowrap;
}
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
  :root {
    color-scheme: light;
    --paper: #fff; --panel: #fff; --ink: #142c38; --muted: #49616d;
    --edge: #ccdae0; --teal: #056b66; --header: #fff; --heat: 5, 107, 102;
  }
  .wrap { width: 100%; } nav { display: none; } section, figure { break-inside: avoid; }
  * { print-color-adjust: exact; }
}
"""

_TOOLTIPS = """
const tip = document.querySelector('#chart-tip');
let anchor;
function show(point) {
  const bounds = point.getBoundingClientRect();
  if (bounds.bottom <= 0 || bounds.top >= window.innerHeight) {
    hide();
    return;
  }
  anchor = point;
  tip.textContent = point.dataset.tip;
  tip.hidden = false;
  const left = bounds.left + (bounds.width - tip.offsetWidth) / 2;
  tip.style.left = Math.max(8, Math.min(left, document.documentElement.clientWidth - tip.offsetWidth - 8)) + 'px';
  const above = bounds.top - tip.offsetHeight - 8;
  tip.style.top = (above >= 8 ? above : bounds.bottom + 8) + 'px';
}
function hide() {
  tip.hidden = true;
  anchor = null;
}
for (const event of ['pointerover', 'pointerdown', 'focusin', 'click']) {
  document.addEventListener(event, ({target, pointerType}) => {
    if (event === 'pointerover' && pointerType === 'touch') return;
    if (event === 'pointerdown' && pointerType !== 'touch') return;
    const point = target.closest('[data-tip]');
    if (point) show(point);
    else if (event === 'pointerdown' || event === 'click') hide();
  });
}
for (const event of ['pointerout', 'focusout']) {
  document.addEventListener(event, ({target, relatedTarget, pointerType}) => {
    if (pointerType === 'touch') return;
    if (anchor && target.closest('[data-tip]') === anchor && !anchor.contains(relatedTarget)) hide();
  });
}
for (const event of ['resize', 'scroll']) {
  window.addEventListener(event, () => {
    if (anchor) show(anchor);
  });
}
document.addEventListener('keydown', event => {
  if (event.key === 'Escape') hide();
  const point = event.target.closest('[data-tip]');
  if (!point) return;
  const chart = point.closest('[data-chart]');
  const stride = Number(chart.dataset.chart);
  const step = {ArrowLeft: -1, ArrowRight: 1, ArrowUp: -stride, ArrowDown: stride}[event.key];
  if (!step) return;
  event.preventDefault();
  const cells = Array.from(chart.querySelectorAll('[data-tip]'));
  const next = cells[cells.indexOf(point) + step];
  if (next) {
    chart.querySelector('[tabindex="0"]').tabIndex = -1;
    next.tabIndex = 0;
    next.focus();
  }
});
"""


@dataclass(slots=True)
class _Usage:
    total: int = 0
    recent: int = 0
    previous: int = 0
    first: datetime | None = None
    last: datetime | None = None
    recent_day: int | None = None
    multiple_days: bool = False
    label: str = ""
    kind: str = ""

    def record(self, when: datetime | None, days_ago: int, label: str = "", kind: str = "") -> None:
        self.total += 1
        if 1 <= days_ago <= _DAYS:
            self.recent += 1
            if self.recent_day is None:
                self.recent_day = days_ago
            elif self.recent_day != days_ago:
                self.multiple_days = True
        self.previous += _DAYS < days_ago <= 2 * _DAYS
        if when:
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
    days = [(start + offset * _DAY, count) for offset, count in enumerate(counts)]
    peak = max(counts)
    bars = "".join(
        f'<div class="day" role="img" {_tooltip(f"{day:%b %d} UTC, {count:,} saved interactions", first=index == 0)}>'
        f'<i style="--fill:{100 * count / (peak or 1):.4f}%"></i></div>'
        for index, (day, count) in enumerate(days)
    )
    caption = (
        f"Peak of {peak:,} on {max(days, key=lambda day: day[1])[0]:%b %d}."
        if peak
        else "No dated interactions in this period."
    )
    return (
        f'<figure><div class="daily" role="group" aria-label="Daily saved interactions" data-chart="0">{bars}</div>'
        f'<div class="axis"><span>{days[0][0]:%b %d}</span>'
        f"<span>{days[-1][0]:%b %d} UTC</span></div>"
        f"<figcaption>{caption}</figcaption></figure>"
        f"<details><summary>Daily counts</summary>{
            _table(('Date (UTC)', 'Interactions'), [(day, f'{count:,}') for day, count in days], numeric=(1,))
        }</details>"
    )


def _tooltip(text: str, *, first: bool) -> str:
    label = escape(text)
    return f'data-tip="{label}" aria-label="{label}" tabindex="{0 if first else -1}"'


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
            description = f"{label} {hour:02d}:00 UTC, {count:,} saved interactions"
            cells.append(
                f'<td style="--fill:{strength:.3f}" {_tooltip(description, first=(day, hour) == (0, 0))}></td>'
            )
        rows.append(f'<tr><th scope="row">{label}</th>{"".join(cells)}</tr>')
    day, hour = max(counts, key=counts.get)
    return (
        f'<figure><table class="heat" aria-label="Interactions by weekday and hour in UTC" data-chart="24">'
        f"<thead><tr><th></th>{columns}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
        f"<figcaption>Busiest slot was {weekdays[day]} at {hour:02d}:00 UTC "
        f"with {peak:,} saved interactions.</figcaption></figure>"
    )


def _date(when):
    return when.strftime("%Y-%m-%d") if when else "Unknown"


def _ranked(items, period):
    return sorted(
        ((key, usage) for key, usage in items.items() if getattr(usage, period)),
        key=lambda item: (-getattr(item[1], period), item[0]),
    )[:_RANK_LIMIT]


def _user_label(usage, user_id):
    return f"@{usage.label.lstrip('@')}" if usage.label else f"Account {user_id}"


def _leaderboard(items, name, row, headers, numeric):
    recent = _ranked(items, "recent")
    all_time = _ranked(items, "total")
    return (
        f"{_bars([(name(key, usage), usage.recent) for key, usage in recent[:5]])}"
        f"<details><summary>Last 30 days · Top {_RANK_LIMIT}</summary>"
        f"{_table(headers, [row(key, usage) for key, usage in recent], numeric=numeric)}</details>"
        f"<details><summary>All time · Top {_RANK_LIMIT}</summary>"
        f"{_table(headers, [row(key, usage) for key, usage in all_time], numeric=numeric)}</details>"
    )


def render_analytics_report(newest_first: list[dict], as_of: datetime) -> BytesIO:
    """Return a named upload with an offline report as of the snapshot time."""
    today = as_of.date()
    start = today - _DAYS * _DAY
    platforms = defaultdict(_Usage)
    users = defaultdict(_Usage)
    chats = defaultdict(_Usage)
    servers = defaultdict(_Usage)
    server_channels = defaultdict(lambda: defaultdict(_Usage))
    chat_users = defaultdict(set)
    chat_active_users = defaultdict(set)
    server_active_users = defaultdict(set)
    daily = [0] * _DAYS
    heat = Counter()
    kinds = Counter()
    triggers = Counter()
    deliveries = Counter()
    recent_rows = []
    today_count = valid_times = discord_without_server = unknown_users = unknown_chats = legacy_platform = 0

    for meta in newest_first:
        platform_key = _value(meta, "platform").lower()
        legacy_platform += not platform_key
        platform = "WhatsApp" if platform_key == "whatsapp" else (platform_key or "telegram").title()
        user_id = _value(meta, "user_id")
        chat_id = _value(meta, "chat_id")
        guild_id = _value(meta, "guild_id") if platform == "Discord" else ""
        chat_type = _chat_type(meta, platform, chat_id, user_id, guild_id)
        kinds[chat_type] += 1
        trigger = _TRIGGERS.get(_value(meta, "trigger"), "Not recorded")
        delivery_types = {
            _DELIVERIES[kind] for kind in _value(meta, "deliveries").split(",") if kind in _DELIVERIES
        }
        triggers[trigger] += 1
        deliveries.update(delivery_types or {"Not recorded"})

        when = _time(meta, as_of)
        days_ago = (today - when.date()).days if when else -1
        in_period = 1 <= days_ago <= _DAYS
        platforms[platform].record(when, days_ago)
        if user_id:
            users[platform, user_id].record(when, days_ago, _value(meta, "username"))
        else:
            unknown_users += 1
        if chat_id:
            chats[platform, chat_id].record(
                when, days_ago, _value(meta, "chat_title"), "" if chat_type == "Unclassified" else chat_type
            )
            if in_period and user_id:
                chat_active_users[platform, chat_id].add(user_id)
            if user_id:
                chat_users[platform, chat_id].add(user_id)
        else:
            unknown_chats += 1
        if guild_id:
            servers[guild_id].record(when, days_ago, _value(meta, "guild_name"))
            if chat_id:
                server_channels[guild_id][chat_id].record(when, days_ago)
            if in_period and user_id:
                server_active_users[guild_id].add(user_id)
        elif platform == "Discord" and chat_type != "Direct message":
            discord_without_server += 1
        if when:
            valid_times += 1
            today_count += days_ago == 0
            if in_period:
                daily[_DAYS - days_ago] += 1
                heat[when.weekday(), when.hour] += 1
            if len(recent_rows) < 25:
                recent_rows.append(
                    (when, platform, user_id, chat_id, trigger, ", ".join(sorted(delivery_types)))
                )

    def chat_label(platform, chat_id):
        chat = chats[platform, chat_id]
        if chat.label and chat.label != "Private Chat":
            return chat.label
        participants = chat_users[platform, chat_id]
        if chat.kind == "Direct message" and len(participants) == 1:
            user_id = next(iter(participants))
            return f"Direct chat with {_user_label(users[platform, user_id], user_id)}"
        return f"Chat {chat_id}"

    def account_label(key, usage):
        return f"{_user_label(usage, key[1])} ({key[0]})"

    def account_row(key, usage):
        platform, uid = key
        return (
            _user_label(usage, uid),
            platform,
            uid,
            f"{usage.recent:,}",
            f"{usage.total:,}",
            _date(usage.first),
            _date(usage.last),
        )

    def chat_row(key, usage):
        platform, cid = key
        return (
            chat_label(platform, cid),
            platform,
            usage.kind or "Unclassified",
            cid,
            f"{len(chat_active_users[key]):,}",
            f"{usage.recent:,}",
            f"{usage.total:,}",
            _date(usage.last),
        )

    total = len(newest_first)
    active_by_platform = Counter()
    previous_by_platform = Counter()
    first_seen = multiple_days = 0
    for (platform, _), user in users.items():
        if user.recent:
            active_by_platform[platform] += 1
            first_seen += user.first.date() >= start
            multiple_days += user.multiple_days
        previous_by_platform[platform] += user.previous > 0
    active_users = active_by_platform.total()

    totals = "".join(
        (
            _metric("Saved interactions", total),
            _metric("Platform accounts", len(users)),
            _metric("Chats & channels", len(chats)),
            _metric("Platforms", len(platforms)),
        )
    )
    ranked_platforms = sorted(platforms.items(), key=lambda item: (-item[1].recent, -item[1].total, item[0]))
    platform_rows = [
        (
            name,
            f"{usage.recent:,}",
            f"{usage.previous:,}",
            f"{active_by_platform[name]:,}",
            f"{previous_by_platform[name]:,}",
            f"{usage.total:,}",
        )
        for name, usage in ranked_platforms
    ]
    account_board = _leaderboard(
        users,
        account_label,
        account_row,
        ("Account", "Platform", "ID", "30 days", "All time", "First seen", "Last seen"),
        (3, 4),
    )
    chat_board = _leaderboard(
        chats,
        lambda key, _: f"{chat_label(*key)} ({key[0]})",
        chat_row,
        ("Chat", "Platform", "Type", "ID", "Active accounts", "30 days", "All time", "Last seen"),
        (4, 5, 6),
    )
    server_panels = []
    # Show the most active servers in the period, then use remaining space for historical servers.
    recent_servers = _ranked(servers, "recent")
    recent_server_ids = {sid for sid, _ in recent_servers}
    other_servers = {sid: usage for sid, usage in servers.items() if sid not in recent_server_ids}
    for sid, usage in (recent_servers + _ranked(other_servers, "total"))[:_RANK_LIMIT]:
        channel_rows = [
            (chat_label("Discord", cid), cid, f"{channel.recent:,}", f"{channel.total:,}")
            for cid, channel in sorted(
                server_channels[sid].items(), key=lambda item: (-item[1].recent, -item[1].total, item[0])
            )[:10]
        ]
        server_name = escape(usage.label or f"Server {sid}")
        server_panels.append(
            f"<details><summary>{server_name} · {usage.recent:,} saved in 30 days</summary>"
            f'<p class="note">ID {escape(sid)} · {len(server_channels[sid]):,} recorded channels · '
            f"{len(server_active_users[sid]):,} active accounts · {usage.total:,} all-time interactions.</p>"
            f"{_table(('Top channels', 'ID', '30 days', 'All time'), channel_rows, numeric=(2, 3))}</details>"
        )
    activity_rows = [
        (
            when.strftime("%Y-%m-%d %H:%M"),
            platform,
            _user_label(users[platform, uid], uid) if uid else "Unknown",
            chat_label(platform, cid) if cid else "Unknown",
            trigger,
            delivery or "Not recorded",
        )
        for when, platform, uid, cid, trigger, delivery in recent_rows
    ]
    earliest = min((usage.first for usage in platforms.values() if usage.first), default=None)
    period = f"{start:%d %b %Y} to {today - _DAY:%d %b %Y} UTC"
    previous_period = f"{start - _DAYS * _DAY:%d %b %Y} to {start - _DAY:%d %b %Y} UTC"
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
<a href="#chats">Chats &amp; servers</a><a href="#interaction-types">Interaction types</a><a href="#activity">Recent activity</a>
</nav>
</div></header>
<main class="wrap">
<section aria-label="All-time totals"><dl class="metrics">{totals}</dl>
<p class="note">One saved interaction is one recorded exchange with a delivered reply or action.
Earliest dated record: {_date(earliest)}.</p></section>
<section id="usage" class="panel">
<h2>Interaction volume</h2>
<p><strong class="num">{sum(daily):,}</strong> interactions in the last 30 complete days, compared with
<strong class="num">{
        sum(usage.previous for usage in platforms.values()):,}</strong> in the previous 30 days.</p>
<p class="note">{period}. Today so far: <strong class="num">{today_count:,}</strong> interactions.</p>
{_daily_chart(start, daily)}
</section>
<div class="grid">
<section id="platforms" class="panel">
<h2>Platforms</h2><p class="note">Share of saved interactions in the last 30 days</p>
{_bars([(name, usage.recent) for name, usage in ranked_platforms if usage.recent], denominator=sum(daily))}
<details><summary>Compare with the previous 30 days</summary>
<p class="note">Previous period: {previous_period}.</p>
{
        _table(
            (
                "Platform",
                "Interactions",
                "Prior interactions",
                "Active accounts",
                "Prior active accounts",
                "All time",
            ),
            platform_rows,
            numeric=(1, 2, 3, 4, 5),
        )
    }
</details>
</section>
<section class="panel"><h2>When chats are active</h2>
<p class="note">Last 30 days (UTC). Brighter cells mean more interactions.</p>
{_heatmap(heat)}
</section>
</div>
<section id="people" class="panel">
<h2>Accounts</h2>
<p><strong class="num">{active_users:,}</strong> active in the last 30 days.
<strong class="num">{first_seen:,}</strong> first seen,
<strong class="num">{active_users - first_seen:,}</strong> returning, and
<strong class="num">{multiple_days:,}</strong> active on multiple days.</p>
<h3>Most active accounts in 30 days</h3>
{account_board}
</section>
<section id="chats" class="panel">
<h2>Chats &amp; servers</h2>
<div class="grid">
<div><p class="note">All-time share by chat type</p>
{_bars(kinds.most_common(), denominator=total)}
</div>
<div><p class="note">Most active chats and channels in 30 days</p>
{chat_board}
</div></div>
<details id="servers"><summary>Discord servers ({len(servers):,} recorded)</summary>
<p class="note">Up to {_RANK_LIMIT} servers. {discord_without_server:,} Discord interactions outside known
direct messages have no saved server ID.</p>
{"".join(server_panels) or '<p class="note">No Discord server activity recorded.</p>'}
</details>
</section>
<section id="interaction-types" class="panel">
<h2>Interaction types</h2>
<p class="note">All-time shares of saved interactions. Older records may lack trigger and delivery details.
An interaction can include multiple delivery types.</p>
<div class="grid">
<div><h3>How it started</h3>{_bars(triggers.most_common(), denominator=total)}</div>
<div><h3>What ShinAI delivered</h3>{_bars(deliveries.most_common(), denominator=total)}</div>
</div>
</section>
<section id="activity" class="panel">
<h2>Recent activity</h2>
<details><summary>Last {len(activity_rows):,} dated interactions</summary>
{_table(("Time (UTC)", "Platform", "Account", "Chat", "Trigger", "Delivered"), activity_rows)}
</details>
</section>
</main>
<footer class="wrap note">
<details><summary>Data coverage</summary>
<p>The same person can have an account on more than one platform. First seen means the first dated
interaction in the saved history.</p>
<p>{legacy_platform:,} records without a platform use the legacy Telegram default.
{total - valid_times:,} interactions have unusable timestamps and only count in all-time totals.
{unknown_users:,} lack account IDs, {unknown_chats:,} lack chat IDs, and {
        kinds["Unclassified"]:,} lack a known chat type.</p>
</details>
</footer>
<div id="chart-tip" aria-hidden="true" hidden></div>
<script>{_TOOLTIPS}</script>
</body>
</html>
"""
    upload = BytesIO(html.encode("utf-8"))
    upload.name = f"shinai-analytics-{as_of:%Y-%m-%d}.html"
    return upload
