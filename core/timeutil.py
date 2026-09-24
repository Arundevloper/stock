"""IST / NSE session helpers. All stored timestamps are epoch seconds (UTC) of the bar START."""
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)
SESSION_START_MIN = 9 * 60 + 15  # 555


def now_ist() -> datetime:
    return datetime.now(IST)


def to_ist(ts: int | float) -> datetime:
    return datetime.fromtimestamp(ts, IST)


def epoch(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=IST)
    return int(dt.timestamp())


def parse_hhmm(s: str) -> time:
    h, m = s.strip().split(":")
    return time(int(h), int(m))


def hhmm_to_min(s: str) -> int:
    t = parse_hhmm(s)
    return t.hour * 60 + t.minute


def minute_of_day(ts: int) -> int:
    d = to_ist(ts)
    return d.hour * 60 + d.minute


def day_str(ts: int | None = None) -> str:
    return (to_ist(ts) if ts is not None else now_ist()).strftime("%Y-%m-%d")


def at_time(d: date | str, t: time | str) -> int:
    """Epoch seconds of `d` at IST clock time `t`."""
    if isinstance(d, str):
        d = date.fromisoformat(d)
    if isinstance(t, str):
        t = parse_hhmm(t)
    return epoch(datetime.combine(d, t, IST))


def in_session(ts: int) -> bool:
    """True if a 1-min bar starting at ts is inside 09:15-15:29."""
    m = minute_of_day(ts)
    return SESSION_START_MIN <= m < 15 * 60 + 30 and to_ist(ts).weekday() < 5


def is_market_open(now: datetime | None = None) -> bool:
    now = now or now_ist()
    return now.weekday() < 5 and MARKET_OPEN <= now.time() < MARKET_CLOSE


def day_bounds(d: date | str) -> tuple[int, int]:
    """[start, end) epoch seconds covering the whole IST calendar day."""
    if isinstance(d, str):
        d = date.fromisoformat(d)
    start = epoch(datetime.combine(d, time(0, 0), IST))
    return start, start + 86400


def days_ago(n: int, ref: int | None = None) -> int:
    ref_dt = to_ist(ref) if ref is not None else now_ist()
    return epoch(ref_dt - timedelta(days=n))
