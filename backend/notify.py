"""One-way Telegram notifications (signals, trade exits, daily P&L, errors)."""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

from core.config import get_settings
from core.timeutil import to_ist

log = logging.getLogger(__name__)
_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="telegram")
_last_sent: dict[str, float] = {}


def _post(text: str) -> None:
    s = get_settings()
    try:
        r = httpx.post(f"https://api.telegram.org/bot{s.telegram_bot_token}/sendMessage",
                       json={"chat_id": s.telegram_chat_id, "text": text, "disable_web_page_preview": True},
                       timeout=10)
        if r.status_code != 200:
            log.warning("Telegram error %s: %s", r.status_code, r.text[:200])
    except httpx.HTTPError as e:
        log.warning("Telegram send failed: %s", e)


def send(text: str, dedupe_key: str | None = None, dedupe_seconds: int = 600) -> None:
    """Fire-and-forget. `dedupe_key` suppresses repeats (e.g. the same error every minute)."""
    if dedupe_key:
        now = time.time()
        if now - _last_sent.get(dedupe_key, 0) < dedupe_seconds:
            return
        _last_sent[dedupe_key] = now
    log.info("notify: %s", text.replace("\n", " | "))
    if get_settings().telegram_enabled:
        _pool.submit(_post, text)


def fmt_price(x: float | None) -> str:
    return "-" if x is None else f"{x:,.2f}"


def signal_text(sig: dict, squareoff: str) -> str:
    icon = "🟢 LONG" if sig["side"] == "long" else "🔴 SHORT"
    prob = f"{sig['prob']:.2f}" if sig.get("prob") is not None else "rule"
    return (f"{icon} {sig['symbol']} | Entry {fmt_price(sig['entry'])} | SL {fmt_price(sig['stop'])} | "
            f"Target {fmt_price(sig['target'])} | Prob {prob} | Qty {sig['qty']} | Valid till {squareoff}")


def trade_closed_text(t: dict) -> str:
    icon = {"target": "✅", "stop": "❌"}.get(t["exit_reason"], "⏹")
    return (f"{icon} {t['symbol']} {t['side'].upper()} closed ({t['exit_reason']}) "
            f"@ {fmt_price(t['exit_price'])} | Net ₹{t['net_pnl']:,.0f} ({t['r_multiple']:+.2f}R) "
            f"at {to_ist(t['exit_time']).strftime('%H:%M')}")


def daily_summary_text(day: str, stats: dict, cumulative: float) -> str:
    return (f"📊 {day} paper P&L\n"
            f"Signals: {stats['signals']} | Trades: {stats['trades']} | Wins: {stats['wins']} | Losses: {stats['losses']}\n"
            f"Net: ₹{stats['net']:,.0f} (costs ₹{stats['costs']:,.0f})\n"
            f"Cumulative: ₹{cumulative:,.0f}")
