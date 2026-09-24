"""Scheduled jobs (IST, Mon-Fri):
  08:50 token check + instrument refresh     09:31 watchlist scan
  every minute 09:15-15:30 data-freshness    15:16 square-off safety net
  15:35 daily P&L to Telegram                16:00 backfill today's official minute candles
  RETRAIN_TIME nightly retrain
"""
from __future__ import annotations

import logging
import subprocess
import sys
import time

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import func, select

from backend import notify, paper, watchlist
from backend.engine import engine
from backend.training import runner
from backend.ws import hub
from core import db
from core.config import ROOT, get_settings
from core.db import PaperTrade
from core.instruments import ensure_instruments
from core.kite import kite_configured, token_status
from core.timeutil import IST, at_time, day_str, is_market_open, parse_hhmm

log = logging.getLogger(__name__)
WEEKDAYS = "mon-fri"


def morning_check() -> None:
    if not kite_configured():
        return
    if not token_status()["valid"]:
        notify.send(f"🔑 Kite login needed for today: {get_settings().public_url}/auth/login",
                    dedupe_key="login", dedupe_seconds=3600)
        return
    ensure_instruments()


def watchlist_job() -> None:
    entries = watchlist.scan()
    engine.invalidate_watchlist()
    hub.broadcast_threadsafe([{"type": "watchlist", "data": entries}])
    if entries:
        notify.send("📋 Watchlist: " + ", ".join(e["symbol"].split(":")[1] for e in entries[:25]))


def health_job() -> None:
    s = get_settings()
    if not is_market_open():
        return
    now = time.time()
    if now < at_time(day_str(), "09:15") + s.stale_data_minutes * 60 + 60:
        return
    last = engine.last_ingest_at
    if last is None or now - last > s.stale_data_minutes * 60:
        ago = "never today" if last is None else f"{(now - last) / 60:.0f} min ago"
        tok = "" if token_status()["valid"] else " (Kite token invalid - log in)"
        notify.send(f"⚠️ No candles from reader - last received {ago}{tok}", dedupe_key="stale", dedupe_seconds=900)
        return
    if s.healthcheck_url:
        try:
            httpx.get(s.healthcheck_url, timeout=5)
        except httpx.HTTPError:
            pass


def squareoff_job() -> None:
    events = engine.square_off("squareoff")
    if events:
        log.info("Square-off closed/cancelled %d trades", len(events))
        hub.broadcast_threadsafe(events)


def eod_summary_job() -> None:
    day = day_str()
    with db.SessionLocal() as s:
        stats = paper.day_stats(s, day)
        cumulative = s.scalar(select(func.coalesce(func.sum(PaperTrade.net_pnl), 0))
                              .where(PaperTrade.status == "closed")) or 0
    notify.send(notify.daily_summary_text(day, stats, float(cumulative)))


def backfill_job() -> None:
    if not token_status()["valid"]:
        return
    r = subprocess.run([sys.executable, "-m", "backfill.historical", "--days", "1"], cwd=str(ROOT),
                       capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        notify.send(f"⚠️ Post-close backfill failed: {r.stderr[-300:]}", dedupe_key="backfill")


async def retrain_job() -> None:
    await runner.run()


def start_scheduler() -> AsyncIOScheduler:
    s = get_settings()
    sch = AsyncIOScheduler(timezone=IST)

    def cron(**kw):
        return CronTrigger(day_of_week=WEEKDAYS, timezone=IST, **kw)

    sch.add_job(morning_check, cron(hour=8, minute=50), id="morning_check")
    sch.add_job(watchlist_job, cron(hour=9, minute=31), id="watchlist")
    sch.add_job(health_job, cron(hour="9-15", minute="*"), id="health")
    sch.add_job(squareoff_job, cron(hour=15, minute=16), id="squareoff")
    sch.add_job(eod_summary_job, cron(hour=15, minute=35), id="eod_summary")
    if s.backfill_after_close:
        sch.add_job(backfill_job, cron(hour=16, minute=0), id="backfill")
    if s.retrain_enabled:
        t = parse_hhmm(s.retrain_time)
        sch.add_job(retrain_job, cron(hour=t.hour, minute=t.minute), id="retrain")
    sch.start()
    log.info("Scheduler started with jobs: %s", ", ".join(j.id for j in sch.get_jobs()))
    return sch
