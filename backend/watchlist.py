"""Morning watchlist scan (09:31): rank the universe by first-15-min relative volume and gap.

Uses the reader's own candles when available (works in replay/demo too); falls back to
kite.quote() when there is no data for today yet.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sqlalchemy import delete, select

from core import db
from core.config import MARKET_SYMBOL, get_settings
from core.db import WatchlistEntry, session_scope
from core.instruments import load_universe
from core.timeutil import at_time, day_bounds, day_str

log = logging.getLogger(__name__)


def symbols_with_data(d0: int, d1: int) -> list[str]:
    with db.engine.connect() as conn:
        rows = conn.execute(select(db.Candle.symbol).where(db.Candle.time >= d0, db.Candle.time < d1).distinct())
        return [r[0] for r in rows if r[0] != MARKET_SYMBOL]


def _scan_from_db(day: str) -> list[dict]:
    d0, d1 = day_bounds(day)
    t930 = at_time(day, "09:30")
    with_data = set(symbols_with_data(d0, t930))
    universe = [s for s in load_universe() if s in with_data] or sorted(with_data)
    rows = []
    for sym in universe:
        df = db.load_candles(sym, d0 - 25 * 86400, t930)
        if df.empty:
            continue
        df = df.assign(day=df.index.strftime("%Y-%m-%d"),
                       tod=df.index.hour * 60 + df.index.minute)
        first15 = df[(df["tod"] >= 555) & (df["tod"] < 570)]
        today = first15[first15["day"] == day]
        if today.empty:
            continue
        prev = df[df["day"] < day]
        prev_close = float(prev["close"].iloc[-1]) if len(prev) else np.nan
        past_vol15 = first15[first15["day"] < day].groupby("day")["volume"].sum().tail(10)
        vol15 = float(today["volume"].sum())
        rows.append({
            "symbol": sym,
            "gap_pct": (float(today["open"].iloc[0]) / prev_close - 1) * 100 if prev_close else None,
            "rvol": vol15 / past_vol15.mean() if len(past_vol15) >= 3 and past_vol15.mean() > 0 else None,
            "turnover_cr": float((today["close"] * today["volume"]).sum()) / 1e7,
        })
    return rows


def _scan_from_quotes() -> list[dict]:
    from core.kite import get_kite

    kite = get_kite()
    universe = load_universe()
    rows = []
    for i in range(0, len(universe), 400):
        quotes = kite.quote(universe[i:i + 400])
        for sym, q in quotes.items():
            prev_close = q.get("ohlc", {}).get("close") or 0
            rows.append({
                "symbol": sym,
                "gap_pct": (q["ohlc"]["open"] / prev_close - 1) * 100 if prev_close else None,
                "rvol": None,
                "turnover_cr": q.get("volume", 0) * q.get("last_price", 0) / 1e7,
            })
    return rows


def scan(day: str | None = None) -> list[dict]:
    s = get_settings()
    day = day or day_str()
    rows = _scan_from_db(day)
    source = "scan"
    if not rows:
        try:
            rows = _scan_from_quotes()
            source = "quote"
        except Exception as e:
            log.warning("Watchlist scan: no candles for %s and quote fallback failed: %s", day, e)
            return []
    df = pd.DataFrame(rows)
    df = df[df["turnover_cr"] >= s.min_turnover_cr] if len(df) else df
    if df.empty:
        log.warning("Watchlist scan: nothing passed the turnover filter")
        return []
    rv = df["rvol"].astype(float).rank(pct=True).fillna(0.5)
    gp = df["gap_pct"].astype(float).abs().rank(pct=True).fillna(0.5)
    df["score"] = (0.6 * rv + 0.4 * gp).round(4)
    df = df.sort_values("score", ascending=False).head(s.watchlist_size)
    entries = [dict(date=day, symbol=r.symbol, score=float(r.score),
                    gap_pct=_f(r.gap_pct), rvol=_f(r.rvol), turnover_cr=_f(r.turnover_cr), source=source)
               for r in df.itertuples()]
    save(day, entries)
    log.info("Watchlist %s: %d symbols (%s)", day, len(entries), source)
    return entries


def save(day: str, entries: list[dict]) -> None:
    with session_scope() as ses:
        ses.execute(delete(WatchlistEntry).where(WatchlistEntry.date == day))
        for e in entries:
            ses.add(WatchlistEntry(**e))


def set_manual(day: str, symbols: list[str]) -> list[dict]:
    entries = [dict(date=day, symbol=sym, score=0.0, source="manual") for sym in symbols]
    save(day, entries)
    return entries


def get(day: str) -> list[dict]:
    with db.SessionLocal() as ses:
        rows = ses.scalars(select(WatchlistEntry).where(WatchlistEntry.date == day)
                           .order_by(WatchlistEntry.score.desc())).all()
        return [r.to_dict() for r in rows]


def _f(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(x) else round(x, 4)
