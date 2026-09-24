"""HTTP API used by the web UI and the reader."""
from __future__ import annotations

import asyncio
import hmac
import json
import time
from collections import defaultdict

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, TypeAdapter, ValidationError
from sqlalchemy import select

from backend import watchlist as wl
from backend.engine import engine
from backend.model_store import models
from backend.paper import day_stats
from backend.training import runner
from backend.ws import hub
from core import db
from core.config import MARKET_SYMBOL, get_settings
from core.db import PaperTrade, Signal
from core.instruments import load_universe, normalize
from core.kite import complete_login, kite_configured, login_url, token_status
from core.timeutil import day_bounds, day_str, is_market_open, now_ist
from ml.features import intraday_vwap, resample

router = APIRouter()


# ------------------------------------------------------------------ helpers

def market_day() -> str:
    """The session the UI should show: day of the newest candle (replay-aware), else today."""
    t = engine.last_candle_time or db.latest_candle_time()
    return day_str(t) if t else day_str()


def _prev_close(symbol: str, day: str) -> float | None:
    d0, _ = day_bounds(day)
    with db.engine.connect() as conn:
        r = conn.execute(select(db.Candle.close).where(db.Candle.symbol == symbol, db.Candle.time < d0)
                         .order_by(db.Candle.time.desc()).limit(1)).first()
    return r[0] if r else None


def _quote(symbol: str, day: str) -> dict:
    last = db.last_candle(symbol)
    pc = _prev_close(symbol, day)
    ltp = last["close"] if last else None
    return {"symbol": symbol, "ltp": ltp, "prev_close": pc, "last_time": last["time"] if last else None,
            "change_pct": round((ltp / pc - 1) * 100, 2) if ltp and pc else None}


# ------------------------------------------------------------------ ingest (reader -> backend)

class CandleIn(BaseModel):
    symbol: str
    time: int
    open: float
    high: float
    low: float
    close: float
    volume: int = 0
    bid: float | None = None
    ask: float | None = None
    buy_qty: int | None = None
    sell_qty: int | None = None


_candles_adapter = TypeAdapter(list[CandleIn])


@router.post("/api/ingest/candles")
async def ingest(request: Request, x_ingest_key: str = Header("")):
    if not hmac.compare_digest(x_ingest_key, get_settings().ingest_key):
        raise HTTPException(401, "bad ingest key")
    body = await request.json()
    if isinstance(body, dict):
        body = body.get("candles", [body])
    try:
        candles = [c.model_dump() for c in _candles_adapter.validate_python(body)]
    except ValidationError as e:
        raise HTTPException(422, str(e)) from e
    events = await asyncio.to_thread(engine.ingest, candles)
    await hub.broadcast(events)
    return {"ok": True, "received": len(candles),
            "signals": sum(1 for e in events if e["type"] == "signal")}


# ------------------------------------------------------------------ market data

@router.get("/api/candles")
def get_candles(symbol: str, tf: int = 5, days: int = 3):
    if tf not in (1, 3, 5, 15, 30, 60):
        raise HTTPException(400, "tf must be 1, 3, 5, 15, 30 or 60")
    last = db.last_candle(symbol)
    if not last:
        return {"symbol": symbol, "tf": tf, "candles": [], "vwap": []}
    end = int(last["time"]) + 60
    df = db.load_candles(symbol, end - int(days * 1.6 + 4) * 86400, end)
    sessions = sorted(set(df.index.strftime("%Y-%m-%d")))[-days:]
    df = df[df.index.strftime("%Y-%m-%d").isin(sessions)]
    bars = resample(df, tf)
    vwap = intraday_vwap(bars) if symbol != MARKET_SYMBOL else None
    ts = [int(t.timestamp()) for t in bars.index]
    out = [{"time": t, "open": r.open, "high": r.high, "low": r.low, "close": r.close, "volume": int(r.volume)}
           for t, r in zip(ts, bars.itertuples())]
    vw = [{"time": t, "value": round(float(v), 2)} for t, v in zip(ts, vwap)] if vwap is not None else []
    return {"symbol": symbol, "tf": tf, "candles": out, "vwap": vw}


@router.get("/api/symbols")
def get_symbols():
    syms = db.list_symbols()
    return {"symbols": syms, "universe": load_universe(), "market": MARKET_SYMBOL}


@router.get("/api/watchlist")
def get_watchlist(day: str | None = None):
    day = day or market_day()
    entries = wl.get(day)
    source = entries[0]["source"] if entries else None
    if not entries:
        d0, d1 = day_bounds(day)
        have = set(wl.symbols_with_data(d0, d1)) or set(db.list_symbols()) - {MARKET_SYMBOL}
        uni = [s for s in load_universe() if s in have] or sorted(have)
        entries = [{"symbol": s, "score": None, "gap_pct": None, "rvol": None} for s in uni]
        source = "universe (no scan yet)"
    rows = [{**e, **_quote(e["symbol"], day)} for e in entries]
    return {"day": day, "source": source, "entries": rows, "market": _quote(MARKET_SYMBOL, day)}


class WatchlistIn(BaseModel):
    symbols: list[str]
    day: str | None = None


@router.post("/api/watchlist")
def set_watchlist(body: WatchlistIn):
    day = body.day or market_day()
    entries = wl.set_manual(day, [normalize(s) for s in body.symbols if s.strip()])
    engine.invalidate_watchlist()
    return {"day": day, "entries": entries}


@router.post("/api/watchlist/scan")
async def scan_watchlist(day: str | None = None):
    entries = await asyncio.to_thread(wl.scan, day or market_day())
    engine.invalidate_watchlist()
    await hub.broadcast([{"type": "watchlist", "data": entries}])
    return {"entries": entries}


# ------------------------------------------------------------------ signals / trades / pnl

@router.get("/api/signals")
def get_signals(day: str | None = None, symbol: str | None = None, days: int = 1,
                include_blocked: bool = True, limit: int = 300):
    """Signals of one session (default: the current market day), or of the last `days` sessions
    ending on `day` when days > 1; optionally for one symbol."""
    day = day or market_day()
    d0, d1 = day_bounds(day)
    d0 -= (max(1, days) - 1) * 86400
    q = select(Signal).where(Signal.time >= d0, Signal.time < d1)
    if symbol:
        q = q.where(Signal.symbol == symbol)
    if not include_blocked:
        q = q.where(Signal.status != "blocked")
    with db.SessionLocal() as s:
        rows = s.scalars(q.order_by(Signal.time.desc()).limit(limit)).all()
        return {"day": day, "signals": [r.to_dict() for r in rows]}


@router.get("/api/signals/{sid}")
def get_signal(sid: int):
    with db.SessionLocal() as s:
        sig = s.get(Signal, sid)
        if not sig:
            raise HTTPException(404)
        trade = s.scalars(select(PaperTrade).where(PaperTrade.signal_id == sid)).first()
        return {**sig.to_dict(), "features": json.loads(sig.features or "{}"),
                "trade": trade.to_dict() if trade else None}


@router.get("/api/setups")
def get_setups():
    return {"setups": list(engine.recent_setups)}


@router.get("/api/trades")
def get_trades(day: str | None = None, days: int = 30):
    with db.SessionLocal() as s:
        q = select(PaperTrade)
        if day:
            d0, d1 = day_bounds(day)
            q = q.where(PaperTrade.signal_time >= d0, PaperTrade.signal_time < d1)
        else:
            ref = db.latest_candle_time() or time.time()  # newest data, so replays of old days show up
            q = q.where((PaperTrade.signal_time >= ref - days * 86400) |
                        PaperTrade.status.in_(("pending", "open")))
        rows = s.scalars(q.order_by(PaperTrade.signal_time.desc()).limit(1000)).all()
        return {"trades": [r.to_dict() for r in rows]}


@router.get("/api/pnl")
def get_pnl():
    with db.SessionLocal() as s:
        trades = s.scalars(select(PaperTrade).where(PaperTrade.status == "closed")
                           .order_by(PaperTrade.exit_time)).all()
        by_day = defaultdict(list)
        for t in trades:
            by_day[day_str(t.signal_time)].append(t)
        days, cum = [], 0.0
        for d in sorted(by_day):
            ts = by_day[d]
            net = sum(t.net_pnl for t in ts)
            cum += net
            days.append({"day": d, "trades": len(ts), "wins": sum(t.exit_reason == "target" for t in ts),
                         "losses": sum(t.net_pnl < 0 for t in ts), "gross": round(sum(t.gross_pnl for t in ts), 2),
                         "costs": round(sum(t.costs for t in ts), 2), "net": round(net, 2), "cumulative": round(cum, 2)})
        equity, eq = [], get_settings().capital
        for t in trades:
            eq += t.net_pnl
            equity.append({"time": int(t.exit_time), "value": round(eq, 2)})
        n = len(trades)
        wins = sum(t.exit_reason == "target" for t in trades)
        peak, max_dd, e = get_settings().capital, 0.0, get_settings().capital
        for t in trades:
            e += t.net_pnl
            peak = max(peak, e)
            max_dd = min(max_dd, e - peak)
        summary = {"trades": n, "wins": wins, "precision": round(wins / n, 4) if n else None,
                   "net": round(cum, 2), "costs": round(sum(t.costs for t in trades), 2),
                   "avg_r": round(sum(t.r_multiple for t in trades) / n, 3) if n else None,
                   "max_dd": round(max_dd, 2)}
        return {"days": days[::-1], "equity": equity, "summary": summary, "capital": get_settings().capital}


@router.post("/api/trades/{tid}/close")
async def close_trade(tid: int):
    events = await asyncio.to_thread(engine.square_off, "manual", None, tid)
    await hub.broadcast(events)
    return {"closed": len(events)}


# ------------------------------------------------------------------ status / control

@router.get("/api/status")
def get_status():
    s = get_settings()
    day = market_day()
    with db.SessionLocal() as ses:
        today = day_stats(ses, day)
    last_ingest = engine.last_ingest_at
    return {
        "server_time": now_ist().isoformat(timespec="seconds"),
        "market_open": is_market_open(),
        "market_day": day,
        "paused": engine.paused,
        "mode": "paper",
        "signal_mode": s.signal_mode,
        "watchlist_mode": s.watchlist_mode,
        "model": models.info(),
        "kite": {"configured": kite_configured(), **token_status()},
        "reader": {"last_ingest_at": last_ingest, "last_candle_time": engine.last_candle_time,
                   "alive": bool(last_ingest and time.time() - last_ingest < s.stale_data_minutes * 60)},
        "today": today,
        "counters": engine.counters.get(day, {}),
        "training": {k: v for k, v in runner.status().items() if k != "log_tail"},
        "telegram": s.telegram_enabled,
        "capital": s.capital,
        "ws_clients": len(hub.clients),
    }


class PauseIn(BaseModel):
    paused: bool


@router.post("/api/control/pause")
async def pause(body: PauseIn):
    engine.set_paused(body.paused)
    await hub.broadcast([{"type": "status", "data": {"paused": engine.paused}}])
    return {"paused": engine.paused}


@router.post("/api/control/squareoff")
async def squareoff_all():
    events = await asyncio.to_thread(engine.square_off, "manual")
    await hub.broadcast(events)
    return {"closed": len(events)}


# ------------------------------------------------------------------ model

@router.get("/api/model")
def get_model():
    return {"info": models.info(), "report": models.report(), "training": runner.status()}


@router.post("/api/model/reload")
def reload_model():
    models.load()
    return models.info()


@router.post("/api/model/train")
async def train_model():
    if runner.running:
        raise HTTPException(409, "training already running")

    async def _run():
        await runner.run()
        await hub.broadcast([{"type": "model", "data": models.info()}])

    asyncio.create_task(_run())
    return {"started": True}


# ------------------------------------------------------------------ Kite login

@router.get("/auth/login")
def kite_login():
    if not kite_configured():
        raise HTTPException(400, "Set KITE_API_KEY and KITE_API_SECRET in .env first")
    return RedirectResponse(login_url())


@router.get("/auth/callback")
def kite_callback(request_token: str | None = None, status: str | None = None):
    """Set this URL as the Redirect URL of your app on developers.kite.trade."""
    if status != "success" or not request_token:
        return RedirectResponse("/?login=failed")
    try:
        complete_login(request_token)
    except Exception as e:
        return RedirectResponse(f"/?login=failed&error={type(e).__name__}")
    return RedirectResponse("/?login=ok")


@router.get("/health")
def health():
    s = get_settings()
    last = engine.last_ingest_at
    return {"ok": True, "model_loaded": models.bundle is not None,
            "reader_alive": bool(last and time.time() - last < s.stale_data_minutes * 60),
            "market_open": is_market_open()}
