"""Live pipeline: candle ingest -> paper-trade update -> (on signal-bar close) features -> setup
filter -> model -> risk rules -> signal + paper trade -> WebSocket / Telegram.

The engine is driven by candle timestamps, not the wall clock, so replaying a past session
behaves exactly like a live one.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque

import numpy as np
import pandas as pd
from sqlalchemy import select

from backend import notify, paper
from backend.model_store import models
from core import db
from core.config import MARKET_SYMBOL, get_settings
from core.costs import position_size
from core.db import PaperTrade, Signal, WatchlistEntry, session_scope
from core.timeutil import SESSION_START_MIN, day_bounds, day_str, minute_of_day, to_ist
from ml.features import build_features, find_setups, model_matrix
from ml.model import predict_proba

log = logging.getLogger(__name__)


def closes_signal_bar(ts: int, tf: int) -> bool:
    """True if the 1-min candle starting at ts is the last minute of a tf-minute bar."""
    return (minute_of_day(ts) + 1 - SESSION_START_MIN) % tf == 0


class Engine:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self._hist: dict[tuple[str, str], pd.DataFrame] = {}
        self._wl: tuple[str, set[str] | None] | None = None
        self.paused = False  # restored from the DB by load_state() at startup
        self.last_ingest_at: float | None = None
        self.last_candle_time: int | None = None
        self.recent_setups: deque[dict] = deque(maxlen=100)
        self.counters: dict[str, dict] = {}

    # ------------------------------------------------------------------ public

    def load_state(self) -> None:
        self.paused = bool(db.get_state("paused", False))

    def set_paused(self, paused: bool) -> None:
        self.paused = paused
        db.set_state("paused", paused)

    def invalidate_watchlist(self) -> None:
        self._wl = None

    def ingest(self, candles: list[dict]) -> list[dict]:
        candles = [c for c in (self._clean(c) for c in candles) if c]
        if not candles:
            return []
        candles.sort(key=lambda c: (c["time"], c["symbol"]))
        with self.lock:
            db.upsert_candles(candles)
            self.last_ingest_at = time.time()
            self.last_candle_time = max(self.last_candle_time or 0, candles[-1]["time"])
            events = [{"type": "candle", "data": c} for c in candles]

            p = models.params
            with session_scope() as s:
                for c in candles:
                    events += paper.on_candle(s, c, p)

            for c in candles:
                if c["symbol"] != MARKET_SYMBOL and closes_signal_bar(c["time"], p.signal_tf):
                    try:
                        events += self.evaluate(c["symbol"], c["time"] + 60, c)
                    except Exception:
                        log.exception("evaluate failed for %s", c["symbol"])

        self._notify_closed(events)
        return events

    # ------------------------------------------------------------------ internals

    @staticmethod
    def _clean(c: dict) -> dict | None:
        try:
            out = {
                "symbol": str(c["symbol"]), "time": int(c["time"]),
                "open": float(c["open"]), "high": float(c["high"]), "low": float(c["low"]),
                "close": float(c["close"]), "volume": int(c.get("volume") or 0),
            }
        except (KeyError, TypeError, ValueError):
            log.warning("bad candle %s", c)
            return None
        for k in ("bid", "ask", "buy_qty", "sell_qty"):
            if c.get(k) is not None:
                out[k] = c[k]
        return out

    def _counter(self, day: str) -> dict:
        if day not in self.counters:
            self.counters = {day: {"bars": 0, "setups": 0, "below_threshold": 0, "signals": 0, "blocked": 0}}
        return self.counters[day]

    def watchlist(self, day: str) -> set[str] | None:
        """Symbols allowed to signal today; None = no restriction."""
        if get_settings().watchlist_mode != "scan":
            return None
        if self._wl and self._wl[0] == day:
            return self._wl[1]
        with db.SessionLocal() as s:
            syms = set(s.scalars(select(WatchlistEntry.symbol).where(WatchlistEntry.date == day)).all())
        self._wl = (day, syms or None)
        return self._wl[1]

    def _history(self, symbol: str, day: str) -> pd.DataFrame:
        """Candles before `day` (cached per day - they don't change intraday)."""
        key = (symbol, day)
        if key not in self._hist:
            if any(k[1] != day for k in self._hist):  # new session: drop yesterday's cache
                self._hist = {}
            d0, _ = day_bounds(day)
            lookback = get_settings().live_lookback_days * 86400
            self._hist[key] = db.load_candles(symbol, d0 - lookback, d0)
        return self._hist[key]

    def _candles_upto(self, symbol: str, T: int) -> pd.DataFrame:
        """History + today's candles strictly before T (never anything later, even in replay)."""
        day = day_str(T - 60)
        d0, _ = day_bounds(day)
        hist = self._history(symbol, day)
        today = db.load_candles(symbol, d0, T)
        if hist.empty:
            return today
        if today.empty:
            return hist
        return pd.concat([hist, today])

    def evaluate(self, symbol: str, T: int, last: dict) -> list[dict]:
        """Run the model on the signal bar that closed at T (epoch s)."""
        s = get_settings()
        p = models.params
        day = day_str(T - 60)
        cnt = self._counter(day)
        cnt["bars"] += 1
        if self.paused:
            return []
        wl = self.watchlist(day)
        if wl is not None and symbol not in wl:
            return []
        if s.signal_mode == "model" and not models.bundle:
            return []

        df1 = self._candles_upto(symbol, T)
        market = self._candles_upto(MARKET_SYMBOL, T)
        f = build_features(df1, market if len(market) else None, p)
        if f.empty or int(f["close_time"].iloc[-1].timestamp()) != T:
            return []
        setups = find_setups(f.iloc[-1:], p)
        if setups.empty:
            return []
        row = setups.iloc[-1]
        cnt["setups"] += 1

        prob = None
        X = None
        if s.signal_mode == "model":
            X = model_matrix(setups, models.bundle["features"])
            prob = float(predict_proba(models.bundle, X)[0])
        side = int(row["side"])
        entry = float(row["close"])
        atr = float(row["atr"])
        stop = round(entry - side * p.stop_atr * atr, 2)
        target = round(entry + side * p.target_atr * atr, 2)
        thr = models.threshold
        passed = prob is None or prob >= thr
        self.recent_setups.appendleft({
            "symbol": symbol, "time": T, "side": "long" if side > 0 else "short", "prob": prob,
            "threshold": thr, "passed": passed, "close": entry, "atr": round(atr, 3),
            "vol_ratio": _num(row.get("vol_ratio_20")), "vwap_dist": _num(row.get("vwap_dist")),
        })
        if not passed:
            cnt["below_threshold"] += 1
            return [{"type": "setup", "data": self.recent_setups[0]}]

        qty = position_size(entry, stop, s.capital, s.risk_per_trade, s.mis_leverage)
        feats = model_matrix(setups).iloc[0].to_dict() if X is None else X.iloc[0].to_dict()
        with session_scope() as ses:
            reason = paper.risk_block_reason(ses, symbol, T, p)
            if not reason and qty < 1:
                reason = "qty < 1"
            sig = Signal(
                symbol=symbol, time=T, side="long" if side > 0 else "short", prob=prob,
                entry=round(entry, 2), stop=stop, target=target, atr=round(atr, 4), qty=qty,
                status="blocked" if reason else "new", reason=reason,
                bid=last.get("bid"), ask=last.get("ask"),
                features=json.dumps({k: _num(v) for k, v in feats.items()}),
                mode="paper", created_at=int(time.time()),
            )
            ses.add(sig)
            ses.flush()
            trade = None
            if not reason:
                trade = PaperTrade(signal_id=sig.id, symbol=symbol, side=sig.side, qty=qty, atr=atr,
                                   signal_time=T, status="pending", bid=sig.bid, ask=sig.ask)
                ses.add(trade)
                ses.flush()
            sig_d = sig.to_dict()
            events = [{"type": "setup", "data": self.recent_setups[0]}, {"type": "signal", "data": sig_d}]
            if trade:
                events.append({"type": "trade", "data": trade.to_dict()})

        if reason:
            cnt["blocked"] += 1
            log.info("Signal %s %s blocked: %s", symbol, sig_d["side"], reason)
        else:
            cnt["signals"] += 1
            log.info("SIGNAL %s %s prob=%s entry=%.2f sl=%.2f tgt=%.2f qty=%d at %s", symbol, sig_d["side"],
                     prob, entry, stop, target, qty, to_ist(T).strftime("%Y-%m-%d %H:%M"))
            notify.send(notify.signal_text(sig_d, p.squareoff))
        return events

    def _notify_closed(self, events: list[dict]) -> None:
        if not get_settings().telegram_trade_updates:
            return
        for e in events:
            if e["type"] == "trade" and e["data"]["status"] == "closed":
                notify.send(notify.trade_closed_text(e["data"]))

    def square_off(self, reason: str = "squareoff", symbol: str | None = None,
                   trade_id: int | None = None) -> list[dict]:
        with self.lock, session_scope() as s:
            events = paper.force_close_all(s, models.params, reason, symbol, trade_id)
        self._notify_closed(events)
        return events


def _num(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(v) else round(v, 6)


engine = Engine()
