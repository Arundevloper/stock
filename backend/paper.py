"""Paper-trading engine. Same fill/exit rules as ml/labels.py so paper results are comparable
with the backtest:
  * entry at the open of the first 1-min candle at/after the signal bar close (+ slippage)
  * stop / target from that fill price and the signal ATR
  * stop and target in the same candle -> stop (conservative); gap through stop fills at open
  * exit at market when max hold (horizon) or square-off time is reached
"""
from __future__ import annotations

import logging

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from core import db
from core.costs import slip, trade_pnl
from core.db import PaperTrade, Signal
from core.params import StrategyParams
from core.timeutil import at_time, day_bounds, day_str
from ml.labels import resolve_path

log = logging.getLogger(__name__)

ACTIVE = ("pending", "open")


def side_int(side: str) -> int:
    return 1 if side == "long" else -1


def deadline(signal_time: int, p: StrategyParams) -> tuple[int, str]:
    sq = at_time(day_str(signal_time), p.squareoff)
    horizon = signal_time + p.horizon_bars * p.signal_tf * 60
    return (horizon, "timeout") if horizon < sq else (sq, "squareoff")


def _event(t: PaperTrade) -> dict:
    return {"type": "trade", "data": t.to_dict()}


def close_trade(s: Session, t: PaperTrade, raw_exit: float, reason: str, exit_time: int,
                p: StrategyParams) -> dict:
    side = side_int(t.side)
    exit_fill = slip(raw_exit, side, False, p.slippage_pct)
    gross, costs, net = trade_pnl(side, t.qty, t.entry_price, exit_fill)
    risk = p.stop_atr * t.atr * t.qty
    t.exit_price = round(exit_fill, 2)
    t.exit_time = exit_time
    t.exit_reason = reason
    t.gross_pnl, t.costs, t.net_pnl = gross, costs, net
    t.r_multiple = round(net / risk, 3) if risk else 0.0
    t.status = "closed"
    if t.signal_id:
        sig = s.get(Signal, t.signal_id)
        if sig:
            sig.status = "closed"
            sig.outcome = reason
    return _event(t)


def on_candle(s: Session, c: dict, p: StrategyParams) -> list[dict]:
    """Advance every pending/open paper trade on `c['symbol']` with one 1-min candle."""
    trades = s.scalars(select(PaperTrade).where(PaperTrade.symbol == c["symbol"],
                                                PaperTrade.status.in_(ACTIVE))).all()
    events = []
    t0, t1 = c["time"], c["time"] + 60
    o, h, lo, cl = c["open"], c["high"], c["low"], c["close"]
    for t in trades:
        side = side_int(t.side)
        dl, dl_reason = deadline(t.signal_time, p)

        if t.status == "pending":
            if t0 < t.signal_time:
                continue
            if day_str(t0) != day_str(t.signal_time) or t0 >= dl:
                t.status = "cancelled"
                t.exit_reason = "expired"
                events.append(_event(t))
                continue
            t.entry_raw = o
            t.entry_price = round(slip(o, side, True, p.slippage_pct), 2)
            t.entry_time = t0
            t.stop = round(o - side * p.stop_atr * t.atr, 2)
            t.target = round(o + side * p.target_atr * t.atr, 2)
            t.status = "open"
            if t.signal_id and (sig := s.get(Signal, t.signal_id)):
                sig.status = "filled"
            events.append(_event(t))

        if t.status != "open":
            continue
        if t0 >= dl:  # missed candles: exit at this candle's open
            events.append(close_trade(s, t, o, dl_reason, t0, p))
            continue
        outcome, _, px = resolve_path(side, t.entry_raw, t.stop, t.target, [o], [h], [lo])
        if outcome == "stop":
            events.append(close_trade(s, t, px, "stop", t1, p))
        elif outcome == "target":
            events.append(close_trade(s, t, px, "target", t1, p))
        elif t1 >= dl:
            events.append(close_trade(s, t, cl, dl_reason, t1, p))
    return events


def force_close_all(s: Session, p: StrategyParams, reason: str = "squareoff",
                    symbol: str | None = None, trade_id: int | None = None) -> list[dict]:
    """Close open trades at the last known price, cancel pending ones."""
    q = select(PaperTrade).where(PaperTrade.status.in_(ACTIVE))
    if symbol:
        q = q.where(PaperTrade.symbol == symbol)
    if trade_id:
        q = q.where(PaperTrade.id == trade_id)
    events = []
    for t in s.scalars(q).all():
        if t.status == "pending":
            t.status = "cancelled"
            t.exit_reason = reason
            events.append(_event(t))
            continue
        last = db.last_candle(t.symbol)
        if not last:
            continue
        events.append(close_trade(s, t, last["close"], reason, int(last["time"]) + 60, p))
    return events


def risk_block_reason(s: Session, symbol: str, T: int, p: StrategyParams) -> str | None:
    """Daily risk rules - identical to ml/backtest.simulate."""
    d0, d1 = day_bounds(day_str(T))
    in_day = (PaperTrade.signal_time >= d0) & (PaperTrade.signal_time < d1)
    if s.scalar(select(func.count()).where(PaperTrade.symbol == symbol, PaperTrade.status.in_(ACTIVE))):
        return "position already open"
    n = s.scalar(select(func.count()).where(in_day, PaperTrade.status != "cancelled"))
    if n >= p.max_trades_per_day:
        return f"max {p.max_trades_per_day} trades/day reached"
    losses = s.scalar(select(func.count()).where(in_day, PaperTrade.status == "closed",
                                                 PaperTrade.net_pnl < 0, PaperTrade.exit_time <= T))
    if losses >= p.max_losses_per_day:
        return f"{p.max_losses_per_day} losses today - stopped"
    return None


def day_stats(s: Session, day: str) -> dict:
    d0, d1 = day_bounds(day)
    sigs = s.scalar(select(func.count()).where(Signal.time >= d0, Signal.time < d1, Signal.status != "blocked"))
    trades = s.scalars(select(PaperTrade).where(PaperTrade.signal_time >= d0, PaperTrade.signal_time < d1,
                                                PaperTrade.status != "cancelled")).all()
    closed = [t for t in trades if t.status == "closed"]
    return {
        "day": day,
        "signals": int(sigs or 0),
        "trades": len(trades),
        "open": sum(1 for t in trades if t.status in ACTIVE),
        "wins": sum(1 for t in closed if t.exit_reason == "target"),
        "losses": sum(1 for t in closed if (t.net_pnl or 0) < 0),
        "gross": round(sum(t.gross_pnl or 0 for t in closed), 2),
        "costs": round(sum(t.costs or 0 for t in closed), 2),
        "net": round(sum(t.net_pnl or 0 for t in closed), 2),
    }
