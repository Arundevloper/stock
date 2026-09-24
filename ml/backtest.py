"""Portfolio backtest of labelled setups: next-bar-open entry, slippage, Zerodha costs,
fixed-risk sizing and the same daily risk rules the live engine enforces."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.costs import position_size, slip, trade_pnl
from core.params import StrategyParams


def simulate(cands: pd.DataFrame, p: StrategyParams, capital: float = 100_000, risk_pct: float = 0.01,
             leverage: float = 5.0, compound: bool = False) -> tuple[pd.DataFrame, dict]:
    """`cands` = labelled setups that passed the model threshold, with columns
    symbol, close_time, side, atr, entry_price, stop_price, exit_time, exit_price, outcome."""
    if cands.empty:
        return pd.DataFrame(), summarize(pd.DataFrame(), capital)
    cands = cands.sort_values("close_time")
    taken = []
    equity = capital
    open_until: dict[str, pd.Timestamp] = {}
    day_trades: dict[str, list] = {}

    for _, r in cands.iterrows():
        T = r["close_time"]
        day = T.strftime("%Y-%m-%d")
        today = day_trades.setdefault(day, [])
        if open_until.get(r["symbol"]) is not None and open_until[r["symbol"]] > T:
            continue  # one position per symbol
        if len(today) >= p.max_trades_per_day:
            continue
        losses = sum(1 for t in today if t["exit_time"] <= T and t["net"] < 0)
        if losses >= p.max_losses_per_day:
            continue

        side = int(r["side"])
        qty = position_size(r["entry_price"], r["stop_price"], equity if compound else capital,
                            risk_pct, leverage)
        if qty < 1:
            continue
        entry_fill = slip(r["entry_price"], side, True, p.slippage_pct)
        exit_fill = slip(r["exit_price"], side, False, p.slippage_pct)
        gross, costs, net = trade_pnl(side, qty, entry_fill, exit_fill)
        risk_rs = abs(r["entry_price"] - r["stop_price"]) * qty
        t = {
            "symbol": r["symbol"], "close_time": T, "exit_time": r["exit_time"], "side": side,
            "prob": r.get("prob", np.nan), "qty": qty, "entry": entry_fill, "exit": exit_fill,
            "outcome": r["outcome"], "win": int(r["outcome"] == "target"),
            "gross": gross, "costs": costs, "net": net, "r": net / risk_rs if risk_rs else 0.0,
        }
        today.append(t)
        taken.append(t)
        open_until[r["symbol"]] = r["exit_time"]
        equity += net

    trades = pd.DataFrame(taken)
    return trades, summarize(trades, capital)


def summarize(trades: pd.DataFrame, capital: float) -> dict:
    if trades.empty:
        return {"trades": 0, "wins": 0, "precision": None, "gross": 0.0, "costs": 0.0, "net": 0.0,
                "avg_r": None, "max_dd": 0.0, "max_dd_pct": 0.0, "worst_streak": 0, "days": 0}
    t = trades.sort_values("exit_time")
    eq = capital + t["net"].cumsum()
    peak = np.maximum.accumulate(np.concatenate([[capital], eq.to_numpy()]))[1:]
    dd = (eq - peak).min()
    streak = worst = 0
    for n in t["net"]:
        streak = streak + 1 if n < 0 else 0
        worst = max(worst, streak)
    return {
        "trades": int(len(t)),
        "wins": int(t["win"].sum()),
        "precision": round(float(t["win"].mean()), 4),
        "gross": round(float(t["gross"].sum()), 2),
        "costs": round(float(t["costs"].sum()), 2),
        "net": round(float(t["net"].sum()), 2),
        "avg_r": round(float(t["r"].mean()), 3),
        "max_dd": round(float(dd), 2),
        "max_dd_pct": round(float(dd / capital * 100), 2),
        "worst_streak": int(worst),
        "days": int(t["close_time"].dt.strftime("%Y-%m-%d").nunique()),
    }


def equity_curve(trades: pd.DataFrame, capital: float) -> list[dict]:
    if trades.empty:
        return []
    t = trades.sort_values("exit_time")
    eq = capital + t["net"].cumsum()
    return [{"time": int(ts.timestamp()), "value": round(float(v), 2)} for ts, v in zip(t["exit_time"], eq)]
