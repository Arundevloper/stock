"""Strict, cost-aware trade labels.

label = 1 if, entering at the next 1-min bar open after the signal bar closes, price reaches
target (target_atr x ATR) BEFORE the stop (stop_atr x ATR) within horizon_bars and before
square-off. Stop and target touched in the same 1-min bar counts as a loss (conservative).
The paper-trading engine (backend/paper.py) applies exactly the same rules.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.params import StrategyParams
from core.timeutil import hhmm_to_min

LABEL_COLS = ["label", "outcome", "entry_time", "entry_price", "stop_price", "target_price",
              "exit_time", "exit_price"]


def exit_deadline(decision_ts: pd.Timestamp, p: StrategyParams) -> pd.Timestamp:
    """Trade must be flat by min(decision + horizon, square-off)."""
    horizon = decision_ts + pd.Timedelta(minutes=p.horizon_bars * p.signal_tf)
    sq = decision_ts.normalize() + pd.Timedelta(minutes=hhmm_to_min(p.squareoff))
    return min(horizon, sq)


def resolve_path(side: int, entry: float, stop: float, target: float,
                 o: np.ndarray, h: np.ndarray, lo: np.ndarray) -> tuple[str, int, float]:
    """Walk a price path (1-min bars) and return (outcome, bar_index, exit_price).
    outcome in {'target', 'stop', 'timeout'}; for timeout bar_index = -1."""
    o, h, lo = np.asarray(o, float), np.asarray(h, float), np.asarray(lo, float)
    if side > 0:
        stop_hit = lo <= stop
        tgt_hit = h >= target
    else:
        stop_hit = h >= stop
        tgt_hit = lo <= target
    i_s = int(np.argmax(stop_hit)) if stop_hit.any() else len(o)
    i_t = int(np.argmax(tgt_hit)) if tgt_hit.any() else len(o)
    if i_s < len(o) and i_s <= i_t:
        # gap through the stop fills at the bar open
        px = min(stop, o[i_s]) if side > 0 else max(stop, o[i_s])
        return "stop", i_s, float(px)
    if i_t < len(o):
        return "target", i_t, float(target)
    return "timeout", -1, float("nan")


def label_setups(setups: pd.DataFrame, df1: pd.DataFrame, p: StrategyParams | None = None) -> pd.DataFrame:
    p = p or StrategyParams()
    out = pd.DataFrame(index=setups.index, columns=LABEL_COLS, dtype=object)
    if setups.empty or df1.empty:
        return out
    t = df1.index
    o = df1["open"].to_numpy(float)
    h = df1["high"].to_numpy(float)
    lo = df1["low"].to_numpy(float)
    c = df1["close"].to_numpy(float)
    one_min = pd.Timedelta(minutes=1)

    for idx, row in setups.iterrows():
        T = row["close_time"]
        j = t.searchsorted(T, side="left")
        if j >= len(t) or t[j].normalize() != T.normalize():
            continue
        deadline = exit_deadline(T, p)
        if t[j] >= deadline:
            continue
        k = t.searchsorted(deadline, side="left")  # bars starting before the deadline
        side = int(row["side"])
        entry = o[j]
        stop = entry - side * p.stop_atr * row["atr"]
        target = entry + side * p.target_atr * row["atr"]
        outcome, i, px = resolve_path(side, entry, stop, target, o[j:k], h[j:k], lo[j:k])
        if outcome == "timeout":
            i, px = k - 1 - j, c[k - 1]
        out.loc[idx] = [int(outcome == "target"), outcome, t[j], entry, stop, target,
                        t[j + i] + one_min, px]
    out["label"] = pd.to_numeric(out["label"])
    for col in ("entry_price", "stop_price", "target_price", "exit_price"):
        out[col] = pd.to_numeric(out[col])
    return out
