"""Load candles from the DB and build the labelled setup dataset."""
from __future__ import annotations

import logging

import pandas as pd

from core import db
from core.config import MARKET_SYMBOL
from core.params import StrategyParams
from ml.features import build_features, find_setups
from ml.labels import label_setups

log = logging.getLogger(__name__)


def symbol_dataset(df1: pd.DataFrame, market: pd.DataFrame | None, p: StrategyParams) -> pd.DataFrame:
    f = build_features(df1, market, p)
    if f.empty:
        return f
    s = find_setups(f, p)
    if s.empty:
        return s
    lab = label_setups(s, df1, p)
    s = s.join(lab)
    return s.dropna(subset=["label"])


def build_dataset(symbols: list[str], p: StrategyParams, start: int | None = None,
                  end: int | None = None) -> pd.DataFrame:
    market = db.load_candles(MARKET_SYMBOL, start, end)
    if market.empty:
        log.warning("No %s candles - market features will be NaN", MARKET_SYMBOL)
        market = None
    frames = []
    for sym in symbols:
        df1 = db.load_candles(sym, start, end)
        if len(df1) < 1000:
            log.info("%s: only %d candles, skipped", sym, len(df1))
            continue
        s = symbol_dataset(df1, market, p)
        log.info("%-18s candles=%7d setups=%5d wins=%4d", sym, len(df1), len(s),
                 int(s["label"].sum()) if len(s) else 0)
        if len(s):
            s = s.assign(symbol=sym)
            frames.append(s)
    if not frames:
        return pd.DataFrame()
    ds = pd.concat(frames).sort_values("close_time")
    ds["entry_time"] = pd.to_datetime(ds["entry_time"])
    ds["exit_time"] = pd.to_datetime(ds["exit_time"])
    ds["session"] = ds["close_time"].dt.strftime("%Y-%m-%d")
    return ds
