"""Shared feature module - used by BOTH training and the live signal engine.

Rules:
  * Input is 1-min OHLCV indexed by tz-aware IST bar-start time (see core.db.load_candles).
  * Output is one row per signal_tf bar. Every column in a row uses only data available at that
    bar's close (`close_time`). tests/test_features.py enforces this by truncating the input.
  * Higher-timeframe values are joined with merge_asof on close time, so a still-forming 15-min
    bar is never used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.params import StrategyParams
from core.timeutil import SESSION_START_MIN, hhmm_to_min

OHLCV = ["open", "high", "low", "close", "volume"]
_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}

BASE_FEATURES = [
    "ret_1", "ret_3", "ret_6", "ret_12",
    "vol_ratio_20", "rvol_tod",
    "vwap_dist", "above_vwap",
    "range_pct", "ret_std_20", "atr_pct",
    "minutes_since_open", "phase",
    "gap_pct", "dist_or_high", "dist_or_low", "or_width_pct",
    "dist_prev_high", "dist_prev_low", "dist_prev_close",
    "breakout_up", "breakout_dn", "vol_spike",
    "trend_15", "ret_15m_1", "trend_d", "prev_day_ret",
    "nifty_ret_open", "nifty_ret_3",
]
# Directional features are also given "in the direction of the trade" (x side) so one model
# can learn longs and shorts together.
DIRECTIONAL = ["ret_1", "ret_3", "ret_6", "vwap_dist", "gap_pct", "trend_15", "trend_d",
               "nifty_ret_open", "nifty_ret_3", "dist_prev_close"]
FEATURES = BASE_FEATURES + ["side"] + [f"{c}_dir" for c in DIRECTIONAL]


def resample(df1: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Aggregate 1-min bars to `minutes` bars labelled by start time (09:15 aligned)."""
    if df1.empty:
        return df1[OHLCV].copy() if set(OHLCV) <= set(df1.columns) else pd.DataFrame(columns=OHLCV)
    if minutes == 1:
        return df1[OHLCV].copy()
    out = df1[OHLCV].resample(f"{minutes}min", label="left", closed="left").agg(_AGG)
    return out.dropna(subset=["open"])


def intraday_vwap(bars: pd.DataFrame) -> pd.Series:
    day = bars.index.normalize()
    tp = (bars["high"] + bars["low"] + bars["close"]) / 3
    vol = bars["volume"].astype(float)
    cum_pv = (tp * vol).groupby(day).cumsum()
    cum_v = vol.groupby(day).cumsum()
    vwap = cum_pv / cum_v.replace(0, np.nan)
    return vwap.fillna(bars["close"])  # index (no volume) -> fall back to close


def _ratio(a, b) -> pd.Series:
    return a / b.replace(0, np.nan) - 1


def build_features(df1: pd.DataFrame, market1: pd.DataFrame | None = None,
                   p: StrategyParams | None = None) -> pd.DataFrame:
    p = p or StrategyParams()
    tf = p.signal_tf
    d = resample(df1, tf)
    if len(d) < 30:
        return pd.DataFrame()
    d.index.name = "ts"

    d["close_time"] = d.index + pd.Timedelta(minutes=tf)
    day = d.index.normalize()
    c, h, lo = d["close"], d["high"], d["low"]
    vol = d["volume"].astype(float)

    # --- returns / volatility
    for n in (1, 3, 6, 12):
        d[f"ret_{n}"] = _ratio(c, c.shift(n))
    r1 = d["ret_1"]
    d["ret_std_20"] = r1.rolling(20, min_periods=10).std()
    d["range_pct"] = (h - lo) / c
    prev_c = c.shift(1)
    tr = pd.concat([h - lo, (h - prev_c).abs(), (lo - prev_c).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / p.atr_period, adjust=False, min_periods=p.atr_period).mean()
    d["atr_pct"] = d["atr"] / c

    # --- volume
    d["vol_ratio_20"] = vol / vol.rolling(p.vol_avg_bars, min_periods=5).mean().shift(1).replace(0, np.nan)
    tod = np.asarray(d.index.hour * 60 + d.index.minute)
    same_slot_avg = vol.groupby(tod).transform(
        lambda s: s.shift(1).rolling(p.rvol_days, min_periods=3).mean())
    d["rvol_tod"] = vol / same_slot_avg.replace(0, np.nan)

    # --- VWAP
    d["vwap"] = intraday_vwap(d)
    d["vwap_dist"] = _ratio(c, d["vwap"])
    d["above_vwap"] = (c > d["vwap"]).astype(int)

    # --- session clock (at decision time = bar close)
    ct = d["close_time"]
    mins = (ct.dt.hour * 60 + ct.dt.minute - SESSION_START_MIN).astype(int)
    d["minutes_since_open"] = mins
    d["phase"] = np.select([mins < 15, mins < 135, mins < 255, mins < 345], [0, 1, 2, 3], 4)

    # --- daily context (previous sessions only, plus today's open)
    daily = df1[OHLCV].groupby(df1.index.normalize()).agg(_AGG)
    daily["prev_close"] = daily["close"].shift(1)
    daily["prev_high"] = daily["high"].shift(1)
    daily["prev_low"] = daily["low"].shift(1)
    daily["prev_day_ret"] = _ratio(daily["close"].shift(1), daily["close"].shift(2))
    daily["sma20_prev"] = daily["close"].shift(1).rolling(20, min_periods=10).mean()
    dm = daily.reindex(day)
    dm.index = d.index
    d["gap_pct"] = _ratio(dm["open"], dm["prev_close"])
    d["dist_prev_high"] = _ratio(c, dm["prev_high"])
    d["dist_prev_low"] = _ratio(c, dm["prev_low"])
    d["dist_prev_close"] = _ratio(c, dm["prev_close"])
    d["trend_d"] = _ratio(dm["prev_close"], dm["sma20_prev"])
    d["prev_day_ret"] = dm["prev_day_ret"]

    # --- opening range (09:15-09:30), running values so early bars never see the future
    in_or = pd.Series(tod < SESSION_START_MIN + 15, index=d.index)
    or_h = h.where(in_or).groupby(day).cummax().groupby(day).ffill()
    or_l = lo.where(in_or).groupby(day).cummin().groupby(day).ffill()
    d["dist_or_high"] = _ratio(c, or_h)
    d["dist_or_low"] = _ratio(c, or_l)
    d["or_width_pct"] = _ratio(or_h, or_l)

    # --- breakouts / spikes
    n = p.breakout_bars
    d["prior_high"] = h.rolling(n, min_periods=n).max().shift(1)
    d["prior_low"] = lo.rolling(n, min_periods=n).min().shift(1)
    d["breakout_up"] = (c > d["prior_high"]).astype(int)
    d["breakout_dn"] = (c < d["prior_low"]).astype(int)
    d["vol_spike"] = (d["vol_ratio_20"] >= p.vol_spike).astype(int)

    # --- higher timeframe trend, joined on completed bars only
    hb = resample(df1, p.trend_tf)
    if len(hb) >= 5:
        ema = hb["close"].ewm(span=20, adjust=False, min_periods=5).mean()
        hf = pd.DataFrame({
            "close_time": hb.index + pd.Timedelta(minutes=p.trend_tf),
            "trend_15": _ratio(hb["close"], ema).to_numpy(),
            "ret_15m_1": _ratio(hb["close"], hb["close"].shift(1)).to_numpy(),
        })
        idx = d.index
        d = pd.merge_asof(d.reset_index(), hf, on="close_time", direction="backward").set_index("ts")
        d.index = idx
    else:
        d["trend_15"] = np.nan
        d["ret_15m_1"] = np.nan

    # --- market (NIFTY)
    if market1 is not None and len(market1) > 0:
        m = resample(market1, tf)
        mday = m.index.normalize()
        m_open = m["open"].groupby(mday).transform("first")
        mf = pd.DataFrame({
            "nifty_ret_open": _ratio(m["close"], m_open),
            "nifty_ret_3": _ratio(m["close"], m["close"].shift(3)),
        }, index=m.index)
        d = d.join(mf, how="left")
    else:
        d["nifty_ret_open"] = np.nan
        d["nifty_ret_3"] = np.nan

    return d


def find_setups(f: pd.DataFrame, p: StrategyParams | None = None) -> pd.DataFrame:
    """Filter feature rows to 'something is happening' bars and assign a side (+1 long / -1 short)."""
    p = p or StrategyParams()
    if f.empty:
        return f.assign(side=pd.Series(dtype=int))
    ct = f["close_time"]
    hhmm = ct.dt.hour * 60 + ct.dt.minute
    in_win = pd.Series(False, index=f.index)
    for a, b in p.windows:
        in_win |= (hhmm >= hhmm_to_min(a)) & (hhmm < hhmm_to_min(b))

    vol_ok = f["vol_ratio_20"] >= p.breakout_min_vol
    spike = f["vol_spike"] == 1
    long_ = ((f["breakout_up"] == 1) & vol_ok) | (spike & (f["close"] > f["open"]) & (f["close"] > f["vwap"]))
    short_ = ((f["breakout_dn"] == 1) & vol_ok) | (spike & (f["close"] < f["open"]) & (f["close"] < f["vwap"]))
    target_pct = p.target_atr * f["atr"] / f["close"] * 100
    ok = in_win & f["atr"].notna() & (target_pct >= p.min_target_pct)

    side = np.where(long_ & ~short_, 1, np.where(short_ & ~long_, -1, 0))
    mask = ok.to_numpy() & (side != 0)
    s = f.loc[mask].copy()
    s["side"] = side[mask]
    return add_directional(s)


def add_directional(s: pd.DataFrame) -> pd.DataFrame:
    for col in DIRECTIONAL:
        s[f"{col}_dir"] = s[col] * s["side"]
    return s


def model_matrix(s: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    cols = features or FEATURES
    X = s.reindex(columns=cols).astype(float)
    return X.replace([np.inf, -np.inf], np.nan)
