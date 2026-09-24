"""Lookahead / consistency tests for the shared feature module."""
import numpy as np
import pandas as pd

from core.params import StrategyParams
from ml.features import BASE_FEATURES, build_features, find_setups

CHECK = BASE_FEATURES + ["atr", "vwap", "prior_high", "prior_low"]


def _assert_rows_equal(a: pd.Series, b: pd.Series, cols, rtol=1e-9):
    for c in cols:
        x, y = float(a[c]), float(b[c])
        if np.isnan(x) and np.isnan(y):
            continue
        assert np.isclose(x, y, rtol=rtol, atol=1e-12), f"{c}: full={x} truncated={y} at {a.name}"


def test_no_lookahead(stock, market):
    """Features for bar t must be identical whether or not the data after t exists."""
    p = StrategyParams()
    full = build_features(stock, market, p)
    rng = np.random.default_rng(0)
    cuts = rng.choice(np.arange(len(full) // 2, len(full)), size=30, replace=False)
    for i in sorted(cuts):
        ct = full["close_time"].iloc[i]
        part = build_features(stock[stock.index < ct], market[market.index < ct], p)
        assert part.index[-1] == full.index[i]
        _assert_rows_equal(full.iloc[i], part.iloc[-1], CHECK)


def test_live_lookback_matches_training(stock, market):
    """The live engine only loads ~40 calendar days; its last row must match full history."""
    p = StrategyParams()
    full = build_features(stock, market, p)
    last_ts = stock.index[-1]
    start = last_ts - pd.Timedelta(days=40)
    part = build_features(stock[stock.index >= start], market[market.index >= start], p)
    _assert_rows_equal(full.iloc[-1], part.iloc[-1], CHECK, rtol=1e-6)


def test_partial_higher_tf_bar_not_used(stock, market):
    """A 5-min bar closing at 09:40 must use the 15-min bar that closed at 09:30, not 09:45."""
    p = StrategyParams()
    f = build_features(stock, market, p)
    day = f.index[-1].normalize()
    row = f.loc[day + pd.Timedelta(hours=9, minutes=35)]
    ref = build_features(stock[stock.index < day + pd.Timedelta(hours=9, minutes=40)], market, p).iloc[-1]
    assert np.isclose(row["trend_15"], ref["trend_15"])


def test_setups_have_side_and_window(stock, market):
    p = StrategyParams(min_target_pct=0.0)
    s = find_setups(build_features(stock, market, p), p)
    assert len(s) > 0
    assert set(s["side"].unique()) <= {1, -1}
    mins = s["close_time"].dt.hour * 60 + s["close_time"].dt.minute
    in_win = ((mins >= 570) & (mins < 690)) | ((mins >= 810) & (mins < 900))
    assert in_win.all()
    assert (s["ret_1_dir"] == s["ret_1"] * s["side"]).all()
