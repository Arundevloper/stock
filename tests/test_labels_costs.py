import numpy as np
import pandas as pd

from core.costs import intraday_costs, position_size, slip, trade_pnl
from core.params import StrategyParams
from core.timeutil import IST
from ml.labels import label_setups, resolve_path


def _path(prices):
    """prices: list of (o, h, l, c) 1-min bars starting 10:00."""
    idx = pd.date_range(pd.Timestamp("2026-03-02 10:00", tz=IST), periods=len(prices), freq="1min")
    return pd.DataFrame(prices, columns=["open", "high", "low", "close"], index=idx).assign(volume=100)


def _setup(side=1, atr=1.0):
    t = pd.Timestamp("2026-03-02 09:55", tz=IST)
    return pd.DataFrame({"close_time": [t + pd.Timedelta(minutes=5)], "side": [side], "atr": [atr]}, index=[t])


def test_target_first_is_win():
    df = _path([(100, 100.5, 99.8, 100.4), (100.4, 101.6, 100.2, 101.5)])
    out = label_setups(_setup(), df, StrategyParams())
    assert out["label"].iloc[0] == 1 and out["outcome"].iloc[0] == "target"
    assert out["exit_price"].iloc[0] == 101.5  # entry 100 + 1.5 * ATR


def test_stop_first_is_loss():
    df = _path([(100, 100.2, 98.9, 99.0), (99.0, 102, 98.5, 101.8)])
    out = label_setups(_setup(), df, StrategyParams())
    assert out["label"].iloc[0] == 0 and out["outcome"].iloc[0] == "stop"


def test_both_in_same_bar_is_loss():
    df = _path([(100, 101.6, 98.9, 100.0)])
    out = label_setups(_setup(), df, StrategyParams())
    assert out["outcome"].iloc[0] == "stop"


def test_short_side():
    df = _path([(100, 100.2, 98.4, 98.5)])
    out = label_setups(_setup(side=-1), df, StrategyParams())
    assert out["outcome"].iloc[0] == "target" and out["exit_price"].iloc[0] == 98.5


def test_timeout_exits_at_horizon_close():
    df = _path([(100, 100.3, 99.7, 100.1)] * 90)
    p = StrategyParams(horizon_bars=12)
    out = label_setups(_setup(), df, p)
    assert out["outcome"].iloc[0] == "timeout" and out["label"].iloc[0] == 0
    assert out["exit_time"].iloc[0] == pd.Timestamp("2026-03-02 11:00", tz=IST)


def test_gap_through_stop_fills_at_open():
    o = np.array([100.0, 98.0]); h = np.array([100.2, 98.2]); lo = np.array([99.5, 97.5])
    outcome, i, px = resolve_path(1, 100, 99, 101.5, o, h, lo)
    assert outcome == "stop" and i == 1 and px == 98.0


def test_costs_reasonable():
    c = intraday_costs(100_000, 100_500)
    assert 40 < c < 120          # ~0.03-0.06% of turnover for a 1-lakh round trip
    g, costs, net = trade_pnl(1, 10, 100, 110)
    assert g == 100 and net == g - costs
    assert slip(100, 1, True, 0.1) > 100 and slip(100, 1, False, 0.1) < 100
    assert slip(100, -1, True, 0.1) < 100


def test_position_size():
    assert position_size(100, 99, 100_000, 0.01, 5) == 1000        # 1000 risk / 1 per share
    assert position_size(100, 99.99, 100_000, 0.01, 5) == 5000     # capped by leverage
    assert position_size(100, 100, 100_000, 0.01, 5) == 0
