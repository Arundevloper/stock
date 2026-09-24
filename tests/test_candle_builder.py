from core.timeutil import at_time
from reader.candle_builder import CandleBuilder

T0 = at_time("2026-03-02", "09:15")


def test_ticks_become_minute_bars_with_volume_deltas():
    b = CandleBuilder(grace_seconds=2)
    b.on_tick("NSE:X", T0 - 300, 99.0, cum_volume=500)      # pre-open: sets baseline only
    b.on_tick("NSE:X", T0 + 1, 100.0, cum_volume=600)
    b.on_tick("NSE:X", T0 + 20, 101.5, cum_volume=900)
    b.on_tick("NSE:X", T0 + 50, 99.5, cum_volume=1000)
    b.on_tick("NSE:X", T0 + 61, 100.2, cum_volume=1100)     # next minute closes the first bar
    out = b.flush_due(T0 + 62)
    assert out == [{"symbol": "NSE:X", "time": T0, "open": 100.0, "high": 101.5, "low": 99.5,
                    "close": 99.5, "volume": 500}]
    out = b.flush_due(T0 + 125)
    assert out[0]["time"] == T0 + 60 and out[0]["volume"] == 100


def test_flush_waits_for_grace_and_ignores_late_ticks():
    b = CandleBuilder(grace_seconds=2)
    b.on_tick("NSE:X", T0 + 5, 100.0, cum_volume=10)
    assert b.flush_due(T0 + 61) == []
    b.on_tick("NSE:X", T0 + 70, 101.0, cum_volume=20)
    b.on_tick("NSE:X", T0 + 30, 50.0, cum_volume=15)        # late tick for a closed minute
    bars = b.flush_due(T0 + 200)
    assert [x["time"] for x in bars] == [T0, T0 + 60]
    assert all(x["low"] >= 100 for x in bars)


def test_depth_extra_and_index_without_volume():
    b = CandleBuilder()
    b.on_tick("NSE:NIFTY 50", T0 + 3, 24000.0)
    b.on_tick("NSE:X", T0 + 3, 10.0, 100, {"bid": 9.95, "ask": 10.05})
    bars = {x["symbol"]: x for x in b.flush_due(T0 + 70)}
    assert bars["NSE:NIFTY 50"]["volume"] == 0
    assert bars["NSE:X"]["bid"] == 9.95 and bars["NSE:X"]["ask"] == 10.05


def test_outside_session_ignored():
    b = CandleBuilder()
    b.on_tick("NSE:X", at_time("2026-03-02", "15:31"), 10.0, 100)
    assert b.flush_due(at_time("2026-03-02", "16:00")) == []
