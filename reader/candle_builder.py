"""Tick -> 1-minute candle aggregation.

Bar volume = cumulative `volume_traded` at the bar's last tick minus the cumulative volume at the
previous tick (ticks before 09:15, e.g. the pre-open auction, only set the baseline).
Bars are emitted by `flush_due()` a couple of seconds after the minute ends, so all symbols of a
minute go to the backend in one batch.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

from core.timeutil import in_session


@dataclass
class Bar:
    symbol: str
    time: int
    open: float
    high: float
    low: float
    close: float
    vol_start: float | None
    vol_end: float | None
    extra: dict = field(default_factory=dict)

    def to_candle(self) -> dict:
        vol = 0
        if self.vol_start is not None and self.vol_end is not None:
            vol = max(0, int(self.vol_end - self.vol_start))
        return {"symbol": self.symbol, "time": self.time, "open": self.open, "high": self.high,
                "low": self.low, "close": self.close, "volume": vol, **self.extra}


class CandleBuilder:
    def __init__(self, grace_seconds: float = 2.0) -> None:
        self.grace = grace_seconds
        self.bars: dict[str, Bar] = {}
        self.last_cum: dict[str, float] = {}
        self.ready: list[dict] = []
        self._lock = threading.Lock()

    def on_tick(self, symbol: str, ts: float, ltp: float, cum_volume: float | None = None,
                extra: dict | None = None) -> None:
        minute = int(ts // 60) * 60
        with self._lock:
            prev_cum = self.last_cum.get(symbol)
            if cum_volume is not None:
                self.last_cum[symbol] = cum_volume
            if not in_session(minute):
                return
            bar = self.bars.get(symbol)
            if bar and minute < bar.time:
                return  # late tick for an already-closed minute
            if bar and minute > bar.time:
                self.ready.append(bar.to_candle())
                bar = None
            if bar is None:
                start = prev_cum if prev_cum is not None else cum_volume
                self.bars[symbol] = Bar(symbol, minute, ltp, ltp, ltp, ltp, start, cum_volume, dict(extra or {}))
                return
            bar.high = max(bar.high, ltp)
            bar.low = min(bar.low, ltp)
            bar.close = ltp
            if cum_volume is not None:
                bar.vol_end = cum_volume
                if bar.vol_start is None:
                    bar.vol_start = cum_volume
            if extra:
                bar.extra.update(extra)

    def flush_due(self, now: float) -> list[dict]:
        """Close every bar whose minute ended more than `grace` seconds ago; return ready candles."""
        with self._lock:
            for sym, bar in list(self.bars.items()):
                if bar.time + 60 + self.grace <= now:
                    self.ready.append(bar.to_candle())
                    del self.bars[sym]
            out, self.ready = self.ready, []
        return out
