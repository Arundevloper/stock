"""Strategy parameters shared by training, backtest, the live signal engine and paper trading.

The trained model bundle stores the exact params it was trained with, and the live engine uses
those — so the setup filter, label geometry and risk rules can never drift between train and live.
"""
from dataclasses import asdict, dataclass, field, fields


@dataclass
class StrategyParams:
    signal_tf: int = 5            # minutes; signals are evaluated on this bar close
    trend_tf: int = 15            # minutes; higher-timeframe trend filter
    atr_period: int = 14

    # --- trade geometry (also the label definition) ---
    stop_atr: float = 1.0         # stop = entry - stop_atr * ATR   (long)
    target_atr: float = 1.5       # target = entry + target_atr * ATR
    horizon_bars: int = 12        # max hold in signal_tf bars (12 x 5m = 60 min), then exit at market
    squareoff: str = "15:15"      # force exit time

    # --- setup filter ---
    breakout_bars: int = 12       # N-bar high/low breakout
    breakout_min_vol: float = 1.5  # breakout needs volume >= this x 20-bar average
    vol_avg_bars: int = 20
    vol_spike: float = 2.0        # volume spike threshold (x 20-bar average)
    min_target_pct: float = 0.4   # skip setups whose target move is < this % (costs eat small moves)
    windows: list = field(default_factory=lambda: [["09:30", "11:30"], ["13:30", "15:00"]])
    rvol_days: int = 10           # sessions used for same-minute relative volume

    # --- risk rules (backtest + live) ---
    max_trades_per_day: int = 3
    max_losses_per_day: int = 2
    slippage_pct: float = 0.05    # per side, adverse

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "StrategyParams":
        if not d:
            return cls()
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})
