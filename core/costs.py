"""Approximate Zerodha NSE equity intraday (MIS) charges. Verify against Zerodha's brokerage
calculator / Console from time to time — rates change."""

BROKERAGE_PCT = 0.0003      # 0.03% or Rs 20 per executed order, whichever is lower
BROKERAGE_CAP = 20.0
STT_SELL_PCT = 0.00025      # 0.025% on sell side
EXCHANGE_PCT = 0.0000297    # NSE transaction charge
SEBI_PCT = 0.000001         # Rs 10 / crore
GST_PCT = 0.18              # on brokerage + exchange + SEBI
STAMP_BUY_PCT = 0.00003     # 0.003% on buy side


def intraday_costs(buy_value: float, sell_value: float) -> float:
    """Total charges in rupees for one round trip."""
    brokerage = min(BROKERAGE_CAP, BROKERAGE_PCT * buy_value) + min(BROKERAGE_CAP, BROKERAGE_PCT * sell_value)
    turnover = buy_value + sell_value
    exchange = EXCHANGE_PCT * turnover
    sebi = SEBI_PCT * turnover
    gst = GST_PCT * (brokerage + exchange + sebi)
    stt = STT_SELL_PCT * sell_value
    stamp = STAMP_BUY_PCT * buy_value
    return round(brokerage + exchange + sebi + gst + stt + stamp, 2)


def slip(price: float, side: int, is_entry: bool, slippage_pct: float) -> float:
    """Adverse slippage. side=+1 long, -1 short. Entry buys (long) pay up; exits sell lower."""
    direction = side if is_entry else -side
    return price * (1 + direction * slippage_pct / 100)


def trade_pnl(side: int, qty: int, entry_fill: float, exit_fill: float) -> tuple[float, float, float]:
    """(gross, costs, net) for a closed trade, fills already including slippage."""
    gross = (exit_fill - entry_fill) * qty * side
    if side > 0:
        costs = intraday_costs(entry_fill * qty, exit_fill * qty)
    else:
        costs = intraday_costs(exit_fill * qty, entry_fill * qty)
    return round(gross, 2), costs, round(gross - costs, 2)


def position_size(entry: float, stop: float, capital: float, risk_pct: float, leverage: float) -> int:
    """qty = (capital * risk) / (entry - stop), capped by MIS buying power."""
    per_share_risk = abs(entry - stop)
    if per_share_risk <= 0 or entry <= 0:
        return 0
    qty_risk = (capital * risk_pct) / per_share_risk
    qty_cap = (capital * leverage) / entry
    return int(max(0, min(qty_risk, qty_cap)))
