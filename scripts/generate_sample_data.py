"""Generate SYNTHETIC 1-min candles so the whole pipeline (train, backtest, backend, UI, replay)
can be exercised without a Kite subscription.

    python -m scripts.generate_sample_data                 # -> data/demo.db, planted edge
    python -m scripts.generate_sample_data --no-edge       # pure noise: model should find nothing

The data is fake. The optional "planted edge" (volume-burst breakouts continue more often when
they agree with the NIFTY direction) exists only to prove the pipeline can find a real pattern.
Results on this data say NOTHING about real markets. It writes to a separate database by default
so it never mixes with real candles.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta

import numpy as np

MINUTES = 375  # 09:15 .. 15:29


def trading_days(n: int, end: date) -> list[date]:
    out, d = [], end
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return out[::-1]


def u_shape(scale_open: float, scale_close: float) -> np.ndarray:
    t = np.arange(MINUTES)
    return 1 + scale_open * np.exp(-t / 20) + scale_close * np.exp(-(MINUTES - t) / 30)


def make_bars(prev_close: float, rets: np.ndarray, sigma: np.ndarray, gap: float,
              vols: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    closes = prev_close * np.exp(gap + np.cumsum(rets))
    opens = np.empty_like(closes)
    opens[0] = prev_close * np.exp(gap)
    opens[1:] = closes[:-1]
    wick_hi = np.exp(np.abs(rng.normal(0, 0.35, MINUTES)) * sigma)
    wick_lo = np.exp(-np.abs(rng.normal(0, 0.35, MINUTES)) * sigma)
    highs = np.maximum(opens, closes) * wick_hi
    lows = np.minimum(opens, closes) * wick_lo
    tick = lambda x: np.round(np.round(x / 0.05) * 0.05, 2)  # noqa: E731
    return np.column_stack([tick(opens), tick(highs), tick(lows), tick(closes), vols])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="sqlite:///data/demo.db", help="target DATABASE_URL")
    ap.add_argument("--days", type=int, default=160)
    ap.add_argument("--symbols", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--end", help="last session date YYYY-MM-DD (default: last weekday before today)")
    ap.add_argument("--no-edge", action="store_true", help="no planted pattern (pure noise)")
    args = ap.parse_args(argv)

    os.environ["DATABASE_URL"] = args.db  # must be set before core.db is imported
    from core import db
    from core.config import MARKET_SYMBOL
    from core.timeutil import at_time

    db.init_db()
    rng = np.random.default_rng(args.seed)
    end = date.fromisoformat(args.end) if args.end else date.today() - timedelta(days=1)
    days = trading_days(args.days, end)

    syms = [f"NSE:DEMO{i + 1:02d}" for i in range(args.symbols)]
    price = {s: float(rng.uniform(200, 3000)) for s in syms}
    beta = {s: float(rng.uniform(0.8, 1.3)) for s in syms}
    idio = {s: float(rng.uniform(0.0007, 0.0011)) for s in syms}
    base_vol = {s: float(rng.uniform(2e4, 1.5e5)) for s in syms}
    nifty = 24000.0
    u_ret = u_shape(1.5, 0.6)
    u_vol = u_shape(2.0, 1.0)
    total = 0

    for d in days:
        t0 = at_time(d, "09:15")
        times = t0 + 60 * np.arange(MINUTES)
        # --- market
        m_sigma = 0.00035 * u_ret
        m_rets = rng.normal(rng.normal(0, 0.00003), m_sigma)
        m_gap = rng.normal(0, 0.004)
        m_bars = make_bars(nifty, m_rets, m_sigma, m_gap, np.zeros(MINUTES), rng)
        nifty = float(m_bars[-1, 3])
        m_since_open = np.cumsum(m_rets)
        rows = [dict(symbol=MARKET_SYMBOL, time=int(t), open=b[0], high=b[1], low=b[2], close=b[3], volume=0)
                for t, b in zip(times, m_bars)]

        # --- stocks
        for s in syms:
            sigma = idio[s] * u_ret
            drift = np.zeros(MINUTES)
            vmult = np.ones(MINUTES)
            for _ in range(rng.poisson(2.5)):
                t = int(rng.integers(20, 330))
                dirn = rng.choice([-1, 1])
                vmult[t:t + 4] *= rng.uniform(3, 6)
                drift[t:t + 4] += dirn * 0.0012
                if args.no_edge:
                    p_cont = 0.5
                else:
                    p_cont = 0.72 if dirn * m_since_open[t] > 0 else 0.40
                cont = rng.random() < p_cont
                drift[t + 4:t + 44] += dirn * (0.0002 if cont else -0.00015)
            rets = beta[s] * m_rets + rng.normal(drift, sigma)
            vols = np.round(base_vol[s] / 60 * u_vol * vmult * rng.lognormal(0, 0.35, MINUTES))
            bars = make_bars(price[s], rets, sigma, rng.normal(0, 0.006), vols, rng)
            price[s] = float(bars[-1, 3])
            rows += [dict(symbol=s, time=int(t), open=b[0], high=b[1], low=b[2], close=b[3], volume=int(b[4]))
                     for t, b in zip(times, bars)]
        total += db.upsert_candles(rows)

    print(f"Wrote {total:,} synthetic 1-min candles for {len(syms)} symbols + NIFTY, "
          f"{days[0]} .. {days[-1]} into {args.db}")
    print(f"Planted edge: {'NO (pure noise)' if args.no_edge else 'yes'}")
    print(f"\nUse it with:  export DATABASE_URL={args.db}")


if __name__ == "__main__":
    sys.exit(main())
