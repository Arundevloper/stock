"""Bulk minute history from Kite (needs the Connect subscription + today's token).

    python -m backfill.historical --days 365              # universe + NIFTY 50, last year
    python -m backfill.historical --days 30 --symbols NSE:RELIANCE,NSE:TCS
    python -m backfill.historical --days 1                # today's session (run after close)

Kite serves at most 60 days of minute data per request, so the range is fetched in chunks.
Existing candles are updated in place (official historical bars replace tick-built ones).
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timedelta

from core import db
from core.instruments import ensure_instruments, normalize, reader_symbols, token_map
from core.kite import get_kite, ist_from_kite
from core.timeutil import IST, epoch

log = logging.getLogger("backfill")
CHUNK_DAYS = 60
REQUEST_GAP = 0.4  # historical API allows ~3 requests/second


def fetch_symbol(kite, symbol: str, token: int, start: datetime, end: datetime) -> int:
    total = 0
    cur = start
    while cur < end:
        chunk_end = min(cur + timedelta(days=CHUNK_DAYS), end)
        for attempt in range(4):
            try:
                data = kite.historical_data(token, cur, chunk_end, "minute")
                break
            except Exception as e:  # network / rate limit
                log.warning("%s %s..%s failed (%s), retrying", symbol, cur.date(), chunk_end.date(), e)
                time.sleep(2 * (attempt + 1))
        else:
            data = []
        rows = [{
            "symbol": symbol, "time": epoch(ist_from_kite(r["date"])),
            "open": r["open"], "high": r["high"], "low": r["low"], "close": r["close"],
            "volume": r.get("volume", 0),
        } for r in data]
        total += db.upsert_candles(rows)
        cur = chunk_end
        time.sleep(REQUEST_GAP)
    return total


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--symbols", help="comma separated (default: universe + NIFTY 50)")
    args = ap.parse_args(argv)
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    db.init_db()

    kite = get_kite()
    ensure_instruments(kite)
    symbols = [normalize(x) for x in args.symbols.split(",")] if args.symbols else reader_symbols()
    tokens = token_map(symbols)
    end = datetime.now(IST).replace(second=0, microsecond=0)
    start = (end - timedelta(days=args.days)).replace(hour=9, minute=0)
    log.info("Backfilling %d symbols from %s to %s", len(tokens), start.date(), end.date())
    for i, (sym, tok) in enumerate(tokens.items(), 1):
        n = fetch_symbol(kite, sym, tok, start, end)
        log.info("[%d/%d] %-20s %7d candles", i, len(tokens), sym, n)


if __name__ == "__main__":
    main()
