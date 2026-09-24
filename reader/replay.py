"""Replay a stored session through the backend exactly like the live reader would.

    python -m reader.replay --date 2026-09-23 --speed 5 --reset

--speed N   candle-minutes per real second (5 = a full session in ~75 s, 0 = as fast as possible)
--reset     delete that day's signals / paper trades first so the replay is repeatable

The engine only ever looks at candles before the current bar close, so a replay of a day that is
already in the database produces the same signals it would have produced live.
"""
from __future__ import annotations

import argparse
import logging
import time
from itertools import groupby

import httpx
from sqlalchemy import select

from core import db
from core.config import get_settings
from core.timeutil import at_time, day_bounds, to_ist
from reader.poster import Poster

log = logging.getLogger("replay")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True)
    ap.add_argument("--speed", type=float, default=5)
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--no-scan", action="store_true", help="don't trigger the 09:31 watchlist scan")
    args = ap.parse_args(argv)
    s = get_settings()
    logging.basicConfig(level=s.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    d0, d1 = day_bounds(args.date)
    if args.reset:
        db.delete_trading_records(d0, d1)
        log.info("Deleted signals / paper trades for %s", args.date)

    with db.engine.connect() as conn:
        rows = conn.execute(select(db.Candle).where(db.Candle.time >= d0, db.Candle.time < d1)
                            .order_by(db.Candle.time, db.Candle.symbol)).mappings().all()
    if not rows:
        raise SystemExit(f"No candles stored for {args.date}")
    log.info("Replaying %d candles for %s at speed %s", len(rows), args.date, args.speed)

    poster = Poster()
    auth = (s.app_username, s.app_password) if s.app_password else None
    scan_at = at_time(args.date, "09:30")
    scanned = args.no_scan
    total_signals = 0
    for t, grp in groupby(rows, key=lambda r: r["time"]):
        batch = [{k: v for k, v in dict(r).items() if v is not None} for r in grp]
        res = poster.post_now(batch)
        total_signals += res.get("signals", 0)
        if res.get("signals"):
            log.info("%s  %d signal(s)", to_ist(t).strftime("%H:%M"), res["signals"])
        if not scanned and t + 60 >= scan_at:
            r = httpx.post(s.backend_url.rstrip("/") + f"/api/watchlist/scan?day={args.date}", auth=auth, timeout=60)
            log.info("Watchlist scan: %s symbols", len(r.json().get("entries", [])) if r.is_success else r.text)
            scanned = True
        if args.speed > 0:
            time.sleep(1 / args.speed)
    log.info("Replay finished: %d signals", total_signals)


if __name__ == "__main__":
    main()
