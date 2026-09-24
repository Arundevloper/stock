"""Run the morning watchlist scan from cron instead of the built-in scheduler.

    python -m scripts.watchlist [--date YYYY-MM-DD]
"""
import argparse
import logging

from backend import watchlist
from core import db


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date")
    args = ap.parse_args(argv)
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    db.init_db()
    for e in watchlist.scan(args.date):
        print(f"{e['symbol']:<20} score={e['score']:.3f} gap={e['gap_pct']}% rvol={e['rvol']} turnover={e['turnover_cr']}cr")


if __name__ == "__main__":
    main()
