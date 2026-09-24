"""Send the daily paper-trading P&L summary to Telegram (cron alternative to the scheduler).

    python -m scripts.eod_summary [--date YYYY-MM-DD]
"""
import argparse
import time

from sqlalchemy import func, select

from backend import notify, paper
from core import db
from core.db import PaperTrade
from core.timeutil import day_str


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date")
    args = ap.parse_args(argv)
    day = args.date or day_str()
    with db.SessionLocal() as s:
        stats = paper.day_stats(s, day)
        cum = s.scalar(select(func.coalesce(func.sum(PaperTrade.net_pnl), 0)).where(PaperTrade.status == "closed"))
    text = notify.daily_summary_text(day, stats, float(cum or 0))
    print(text)
    notify.send(text)
    time.sleep(3)  # let the background sender finish


if __name__ == "__main__":
    main()
