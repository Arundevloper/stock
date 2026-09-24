"""Database models and helpers. SQLite by default; set DATABASE_URL for PostgreSQL/TimescaleDB."""
from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from contextlib import contextmanager

import pandas as pd
from sqlalchemy import (
    BigInteger, Float, ForeignKey, Integer, String, Text, create_engine, delete, event, func, select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from core.config import get_settings
from core.timeutil import IST


class Base(DeclarativeBase):
    pass


class Candle(Base):
    """1-minute bars. `time` = bar start, epoch seconds. Higher timeframes are resampled on the fly."""
    __tablename__ = "candles"
    symbol: Mapped[str] = mapped_column(String(48), primary_key=True)
    time: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[int] = mapped_column(BigInteger, default=0)
    # last depth snapshot of the minute (live reader only; historical backfill leaves these NULL)
    bid: Mapped[float | None] = mapped_column(Float, nullable=True)
    ask: Mapped[float | None] = mapped_column(Float, nullable=True)
    buy_qty: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    sell_qty: Mapped[int | None] = mapped_column(BigInteger, nullable=True)


class Signal(Base):
    __tablename__ = "signals"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(48), index=True)
    time: Mapped[int] = mapped_column(BigInteger, index=True)   # decision time (signal bar close)
    side: Mapped[str] = mapped_column(String(8))                 # long / short
    prob: Mapped[float | None] = mapped_column(Float, nullable=True)
    entry: Mapped[float] = mapped_column(Float)                  # estimate (bar close); fill = next bar open
    stop: Mapped[float] = mapped_column(Float)
    target: Mapped[float] = mapped_column(Float)
    atr: Mapped[float] = mapped_column(Float)
    qty: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), default="new")  # new / blocked / filled / closed / cancelled
    reason: Mapped[str | None] = mapped_column(String(120), nullable=True)
    outcome: Mapped[str | None] = mapped_column(String(16), nullable=True)
    bid: Mapped[float | None] = mapped_column(Float, nullable=True)
    ask: Mapped[float | None] = mapped_column(Float, nullable=True)
    features: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON snapshot for debugging
    mode: Mapped[str] = mapped_column(String(8), default="paper")
    created_at: Mapped[int] = mapped_column(BigInteger)

    def to_dict(self) -> dict:
        d = {c.name: getattr(self, c.name) for c in self.__table__.columns if c.name != "features"}
        return d


class PaperTrade(Base):
    __tablename__ = "paper_trades"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    signal_id: Mapped[int | None] = mapped_column(ForeignKey("signals.id"), nullable=True)
    symbol: Mapped[str] = mapped_column(String(48), index=True)
    side: Mapped[str] = mapped_column(String(8))
    qty: Mapped[int] = mapped_column(Integer)
    atr: Mapped[float] = mapped_column(Float)
    signal_time: Mapped[int] = mapped_column(BigInteger, index=True)
    status: Mapped[str] = mapped_column(String(16), default="pending")  # pending / open / closed / cancelled
    entry_time: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    entry_raw: Mapped[float | None] = mapped_column(Float, nullable=True)    # bar open
    entry_price: Mapped[float | None] = mapped_column(Float, nullable=True)  # incl. slippage
    stop: Mapped[float | None] = mapped_column(Float, nullable=True)
    target: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_time: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_reason: Mapped[str | None] = mapped_column(String(16), nullable=True)  # target/stop/timeout/squareoff/manual
    gross_pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    costs: Mapped[float | None] = mapped_column(Float, nullable=True)
    net_pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    r_multiple: Mapped[float | None] = mapped_column(Float, nullable=True)
    bid: Mapped[float | None] = mapped_column(Float, nullable=True)
    ask: Mapped[float | None] = mapped_column(Float, nullable=True)

    def to_dict(self) -> dict:
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class WatchlistEntry(Base):
    __tablename__ = "watchlist"
    date: Mapped[str] = mapped_column(String(10), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(48), primary_key=True)
    score: Mapped[float] = mapped_column(Float, default=0)
    gap_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    rvol: Mapped[float | None] = mapped_column(Float, nullable=True)
    turnover_cr: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(16), default="scan")

    def to_dict(self) -> dict:
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class AppState(Base):
    __tablename__ = "app_state"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(Text)


# ---------------------------------------------------------------- engine / session

_settings = get_settings()
_is_sqlite = _settings.database_url.startswith("sqlite")
engine = create_engine(
    _settings.database_url,
    connect_args={"check_same_thread": False, "timeout": 30} if _is_sqlite else {},
    pool_pre_ping=True,
)

if _is_sqlite:
    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_conn, _):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA busy_timeout=30000")
        cur.close()

SessionLocal = sessionmaker(engine, expire_on_commit=False)


def init_db() -> None:
    Base.metadata.create_all(engine)


@contextmanager
def session_scope() -> Iterable[Session]:
    s = SessionLocal()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def _insert_fn():
    if engine.dialect.name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    else:
        from sqlalchemy.dialects.sqlite import insert
    return insert


# ---------------------------------------------------------------- candles

CANDLE_COLS = ["symbol", "time", "open", "high", "low", "close", "volume", "bid", "ask", "buy_qty", "sell_qty"]


def upsert_candles(rows: Sequence[dict]) -> int:
    """Insert or update 1-min candles. Depth columns are only overwritten by non-NULL values,
    so a historical backfill never erases depth captured live."""
    if not rows:
        return 0
    clean = [{c: r.get(c) for c in CANDLE_COLS} for r in rows]
    for r in clean:
        r["volume"] = int(r["volume"] or 0)
    insert = _insert_fn()
    stmt = insert(Candle)
    ex = stmt.excluded
    stmt = stmt.on_conflict_do_update(
        index_elements=["symbol", "time"],
        set_={
            "open": ex.open, "high": ex.high, "low": ex.low, "close": ex.close, "volume": ex.volume,
            "bid": func.coalesce(ex.bid, Candle.bid), "ask": func.coalesce(ex.ask, Candle.ask),
            "buy_qty": func.coalesce(ex.buy_qty, Candle.buy_qty),
            "sell_qty": func.coalesce(ex.sell_qty, Candle.sell_qty),
        },
    )
    with engine.begin() as conn:
        for i in range(0, len(clean), 2000):
            conn.execute(stmt, clean[i:i + 2000])
    return len(clean)


def load_candles(symbol: str, start: int | None = None, end: int | None = None,
                 with_depth: bool = False) -> pd.DataFrame:
    """1-min candles for `symbol` with start <= time < end, indexed by IST bar-start datetime `ts`."""
    cols = [Candle.time, Candle.open, Candle.high, Candle.low, Candle.close, Candle.volume]
    if with_depth:
        cols += [Candle.bid, Candle.ask, Candle.buy_qty, Candle.sell_qty]
    q = select(*cols).where(Candle.symbol == symbol)
    if start is not None:
        q = q.where(Candle.time >= start)
    if end is not None:
        q = q.where(Candle.time < end)
    q = q.order_by(Candle.time)
    with engine.connect() as conn:
        df = pd.read_sql(q, conn)
    return frame_from_rows(df)


def frame_from_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        idx = pd.DatetimeIndex([], tz=IST, name="ts")
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"], index=idx)
    df = df.copy()
    df.index = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(IST)
    df.index.name = "ts"
    df["volume"] = df["volume"].astype(float)
    return df


def list_symbols() -> list[str]:
    with engine.connect() as conn:
        return [r[0] for r in conn.execute(select(Candle.symbol).distinct().order_by(Candle.symbol))]


def last_candle(symbol: str) -> dict | None:
    with engine.connect() as conn:
        r = conn.execute(
            select(Candle).where(Candle.symbol == symbol).order_by(Candle.time.desc()).limit(1)
        ).mappings().first()
    return dict(r) if r else None


def latest_candle_time() -> int | None:
    with engine.connect() as conn:
        return conn.execute(select(func.max(Candle.time))).scalar()


def delete_trading_records(start: int, end: int) -> None:
    """Remove signals / paper trades in [start, end) — used by replay --reset."""
    with session_scope() as s:
        s.execute(delete(PaperTrade).where(PaperTrade.signal_time >= start, PaperTrade.signal_time < end))
        s.execute(delete(Signal).where(Signal.time >= start, Signal.time < end))


# ---------------------------------------------------------------- app state (key/value)

def get_state(key: str, default=None):
    with SessionLocal() as s:
        row = s.get(AppState, key)
        return json.loads(row.value) if row else default


def set_state(key: str, value) -> None:
    with session_scope() as s:
        row = s.get(AppState, key)
        if row:
            row.value = json.dumps(value)
        else:
            s.add(AppState(key=key, value=json.dumps(value)))
