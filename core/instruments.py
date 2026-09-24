"""Instrument master (symbol <-> instrument_token) and the trading universe."""
from __future__ import annotations

import logging
from datetime import date
from functools import lru_cache

import pandas as pd

from core.config import MARKET_SYMBOL, get_settings

log = logging.getLogger(__name__)


def _csv_path():
    return get_settings().data_dir / "instruments_NSE.csv"


def download_instruments(kite=None) -> pd.DataFrame:
    """Download the NSE instrument dump (do this daily - tokens can change)."""
    from core.kite import get_kite

    kite = kite or get_kite()
    df = pd.DataFrame(kite.instruments("NSE"))
    df.to_csv(_csv_path(), index=False)
    _load.cache_clear()
    log.info("Downloaded %d NSE instruments", len(df))
    return df


@lru_cache
def _load() -> pd.DataFrame:
    p = _csv_path()
    if not p.exists():
        return pd.DataFrame(columns=["instrument_token", "tradingsymbol", "segment"])
    df = pd.read_csv(p)
    df["symbol"] = "NSE:" + df["tradingsymbol"].astype(str)
    return df


def instruments_fresh() -> bool:
    p = _csv_path()
    return p.exists() and date.fromtimestamp(p.stat().st_mtime) == date.today()


def ensure_instruments(kite=None) -> pd.DataFrame:
    if not instruments_fresh():
        try:
            return download_instruments(kite)
        except Exception as e:  # stale file is better than nothing
            log.warning("Instrument download failed (%s); using cached file", e)
    return _load()


def token_map(symbols: list[str]) -> dict[str, int]:
    """symbol -> instrument_token. Unknown symbols are logged and skipped."""
    df = _load()
    lookup = dict(zip(df["symbol"], df["instrument_token"]))
    out = {}
    for s in symbols:
        if s in lookup:
            out[s] = int(lookup[s])
        else:
            log.warning("Symbol %s not found in instrument list - skipped", s)
    return out


def normalize(sym: str) -> str:
    sym = sym.strip().upper()
    return sym if ":" in sym else f"NSE:{sym}"


def load_universe() -> list[str]:
    """Symbols from config/universe.txt (one per line, # comments allowed)."""
    p = get_settings().universe_file
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(normalize(line))
    return list(dict.fromkeys(out))


def reader_symbols() -> list[str]:
    """Everything the live reader should stream: universe + market index."""
    return list(dict.fromkeys(load_universe() + [MARKET_SYMBOL]))
