"""Live reader: Zerodha KiteTicker (MODE_FULL) -> 1-min candles -> POST /api/ingest/candles.

    python -m reader.kite_reader

Run it under systemd / docker with restart=always. It exits (and gets restarted) when the daily
access token changes or expires, because KiteTicker's twisted reactor cannot be restarted in-process.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime

from core.config import get_settings
from core.instruments import ensure_instruments, reader_symbols, token_map
from core.kite import get_kite, load_token, token_status
from reader.candle_builder import CandleBuilder
from reader.poster import Poster

log = logging.getLogger("reader")


def tick_time(t: dict) -> float:
    """Exchange timestamp if sane, else wall clock. kiteconnect builds naive datetimes with
    datetime.fromtimestamp(), so .timestamp() round-trips them regardless of the server TZ."""
    now = time.time()
    ts = t.get("exchange_timestamp") or t.get("last_trade_time")
    if isinstance(ts, datetime):
        e = ts.timestamp()
        if abs(e - now) < 120:
            return e
    return now


def depth_extra(t: dict) -> dict:
    depth = t.get("depth") or {}
    buy, sell = depth.get("buy") or [], depth.get("sell") or []
    bid = buy[0].get("price") if buy else None
    ask = sell[0].get("price") if sell else None
    return {
        "bid": bid or None, "ask": ask or None,
        "buy_qty": t.get("total_buy_quantity"), "sell_qty": t.get("total_sell_quantity"),
    }


def wait_for_token() -> None:
    s = get_settings()
    while not token_status()["valid"]:
        log.info("Waiting for today's Kite login (open %s/auth/login or run python -m auth.login)", s.public_url)
        time.sleep(30)


def main() -> None:
    from kiteconnect import KiteTicker

    s = get_settings()
    logging.basicConfig(level=s.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    wait_for_token()
    kite = get_kite()
    ensure_instruments(kite)
    sym_tokens = token_map(reader_symbols())
    if not sym_tokens:
        raise SystemExit("No instruments resolved - check config/universe.txt")
    by_token = {tok: sym for sym, tok in sym_tokens.items()}
    tokens = list(by_token)
    log.info("Streaming %d instruments", len(tokens))

    builder = CandleBuilder()
    poster = Poster()
    token_mtime = s.token_file.stat().st_mtime
    state = {"fatal": None, "ticks": 0}

    kws = KiteTicker(s.kite_api_key, load_token()["access_token"], reconnect=True,
                     reconnect_max_tries=300, reconnect_max_delay=30)

    def on_ticks(ws, ticks):
        for t in ticks:
            sym = by_token.get(t.get("instrument_token"))
            if sym is None or not t.get("last_price"):
                continue
            state["ticks"] += 1
            builder.on_tick(sym, tick_time(t), float(t["last_price"]), t.get("volume_traded"),
                            depth_extra(t) if "depth" in t else None)

    def on_connect(ws, response):
        log.info("Connected; subscribing %d tokens in MODE_FULL", len(tokens))
        ws.subscribe(tokens)
        ws.set_mode(ws.MODE_FULL, tokens)

    def on_close(ws, code, reason):
        log.warning("Websocket closed: %s %s", code, reason)

    def on_error(ws, code, reason):
        log.error("Websocket error: %s %s", code, reason)
        if code == 403 or "403" in str(reason):
            state["fatal"] = "access token rejected (403)"

    def on_reconnect(ws, attempts):
        log.warning("Reconnecting (attempt %d)", attempts)

    def on_noreconnect(ws):
        state["fatal"] = "gave up reconnecting"

    kws.on_ticks = on_ticks
    kws.on_connect = on_connect
    kws.on_close = on_close
    kws.on_error = on_error
    kws.on_reconnect = on_reconnect
    kws.on_noreconnect = on_noreconnect
    kws.connect(threaded=True)

    last_check = last_log = time.time()
    while True:
        time.sleep(1)
        now = time.time()
        poster.submit(builder.flush_due(now))
        if state["fatal"]:
            log.error("Fatal: %s - exiting for restart", state["fatal"])
            time.sleep(30)
            os._exit(1)
        if now - last_check > 30:
            last_check = now
            try:
                if s.token_file.stat().st_mtime != token_mtime:
                    log.info("Access token changed - exiting for restart")
                    os._exit(0)
            except FileNotFoundError:
                pass
            if not token_status()["valid"]:
                log.info("Access token expired - exiting for restart")
                os._exit(0)
        if now - last_log > 300:
            last_log = now
            log.info("ticks in last 5 min: %d", state["ticks"])
            state["ticks"] = 0


if __name__ == "__main__":
    main()
