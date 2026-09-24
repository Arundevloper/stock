"""End-to-end: history in the DB, one session ingested through the API in rule mode."""
import os

import pytest

from tests.conftest import synthetic_1min


@pytest.fixture(scope="module")
def client():
    os.environ["SIGNAL_MODE"] = "rule"
    os.environ["WATCHLIST_MODE"] = "universe"
    from core.config import get_settings
    get_settings.cache_clear()
    from fastapi.testclient import TestClient

    from backend.main import app
    from core import db
    db.init_db()
    with TestClient(app) as c:
        yield c


def _rows(df, symbol):
    return [{"symbol": symbol, "time": int(t), "open": r.open, "high": r.high, "low": r.low,
             "close": r.close, "volume": int(r.volume)} for t, r in zip(df["time"], df.itertuples())]


def test_session_through_api(client):
    from core import db
    from core.config import MARKET_SYMBOL

    stock = synthetic_1min(days=30, seed=5)
    market = synthetic_1min(days=30, seed=6, start_price=24000, volume=False)
    last_day = stock.index[-1].normalize()
    hist = stock.index < last_day
    db.upsert_candles(_rows(stock[hist], "NSE:TEST") + _rows(market[hist], MARKET_SYMBOL))

    today = _rows(stock[~hist], "NSE:TEST")
    mk = _rows(market[~hist], MARKET_SYMBOL)
    assert client.post("/api/ingest/candles", json={"candles": today[:1]}, headers={"X-Ingest-Key": "nope"}).status_code == 401
    signals = 0
    for a, b in zip(today, mk):
        r = client.post("/api/ingest/candles", json={"candles": [a, b]}, headers={"X-Ingest-Key": "test-key"})
        assert r.status_code == 200, r.text
        signals += r.json()["signals"]
    assert signals > 0, "rule mode should emit at least one signal on a full session"

    trades = client.get("/api/trades?days=400").json()["trades"]
    assert trades and all(t["status"] in ("closed", "cancelled") for t in trades), "all flat after square-off"
    assert len([t for t in trades if t["status"] != "cancelled"]) <= 3

    for path in ("/api/status", "/api/watchlist", "/api/signals", "/api/pnl", "/api/setups", "/api/model",
                 "/api/symbols", "/api/candles?symbol=NSE:TEST&tf=5&days=2", "/health", "/"):
        assert client.get(path).status_code == 200, path
    pnl = client.get("/api/pnl").json()
    assert pnl["summary"]["trades"] == len([t for t in trades if t["status"] == "closed"])
