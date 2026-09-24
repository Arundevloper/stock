import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# tests never touch the real database, model or token
_TMP = Path(tempfile.mkdtemp(prefix="intraday-tests-"))
os.environ.update({
    "DATABASE_URL": f"sqlite:///{_TMP / 'test.db'}",
    "DATA_DIR": str(_TMP),
    "MODEL_PATH": str(_TMP / "model.pkl"),
    "REPORT_PATH": str(_TMP / "report.json"),
    "SCHEDULER_ENABLED": "false",
    "INGEST_KEY": "test-key",
    "TELEGRAM_BOT_TOKEN": "",
    "APP_PASSWORD": "",
})

from core.timeutil import IST  # noqa: E402


def synthetic_1min(days: int = 45, seed: int = 0, start_price: float = 1000.0, volume: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames, price = [], start_price
    d = pd.Timestamp("2026-03-02", tz=IST)
    n = 0
    while n < days:
        if d.weekday() < 5:
            idx = pd.date_range(d + pd.Timedelta(hours=9, minutes=15), periods=375, freq="1min")
            r = rng.normal(0, 0.0012, 375)
            close = price * np.exp(rng.normal(0, 0.005) + np.cumsum(r))
            open_ = np.r_[price * np.exp(rng.normal(0, 0.003)), close[:-1]]
            high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.0005, 375)))
            low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.0005, 375)))
            vol = rng.lognormal(8, 0.8, 375).round() if volume else np.zeros(375)
            frames.append(pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                                        "volume": vol}, index=idx))
            price = close[-1]
            n += 1
        d += pd.Timedelta(days=1)
    df = pd.concat(frames)
    df.index.name = "ts"
    df["time"] = (df.index - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)
    return df


@pytest.fixture(scope="session")
def stock():
    return synthetic_1min(seed=1)


@pytest.fixture(scope="session")
def market():
    return synthetic_1min(seed=2, start_price=24000, volume=False)
