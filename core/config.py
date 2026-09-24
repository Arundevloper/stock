"""Runtime settings, read from environment variables / .env at the project root."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent

MARKET_SYMBOL = "NSE:NIFTY 50"  # index used for market-context features


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", extra="ignore", protected_namespaces=()
    )

    # --- Zerodha Kite Connect ---
    kite_api_key: str = ""
    kite_api_secret: str = ""

    # --- storage ---
    database_url: str = f"sqlite:///{ROOT / 'data' / 'trading.db'}"
    data_dir: Path = ROOT / "data"
    model_path: Path = ROOT / "models" / "model.pkl"
    report_path: Path = ROOT / "models" / "report.json"
    universe_file: Path = ROOT / "config" / "universe.txt"

    # --- backend / web ---
    backend_url: str = "http://127.0.0.1:8000"  # used by the reader to POST candles
    public_url: str = "http://127.0.0.1:8000"   # used in Telegram links (login reminder)
    ingest_key: str = "change-me"               # shared secret between reader and backend
    app_username: str = ""                      # optional HTTP basic auth for the web UI
    app_password: str = ""

    # --- trading ---
    paper_mode: bool = True
    signal_mode: str = "model"          # "model" = ML filter, "rule" = every setup (base-rule paper test)
    signal_threshold: float | None = None  # override the threshold stored in the model bundle
    capital: float = 100_000
    risk_per_trade: float = 0.01        # 1% of capital risked per trade
    mis_leverage: float = 5.0           # caps qty at capital * leverage / price
    watchlist_mode: str = "scan"        # "scan" = 9:31 morning scan, "universe" = trade the whole universe
    watchlist_size: int = 25
    min_turnover_cr: float = 2.0        # first-15-min turnover filter for the scan, in crore rupees
    live_lookback_days: int = 40        # calendar days of history loaded for live features

    # --- telegram ---
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    telegram_trade_updates: bool = True

    # --- ops ---
    scheduler_enabled: bool = True
    retrain_enabled: bool = True
    retrain_time: str = "20:00"
    backfill_after_close: bool = True
    stale_data_minutes: int = 3
    healthcheck_url: str = ""
    sentry_dsn: str = ""
    log_level: str = "INFO"

    @property
    def token_file(self) -> Path:
        return self.data_dir / "access_token.json"

    @property
    def telegram_enabled(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    (s.data_dir / "logs").mkdir(exist_ok=True)
    s.model_path.parent.mkdir(parents=True, exist_ok=True)
    return s
