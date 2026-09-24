"""Kite Connect client + daily access-token storage.

Kite access tokens expire every morning (~06:00 IST). The token is stored in data/access_token.json;
the reader watches that file and restarts when it changes.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, time, timedelta

from core.config import get_settings
from core.timeutil import IST, now_ist

log = logging.getLogger(__name__)

TOKEN_EXPIRY = time(6, 0)


class KiteNotReady(RuntimeError):
    pass


def _expiry_after(saved: datetime) -> datetime:
    exp = saved.replace(hour=TOKEN_EXPIRY.hour, minute=TOKEN_EXPIRY.minute, second=0, microsecond=0)
    if saved >= exp:
        exp += timedelta(days=1)
    return exp


def load_token() -> dict | None:
    f = get_settings().token_file
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def token_status() -> dict:
    tok = load_token()
    if not tok:
        return {"valid": False, "user": None, "saved_at": None, "expires_at": None}
    saved = datetime.fromisoformat(tok["saved_at"])
    exp = _expiry_after(saved)
    return {
        "valid": now_ist() < exp,
        "user": tok.get("user_id"),
        "saved_at": saved.isoformat(),
        "expires_at": exp.isoformat(),
    }


def save_token(access_token: str, user_id: str | None = None) -> None:
    f = get_settings().token_file
    f.write_text(json.dumps({
        "access_token": access_token,
        "user_id": user_id,
        "saved_at": now_ist().isoformat(),
    }))
    log.info("Saved Kite access token for %s", user_id)


def kite_configured() -> bool:
    s = get_settings()
    return bool(s.kite_api_key and s.kite_api_secret)


def get_kite(require_token: bool = True):
    """KiteConnect instance with today's access token."""
    from kiteconnect import KiteConnect

    s = get_settings()
    if not s.kite_api_key:
        raise KiteNotReady("KITE_API_KEY is not set")
    kite = KiteConnect(api_key=s.kite_api_key)
    if require_token:
        st = token_status()
        if not st["valid"]:
            raise KiteNotReady("No valid Kite access token for today - log in first")
        kite.set_access_token(load_token()["access_token"])
    return kite


def login_url() -> str:
    return get_kite(require_token=False).login_url()


def complete_login(request_token: str) -> dict:
    """Exchange request_token (from the redirect URL) for an access token and save it."""
    s = get_settings()
    kite = get_kite(require_token=False)
    data = kite.generate_session(request_token, api_secret=s.kite_api_secret)
    save_token(data["access_token"], data.get("user_id"))
    return {"user_id": data.get("user_id"), "user_name": data.get("user_name")}


def ist_from_kite(dt) -> datetime:
    """Kite historical candles carry tz-aware datetimes; be defensive about naive ones."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=IST)
    return dt.astimezone(IST)
