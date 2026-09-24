"""FastAPI app: REST API + WebSocket + static web UI.

    uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from backend.api import router
from backend.engine import engine
from backend.model_store import models
from backend.ws import hub
from core import db
from core.config import ROOT, get_settings

settings = get_settings()
logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("backend")

if settings.sentry_dsn:
    try:
        import sentry_sdk
        sentry_sdk.init(dsn=settings.sentry_dsn, traces_sample_rate=0)
    except ImportError:
        log.warning("SENTRY_DSN set but sentry-sdk is not installed")

FRONTEND = ROOT / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    engine.load_state()
    models.load()
    hub.loop = asyncio.get_running_loop()
    if not settings.paper_mode:
        log.warning("PAPER_MODE=false but live order execution is not implemented - running in paper mode")
    if settings.ingest_key == "change-me":
        log.warning("INGEST_KEY is the default 'change-me' - set a secret in .env")
    sched = None
    if settings.scheduler_enabled:
        from backend.jobs import start_scheduler
        sched = start_scheduler()
    log.info("Backend up. Signal mode=%s, model=%s, paused=%s", settings.signal_mode,
             "loaded" if models.bundle else "none", engine.paused)
    yield
    if sched:
        sched.shutdown(wait=False)


app = FastAPI(title="Intraday ML Trading", lifespan=lifespan)


# ------------------------------------------------------------------ optional basic auth

def _session_token() -> str:
    key = f"{settings.app_username}:{settings.app_password}".encode()
    return hmac.new(key, b"intraday-session", hashlib.sha256).hexdigest()


OPEN_PATHS = ("/health", "/api/ingest/")


@app.middleware("http")
async def basic_auth(request: Request, call_next):
    if not settings.app_password or request.url.path.startswith(OPEN_PATHS):
        return await call_next(request)
    if hmac.compare_digest(request.cookies.get("sess", ""), _session_token()):
        return await call_next(request)
    auth = request.headers.get("authorization", "")
    if auth.startswith("Basic "):
        try:
            user, _, pw = base64.b64decode(auth[6:]).decode().partition(":")
        except ValueError:
            user = pw = ""
        if hmac.compare_digest(user, settings.app_username) and hmac.compare_digest(pw, settings.app_password):
            resp = await call_next(request)
            resp.set_cookie("sess", _session_token(), httponly=True, samesite="lax", max_age=30 * 86400)
            return resp
    return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="intraday"'})


# ------------------------------------------------------------------ routes

app.include_router(router)


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    if settings.app_password and not hmac.compare_digest(ws.cookies.get("sess", ""), _session_token()):
        await ws.close(code=4401)
        return
    await hub.connect(ws)
    try:
        while True:
            msg = await ws.receive_text()
            if msg == "ping":
                await ws.send_text('{"type":"pong"}')
    except WebSocketDisconnect:
        pass
    finally:
        hub.disconnect(ws)


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND), name="static")
