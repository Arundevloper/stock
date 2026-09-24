"""WebSocket fan-out to browsers."""
from __future__ import annotations

import asyncio
import json
import logging

from fastapi import WebSocket

log = logging.getLogger(__name__)


class Hub:
    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()
        self.loop: asyncio.AbstractEventLoop | None = None

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.clients.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self.clients.discard(ws)

    async def broadcast(self, events: list[dict]) -> None:
        if not events or not self.clients:
            return
        payloads = [json.dumps(e, default=str) for e in events]
        dead = []
        for ws in list(self.clients):
            try:
                for p in payloads:
                    await ws.send_text(p)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    def broadcast_threadsafe(self, events: list[dict]) -> None:
        """For code running in worker threads / the scheduler."""
        if self.loop and events:
            asyncio.run_coroutine_threadsafe(self.broadcast(events), self.loop)


hub = Hub()
