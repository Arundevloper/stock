"""Background thread that POSTs candle batches to the backend with retries."""
from __future__ import annotations

import json
import logging
import queue
import threading
import time

import httpx

from core.config import get_settings

log = logging.getLogger(__name__)


class Poster:
    def __init__(self, max_retries: int = 6) -> None:
        s = get_settings()
        self.url = s.backend_url.rstrip("/") + "/api/ingest/candles"
        self.headers = {"X-Ingest-Key": s.ingest_key}
        self.spool = s.data_dir / "spool.jsonl"
        self.max_retries = max_retries
        self.q: queue.Queue[list[dict]] = queue.Queue()
        self.client = httpx.Client(timeout=30)
        threading.Thread(target=self._run, daemon=True, name="poster").start()

    def submit(self, candles: list[dict]) -> None:
        if candles:
            self.q.put(candles)

    def post_now(self, candles: list[dict]) -> dict | None:
        """Synchronous post (used by replay)."""
        r = self.client.post(self.url, json={"candles": candles}, headers=self.headers)
        r.raise_for_status()
        return r.json()

    def _run(self) -> None:
        while True:
            batch = self.q.get()
            for attempt in range(self.max_retries):
                try:
                    res = self.post_now(batch)
                    if res and res.get("signals"):
                        log.info("backend produced %d signal(s)", res["signals"])
                    break
                except httpx.HTTPError as e:
                    wait = min(30, 2 ** attempt)
                    log.warning("POST failed (%s), retry in %ss", e, wait)
                    time.sleep(wait)
            else:
                # backend unreachable: keep the candles; the post-close backfill also repairs gaps
                with open(self.spool, "a") as fh:
                    for c in batch:
                        fh.write(json.dumps(c) + "\n")
                log.error("Spooled %d candles to %s", len(batch), self.spool)
