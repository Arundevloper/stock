"""Runs `python -m ml.train` as a subprocess (so a long fit never blocks the API) and hot-reloads
the model when it succeeds."""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time

from backend import notify
from backend.model_store import models
from core.config import ROOT, get_settings

log = logging.getLogger(__name__)


class TrainingRunner:
    def __init__(self) -> None:
        self.running = False
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.exit_code: int | None = None

    @property
    def log_path(self):
        return get_settings().data_dir / "logs" / "train.log"

    def status(self) -> dict:
        tail = ""
        if self.log_path.exists():
            tail = "".join(self.log_path.read_text(errors="replace").splitlines(keepends=True)[-40:])
        return {"running": self.running, "started_at": self.started_at, "finished_at": self.finished_at,
                "exit_code": self.exit_code, "log_tail": tail}

    async def run(self, extra_args: list[str] | None = None) -> int:
        if self.running:
            return -1
        self.running, self.started_at, self.exit_code = True, time.time(), None
        try:
            with open(self.log_path, "w") as fh:
                proc = await asyncio.create_subprocess_exec(
                    sys.executable, "-m", "ml.train", *(extra_args or []),
                    cwd=str(ROOT), stdout=fh, stderr=asyncio.subprocess.STDOUT, env=os.environ.copy())
                self.exit_code = await proc.wait()
        finally:
            self.running = False
            self.finished_at = time.time()

        if self.exit_code == 0:
            models.load()
            rep = models.report() or {}
            o = rep.get("oos", {})
            notify.send(f"🧠 Retrained {rep.get('model_kind')} | thr {rep.get('threshold')} | OOS trades {o.get('trades')} "
                        f"| precision {o.get('precision')} | net ₹{(o.get('net') or 0):,.0f}"
                        + ("".join(f"\n⚠️ {w}" for w in rep.get("warnings", []))))
        else:
            notify.send(f"⚠️ Retrain failed (exit {self.exit_code}). See data/logs/train.log", dedupe_key="train-fail")
        log.info("Training finished with exit code %s", self.exit_code)
        return self.exit_code


runner = TrainingRunner()
