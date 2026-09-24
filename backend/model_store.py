"""Holds the currently loaded model bundle; hot-reloadable after retraining."""
from __future__ import annotations

import json
import logging
import threading

from core.config import get_settings
from core.params import StrategyParams
from ml.model import load_bundle

log = logging.getLogger(__name__)


class ModelStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.bundle: dict | None = None
        self.error: str | None = None

    def load(self) -> bool:
        path = get_settings().model_path
        try:
            b = load_bundle(path)
        except Exception as e:  # corrupt / incompatible pickle
            log.exception("Failed to load model")
            self.error = str(e)
            return False
        with self._lock:
            self.bundle = b
            self.error = None if b else "no model trained yet"
        if b:
            log.info("Loaded %s model trained %s, threshold %.2f", b["kind"], b["trained_at"], b["threshold"])
        return b is not None

    @property
    def params(self) -> StrategyParams:
        """Strategy params: from the model bundle if present so live == training."""
        b = self.bundle
        return StrategyParams.from_dict(b["params"]) if b else StrategyParams()

    @property
    def threshold(self) -> float | None:
        s = get_settings()
        if s.signal_threshold is not None:
            return s.signal_threshold
        return self.bundle["threshold"] if self.bundle else None

    def info(self) -> dict:
        b = self.bundle
        if not b:
            return {"loaded": False, "error": self.error}
        return {"loaded": True, "kind": b["kind"], "threshold": self.threshold, "trained_at": b["trained_at"],
                "metrics": b.get("metrics", {}), "n_features": len(b["features"])}

    def report(self) -> dict | None:
        p = get_settings().report_path
        return json.loads(p.read_text()) if p.exists() else None


models = ModelStore()
