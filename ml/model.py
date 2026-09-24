"""Model construction, calibration and the saved bundle format."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from core.timeutil import IST

log = logging.getLogger(__name__)

BUNDLE_VERSION = 1

DEFAULT_XGB = dict(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                   colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0)


def available_kinds() -> list[str]:
    kinds = ["hgb", "logreg"]
    try:
        import xgboost  # noqa: F401
        kinds.insert(0, "xgb")
    except ImportError:
        pass
    return kinds


def make_estimator(kind: str, pos_weight: float, hp: dict | None = None):
    hp = dict(hp or {})
    if kind == "xgb":
        from xgboost import XGBClassifier
        params = {**DEFAULT_XGB, **hp}
        return XGBClassifier(**params, scale_pos_weight=pos_weight, eval_metric="logloss",
                             n_jobs=2, verbosity=0, tree_method="hist")
    if kind == "hgb":
        params = dict(max_iter=300, learning_rate=0.05, max_depth=3, min_samples_leaf=30,
                      l2_regularization=1.0)
        params.update(hp)
        return HistGradientBoostingClassifier(**params, class_weight="balanced")
    if kind == "logreg":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=hp.get("C", 0.5), class_weight="balanced", max_iter=2000))
    raise ValueError(f"unknown model kind {kind!r}")


def fit_calibrated(X: pd.DataFrame, y: pd.Series, kind: str, hp: dict | None = None):
    """Fit + calibrate so that predict_proba ~ real hit rate. Rows must be in time order; the
    calibration folds are time-ordered too (TimeSeriesSplit), never shuffled."""
    pos = max(1, int(y.sum()))
    neg = max(1, len(y) - pos)
    est = make_estimator(kind, neg / pos, hp)
    n_splits = 3 if len(y) >= 300 and pos >= 30 else 2
    clf = CalibratedClassifierCV(est, method="sigmoid", cv=TimeSeriesSplit(n_splits=n_splits))
    clf.fit(X, y)
    return clf


def feature_importance(clf, features: list[str]) -> list[dict]:
    imps = []
    for cc in getattr(clf, "calibrated_classifiers_", []):
        est = cc.estimator
        if hasattr(est, "feature_importances_"):
            imps.append(np.asarray(est.feature_importances_, float))
        elif hasattr(est, "steps"):
            lr = est.steps[-1][1]
            if hasattr(lr, "coef_"):
                imps.append(np.abs(lr.coef_[0]))
    if not imps:
        return []
    imp = np.mean(imps, axis=0)
    if imp.sum() > 0:
        imp = imp / imp.sum()
    order = np.argsort(imp)[::-1]
    return [{"feature": features[i], "importance": round(float(imp[i]), 4)} for i in order]


def make_bundle(clf, features, threshold, params: dict, kind: str, metrics: dict) -> dict:
    return {
        "version": BUNDLE_VERSION,
        "model": clf,
        "features": list(features),
        "threshold": float(threshold),
        "params": params,
        "kind": kind,
        "trained_at": datetime.now(IST).isoformat(timespec="seconds"),
        "metrics": metrics,
    }


def save_bundle(bundle: dict, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    joblib.dump(bundle, tmp)
    tmp.replace(path)  # atomic swap so the backend never loads a half-written file


def load_bundle(path: Path) -> dict | None:
    if not path.exists():
        return None
    b = joblib.load(path)
    if b.get("version") != BUNDLE_VERSION:
        log.warning("Model bundle version %s != %s", b.get("version"), BUNDLE_VERSION)
    return b


def predict_proba(bundle_or_clf, X: pd.DataFrame) -> np.ndarray:
    clf = bundle_or_clf["model"] if isinstance(bundle_or_clf, dict) else bundle_or_clf
    return clf.predict_proba(X)[:, 1]
