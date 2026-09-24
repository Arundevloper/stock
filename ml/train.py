"""Walk-forward training, calibration, threshold sweep and cost-aware backtest.

    python -m ml.train                         # all symbols in the DB, xgboost
    python -m ml.train --model logreg          # baseline
    python -m ml.train --tune 40               # optuna tuning on pre-OOS data only
    python -m ml.train --test-days 20 --min-train-days 60

Writes models/model.pkl (bundle: calibrated model + features + threshold + strategy params)
and models/report.json (shown on the web UI's Model tab).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from core import db
from core.config import MARKET_SYMBOL, get_settings
from core.params import StrategyParams
from core.timeutil import IST
from ml.backtest import equity_curve, simulate
from ml.dataset import build_dataset
from ml.features import FEATURES, model_matrix
from ml.model import (available_kinds, feature_importance, fit_calibrated, make_bundle,
                      make_estimator, save_bundle)

log = logging.getLogger("train")

THRESHOLDS = [round(x, 2) for x in np.arange(0.40, 0.901, 0.05)]


def walk_forward_windows(sessions: list[str], min_train: int, test_days: int, train_window: int):
    out = []
    for i in range(min_train, len(sessions), test_days):
        test = sessions[i:i + test_days]
        if len(test) < max(3, test_days // 3):
            break
        train = sessions[max(0, i - train_window):i] if train_window else sessions[:i]
        out.append((train, test))
    return out


def tune(ds: pd.DataFrame, kind: str, n_trials: int) -> dict:
    try:
        import optuna
    except ImportError:
        log.warning("optuna not installed - skipping tuning (pip install optuna)")
        return {}
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cut = int(len(ds) * 0.75)
    tr, va = ds.iloc[:cut], ds.iloc[cut:]
    Xtr, ytr, Xva, yva = model_matrix(tr), tr["label"], model_matrix(va), va["label"]
    pos_w = (len(ytr) - ytr.sum()) / max(1, ytr.sum())

    def objective(trial):
        if kind == "xgb":
            hp = dict(max_depth=trial.suggest_int("max_depth", 2, 5),
                      learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                      n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
                      min_child_weight=trial.suggest_float("min_child_weight", 1, 20, log=True),
                      subsample=trial.suggest_float("subsample", 0.6, 1.0),
                      colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0))
        elif kind == "hgb":
            hp = dict(max_depth=trial.suggest_int("max_depth", 2, 6),
                      learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                      max_iter=trial.suggest_int("max_iter", 100, 600, step=50),
                      min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 100))
        else:
            hp = dict(C=trial.suggest_float("C", 0.01, 10, log=True))
        est = make_estimator(kind, pos_w, hp)
        est.fit(Xtr, ytr)
        return average_precision_score(yva, est.predict_proba(Xva)[:, 1])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    log.info("Tuning best PR-AUC %.4f params %s", study.best_value, study.best_params)
    return study.best_params


def sweep(oos: pd.DataFrame, p: StrategyParams, s, n_windows: int) -> list[dict]:
    rows = []
    for thr in THRESHOLDS:
        sel = oos[oos["prob"] >= thr]
        trades, m = simulate(sel, p, s.capital, s.risk_per_trade, s.mis_leverage)
        rows.append({
            "threshold": thr,
            "candidates": int(len(sel)),
            "candidates_per_window": round(len(sel) / max(1, n_windows), 1),
            "precision": round(float(sel["label"].mean()), 4) if len(sel) else None,
            "trades": m["trades"],
            "trade_precision": m["precision"],
            "net_pnl": m["net"],
            "max_dd": m["max_dd"],
            "avg_r": m["avg_r"],
        })
    return rows


def choose_threshold(rows: list[dict], min_per_window: float) -> tuple[float, str]:
    ok = [r for r in rows if r["precision"] is not None and r["candidates_per_window"] >= min_per_window]
    if ok:
        best = max(ok, key=lambda r: (r["precision"], r["net_pnl"]))
        return best["threshold"], f"highest precision with >= {min_per_window} candidates/window"
    ok = [r for r in rows if r["precision"] is not None and r["candidates"] >= 20]
    if ok:
        best = max(ok, key=lambda r: (r["precision"], r["net_pnl"]))
        return best["threshold"], "fallback: highest precision with >= 20 candidates total (too little data)"
    return 0.6, "fallback: default 0.60 (not enough out-of-sample candidates)"


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    return o


def run(args) -> dict:
    s = get_settings()
    p = StrategyParams.from_dict(json.loads(args.params)) if args.params else StrategyParams()
    t0 = time.time()

    symbols = args.symbols.split(",") if args.symbols else [x for x in db.list_symbols() if x != MARKET_SYMBOL]
    if not symbols:
        raise SystemExit("No candles in the database. Run backfill (or scripts.generate_sample_data) first.")
    log.info("Building dataset for %d symbols ...", len(symbols))
    ds = build_dataset(symbols, p)
    if len(ds) < 200 or ds["label"].sum() < 20:
        raise SystemExit(f"Not enough labelled setups ({len(ds)} rows, {int(ds['label'].sum()) if len(ds) else 0} wins). "
                         "Backfill more history or loosen the setup filter.")
    ds = ds.reset_index(drop=True)
    sessions = sorted(ds["session"].unique())
    base_rate = float(ds["label"].mean())
    log.info("Dataset: %d setups over %d sessions, base win rate %.3f", len(ds), len(sessions), base_rate)

    windows = walk_forward_windows(sessions, args.min_train_days, args.test_days, args.train_window_days)
    warnings = []
    if not windows:
        cut = int(len(sessions) * 0.7)
        windows = [(sessions[:cut], sessions[cut:])]
        warnings.append(f"Only {len(sessions)} sessions: used a single 70/30 time split instead of walk-forward.")

    hp = {}
    if args.tune:
        pre_oos = ds[ds["session"].isin(windows[0][0])]
        hp = tune(pre_oos, args.model, args.tune)

    oos_parts, win_rows = [], []
    for wi, (train_s, test_s) in enumerate(windows):
        test = ds[ds["session"].isin(test_s)]
        train = ds[ds["session"].isin(train_s)]
        train = train[train["exit_time"] < test["close_time"].min()]  # purge label overlap
        if train["label"].sum() < 10 or test.empty:
            continue
        clf = fit_calibrated(model_matrix(train), train["label"], args.model, hp)
        prob = clf.predict_proba(model_matrix(test))[:, 1]
        test = test.assign(prob=prob, window=wi)
        oos_parts.append(test)
        y = test["label"]
        win_rows.append({
            "window": wi, "train_start": train_s[0], "train_end": train_s[-1],
            "test_start": test_s[0], "test_end": test_s[-1],
            "n_train": int(len(train)), "n_test": int(len(test)), "base_rate": round(float(y.mean()), 4),
            "auc": round(float(roc_auc_score(y, prob)), 4) if y.nunique() == 2 else None,
        })
        log.info("window %2d  test %s..%s  n=%4d  base=%.3f  auc=%s", wi, test_s[0], test_s[-1],
                 len(test), y.mean(), win_rows[-1]["auc"])

    if not oos_parts:
        raise SystemExit("No usable walk-forward windows (too few wins per training window).")
    oos = pd.concat(oos_parts)
    n_windows = len(win_rows)

    sweep_rows = sweep(oos, p, s, n_windows)
    threshold, why = choose_threshold(sweep_rows, args.min_trades_per_window)
    if s.signal_threshold is not None:
        warnings.append(f"SIGNAL_THRESHOLD={s.signal_threshold} in .env overrides the chosen {threshold}.")

    sel = oos[oos["prob"] >= threshold]
    trades, oos_metrics = simulate(sel, p, s.capital, s.risk_per_trade, s.mis_leverage)
    for w in win_rows:
        ws = sel[sel["window"] == w["window"]]
        wt = trades[trades["close_time"].dt.strftime("%Y-%m-%d").between(w["test_start"], w["test_end"])] \
            if len(trades) else trades
        w["candidates"] = int(len(ws))
        w["precision"] = round(float(ws["label"].mean()), 4) if len(ws) else None
        w["trades"] = int(len(wt))
        w["net_pnl"] = round(float(wt["net"].sum()), 2) if len(wt) else 0.0

    rule_trades, rule_metrics = simulate(oos, p, s.capital, s.risk_per_trade, s.mis_leverage)

    oos_auc = roc_auc_score(oos["label"], oos["prob"]) if oos["label"].nunique() == 2 else None
    precs = [w["precision"] for w in win_rows if w["precision"] is not None and w["candidates"] >= 5]
    if oos_metrics["precision"] and oos_metrics["precision"] >= 0.8:
        warnings.append("Out-of-sample precision >= 80% - almost certainly a data leak or synthetic data.")
    if len(precs) >= 3 and np.std(precs) > 0.15:
        warnings.append(f"Precision unstable across windows (std {np.std(precs):.2f}).")
    if oos_metrics["trades"] < 50:
        warnings.append(f"Only {oos_metrics['trades']} out-of-sample trades - too few to trust.")
    if rule_metrics["net"] <= 0 and oos_metrics["net"] <= 0:
        warnings.append("Neither the base rule nor the model is profitable after costs out-of-sample.")

    log.info("Fitting final model on all %d rows ...", len(ds))
    final = fit_calibrated(model_matrix(ds), ds["label"], args.model, hp)
    importance = feature_importance(final, FEATURES)

    report = {
        "trained_at": pd.Timestamp.now(tz=IST).isoformat(timespec="seconds"),
        "model_kind": args.model, "hyperparams": hp, "threshold": threshold, "threshold_reason": why,
        "params": p.to_dict(), "symbols": symbols,
        "data": {"setups": int(len(ds)), "wins": int(ds["label"].sum()), "base_rate": round(base_rate, 4),
                 "sessions": len(sessions), "first": sessions[0], "last": sessions[-1]},
        "oos": {**oos_metrics, "auc": round(float(oos_auc), 4) if oos_auc is not None else None,
                "candidates": int(len(sel)), "equity": equity_curve(trades, s.capital)},
        "baseline_rule": {**rule_metrics, "equity": equity_curve(rule_trades, s.capital)},
        "sweep": sweep_rows, "windows": win_rows, "importance": importance[:25],
        "warnings": warnings, "capital": s.capital, "duration_s": round(time.time() - t0, 1),
    }
    report = _clean(report)
    bundle = make_bundle(final, FEATURES, threshold, p.to_dict(), args.model,
                         {k: report["oos"][k] for k in ("trades", "precision", "net", "max_dd", "auc")})

    _print_summary(report)
    if not args.no_save:
        save_bundle(bundle, s.model_path)
        s.report_path.write_text(json.dumps(report, indent=1))
        log.info("Saved %s and %s", s.model_path, s.report_path)
    return report


def _print_summary(r: dict) -> None:
    print("\n=== Threshold sweep (out-of-sample, walk-forward) ===")
    print(f"{'thr':>5} {'cand':>6} {'c/win':>6} {'prec':>6} {'trades':>7} {'t.prec':>7} {'net Rs':>11} {'maxDD':>10}")
    for x in r["sweep"]:
        pr = f"{x['precision']:.3f}" if x["precision"] is not None else "  -  "
        tp = f"{x['trade_precision']:.3f}" if x["trade_precision"] is not None else "  -  "
        mark = " <" if x["threshold"] == r["threshold"] else ""
        print(f"{x['threshold']:>5.2f} {x['candidates']:>6} {x['candidates_per_window']:>6} {pr:>6} "
              f"{x['trades']:>7} {tp:>7} {x['net_pnl']:>11,.0f} {x['max_dd']:>10,.0f}{mark}")
    o, b = r["oos"], r["baseline_rule"]
    print(f"\nChosen threshold {r['threshold']} ({r['threshold_reason']})")
    print(f"Model @thr : trades={o['trades']} precision={o['precision']} net=Rs {o['net']:,} maxDD=Rs {o['max_dd']:,} AUC={o['auc']}")
    print(f"Base rule  : trades={b['trades']} precision={b['precision']} net=Rs {b['net']:,} maxDD=Rs {b['max_dd']:,}")
    for w in r["warnings"]:
        print(f"WARNING: {w}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=available_kinds()[0], choices=available_kinds())
    ap.add_argument("--symbols", help="comma separated, default: every symbol in the DB")
    ap.add_argument("--test-days", type=int, default=20, help="sessions per walk-forward test window")
    ap.add_argument("--min-train-days", type=int, default=60, help="sessions before the first test window")
    ap.add_argument("--train-window-days", type=int, default=0, help="rolling train window (0 = expanding)")
    ap.add_argument("--min-trades-per-window", type=float, default=30)
    ap.add_argument("--tune", type=int, default=0, help="optuna trials (0 = off)")
    ap.add_argument("--params", help="JSON overrides for StrategyParams")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=get_settings().log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    db.init_db()
    try:
        run(args)
    except SystemExit as e:
        log.error("%s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
