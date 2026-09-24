# Intraday ML Trading System (Zerodha)

Streams live NSE data from Zerodha Kite Connect, builds 1-min candles, computes intraday features,
runs a calibrated classifier on filtered setups, and emits **entry / stop-loss / target** signals to a
web dashboard and Telegram. Every signal is paper-traded automatically with the same rules as the
backtest. **Signal-only + paper trading. Nothing here places real orders.**

> Not financial advice. Price/volume models give a small edge at best. Expect 55–65% precision on
> well-filtered setups and 1–5 signals a day. If you see 80%+ out-of-sample, look for a data leak.

---

## Architecture

```
Zerodha KiteTicker (MODE_FULL ticks)
        │
        ▼
reader/kite_reader.py ── ticks → 1-min candles ── POST /api/ingest/candles ──┐
reader/replay.py      ── replays a stored day the same way (testing) ───────┤
                                                                            ▼
backend (FastAPI, one process)                                                    
  upsert candle → advance paper trades → on 5-min bar close:
  ml/features.py → setup filter → model.predict_proba → threshold → risk rules
  → Signal + PaperTrade (DB) → WebSocket /ws/live → browser,  Telegram
  scheduler: 08:50 token check · 09:31 watchlist · 15:16 square-off · 15:35 P&L · 16:00 backfill · nightly retrain
        │
        ▼
frontend/ (plain HTML + JS + Lightweight Charts, served by the backend at /)
```

| Layer | Stack |
|---|---|
| Data | Kite Connect (`kiteconnect`): KiteTicker websocket, `historical_data`, `quote` |
| Backend | Python 3.12, FastAPI, SQLAlchemy 2, APScheduler, httpx |
| Storage | SQLite (WAL) by default; PostgreSQL/TimescaleDB via `DATABASE_URL` |
| ML | pandas, scikit-learn, XGBoost (HistGradientBoosting / LogisticRegression fallbacks), optional optuna |
| Frontend | Vanilla HTML/CSS/JS, TradingView Lightweight Charts 4.2 (CDN), no build step |
| Alerts / ops | Telegram Bot API, `/health` for Uptime Kuma, Healthchecks.io ping, optional Sentry |
| Deploy | Docker Compose (backend + reader) or systemd; nginx for HTTPS/WebSocket |

The single most important design rule: **`ml/features.py` is shared by training and the live engine**,
and the trained model bundle stores the exact `StrategyParams` it was trained with (setup filter, ATR
multiples, horizon, risk rules). Live always uses those, so training and live can't drift apart.

---

## Quick start — demo without Kite (about 5 minutes)

Uses **synthetic** candles in a separate `data/demo.db`. The fake data contains a planted pattern, so
the pipeline has something to find. The results tell you nothing about real markets.

```bash
make install            # venv + deps, copies .env.example -> .env
make demo-data          # 160 sessions x 10 fake symbols + NIFTY into data/demo.db
make demo-train         # walk-forward training -> models/demo_model.pkl (never touches the real model)
make demo               # backend + UI at http://127.0.0.1:8000
# second terminal: replay the last synthetic session through the live pipeline
make demo-replay DATE=$(date -d yesterday +%F)   # use the last weekday printed by demo-data
```

Watch signals, paper trades and P&L fill in on the dashboard. Run
`python -m scripts.generate_sample_data --no-edge` and retrain: the model should find **no** edge
(precision ≈ base rate). That's a useful check that the pipeline doesn't invent patterns.

---

## Going live with Zerodha (paper trading)

1. **Kite Connect app.** On developers.kite.trade create an app on the Connect plan (₹500/month, live + historical data).
   Set **Redirect URL** to `<PUBLIC_URL>/auth/callback` (for example `http://127.0.0.1:8000/auth/callback`).
2. **Configure.** `cp .env.example .env` and set `KITE_API_KEY`, `KITE_API_SECRET`, a random `INGEST_KEY`,
   and optionally `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID`. Edit `config/universe.txt` (liquid F&O names).
3. **Log in** (every morning; tokens expire around 06:00): click **Login to Kite** in the UI, or run `make login`.
4. **Backfill history** (6–12 months): `python -m backfill.historical --days 365`. Kite serves 60 days of
   minute data per request, so this loops in chunks at about 3 requests/s. Expect roughly 10–20 minutes for 40 symbols.
5. **Sanity-check the base rule** before trusting any ML. Run training and compare the *Base rule* row
   against the model in the report, or paper-trade the rule alone with `SIGNAL_MODE=rule`. If the base
   rule shows no edge, ML on the same inputs won't fix it. A 30-minute Streak backtest of the same rule is a good second opinion.
6. **Train:** `make train`. Read the threshold sweep, walk-forward windows and warnings (terminal output or the **Model** tab).
7. **Run:** `make run` (backend + UI) and `make reader` (tick stream). Or use `docker compose up -d --build` for both.
8. **Paper trade for at least a month or 50 trades.** Live precision should land within about 5–10 points of
   the backtest. A bigger gap means a leak or overfitting. Verify P&L against Zerodha Console now and then.

### Daily routine (automatic once running)

| IST | What happens |
|---|---|
| 08:50 | Token check. If there's no valid token, Telegram sends a login link. Instrument list refreshed. |
| 09:15 | Reader starts building candles (it waits for the token, then connects). |
| 09:31 | Watchlist scan: top `WATCHLIST_SIZE` by first-15-min relative volume and gap, with a turnover filter. |
| 09:30–11:30, 13:30–15:00 | Signal windows. Setups are evaluated at every 5-min bar close. |
| 15:15 | Paper positions squared off (15:16 safety-net job for anything left). |
| 15:35 | Daily P&L summary on Telegram. |
| 16:00 | Official minute candles re-downloaded for the day (repairs gaps in tick-built bars). |
| 20:00 | Nightly retrain; the new model is hot-reloaded and the report is sent to Telegram. |

---

## How a signal is made

1. **Candle.** The reader buckets ticks by minute. Bar volume is the difference in cumulative `volume_traded`.
   The last bid/ask and total buy/sell quantity of each minute are stored too.
2. **Setup filter** (`find_setups`), checked at each 5-min bar close:
   * inside a trading window (09:30–11:30 or 13:30–15:00), and
   * a 12-bar high/low breakout with volume ≥ 1.5× average, or a volume spike ≥ 2× on a bar that closes
     in the spike's direction on the right side of VWAP, and
   * target move (1.5 × ATR) ≥ 0.4% of price, so costs don't eat the move.
3. **Features** (past bars only): returns over 1/3/6/12 bars, volume vs 20-bar average and vs the same minute
   on previous days, VWAP distance, range, volatility, ATR%, minutes since open, session phase, gap %,
   opening-range position, previous-day high/low/close distances, breakout flags, 15-min trend (completed
   bars only), daily trend, NIFTY move since open and over 15 min. Directional features are also given
   multiplied by the trade side.
4. **Model:** calibrated XGBoost gives `P(target before stop)`. If it's below the threshold, the signal is dropped.
5. **Risk rules:** at most 3 trades a day, stop after 2 losses, one position per symbol.
6. **Levels:** entry = next bar open, stop = entry ∓ 1×ATR(14, 5-min), target = entry ± 1.5×ATR,
   qty = `capital × 1% / (entry − stop)`, capped by MIS leverage.

The **label** used in training is the same trade: 1 if the target is hit before the stop within 12 bars
(60 min) and before 15:15, entering at the next 1-min open. If stop and target are both hit in the same
1-min bar, it counts as a loss. The paper engine (`backend/paper.py`) applies identical rules.

## Training and evaluation

`python -m ml.train [--model xgb|hgb|logreg] [--tune 40] [--test-days 20] [--min-train-days 60]`

* **Walk-forward by session:** train on everything before a 20-session test window, test on that window,
  roll forward. Label overlap at the boundary is purged. There are no random splits.
* **Calibration:** `CalibratedClassifierCV(sigmoid)` with time-ordered folds, so 0.70 means about 70%.
* **Threshold sweep:** 0.40–0.90. Picks the highest precision with ≥ 30 candidates per window.
* **Backtest:** next-bar-open entry, 0.05% slippage per side, Zerodha intraday charges (`core/costs.py`),
  fixed 1% risk sizing, the same daily risk rules. Reports net P&L, max drawdown, worst losing streak, and the
  base rule's result over the same out-of-sample period.
* **Warnings** flag suspiciously high precision (leak), unstable windows, too few trades, and no edge after costs.

Strategy parameters live in `core/params.py`. Override them at training time with
`--params '{"target_atr": 2.0, "horizon_bars": 18}'`. The live engine picks them up from the model bundle.

## Tests

`make test` runs the leak tests, which are the ones that matter most:

* `test_no_lookahead` recomputes features on data truncated at 30 random bar closes and asserts every
  feature is identical to the full-history value.
* `test_live_lookback_matches_training` checks that the live engine's 40-day window gives the same features as full history.
* Label edge cases (same-bar stop and target, gaps, timeouts), costs, sizing, the tick→candle builder, and
  a full session ingested through the API in rule mode (signals, paper fills, square-off).

---

## Web UI

`http://127.0.0.1:8000`: watchlist with live prices, candlestick chart (1/5/15-min, VWAP, signal and exit
markers, entry/SL/target lines for open trades), live signal feed (click a signal to see the model's
inputs), today's stats, and tabs for **Paper trades**, **P&L** (equity curve, daily table), **Setups**
(every evaluated setup with its probability) and **Model** (walk-forward report, threshold sweep, feature
importance, one-click retrain). Also: Pause/Resume signals, a manual watchlist scan, and manual close of a paper trade.

## API

| Method | Path | |
|---|---|---|
| POST | `/api/ingest/candles` | reader → backend; header `X-Ingest-Key`; body `{"candles": [...]}` |
| GET | `/api/candles?symbol=&tf=5&days=3` | resampled bars + VWAP |
| GET | `/api/watchlist` · POST `/api/watchlist` · POST `/api/watchlist/scan` | day's watchlist, manual set, rescan |
| GET | `/api/signals?day=&symbol=&days=` · `/api/signals/{id}` | signals; detail includes features + trade |
| GET | `/api/trades` · POST `/api/trades/{id}/close` | paper trades |
| GET | `/api/pnl` · `/api/setups` · `/api/status` · `/api/model` | |
| POST | `/api/control/pause` · `/api/control/squareoff` · `/api/model/train` · `/api/model/reload` | |
| GET | `/auth/login` · `/auth/callback` | Kite login flow |
| GET | `/health` | for Uptime Kuma (no auth) |
| WS | `/ws/live` | events: `candle`, `setup`, `signal`, `trade`, `watchlist`, `model`, `status` |

## Project layout

```
core/       config, DB models, costs, strategy params, Kite client + token store, instruments, IST helpers
reader/     kite_reader.py (live), candle_builder.py, poster.py, replay.py
backfill/   historical.py (bulk minute history)
auth/       login.py (terminal login)
ml/         features.py (shared), labels.py, dataset.py, model.py, backtest.py, train.py
backend/    main.py, api.py, engine.py (live pipeline), paper.py, watchlist.py, jobs.py, notify.py, ws.py
frontend/   index.html, style.css, app.js
scripts/    generate_sample_data.py, watchlist.py, eod_summary.py
config/     universe.txt
deploy/     systemd units, nginx.conf
tests/
```

## Deployment

A 2 GB Linux VPS is enough. `docker compose up -d --build` runs the backend and reader, with `data/`, `models/`
and `config/` as volumes. Put nginx in front (`deploy/nginx.conf`, WebSocket upgrade included), get a cert
with certbot, and **set `APP_USERNAME` / `APP_PASSWORD`** so the dashboard isn't public. Run one uvicorn
worker only: the engine, paper book and WebSocket hub live in one process. Point Uptime Kuma at `/health`.

For PostgreSQL/TimescaleDB, run `docker compose --profile postgres up -d`, `pip install psycopg[binary]`, and set `DATABASE_URL`.
Optionally make candles a hypertable: `SELECT create_hypertable('candles', by_range('time', 604800));`.

## Not included (on purpose, or next steps)

* **Live order execution.** Not implemented. `PAPER_MODE=false` is ignored with a warning. Add it only after
  paper results hold: MIS entry, then an SL-M stop, with OCO handling through order postbacks, and check SEBI's
  retail algo rules via Zerodha first.
* Depth features (spread, buy/sell ratio) are stored per minute but not used by the model, because historical
  data has no depth. After a few months of live collection they can be added to `features.py`.
* India VIX filter, headline sentiment, delivery %/FII flows, two-way Telegram commands (`/status`, `/pause`).
* NSE holidays aren't encoded. The scheduler runs Mon–Fri and the data-freshness alert will fire on holidays.
