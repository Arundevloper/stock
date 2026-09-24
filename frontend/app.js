'use strict';
/* Intraday ML Desk - vanilla JS client for the FastAPI backend.
 * REST for initial loads, WebSocket /ws/live for candles, signals, trades and setups.
 * Times from the API are epoch seconds (UTC); Lightweight Charts renders UTC, so chart
 * times are shifted by +5:30 to display IST. */

const IST = 19800;
const MAX_SETUPS = 100;
const RUPEES = { type: 'price', precision: 0, minMove: 1 };

const state = {
  symbol: null, tf: 5, days: 3, vwapOn: true,
  bars: [], chartSignals: [], trades: [], daySignals: [], setups: [],
  watch: new Map(), market: 'NSE:NIFTY 50', status: null, day: null, threshold: null,
};

// ------------------------------------------------------------------ helpers
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const nf2 = new Intl.NumberFormat('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const nf0 = new Intl.NumberFormat('en-IN', { maximumFractionDigits: 0 });
const px = v => (v == null ? '–' : nf2.format(v));
const inr = v => (v == null ? '–' : (v < 0 ? '−₹' : '₹') + nf0.format(Math.abs(v)));
const pct = (v, d = 2) => (v == null ? '–' : (v > 0 ? '+' : '') + Number(v).toFixed(d) + '%');
const num = (v, d = 2) => (v == null ? '–' : Number(v).toFixed(d));
const hhmm = ts => new Date(ts * 1000).toLocaleTimeString('en-GB', { timeZone: 'Asia/Kolkata', hour: '2-digit', minute: '2-digit' });
const dmy = ts => new Date(ts * 1000).toLocaleDateString('en-GB', { timeZone: 'Asia/Kolkata', day: '2-digit', month: 'short' });
const short = s => (s ? s.split(':').pop() : '');
const dirCls = v => (v > 0 ? 'up' : v < 0 ? 'down' : '');
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const cssVar = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };

async function api(path, opts = {}) {
  const r = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...opts });
  if (!r.ok) throw new Error(`${r.status}: ${(await r.text()).slice(0, 200)}`);
  return r.json();
}

function toast(html, kind = 'info', ms = 6000) {
  const el = document.createElement('div');
  el.className = `toast ${kind}`;
  el.innerHTML = html;
  $('#toasts').prepend(el);
  setTimeout(() => el.remove(), ms);
}

function store(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (_) { /* private mode */ } }
function recall(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch (_) { return d; } }

// ------------------------------------------------------------------ chart
let chart, candles, volume, vwap, priceLines = [];

function chartTheme() {
  return {
    layout: { background: { type: 'solid', color: cssVar('--panel') }, textColor: cssVar('--muted'), fontFamily: 'Inter, system-ui, sans-serif', fontSize: 11 },
    grid: { vertLines: { color: cssVar('--grid') }, horzLines: { color: cssVar('--grid') } },
    rightPriceScale: { borderColor: cssVar('--border') },
    timeScale: { borderColor: cssVar('--border'), timeVisible: true, secondsVisible: false, rightOffset: 4 },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
  };
}

function initChart() {
  chart = LightweightCharts.createChart($('#chart'), { ...chartTheme(), autoSize: true });
  const up = cssVar('--green'), down = cssVar('--red');
  candles = chart.addCandlestickSeries({ upColor: up, downColor: down, borderVisible: false, wickUpColor: up, wickDownColor: down });
  volume = chart.addHistogramSeries({ priceScaleId: 'vol', priceFormat: { type: 'volume' }, lastValueVisible: false, priceLineVisible: false });
  chart.priceScale('vol').applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });
  vwap = chart.addLineSeries({ color: cssVar('--amber'), lineWidth: 2, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
  chart.subscribeCrosshairMove(showLegend);
  matchMedia('(prefers-color-scheme: light)').addEventListener('change', () => chart.applyOptions(chartTheme()));
}

const toCandle = b => ({ time: b.time + IST, open: b.open, high: b.high, low: b.low, close: b.close });
const toVol = b => ({ time: b.time + IST, value: b.volume, color: b.close >= b.open ? 'rgba(34,192,122,.35)' : 'rgba(240,80,110,.35)' });

function showLegend(p) {
  const el = $('#legend');
  const d = p && p.time ? p.seriesData.get(candles) : null;
  const b = d || (state.bars.length ? toCandle(state.bars.at(-1)) : null);
  if (!b) { el.textContent = ''; return; }
  const v = p && p.time ? p.seriesData.get(volume) : null;
  el.innerHTML = `O ${px(b.open)} H ${px(b.high)} L ${px(b.low)} C <span class="${dirCls(b.close - b.open)}">${px(b.close)}</span>` +
    (v ? ` V ${nf0.format(v.value)}` : '');
}

async function loadChart(keepRange = false) {
  if (!state.symbol) return;
  const range = keepRange ? chart.timeScale().getVisibleLogicalRange() : null;
  const q = `symbol=${encodeURIComponent(state.symbol)}`;
  const [d, sigs] = await Promise.all([
    api(`/api/candles?${q}&tf=${state.tf}&days=${state.days}`),
    api(`/api/signals?${q}&days=${state.days + 4}`),
  ]);
  state.bars = d.candles;
  state.chartSignals = sigs.signals;
  candles.setData(d.candles.map(toCandle));
  volume.setData(d.candles.map(toVol));
  setVwap(d.vwap);
  updateMarkers();
  updatePriceLines();
  updateHeader();
  showLegend(null);
  if (range) chart.timeScale().setVisibleLogicalRange(range);
  else {
    const n = d.candles.length;
    chart.timeScale().setVisibleLogicalRange({ from: Math.max(0, n - (state.tf === 1 ? 240 : 160)), to: n + 4 });
  }
}

function setVwap(points) {
  const on = state.vwapOn && state.symbol !== state.market;
  vwap.setData(on ? points.map(p => ({ time: p.time + IST, value: p.value })) : []);
}

// Server is the source of truth: on a live candle, re-fetch today's bars and merge the tail.
const refreshTail = debounce(async () => {
  if (!state.symbol) return;
  const d = await api(`/api/candles?symbol=${encodeURIComponent(state.symbol)}&tf=${state.tf}&days=1`);
  const lastT = state.bars.length ? state.bars.at(-1).time : -1;
  for (const b of d.candles) {
    if (b.time < lastT) continue;
    if (b.time === lastT) state.bars[state.bars.length - 1] = b; else state.bars.push(b);
    candles.update(toCandle(b));
    volume.update(toVol(b));
  }
  if (state.vwapOn && state.symbol !== state.market) {
    for (const p of d.vwap) if (p.time >= lastT) vwap.update({ time: p.time + IST, value: p.value });
  }
  updateHeader();
  showLegend(null);
}, 400);

function updateMarkers() {
  const tfs = state.tf * 60;
  const bucket = t => t - (t % tfs) + IST;
  const have = new Set(state.bars.map(b => b.time + IST));
  const green = cssVar('--green'), red = cssVar('--red'), gray = cssVar('--gray');
  const ms = [];
  for (const s of state.chartSignals) {
    const t = bucket(s.time - 60);
    if (!have.has(t)) continue;
    const long = s.side === 'long';
    ms.push({
      time: t, position: long ? 'belowBar' : 'aboveBar', shape: long ? 'arrowUp' : 'arrowDown',
      color: s.status === 'blocked' ? gray : long ? green : red,
      text: s.prob != null ? s.prob.toFixed(2) : 'rule',
    });
  }
  for (const tr of state.trades) {
    if (tr.symbol !== state.symbol || tr.status !== 'closed') continue;
    const t = bucket(tr.exit_time - 60);
    if (!have.has(t)) continue;
    const label = { target: 'TGT', stop: 'SL' }[tr.exit_reason] || 'EXIT';
    ms.push({ time: t, position: tr.side === 'long' ? 'aboveBar' : 'belowBar', shape: 'circle', color: tr.net_pnl >= 0 ? green : red, text: label });
  }
  ms.sort((a, b) => a.time - b.time);
  candles.setMarkers(ms);
}

function updatePriceLines() {
  priceLines.forEach(l => candles.removePriceLine(l));
  priceLines = [];
  const t = state.trades.find(x => x.symbol === state.symbol && x.status === 'open');
  if (!t) return;
  const S = LightweightCharts.LineStyle;
  priceLines.push(candles.createPriceLine({ price: t.entry_price, color: cssVar('--blue'), lineStyle: S.Dashed, lineWidth: 1, title: 'entry' }));
  priceLines.push(candles.createPriceLine({ price: t.stop, color: cssVar('--red'), lineStyle: S.Solid, lineWidth: 1, title: 'SL' }));
  priceLines.push(candles.createPriceLine({ price: t.target, color: cssVar('--green'), lineStyle: S.Solid, lineWidth: 1, title: 'target' }));
}

function updateHeader() {
  $('#chart-symbol').textContent = short(state.symbol);
  const w = state.watch.get(state.symbol);
  const last = state.bars.at(-1);
  const ltp = last ? last.close : w?.ltp;
  $('#chart-ltp').textContent = px(ltp);
  const chg = w?.prev_close && ltp ? (ltp / w.prev_close - 1) * 100 : null;
  const el = $('#chart-chg');
  el.textContent = pct(chg);
  el.className = `chg mono ${dirCls(chg)}`;
}

function selectSymbol(sym) {
  if (!sym) return;
  state.symbol = sym;
  store('symbol', sym);
  $$('.wl-row').forEach(r => r.classList.toggle('active', r.dataset.symbol === sym));
  loadChart().catch(e => toast(`Chart: ${esc(e.message)}`, 'bad'));
}

// ------------------------------------------------------------------ watchlist
async function loadWatchlist() {
  const d = await api('/api/watchlist');
  state.watch = new Map(d.entries.map(e => [e.symbol, e]));
  state.watch.set(d.market.symbol, d.market);
  state.market = d.market.symbol;
  $('#wl-source').textContent = d.source ? `${d.day} · ${d.source}` : d.day;
  $('#wl-market').innerHTML = d.market.ltp != null ? wlRow(d.market, true) : '';
  $('#watchlist').innerHTML = d.entries.map(e => wlRow(e)).join('') || '<li class="empty">No symbols with data yet.</li>';
  $$('.wl-row').forEach(r => r.addEventListener('click', () => selectSymbol(r.dataset.symbol)));
  if (!state.symbol) selectSymbol(recall('symbol', null) || d.entries[0]?.symbol || d.market.symbol);
  else $$('.wl-row').forEach(r => r.classList.toggle('active', r.dataset.symbol === state.symbol));
}

function wlRow(e, market = false) {
  const meta = market ? 'Index' : [
    e.gap_pct != null ? `gap ${pct(e.gap_pct, 1)}` : null,
    e.rvol != null ? `rvol ${num(e.rvol, 1)}×` : null,
  ].filter(Boolean).join(' · ');
  return `<li class="wl-row${market ? ' market' : ''}${e.symbol === state.symbol ? ' active' : ''}" data-symbol="${esc(e.symbol)}">
    <span class="name">${esc(short(e.symbol))}</span><span class="px">${px(e.ltp)}</span>
    <span class="meta">${esc(meta)}</span><span class="chg ${dirCls(e.change_pct)}">${pct(e.change_pct)}</span></li>`;
}

function onCandle(c) {
  const w = state.watch.get(c.symbol);
  if (w) {
    w.ltp = c.close;
    w.change_pct = w.prev_close ? (c.close / w.prev_close - 1) * 100 : null;
    const row = $(`.wl-row[data-symbol="${CSS.escape(c.symbol)}"]`);
    if (row) {
      row.querySelector('.px').textContent = px(w.ltp);
      const ch = row.querySelector('.chg');
      ch.textContent = pct(w.change_pct);
      ch.className = `chg ${dirCls(w.change_pct)}`;
    }
  }
  if (c.symbol === state.symbol) refreshTail();
}

async function loadSymbols() {
  const d = await api('/api/symbols');
  $('#symbol-list').innerHTML = d.symbols.map(s => `<option value="${esc(s)}">`).join('');
}

// ------------------------------------------------------------------ signals (right panel)
async function loadDaySignals() {
  const d = await api('/api/signals');
  state.day = d.day;
  state.daySignals = d.signals;
  renderSignals();
}

function renderSignals() {
  const list = state.daySignals;
  $('#signals').innerHTML = list.map(sigCard).join('');
  $('#signals-empty').hidden = list.length > 0;
  const live = list.filter(s => s.status !== 'blocked').length;
  $('#sig-count').textContent = list.length ? `${live} taken · ${list.length - live} blocked` : '';
  $$('#signals .sig').forEach(el => el.addEventListener('click', () => {
    selectSymbol(el.dataset.symbol);
    openSignal(+el.dataset.id);
  }));
}

function sigCard(s, fresh = false) {
  const thr = state.threshold;
  const status = s.status === 'closed' ? (s.outcome || 'closed') : s.status;
  const probBar = s.prob == null ? '<div class="prob">rule mode (no model filter)</div>' : `
    <div class="prob"><span>p ${s.prob.toFixed(2)}</span><div class="track"><div class="fill" style="width:${Math.min(100, s.prob * 100)}%"></div>
    ${thr ? `<div class="thr" style="left:${thr * 100}%"></div>` : ''}</div></div>`;
  return `<li class="sig ${s.side} ${s.status === 'blocked' ? 'blocked' : ''} ${fresh ? 'fresh' : ''}" data-id="${s.id}" data-symbol="${esc(s.symbol)}">
    <div class="sig-top"><span class="tag ${s.side}">${s.side.toUpperCase()}</span><span class="name">${esc(short(s.symbol))}</span>
      <span class="tag ${esc(status)}">${esc(status)}</span><span class="time">${hhmm(s.time)}</span></div>
    <div class="sig-levels"><div><span>Entry</span>${px(s.entry)}</div><div><span>SL</span>${px(s.stop)}</div>
      <div><span>Target</span>${px(s.target)}</div><div><span>Qty</span>${s.qty}</div></div>
    ${probBar}
    ${s.reason ? `<div class="sig-foot">${esc(s.reason)}</div>` : ''}</li>`;
}

async function openSignal(id) {
  const d = await api(`/api/signals/${id}`);
  $('#sig-dialog-title').textContent = `${d.side.toUpperCase()} ${short(d.symbol)} · ${dmy(d.time)} ${hhmm(d.time)}`;
  const t = d.trade;
  const tradeHtml = t ? `<div class="kpis">
      <div class="kpi"><span>Fill</span><b>${px(t.entry_price)}</b></div>
      <div class="kpi"><span>Exit</span><b>${px(t.exit_price)}</b></div>
      <div class="kpi"><span>Reason</span><b>${esc(t.exit_reason || t.status)}</b></div>
      <div class="kpi"><span>Net</span><b class="${dirCls(t.net_pnl)}">${inr(t.net_pnl)}</b></div></div>` : '';
  const feats = Object.entries(d.features).map(([k, v]) => `<div><span>${esc(k)}</span><b>${v == null ? '–' : (+v).toFixed(4)}</b></div>`).join('');
  $('#sig-dialog-content').innerHTML = `
    <div class="kpis">
      <div class="kpi"><span>Prob</span><b>${d.prob == null ? 'rule' : d.prob.toFixed(3)}</b></div>
      <div class="kpi"><span>Entry est.</span><b>${px(d.entry)}</b></div>
      <div class="kpi"><span>SL / Target</span><b>${px(d.stop)} / ${px(d.target)}</b></div>
      <div class="kpi"><span>ATR · Qty</span><b>${num(d.atr)} · ${d.qty}</b></div></div>
    ${d.reason ? `<p class="warning">Blocked: ${esc(d.reason)}</p>` : ''}
    ${tradeHtml}
    <h3>Model inputs at signal time</h3><div class="feat-grid">${feats}</div>`;
  $('#sig-dialog').showModal();
}

function onSignal(s) {
  if (state.day && dayOf(s.time - 60) !== state.day) { loadDaySignals(); }
  else {
    const i = state.daySignals.findIndex(x => x.id === s.id);
    if (i >= 0) state.daySignals[i] = s; else state.daySignals.unshift(s);
    renderSignals();
    $(`#signals .sig[data-id="${s.id}"]`)?.classList.add('fresh');
  }
  if (s.symbol === state.symbol) {
    const i = state.chartSignals.findIndex(x => x.id === s.id);
    if (i >= 0) state.chartSignals[i] = s; else state.chartSignals.push(s);
    updateMarkers();
  }
  if (s.status === 'new') {
    toast(`<b>${s.side === 'long' ? '🟢 LONG' : '🔴 SHORT'} ${esc(short(s.symbol))}</b><br>
      Entry ${px(s.entry)} · SL ${px(s.stop)} · Tgt ${px(s.target)} · Qty ${s.qty}${s.prob != null ? ` · p ${s.prob.toFixed(2)}` : ''}`, s.side === 'long' ? 'good' : 'bad', 10000);
  }
  refreshStatus();
}

const dayOf = ts => new Date(ts * 1000).toLocaleDateString('en-CA', { timeZone: 'Asia/Kolkata' });

// ------------------------------------------------------------------ trades
async function loadTrades() {
  const d = await api('/api/trades?days=30');
  state.trades = d.trades;
  renderTrades();
  updateMarkers();
  updatePriceLines();
}

function renderTrades() {
  const rows = state.trades.map(t => `<tr>
    <td class="mono">${dmy(t.signal_time)} ${hhmm(t.signal_time)}</td>
    <td><a href="#" data-sym="${esc(t.symbol)}">${esc(short(t.symbol))}</a></td>
    <td><span class="tag ${t.side}">${t.side}</span></td>
    <td class="r">${t.qty}</td><td class="r">${px(t.entry_price)}</td><td class="r">${px(t.stop)}</td>
    <td class="r">${px(t.target)}</td><td class="r">${px(t.exit_price)}</td>
    <td>${t.exit_reason ? `<span class="tag ${esc(t.exit_reason)}">${esc(t.exit_reason)}</span>` : ''}</td>
    <td class="r ${dirCls(t.net_pnl)}">${t.net_pnl == null ? '–' : inr(t.net_pnl)}</td>
    <td class="r ${dirCls(t.r_multiple)}">${t.r_multiple == null ? '–' : num(t.r_multiple)}</td>
    <td><span class="tag ${t.status}">${t.status}</span></td>
    <td>${t.status === 'open' || t.status === 'pending' ? `<button class="btn btn-ghost btn-xs" data-close="${t.id}">Close</button>` : ''}</td></tr>`);
  $('#trades-table tbody').innerHTML = rows.join('');
  $('#trades-empty').hidden = state.trades.length > 0;
  $$('#trades-table [data-sym]').forEach(a => a.addEventListener('click', e => { e.preventDefault(); selectSymbol(a.dataset.sym); }));
  $$('#trades-table [data-close]').forEach(b => b.addEventListener('click', async () => {
    b.disabled = true;
    try { await api(`/api/trades/${b.dataset.close}/close`, { method: 'POST' }); } catch (e) { toast(esc(e.message), 'bad'); }
  }));
}

function onTrade(t) {
  const i = state.trades.findIndex(x => x.id === t.id);
  if (i >= 0) state.trades[i] = t; else state.trades.unshift(t);
  renderTrades();
  if (t.symbol === state.symbol) { updateMarkers(); updatePriceLines(); }
  if (t.status === 'closed') {
    toast(`<b>${esc(short(t.symbol))} ${t.side} closed · ${esc(t.exit_reason)}</b><br>Net ${inr(t.net_pnl)} (${num(t.r_multiple)}R)`, t.net_pnl >= 0 ? 'good' : 'bad');
    refreshPnl();
  }
  refreshStatus();
}

// ------------------------------------------------------------------ P&L tab
let pnlChart, pnlSeries;
async function loadPnl() {
  const d = await api('/api/pnl');
  const s = d.summary;
  $('#pnl-kpis').innerHTML = [
    ['Closed trades', s.trades], ['Win rate', s.precision == null ? '–' : `${(s.precision * 100).toFixed(1)}%`],
    ['Net P&L', `<span class="${dirCls(s.net)}">${inr(s.net)}</span>`], ['Costs paid', inr(s.costs)],
    ['Avg R', num(s.avg_r)], ['Max drawdown', inr(s.max_dd)],
  ].map(([k, v]) => `<div class="kpi"><span>${k}</span><b>${v}</b></div>`).join('');
  $('#pnl-table tbody').innerHTML = d.days.map(r => `<tr><td class="mono">${r.day}</td><td class="r">${r.trades}</td>
    <td class="r">${r.wins}</td><td class="r">${r.losses}</td><td class="r">${inr(r.costs)}</td>
    <td class="r ${dirCls(r.net)}">${inr(r.net)}</td><td class="r ${dirCls(r.cumulative)}">${inr(r.cumulative)}</td></tr>`).join('');
  if (!pnlChart) {
    pnlChart = LightweightCharts.createChart($('#pnl-chart'), { ...chartTheme(), autoSize: true });
    pnlSeries = pnlChart.addBaselineSeries({ baseValue: { type: 'price', price: d.capital }, lineWidth: 2, priceFormat: RUPEES });
  }
  pnlSeries.setData(dedupeTimes(d.equity));
  pnlChart.timeScale().fitContent();
}
const refreshPnl = debounce(() => { if ($('#tab-pnl').classList.contains('on')) loadPnl(); }, 800);

// equity points can share an exit second; charts need strictly increasing times
function dedupeTimes(points) {
  const out = [];
  for (const p of points) {
    const t = p.time + IST;
    if (out.length && out.at(-1).time >= t) out.at(-1).value = p.value; else out.push({ time: t, value: p.value });
  }
  return out;
}

// ------------------------------------------------------------------ setups tab
async function loadSetups() {
  state.setups = (await api('/api/setups')).setups;
  renderSetups();
}
function renderSetups() {
  $('#setups-table tbody').innerHTML = state.setups.map(s => `<tr>
    <td class="mono">${dmy(s.time)} ${hhmm(s.time)}</td><td>${esc(short(s.symbol))}</td>
    <td><span class="tag ${s.side}">${s.side}</span></td>
    <td class="r">${s.prob == null ? 'rule' : s.prob.toFixed(3)}</td><td class="r">${num(s.threshold)}</td>
    <td class="r">${px(s.close)}</td><td class="r">${num(s.atr)}</td><td class="r">${num(s.vol_ratio, 1)}</td>
    <td class="r">${s.vwap_dist == null ? '–' : pct(s.vwap_dist * 100)}</td>
    <td>${s.passed ? '<span class="tag new">signal</span>' : '<span class="tag closed">below thr</span>'}</td></tr>`).join('')
    || '<tr><td colspan="10" class="empty">No setups evaluated since the backend started.</td></tr>';
}
function onSetup(s) {
  state.setups.unshift(s);
  state.setups.length = Math.min(state.setups.length, MAX_SETUPS);
  if ($('#tab-setups').classList.contains('on')) renderSetups();
}

// ------------------------------------------------------------------ model tab
let modelChart, modelSeries, baseSeries, trainPoll;
async function loadModel() {
  const d = await api('/api/model');
  const r = d.report, info = d.info;
  renderTraining(d.training);
  if (!r) {
    $('#model-meta').innerHTML = 'No model trained yet. Backfill history, then click <b>Retrain now</b> (or run <code>python -m ml.train</code>).';
    return;
  }
  state.threshold = info.threshold;
  $('#model-meta').innerHTML = `${esc(r.model_kind)} · trained ${esc(r.trained_at.replace('T', ' ').slice(0, 16))} ·
    ${r.data.setups} setups over ${r.data.sessions} sessions (${esc(r.data.first)} → ${esc(r.data.last)}) ·
    threshold <b>${num(info.threshold)}</b> <span class="muted">(${esc(r.threshold_reason)})</span>`;
  $('#model-warnings').innerHTML = (r.warnings || []).map(w => `<div class="warning">⚠ ${esc(w)}</div>`).join('');
  const o = r.oos, b = r.baseline_rule;
  $('#model-kpis').innerHTML = [
    ['OOS trades', o.trades], ['Precision', o.precision == null ? '–' : `${(o.precision * 100).toFixed(1)}%`],
    ['Base win rate', `${(r.data.base_rate * 100).toFixed(1)}%`], ['OOS AUC', num(o.auc, 3)],
    ['Net after costs', `<span class="${dirCls(o.net)}">${inr(o.net)}</span>`], ['Max drawdown', inr(o.max_dd)],
    ['Base rule net', `<span class="${dirCls(b.net)}">${inr(b.net)}</span>`], ['Worst losing streak', o.worst_streak],
  ].map(([k, v]) => `<div class="kpi"><span>${k}</span><b>${v}</b></div>`).join('');
  $('#sweep-table tbody').innerHTML = r.sweep.map(x => `<tr class="${x.threshold === r.threshold ? 'hl' : ''}">
    <td class="r">${num(x.threshold)}</td><td class="r">${x.candidates}</td><td class="r">${x.candidates_per_window}</td>
    <td class="r">${x.precision == null ? '–' : (x.precision * 100).toFixed(1) + '%'}</td><td class="r">${x.trades}</td>
    <td class="r ${dirCls(x.net_pnl)}">${inr(x.net_pnl)}</td><td class="r">${inr(x.max_dd)}</td></tr>`).join('');
  $('#windows-table tbody').innerHTML = r.windows.map(w => `<tr>
    <td class="mono">${esc(w.test_start)} → ${esc(w.test_end)}</td><td class="r">${w.n_test}</td>
    <td class="r">${(w.base_rate * 100).toFixed(1)}%</td><td class="r">${num(w.auc, 3)}</td><td class="r">${w.candidates}</td>
    <td class="r">${w.precision == null ? '–' : (w.precision * 100).toFixed(1) + '%'}</td>
    <td class="r ${dirCls(w.net_pnl)}">${inr(w.net_pnl)}</td></tr>`).join('');
  const maxImp = Math.max(...r.importance.map(i => i.importance), 1e-9);
  $('#importance').innerHTML = r.importance.slice(0, 15).map(i => `<div class="bar-row"><span>${esc(i.feature)}</span>
    <div class="track"><div class="fill" style="width:${(i.importance / maxImp) * 100}%"></div></div><span class="v">${(i.importance * 100).toFixed(1)}</span></div>`).join('');
  if (!modelChart) {
    modelChart = LightweightCharts.createChart($('#model-chart'), { ...chartTheme(), autoSize: true });
    baseSeries = modelChart.addLineSeries({ color: cssVar('--gray'), lineWidth: 1, priceLineVisible: false, priceFormat: RUPEES });
    modelSeries = modelChart.addLineSeries({ color: cssVar('--blue'), lineWidth: 2, priceLineVisible: false, priceFormat: RUPEES });
  }
  modelSeries.setData(dedupeTimes(o.equity || []));
  baseSeries.setData(dedupeTimes(b.equity || []));
  modelChart.timeScale().fitContent();
  renderSignals();
}

function renderTraining(t) {
  const btn = $('#btn-train');
  btn.disabled = t.running;
  btn.textContent = t.running ? 'Training…' : 'Retrain now';
  $('#train-log').textContent = t.log_tail || 'No training run from the UI yet.';
}

async function startTraining() {
  try {
    await api('/api/model/train', { method: 'POST' });
    toast('Training started — this can take a few minutes.');
    clearInterval(trainPoll);
    trainPoll = setInterval(async () => {
      const d = await api('/api/model');
      renderTraining(d.training);
      if (!d.training.running) {
        clearInterval(trainPoll);
        toast(d.training.exit_code === 0 ? 'Model retrained and reloaded.' : 'Training failed — see the log.', d.training.exit_code === 0 ? 'good' : 'bad');
        loadModel();
      }
    }, 3000);
    renderTraining({ running: true, log_tail: 'Starting…' });
  } catch (e) { toast(esc(e.message), 'bad'); }
}

// ------------------------------------------------------------------ status
function pill(id, cls, text) {
  const el = $(`#pill-${id}`);
  el.className = `pill ${cls}`;
  el.querySelector('em').textContent = text;
}

async function loadStatus() {
  const s = await api('/api/status');
  state.status = s;
  pill('market', s.market_open ? 'ok' : '', s.market_open ? 'open' : 'closed');
  if (!s.kite.configured) pill('kite', '', 'not configured');
  else if (s.kite.valid) pill('kite', 'ok', s.kite.user || 'logged in');
  else pill('kite', 'bad', 'login needed');
  $('#btn-login').hidden = !s.kite.configured || s.kite.valid;
  const r = s.reader;
  if (r.alive) pill('reader', 'ok', r.last_candle_time ? `bar ${hhmm(r.last_candle_time)}` : 'live');
  else pill('reader', s.market_open ? 'bad' : '', r.last_ingest_at ? `idle ${hhmm(r.last_ingest_at)}` : 'no data');
  if (s.signal_mode === 'rule') pill('model', 'warn', 'rule mode');
  else if (s.model.loaded) pill('model', 'ok', `${s.model.kind} @ ${num(s.model.threshold)}`);
  else pill('model', 'bad', 'not trained');
  state.threshold = s.model.threshold ?? state.threshold;
  const pb = $('#btn-pause');
  pb.textContent = s.paused ? 'Resume signals' : 'Pause signals';
  pb.classList.toggle('paused', s.paused);
  const t = s.today, c = s.counters || {};
  $('#today-day').textContent = s.market_day;
  $('#today').innerHTML = [
    ['Signals', t.signals], ['Trades', `${t.trades}${t.open ? ` <span class="muted small">(${t.open} open)</span>` : ''}`],
    ['Wins / losses', `${t.wins} / ${t.losses}`], ['Net', `<span class="${dirCls(t.net)}">${inr(t.net)}</span>`],
    ['Setups seen', c.setups ?? '–'], ['Below threshold', c.below_threshold ?? '–'],
  ].map(([k, v]) => `<div class="kpi"><span>${k}</span><b>${v}</b></div>`).join('');
  if (state.day && s.market_day !== state.day) { loadDaySignals(); loadWatchlist(); loadChart(); }
}
const refreshStatus = debounce(() => loadStatus().catch(() => {}), 500);

// ------------------------------------------------------------------ websocket
let ws, pingTimer;
function connectWS() {
  ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/live`);
  ws.onopen = () => {
    pill('ws', 'ok', 'connected');
    clearInterval(pingTimer);
    pingTimer = setInterval(() => ws.readyState === 1 && ws.send('ping'), 25000);
  };
  ws.onclose = () => {
    pill('ws', 'bad', 'reconnecting');
    clearInterval(pingTimer);
    setTimeout(connectWS, 2000);
  };
  ws.onmessage = ev => {
    const m = JSON.parse(ev.data);
    switch (m.type) {
      case 'candle': onCandle(m.data); break;
      case 'signal': onSignal(m.data); break;
      case 'trade': onTrade(m.data); break;
      case 'setup': onSetup(m.data); break;
      case 'watchlist': loadWatchlist(); break;
      case 'model': loadModel(); break;
      case 'status': refreshStatus(); break;
      default: break;
    }
  };
}

// ------------------------------------------------------------------ wiring
function wire() {
  $$('#tf-seg button').forEach(b => b.addEventListener('click', () => {
    $$('#tf-seg button').forEach(x => x.classList.toggle('on', x === b));
    state.tf = +b.dataset.tf; store('tf', state.tf); loadChart();
  }));
  $$('#days-seg button').forEach(b => b.addEventListener('click', () => {
    $$('#days-seg button').forEach(x => x.classList.toggle('on', x === b));
    state.days = +b.dataset.days; store('days', state.days); loadChart();
  }));
  $('#vwap-toggle').addEventListener('change', e => { state.vwapOn = e.target.checked; loadChart(true); });
  $$('#tabs button').forEach(b => b.addEventListener('click', () => {
    $$('#tabs button').forEach(x => x.classList.toggle('on', x === b));
    $$('.tab').forEach(t => t.classList.toggle('on', t.id === `tab-${b.dataset.tab}`));
    store('tab', b.dataset.tab);
    ({ pnl: loadPnl, setups: loadSetups, model: loadModel, trades: loadTrades })[b.dataset.tab]?.().catch(e => toast(esc(e.message), 'bad'));
  }));
  $('#symbol-search').addEventListener('change', e => {
    const v = e.target.value.trim().toUpperCase();
    if (v) selectSymbol(v.includes(':') ? v : `NSE:${v}`);
    e.target.value = '';
  });
  $('#btn-pause').addEventListener('click', async () => {
    const paused = !state.status?.paused;
    await api('/api/control/pause', { method: 'POST', body: JSON.stringify({ paused }) });
    toast(paused ? 'Signals paused.' : 'Signals resumed.');
    loadStatus();
  });
  $('#btn-scan').addEventListener('click', async () => {
    const d = await api('/api/watchlist/scan', { method: 'POST' });
    toast(d.entries.length ? `Watchlist: ${d.entries.length} symbols` : 'Scan found nothing (no data for the first 15 minutes yet?)');
    loadWatchlist();
  });
  $('#btn-train').addEventListener('click', startTraining);
  $('#btn-reload').addEventListener('click', async () => { await api('/api/model/reload', { method: 'POST' }); loadModel(); loadStatus(); toast('Model reloaded.'); });

  const q = new URLSearchParams(location.search);
  if (q.get('login') === 'ok') toast('Kite login successful — token saved for today.', 'good');
  if (q.get('login') === 'failed') toast(`Kite login failed ${esc(q.get('error') || '')}`, 'bad');
  if (q.has('login')) history.replaceState(null, '', '/');
}

function restorePrefs() {
  state.tf = recall('tf', 5);
  state.days = recall('days', 3);
  $$('#tf-seg button').forEach(b => b.classList.toggle('on', +b.dataset.tf === state.tf));
  $$('#days-seg button').forEach(b => b.classList.toggle('on', +b.dataset.days === state.days));
  const tab = recall('tab', 'trades');
  $(`#tabs button[data-tab="${tab}"]`)?.click();
}

async function main() {
  if (!window.LightweightCharts) {
    document.body.innerHTML = '<p class="empty">Could not load the chart library from the CDN. Check your internet connection.</p>';
    return;
  }
  initChart();
  wire();
  restorePrefs();
  try {
    await loadStatus();
    await Promise.all([loadModel(), loadTrades(), loadDaySignals(), loadSymbols()]);
    await loadWatchlist();
  } catch (e) {
    toast(`Failed to load: ${esc(e.message)}`, 'bad', 15000);
  }
  connectWS();
  setInterval(() => loadStatus().catch(() => pill('ws', 'bad', 'backend down')), 10000);
}

main();
