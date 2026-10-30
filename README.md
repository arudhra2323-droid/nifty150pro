# Nifty150 Pro — working core

A FastAPI backend plus a single-page dark-navy frontend for bullish swing-trade screening.
It runs end-to-end in **DEMO mode** with no credentials.

> Not financial advice. A 70% probability threshold is a model filter, not a guarantee.

## Run locally

```bash
cd nifty150pro
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # MARKET_DATA_PROVIDER=demo
uvicorn backend.app:app --reload --port 8000
```

Open http://localhost:8000. Pages: `#/dashboard`, `#/screener`, `#/universe`, `#/strategies`, `#/stock/RELIANCE`.

Run tests:

```bash
python -m pytest -q
```

## What is implemented

- **Indicators** (`backend/indicators.py`): EMA, RSI (Wilder), MACD, ATR, Supertrend. Support/resistance and relative volume are in `analysis.py`.
- **Five strategy modules** (`backend/strategies.py`), each with version, explicit conditions, entry, stop, target and invalidation:
  1. Trend Pullback Confirmation
  2. Breakout with Volume Confirmation
  3. Supertrend + weekly trend continuation
  4. RSI Recovery Near Support
  5. Fundamental Quality + Relative Strength. **Not evaluated** because no fundamentals provider exists. Missing data is never treated as positive.
- **Backtesting** (`backend/analysis.py`): pooled across the universe, no overlapping trades per symbol, next-bar-open entry, 1.5×ATR stop, 2R target, 15-session horizon, 0.25% round-trip cost, stop checked first when both levels are hit in the same bar, chronological 60/40 in-sample/out-of-sample split.
- **Probability validation**: a probability is shown as validated only when the data is live and there are at least 30 out-of-sample trades. Otherwise it shows as "not validated". Wilson 95% lower bound is also reported.
- **Ranking**: expected net return = p·(2R − cost) − (1−p)·(R + cost), with R = stop distance. Risk-adjusted score = expected net return ÷ (stop distance + cost).
- **Explanation endpoint** `GET /api/stocks/{symbol}/explanation`: structured evidence (actual readings, interpretation, contribution), strategy match with satisfied and unsatisfied conditions, probability meaning, trade plan, what could go wrong, reasons not to trade, and a decision state. The AI narrative is intentionally `null` and not wired up.
- **Data safety**: demo data is labeled `DEMO` everywhere. Demo signals are shown as "Signal, not validated", never as Top 5 picks. Live mode is not silently substituted.

## What is NOT built yet

| Area | Status |
|---|---|
| Live market data adapter | Interface only (`market_data.py`). Raises a clear error if `live` is chosen. |
| Official Nifty 150 constituents | Demo sample of 30 symbols. Needs a licensed index-constituent source. |
| Market cap, fundamentals, news | Need provider adapters. Shown as unavailable. |
| Watchlists, paper trading, portfolio | Not built. Needs PostgreSQL and auth. |
| Alerts and Web Push / PWA | Not built. Needs VAPID keys and a service worker. |
| Settings page, Alerts page, Portfolio page | Not built. |
| Strategy 5 | Needs fundamentals data (see above). |
| Chart timeframes 5m / 15m / 1h / weekly | Only daily candles. Intraday needs a live provider. |
| AI narrative | Not configured. |

## Connections needed from you

1. **Market-data provider** with licensed NSE end-of-day OHLCV and ideally intraday data. Examples include broker APIs (Zerodha Kite Connect, Angel One SmartAPI), NSE data vendors, or Global Datafeeds. Check each provider's redistribution terms. Once chosen, I will write its adapter in `market_data.py` and set `MARKET_DATA_PROVIDER` plus its key.
2. **Nifty 150 constituent list** from NSE index files or a vendor.
3. **Fundamentals provider** for revenue, earnings, ROE, debt and valuation.
4. **News and corporate announcements** provider (NSE corporate announcements feed or a licensed news API).
5. **PostgreSQL** database and a VAPID key pair for push notifications.

Put every key in `.env`. Never paste keys into chat.
