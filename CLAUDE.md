# MNQ Alert System — Project Guide

## What this is
A Python app that monitors MNQ (Micro Nasdaq futures) in real time and sends
Pushover push notifications when price approaches key intraday levels:
- **IBH / IBL** — Initial Balance High/Low (9:30–10:30 AM ET, locked after 10:30)
- **VWAP** — Session volume-weighted average price (updates on every trade tick)

Alerts fire once when price enters within 15 points of a level, then reset
after price exits, preventing notification spam.

## Architecture

```
main.py           — Event-driven loop; iterates over trade_stream()
market_data.py    — Databento Live feed; accumulates session trades
levels.py         — VWAP and IB calculations from trade tick data
alert_manager.py  — Per-level state machine (in_zone tracking)
notifications.py  — Pushover API integration
config.py         — All tunables; credentials loaded from .env
```

## Data source
**Databento Live API** — `GLBX.MDP3` dataset, `trades` schema, symbol `MNQ.c.0`
(continuous front-month, auto-rolled). Requires a CME Globex MDP 3.0 standard
subscription. Each `TradeMsg` has `price` (fixed-point int ÷ 1e9) and `size`.

The live feed replaced a historical REST API approach that polled every 30s.

## Key design decisions

- **Event-driven, not polling.** `trade_stream()` blocks on each incoming trade.
  VWAP recalculates on every tick; alerts fire with no artificial delay.
- **VWAP from ticks.** `Σ(Price × Size) / Σ(Size)` over raw trades — more
  accurate than the bar-based typical price approximation `(H+L+C)/3`.
- **IB from trade prices.** High/low during 9:30–10:30 AM from actual trade
  prices, not OHLCV bar H/L.
- **IB is locked.** After 10:30 AM ET, IBH/IBL are fixed for the session.
  VWAP continues to drift and is updated on every trade.
- **Console output throttled** to once per 5 seconds to stay readable at
  tick frequency.
- **Futures trade 24/5.** The live feed streams outside RTH too. Trades outside
  9:30 AM – 4:00 PM ET are consumed but skipped for alert processing.
- **Auto-reconnect.** `trade_stream()` catches feed errors and reconnects.

## Running locally

```bash
cd mnq_alerts
cp .env.example .env        # fill in DATABENTO_API_KEY, PUSHOVER_TOKEN, PUSHOVER_USER_KEY
pip install -r requirements.txt
python main.py
```

Requires Python 3.9+. The `databento` SDK includes a compiled Rust extension
(`databento-dbn`) — pre-built wheels are available for common platforms.

## Configuration (config.py)

| Setting | Default | Description |
|---|---|---|
| `ALERT_THRESHOLD_POINTS` | 15 | Points from a level to trigger alert |
| `MARKET_OPEN` | 9:30 AM ET | RTH start |
| `MARKET_CLOSE` | 4:00 PM ET | RTH end |
| `IB_END` | 10:30 AM ET | IB window close |
| `DATABENTO_DATASET` | `GLBX.MDP3` | CME Globex futures |
| `DATABENTO_SYMBOL` | `MNQ.c.0` | Continuous front-month MNQ |

## Credentials (.env)

```
DATABENTO_API_KEY=...
PUSHOVER_TOKEN=...
PUSHOVER_USER_KEY=...
```

Pushover notifications use priority 1 (high — bypasses quiet hours).
If credentials are missing, notifications fall back to stdout.

## Display timezone
All timestamps are displayed in **Pacific Time (PT)**. Market hours shown as
6:30 AM – 1:00 PM PT.
