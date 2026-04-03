# MNQ Alert System — Project Guide

## What this is
A Python app that monitors MNQ (Micro Nasdaq futures) in real time and sends
Pushover push notifications when price approaches key intraday levels:
- **IBH / IBL** — Initial Balance High/Low (9:30–10:30 AM ET, locked after 10:30)
- **VWAP** — Session volume-weighted average price (updates on every trade tick)

Alerts fire once when price enters within 10 points of a level, then reset
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
| `ALERT_THRESHOLD_POINTS` | 7 | Points from a level to trigger alert |
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

# Optional — paper trading (disabled by default)
IBKR_TRADING_ENABLED=false
IBKR_PORT=4002
IBKR_ACCOUNT=DU1234567
```

Pushover notifications use priority 1 (high — bypasses quiet hours).
If credentials are missing, notifications fall back to stdout.

## Display timezone
Timestamps auto-detect the system's local timezone. Override with `DISPLAY_TZ`
in `.env` (e.g. `DISPLAY_TZ=America/Los_Angeles`). Useful on EC2 where the
system timezone defaults to UTC.

## Deploying to AWS EC2

### One-time instance setup

```bash
# 1. Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 2. Clone the repo
git clone https://github.com/anthonyverghese/future-of-trading-futures.git
cd future-of-trading-futures

# 3. Create Python 3.11 environment and install dependencies
conda create -n mnq python=3.11 -y
conda activate mnq
pip install -r mnq_alerts/requirements.txt

# 4. Create .env with credentials
cat > mnq_alerts/.env <<EOF
DATABENTO_API_KEY=...
PUSHOVER_TOKEN=...
PUSHOVER_USER_KEY=...
DISPLAY_TZ=America/Los_Angeles
EOF
```

### Install systemd services and timers

```bash
# Copy all service and timer files
sudo cp mnq-alerts.service mnq-alerts.timer mnq-backup.service mnq-backup.timer /etc/systemd/system/

sudo systemctl daemon-reload

# App: starts at 9:30 AM ET weekdays, exits at 4 PM ET
sudo systemctl enable mnq-alerts.timer
sudo systemctl start mnq-alerts.timer

# Backup: copies alerts_log.db to S3 at 4:15 PM ET weekdays
sudo systemctl enable mnq-backup.timer
sudo systemctl start mnq-backup.timer

# Check logs
sudo journalctl -u mnq-alerts -f
```

### Paper trading with IB Gateway (optional)

Enables automated bracket order submission (market entry + limit TP + stop SL)
when alerts fire. Disabled by default — requires IB Gateway running in Docker.

Uses [gnzsnz/ib-gateway-docker](https://github.com/gnzsnz/ib-gateway-docker) for
IB Gateway + IBC + VNC in a single container. 2FA required once per week via VNC.

```bash
# 1. Configure IB Gateway credentials
cp ib-gateway.env.example ib-gateway.env
vi ib-gateway.env   # set TWS_USERID, TWS_PASSWORD, VNC_SERVER_PASSWORD

# 2. Enable trading in the app's .env
echo "IBKR_TRADING_ENABLED=true" >> mnq_alerts/.env
echo "IBKR_PORT=4002" >> mnq_alerts/.env
echo "IBKR_ACCOUNT=DU1234567" >> mnq_alerts/.env  # your paper account ID

# 3. Run setup (installs Docker, starts container)
chmod +x setup-ib-gateway.sh
./setup-ib-gateway.sh

# 4. Complete 2FA via VNC (from your local machine):
#    ssh -NL 5900:localhost:5900 -i FuturesTrader.pem ec2-user@<EC2_IP>
#    Then open vnc://localhost:5900 and approve the IBKR Mobile prompt

# 5. Restart the trading app
sudo systemctl restart mnq-alerts
```

Weekly maintenance: VNC in on Sunday evening to approve the 2FA prompt.
IB Gateway auto-restarts daily at 11:59 PM ET without re-prompting.

Risk controls (validated over 214 days in `bot_risk_backtest.py`):
- 1 position at a time (no stacking)
- $150/day loss limit — stops trading for the day
- 3 consecutive losses — stops trading for the day
- Target: +12 pts ($24), Stop: -25 pts ($50)

### Updating the app after a code change

```bash
cd future-of-trading-futures
git pull
sudo systemctl restart mnq-alerts
```

### SQLite on EC2
Both databases live on the EBS root volume and persist across reboots.
`alerts_log.db` is automatically backed up to S3 at 4:15 PM ET each weekday
via the `mnq-backup.timer` systemd unit.
