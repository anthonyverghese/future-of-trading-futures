# MNQ Alert System

Monitors MNQ Micro Nasdaq futures during Regular Trading Hours and sends push notifications when price approaches key intraday levels (IBH, IBL, VWAP).

## How It Works

- Runs Monday–Friday, 9:30 AM – 4:00 PM ET
- After 10:30 AM, locks in the **Initial Balance High (IBH)** and **Initial Balance Low (IBL)**
- Recalculates **VWAP** on every tick
- Sends a push notification when MNQ is within **15 points** of any level
- Will not re-alert until price moves more than 15 points away and re-approaches

## Prerequisites

- Python 3.8+
- A [Pushover](https://pushover.net) account + app (~$5 one-time) for push notifications
- A [Databento](https://databento.com) account + API key for real-time MNQ data (~$5–15/month)

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Configure credentials**

Create a `.env` file in the `mnq_alerts/` directory with the following fields:
```
PUSHOVER_TOKEN=       # Pushover app API token (from pushover.net/apps)
PUSHOVER_USER_KEY=    # Pushover user key (from your Pushover dashboard)
DATABENTO_API_KEY=    # Databento API key (from databento.com → API Keys)
```

**3. Run**
```bash
python main.py
```

The app will sleep until 9:30 AM ET and start monitoring automatically. Press `Ctrl+C` to stop.

## Files

| File | Purpose |
|---|---|
| `main.py` | Entry point and main polling loop |
| `config.py` | All settings (thresholds, timing, credentials) |
| `market_data.py` | Fetches real-time bars from Databento |
| `levels.py` | Calculates IBH/IBL and VWAP |
| `alert_manager.py` | Alert state tracking and notification logic |
| `notifications.py` | Sends push notifications via Pushover |
