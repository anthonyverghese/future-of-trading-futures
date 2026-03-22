# MNQ Alert System

Monitors MNQ Micro Nasdaq futures during Regular Trading Hours and sends push notifications when price approaches key intraday levels (IBH, IBL, VWAP).

## How It Works

- Runs Monday–Friday, 9:30 AM – 4:00 PM ET
- After 10:30 AM, locks in the **Initial Balance High (IBH)** and **Initial Balance Low (IBL)**
- Recalculates **VWAP** on every tick
- Sends a push notification when MNQ is within **7 points** of any level
- Will not re-alert until price moves more than 20 points away from the level and re-approaches

## Prerequisites

- Python 3.11+
- A [Pushover](https://pushover.net) account + app (~$5 one-time) for push notifications
- A [Databento](https://databento.com) CME Globex MDP 3.0 subscription + API key for real-time MNQ data

## Local Setup

**1. Create a Python 3.11 environment**
```bash
conda create -n mnq python=3.11 -y
conda activate mnq
```

**2. Install dependencies**
```bash
pip install -r mnq_alerts/requirements.txt
```

**3. Configure credentials**

Create `mnq_alerts/.env`:
```
DATABENTO_API_KEY=    # from databento.com → API Keys
PUSHOVER_TOKEN=       # from pushover.net/apps
PUSHOVER_USER_KEY=    # from your Pushover dashboard
```

**4. Run**
```bash
python mnq_alerts/main.py
```

## Deploying to AWS EC2

### One-time instance setup

```bash
# 1. Set timezone to Eastern (required for correct scheduling)
sudo timedatectl set-timezone America/New_York

# 2. Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

# 3. Clone the repo
git clone https://github.com/anthonyverghese/future-of-trading-futures.git
cd future-of-trading-futures

# 4. Create Python 3.11 environment and install dependencies
conda create -n mnq python=3.11 -y
conda activate mnq
pip install -r mnq_alerts/requirements.txt

# 5. Create .env with credentials
cat > mnq_alerts/.env << EOF
DATABENTO_API_KEY=...
PUSHOVER_TOKEN=...
PUSHOVER_USER_KEY=...
DISPLAY_TZ=America/Los_Angeles
EOF
```

### Install systemd services and timers

```bash
sudo cp mnq-alerts.service mnq-alerts.timer mnq-backup.service mnq-backup.timer /etc/systemd/system/
sudo systemctl daemon-reload

# App: starts at 9:30 AM ET weekdays, exits at 4 PM ET
sudo systemctl enable mnq-alerts.timer
sudo systemctl start mnq-alerts.timer

# Backup: copies alerts_log.db to S3 at 4:15 PM ET weekdays
sudo systemctl enable mnq-backup.timer
sudo systemctl start mnq-backup.timer
```

### Set up S3 backup

1. Create an S3 bucket named `mnq-alerts-backup` in the AWS console
2. Attach an IAM role to your EC2 instance with `AmazonS3FullAccess` (EC2 → Instance → Actions → Security → Modify IAM role)

`alerts_log.db` will then be backed up automatically each weekday at 4:15 PM ET.

### Verify everything is running

```bash
sudo systemctl status mnq-alerts.timer mnq-backup.timer
sudo journalctl -u mnq-alerts -f
```

### Update after a code change

```bash
cd future-of-trading-futures
git pull
sudo systemctl restart mnq-alerts
```

## Files

| File | Purpose |
|---|---|
| `mnq_alerts/main.py` | Entry point and main event-driven loop |
| `mnq_alerts/config.py` | All settings (thresholds, timing, credentials) |
| `mnq_alerts/market_data.py` | Databento Live feed integration |
| `mnq_alerts/levels.py` | Calculates IBH/IBL and VWAP from trade ticks |
| `mnq_alerts/alert_manager.py` | Alert state tracking and notification logic |
| `mnq_alerts/notifications.py` | Sends push notifications via Pushover |
| `mnq_alerts/cache.py` | SQLite session cache and alerts log |
| `mnq_alerts/outcome_tracker.py` | Evaluates recommendation correctness |
| `mnq_alerts/clear_cache.py` | Deletes the session cache |
| `mnq-alerts.service` | systemd service for the app |
| `mnq-alerts.timer` | systemd timer — starts app at 9:30 AM ET weekdays |
| `mnq-backup.service` | systemd service for S3 backup |
| `mnq-backup.timer` | systemd timer — backs up to S3 at 4:15 PM ET weekdays |
