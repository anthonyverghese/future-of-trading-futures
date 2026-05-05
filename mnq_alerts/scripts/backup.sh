#!/bin/bash
# Backup alerts_log.db + today's parquet to S3, then verify and clean.
# If today's parquet didn't make it to S3, page via Pushover.
# Sourced credentials: PUSHOVER_TOKEN, PUSHOVER_USER_KEY (from mnq_alerts/.env).
set -euo pipefail

REPO="/home/ec2-user/future-of-trading-futures"
DATA_DIR="$REPO/mnq_alerts/data_cache"
DB="$REPO/mnq_alerts/alerts_log.db"
BUCKET="s3://mnq-alerts-backup"
TODAY=$(date +%Y-%m-%d)
SQLITE3=/home/ec2-user/miniconda3/bin/sqlite3

aws s3 cp "$DB" "$BUCKET/alerts_log.db"
aws s3 sync "$DATA_DIR/" "$BUCKET/data_cache/" \
    --exclude "*" --include "MNQ_*.parquet"
echo "[backup] S3 upload complete — verifying today's parquet"

# Skip parquet check on days the bot didn't run (US market holiday, service down).
ALERT_N=$($SQLITE3 "$DB" "SELECT COUNT(*) FROM alerts WHERE date='$TODAY'" 2>/dev/null || echo 0)
TRADE_N=$($SQLITE3 "$DB" "SELECT COUNT(*) FROM bot_trades WHERE date='$TODAY'" 2>/dev/null || echo 0)

if [ "$ALERT_N" -eq 0 ] && [ "$TRADE_N" -eq 0 ]; then
    echo "[backup] No alerts/trades today (holiday or service down) — skipping parquet check"
elif ! aws s3 ls "$BUCKET/data_cache/MNQ_$TODAY.parquet" > /dev/null 2>&1; then
    echo "[backup] ERROR: MNQ_$TODAY.parquet NOT in S3 (alerts=$ALERT_N trades=$TRADE_N)"
    if [ -n "${PUSHOVER_TOKEN:-}" ] && [ -n "${PUSHOVER_USER_KEY:-}" ]; then
        curl -s --max-time 10 \
            --form-string "token=$PUSHOVER_TOKEN" \
            --form-string "user=$PUSHOVER_USER_KEY" \
            --form-string "title=MNQ parquet missing" \
            --form-string "message=MNQ_$TODAY.parquet not uploaded to S3 (alerts=$ALERT_N, trades=$TRADE_N). Check mnq-alerts journal for export_daily_parquet errors." \
            --form-string "priority=1" \
            https://api.pushover.net/1/messages.json > /dev/null || true
    fi
    exit 1
fi

echo "[backup] cleaning old parquets"
find "$DATA_DIR/" -name "MNQ_*.parquet" -mtime +7 -delete
echo "[backup] Done"
