#!/bin/bash
# Pull alerts_log.db from S3 backup to local machine for analysis.
# Run after market close (backup fires at 4:15 PM ET).
#
# Usage: ./scripts/sync-db.sh

set -euo pipefail

BUCKET="s3://mnq-alerts-backup/alerts_log.db"
LOCAL="$(dirname "$0")/../mnq_alerts/alerts_log.db"

echo "Pulling alerts_log.db from S3..."
aws s3 cp "$BUCKET" "$LOCAL"
echo "Done — $(wc -c < "$LOCAL" | tr -d ' ') bytes → $LOCAL"
