#!/bin/bash
# SessionStart hook: inject yesterday's alert stats into context.
# Gracefully skips if alerts_log.db doesn't exist (e.g., local dev — DB is on EC2).

DB="$(dirname "$(dirname "$0")")/../mnq_alerts/alerts_log.db"

if [ ! -f "$DB" ]; then
  echo '{"hookSpecificOutput":{"hookEventName":"SessionStart","additionalContext":"alerts_log.db not found locally — SQLite stats unavailable. DB lives on EC2."}}'
  exit 0
fi

STATS=$(sqlite3 "$DB" "
  SELECT
    COUNT(*) as total,
    SUM(CASE WHEN outcome='correct' THEN 1 ELSE 0 END) as wins,
    SUM(CASE WHEN outcome='incorrect' THEN 1 ELSE 0 END) as losses
  FROM alerts
  WHERE date = date('now', '-1 day');
")

IFS='|' read -r TOTAL WINS LOSSES <<< "$STATS"

if [ "$TOTAL" -gt 0 ] 2>/dev/null; then
  DECIDED=$((WINS + LOSSES))
  if [ "$DECIDED" -gt 0 ]; then
    WR=$(echo "scale=1; $WINS * 100 / $DECIDED" | bc)
    MSG="Yesterday: ${TOTAL} alerts, ${WINS}W/${LOSSES}L (${WR}% win rate)"
  else
    MSG="Yesterday: ${TOTAL} alerts (all inconclusive)"
  fi
else
  MSG="No alerts recorded yesterday"
fi

echo "{\"hookSpecificOutput\":{\"hookEventName\":\"SessionStart\",\"additionalContext\":\"Daily alert stats: $MSG\"}}"
