#!/bin/bash
# setup-ib-gateway.sh — Install Docker and launch IB Gateway container on EC2
#
# Run once on EC2:
#   chmod +x setup-ib-gateway.sh
#   ./setup-ib-gateway.sh
#
# Prerequisites: fill in ib-gateway.env and mnq_alerts/.env with credentials first.

set -euo pipefail

REPO_DIR="/home/ec2-user/future-of-trading-futures"

# ── Install Docker ───────────────────────────────────────────────────────────
if ! command -v docker &> /dev/null; then
    echo "=== Installing Docker ==="
    sudo dnf install -y docker
    sudo systemctl enable --now docker
    sudo usermod -aG docker ec2-user
fi

# ── Install Docker Compose plugin ────────────────────────────────────────────
if ! sudo docker compose version &> /dev/null; then
    echo "=== Installing Docker Compose plugin ==="
    sudo mkdir -p /usr/local/lib/docker/cli-plugins
    sudo curl -SL "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-$(uname -m)" \
        -o /usr/local/lib/docker/cli-plugins/docker-compose
    sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
fi

# ── Validate configs ─────────────────────────────────────────────────────────
ENV_FILE="${REPO_DIR}/ib-gateway.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: ${ENV_FILE} not found."
    echo "Copy ib-gateway.env.example and fill in your credentials."
    exit 1
fi

if grep -q "TWS_USERID=$" "$ENV_FILE" || grep -q "TWS_PASSWORD=$" "$ENV_FILE"; then
    echo "ERROR: TWS_USERID and TWS_PASSWORD must be set in ${ENV_FILE}"
    exit 1
fi

APP_ENV="${REPO_DIR}/mnq_alerts/.env"
if ! grep -q "IBKR_TRADING_ENABLED=true" "$APP_ENV" 2>/dev/null; then
    echo "WARNING: IBKR_TRADING_ENABLED=true not found in ${APP_ENV}"
    echo "The bot will run but won't submit orders."
fi

if ! grep -q "IBKR_ACCOUNT=" "$APP_ENV" 2>/dev/null; then
    echo "WARNING: IBKR_ACCOUNT not set in ${APP_ENV} — no account safety check"
fi

# ── Update systemd services ──────────────────────────────────────────────────
echo "=== Updating systemd services ==="
sudo cp "${REPO_DIR}/mnq-alerts.service" "${REPO_DIR}/mnq-alerts.timer" /etc/systemd/system/
sudo cp "${REPO_DIR}/mnq-backup.service" "${REPO_DIR}/mnq-backup.timer" /etc/systemd/system/
sudo cp "${REPO_DIR}/mnq-2fa-reminder.service" "${REPO_DIR}/mnq-2fa-reminder.timer" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable mnq-2fa-reminder.timer
sudo systemctl start mnq-2fa-reminder.timer

# ── Launch IB Gateway container ──────────────────────────────────────────────
echo "=== Starting IB Gateway container ==="
cd "$REPO_DIR"
sudo docker compose up -d ib-gateway

echo ""
echo "=== Setup complete ==="
echo ""
echo "IB Gateway is starting. Check logs:"
echo "  sudo docker compose logs -f ib-gateway"
echo ""
echo "Next: VNC in to approve 2FA (from your local machine):"
echo "  ssh -NL 5900:localhost:5900 -i FuturesTrader.pem ec2-user@<EC2_IP>"
echo "  Then open vnc://localhost:5900"
echo ""
echo "After 2FA is approved, restart the trading app:"
echo "  sudo systemctl restart mnq-alerts"
