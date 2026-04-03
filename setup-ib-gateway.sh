#!/bin/bash
# setup-ib-gateway.sh — Install Docker and launch IB Gateway container on EC2
#
# Run once on EC2:
#   chmod +x setup-ib-gateway.sh
#   ./setup-ib-gateway.sh
#
# Prerequisites: fill in ib-gateway.env with your IBKR credentials first.

set -euo pipefail

REPO_DIR="/home/ec2-user/future-of-trading-futures"

# ── Install Docker ───────────────────────────────────────────────────────────
if ! command -v docker &> /dev/null; then
    echo "=== Installing Docker ==="
    sudo dnf install -y docker
    sudo systemctl enable --now docker
    sudo usermod -aG docker ec2-user
    echo "=== Docker installed. You may need to log out and back in for group changes. ==="
fi

# ── Install Docker Compose plugin ────────────────────────────────────────────
if ! docker compose version &> /dev/null; then
    echo "=== Installing Docker Compose plugin ==="
    sudo mkdir -p /usr/local/lib/docker/cli-plugins
    sudo curl -SL "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-$(uname -m)" \
        -o /usr/local/lib/docker/cli-plugins/docker-compose
    sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
fi

# ── Validate config ──────────────────────────────────────────────────────────
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

# ── Sunday 2FA reminder (Pushover notification) ─────────────────────────────
echo "=== Installing 2FA reminder timer ==="
sudo cp "${REPO_DIR}/mnq-2fa-reminder.service" "${REPO_DIR}/mnq-2fa-reminder.timer" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now mnq-2fa-reminder.timer

# ── Launch IB Gateway container ──────────────────────────────────────────────
echo "=== Starting IB Gateway container ==="
cd "$REPO_DIR"
docker compose up -d ib-gateway

echo ""
echo "=== Setup complete ==="
echo ""
echo "IB Gateway is starting. Check logs:"
echo "  docker compose logs -f ib-gateway"
echo ""
echo "VNC access for weekly 2FA (from your local machine):"
echo "  ssh -NL 5900:localhost:5900 -i FuturesTrader.pem ec2-user@<EC2_IP>"
echo "  Then open vnc://localhost:5900 (password: in ib-gateway.env)"
echo ""
echo "API available at localhost:4002 (paper trading)"
