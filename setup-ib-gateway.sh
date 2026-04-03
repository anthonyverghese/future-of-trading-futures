#!/bin/bash
# setup-ib-gateway.sh — Install IB Gateway + IBC + Xvfb on Amazon Linux 2023 EC2
#
# Run once on EC2:
#   chmod +x setup-ib-gateway.sh
#   ./setup-ib-gateway.sh
#
# After running, configure credentials:
#   1. Edit /home/ec2-user/ibc/config.ini — set IbLoginId and IbPassword
#   2. Edit /home/ec2-user/future-of-trading-futures/mnq_alerts/.env — add IBKR lines
#   3. Install systemd services (see bottom of script)

set -euo pipefail

echo "=== Installing dependencies ==="
sudo dnf install -y java-11-amazon-corretto-headless xorg-x11-server-Xvfb unzip wget

# ── IB Gateway (latest stable) ───────────────────────────────────────────────
# Download from IBKR's site. The "stable" channel is recommended for automated trading.
IB_GATEWAY_VERSION="1030"  # 10.30 — update if newer stable is available
IB_GATEWAY_INSTALLER="ibgateway-${IB_GATEWAY_VERSION}-standalone-linux-x64.sh"

echo "=== Downloading IB Gateway ${IB_GATEWAY_VERSION} ==="
cd /tmp
if [ ! -f "${IB_GATEWAY_INSTALLER}" ]; then
    wget -q "https://download2.interactivebrokers.com/installers/ibgateway/stable-standalone/ibgateway-${IB_GATEWAY_VERSION}-standalone-linux-x64.sh"
fi

echo "=== Installing IB Gateway ==="
chmod +x "${IB_GATEWAY_INSTALLER}"
# Silent install to /home/ec2-user/Jts
yes "" | sh "${IB_GATEWAY_INSTALLER}" -q -dir /home/ec2-user/Jts 2>/dev/null || true

# ── IBC (IB Controller — auto-login and auto-restart) ────────────────────────
IBC_VERSION="3.19.0"
IBC_ZIP="IBCLinux-${IBC_VERSION}.zip"

echo "=== Downloading IBC ${IBC_VERSION} ==="
cd /tmp
if [ ! -f "${IBC_ZIP}" ]; then
    wget -q "https://github.com/IbcAlpha/IBC/releases/download/${IBC_VERSION}/${IBC_ZIP}"
fi

echo "=== Installing IBC ==="
mkdir -p /home/ec2-user/ibc
unzip -o "/tmp/${IBC_ZIP}" -d /home/ec2-user/ibc
chmod +x /home/ec2-user/ibc/*.sh

# ── IBC config ───────────────────────────────────────────────────────────────
# Copy our config template if no config exists yet
if [ ! -f /home/ec2-user/ibc/config.ini ]; then
    cp /home/ec2-user/future-of-trading-futures/ibc-config.ini /home/ec2-user/ibc/config.ini
    echo "=== Created /home/ec2-user/ibc/config.ini — EDIT THIS with your IBKR credentials ==="
else
    echo "=== IBC config already exists at /home/ec2-user/ibc/config.ini ==="
fi

# ── Systemd services ─────────────────────────────────────────────────────────
echo "=== Installing systemd services ==="
sudo cp /home/ec2-user/future-of-trading-futures/xvfb.service /etc/systemd/system/
sudo cp /home/ec2-user/future-of-trading-futures/ib-gateway.service /etc/systemd/system/
sudo systemctl daemon-reload

# Enable Xvfb (always running) and IB Gateway timer (starts before market open)
sudo systemctl enable --now xvfb.service
sudo systemctl enable ib-gateway.service

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit /home/ec2-user/ibc/config.ini"
echo "     - Set IbLoginId=YOUR_IBKR_USERNAME"
echo "     - Set IbPassword=YOUR_IBKR_PASSWORD"
echo ""
echo "  2. Add to mnq_alerts/.env:"
echo "     IBKR_TRADING_ENABLED=true"
echo "     IBKR_PORT=4002"
echo ""
echo "  3. Test IB Gateway manually:"
echo "     sudo systemctl start ib-gateway"
echo "     sudo journalctl -u ib-gateway -f"
echo ""
echo "  4. Test the trading app:"
echo "     sudo systemctl restart mnq-alerts"
echo "     sudo journalctl -u mnq-alerts -f"
