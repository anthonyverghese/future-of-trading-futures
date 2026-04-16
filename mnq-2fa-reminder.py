#!/usr/bin/env python3
"""Send a Pushover reminder to approve IB Gateway 2FA via VNC."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "mnq_alerts"))

from notifications import send_notification, PRIORITY_HIGH

send_notification(
    title="IB Gateway 2FA Reminder",
    message=(
        "Approve tonight's IB Gateway 2FA login before tomorrow's open.\n\n"
        "1. ssh -NL 5900:localhost:5900 -i FuturesTrader.pem ec2-user@<EC2_IP>\n"
        "2. Open vnc://localhost:5900\n"
        "3. Approve the prompt on IBKR Mobile"
    ),
    priority=PRIORITY_HIGH,
)
