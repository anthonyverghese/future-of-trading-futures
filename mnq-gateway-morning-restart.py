#!/usr/bin/env python3
"""mnq-gateway-morning-restart.py — Pre-market docker restart of IB Gateway.

Fires via the matching systemd timer every trading day at 07:30 ET to
recycle the gateway JVM before mnq-alerts.timer triggers at 09:30 ET.
Two consecutive nights (2026-04-13 and 2026-04-14) the gateway got
stuck at ~00:47 ET after IBKR force-disconnected the paper session and
IBC's Re-login attempt landed on an "Unrecognized Username or Password"
dialog. `docker restart` (preserves container filesystem → IBC's
autorestart file from the 11:59 PM cycle is still consumable → silent
re-auth, no 2FA) reliably clears the stuck state.

Notifies via Pushover on any failure so the user has ~2 hours to
recover manually before market open.
"""

from __future__ import annotations

import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "mnq_alerts"))
from notifications import PRIORITY_HIGH, send_notification

CONTAINER = "future-of-trading-futures-ib-gateway-1"
BOOT_WAIT_SECS = 45
LOG_SCRAPE_WINDOW_SECS = BOOT_WAIT_SECS + 15

# Patterns that indicate the restart failed. Keep these narrow — matching
# on intermediate dialog frames like "Password Notice" would false-positive
# on healthy runs where IBC auto-dismisses the nag.
BAD_PATTERNS = (
    "Unrecognized Username or Password",
    "Too many failed login attempts",
)

SUCCESS_MARKER = "Login has completed"


def run(cmd: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def notify_fail(reason: str) -> None:
    print(f"[morning-restart] FAIL: {reason}", flush=True)
    try:
        send_notification(
            title="Morning Gateway Restart Failed",
            message=(
                f"{reason}\n\n"
                "Manual recovery needed before 09:30 ET market open.\n"
                f"  sudo docker logs --tail 40 {CONTAINER}\n"
                f"  sudo docker restart {CONTAINER}\n"
                "VNC if a dialog is stuck."
            ),
            priority=PRIORITY_HIGH,
        )
    except Exception as exc:
        print(f"[morning-restart] Pushover send error: {exc}", flush=True)


def main() -> int:
    print(f"[morning-restart] Restarting {CONTAINER}", flush=True)
    try:
        result = run(["sudo", "docker", "restart", CONTAINER])
    except Exception as exc:
        notify_fail(f"docker restart raised: {exc}")
        return 1
    if result.returncode != 0:
        notify_fail(
            f"docker restart exit {result.returncode}: {result.stderr.strip()}"
        )
        return 1

    print(
        f"[morning-restart] Waiting {BOOT_WAIT_SECS}s for gateway to initialize",
        flush=True,
    )
    time.sleep(BOOT_WAIT_SECS)

    try:
        inspect = run(
            ["sudo", "docker", "inspect", "-f", "{{.State.Running}}", CONTAINER]
        )
    except Exception as exc:
        notify_fail(f"docker inspect raised: {exc}")
        return 1
    if inspect.stdout.strip() != "true":
        notify_fail(f"Container not running after restart: {inspect.stdout.strip()}")
        return 1

    try:
        ss = run(["ss", "-tln"], timeout=5)
    except Exception as exc:
        notify_fail(f"ss raised: {exc}")
        return 1
    if "127.0.0.1:4002" not in ss.stdout:
        notify_fail("Port 4002 not listening on host after restart")
        return 1

    try:
        logs = run(
            [
                "sudo",
                "docker",
                "logs",
                "--since",
                f"{LOG_SCRAPE_WINDOW_SECS}s",
                CONTAINER,
            ],
            timeout=15,
        )
    except Exception as exc:
        notify_fail(f"docker logs raised: {exc}")
        return 1
    combined = logs.stdout + logs.stderr
    for pat in BAD_PATTERNS:
        if pat in combined:
            notify_fail(f"Bad marker in gateway logs: {pat!r}")
            return 1
    if SUCCESS_MARKER not in combined:
        notify_fail(
            f"{SUCCESS_MARKER!r} not seen in gateway logs within "
            f"{BOOT_WAIT_SECS}s boot window"
        )
        return 1

    print(
        "[morning-restart] Success — gateway logged in, port 4002 listening",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        notify_fail(f"Unhandled exception: {exc}\n{traceback.format_exc()[:500]}")
        sys.exit(1)
