"""
ib-gateway-watchdog.py — Monitors IB Gateway login attempts and stops the
container after too many failures, sending a Pushover notification.

Watches `docker logs --follow` for IBC login events and tracks failure count.
After 4 consecutive failures, stops the container and notifies the user.
The container stays stopped until manually restarted.

Designed to run as a systemd service alongside the IB Gateway container.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Load Pushover credentials from .env
ENV_PATH = Path(__file__).parent / "mnq_alerts" / ".env"
PUSHOVER_TOKEN = ""
PUSHOVER_USER_KEY = ""
if ENV_PATH.exists():
    for line in ENV_PATH.read_text().splitlines():
        if line.startswith("PUSHOVER_TOKEN="):
            PUSHOVER_TOKEN = line.split("=", 1)[1].strip().strip('"')
        elif line.startswith("PUSHOVER_USER_KEY="):
            PUSHOVER_USER_KEY = line.split("=", 1)[1].strip().strip('"')

CONTAINER_NAME = "future-of-trading-futures-ib-gateway-1"
MAX_FAILURES = 4

# Patterns to track
LOGIN_ATTEMPT_RE = re.compile(r"IBC: Login attempt: (\d+)")
LOGIN_FAILED_RE = re.compile(
    r"IBC:.*(Too many failed login attempts|Login.*failed|password.*incorrect|Invalid login)",
    re.IGNORECASE,
)
LOGIN_SUCCESS_RE = re.compile(
    r"IBC:.*(Login has completed|Logged in|Connected to)",
    re.IGNORECASE,
)


def send_pushover(title: str, message: str) -> None:
    """Send a Pushover notification."""
    if not PUSHOVER_TOKEN or not PUSHOVER_USER_KEY:
        print(f"[watchdog] No Pushover creds — would have sent: {title}: {message}")
        return
    try:
        import urllib.parse
        import urllib.request

        data = urllib.parse.urlencode(
            {
                "token": PUSHOVER_TOKEN,
                "user": PUSHOVER_USER_KEY,
                "title": title,
                "message": message,
                "priority": 1,
            }
        ).encode()
        req = urllib.request.Request(
            "https://api.pushover.net/1/messages.json", data=data
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            print(f"[watchdog] Pushover sent: {resp.status}")
    except Exception as e:
        print(f"[watchdog] Pushover error: {e}")


def stop_container() -> None:
    """Stop the IB Gateway container."""
    try:
        subprocess.run(
            ["sudo", "docker", "stop", CONTAINER_NAME],
            check=True,
            capture_output=True,
            timeout=30,
        )
        print(f"[watchdog] Stopped container {CONTAINER_NAME}")
    except Exception as e:
        print(f"[watchdog] Failed to stop container: {e}")


def main() -> None:
    print(f"[watchdog] Starting — monitoring {CONTAINER_NAME}")
    print(
        f"[watchdog] Will stop container after {MAX_FAILURES} consecutive login failures"
    )

    failure_count = 0
    last_attempt_seen = 0

    # Tail container logs
    proc = subprocess.Popen(
        ["sudo", "docker", "logs", "--follow", "--tail", "0", CONTAINER_NAME],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    try:
        for line in proc.stdout:
            line = line.rstrip()

            # Track login attempts (extracted from "Login attempt: N")
            m = LOGIN_ATTEMPT_RE.search(line)
            if m:
                attempt_num = int(m.group(1))
                # If attempt number reset to 1, login was successful
                if attempt_num == 1 and last_attempt_seen > 0:
                    if failure_count > 0:
                        print(f"[watchdog] Login succeeded — resetting failure count")
                    failure_count = 0
                last_attempt_seen = attempt_num
                continue

            # Detect explicit failure messages
            if LOGIN_FAILED_RE.search(line):
                failure_count += 1
                print(
                    f"[watchdog] Login failure detected ({failure_count}/{MAX_FAILURES}): "
                    f"{line[:120]}"
                )

                if failure_count >= MAX_FAILURES:
                    print(
                        f"[watchdog] Reached {MAX_FAILURES} failures — stopping container"
                    )
                    stop_container()
                    send_pushover(
                        "IB Gateway Login Failed",
                        f"IB Gateway failed to log in {MAX_FAILURES} times. "
                        f"Container stopped. Manual intervention needed: VNC in, "
                        f"check credentials, then restart container.",
                    )
                    # Exit so systemd can restart us when container comes back
                    sys.exit(0)
                continue

            # Detect successful login
            if LOGIN_SUCCESS_RE.search(line):
                if failure_count > 0:
                    print(f"[watchdog] Login succeeded — resetting failure count")
                failure_count = 0
                continue

    except KeyboardInterrupt:
        print("[watchdog] Stopped by user")
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
