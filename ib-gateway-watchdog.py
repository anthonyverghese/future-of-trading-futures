"""
ib-gateway-watchdog.py — Monitors IB Gateway login attempts and stops the
container after too many failures, sending a Pushover notification.

Polls `docker logs --since` every 30 seconds. After 4 consecutive login
failures, stops the container and notifies the user. The container stays
stopped until manually restarted.

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
POLL_INTERVAL_SECS = 30

# Patterns to track
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
        print(
            f"[watchdog] No Pushover creds — would have sent: {title}: {message}",
            flush=True,
        )
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
            print(f"[watchdog] Pushover sent: {resp.status}", flush=True)
    except Exception as e:
        print(f"[watchdog] Pushover error: {e}", flush=True)


def stop_container() -> None:
    """Stop the IB Gateway container."""
    try:
        subprocess.run(
            ["sudo", "docker", "stop", CONTAINER_NAME],
            check=True,
            capture_output=True,
            timeout=30,
        )
        print(f"[watchdog] Stopped container {CONTAINER_NAME}", flush=True)
    except Exception as e:
        print(f"[watchdog] Failed to stop container: {e}", flush=True)


def container_running() -> bool:
    """Check if the container is currently running."""
    try:
        result = subprocess.run(
            ["sudo", "docker", "inspect", "-f", "{{.State.Running}}", CONTAINER_NAME],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() == "true"
    except Exception:
        return False


def fetch_recent_logs(since_secs: int) -> str:
    """Fetch container logs from the last N seconds."""
    try:
        result = subprocess.run(
            ["sudo", "docker", "logs", "--since", f"{since_secs}s", CONTAINER_NAME],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return result.stdout + result.stderr
    except Exception as e:
        print(f"[watchdog] Failed to fetch logs: {e}", flush=True)
        return ""


def main() -> None:
    print(f"[watchdog] Starting — monitoring {CONTAINER_NAME}", flush=True)
    print(
        f"[watchdog] Polling every {POLL_INTERVAL_SECS}s; will stop container "
        f"after {MAX_FAILURES} consecutive login failures",
        flush=True,
    )

    failure_count = 0
    seen_lines: set[str] = set()  # dedupe across polls (since windows overlap)

    while True:
        if not container_running():
            # Container is stopped — clear failure count and seen lines, wait
            if failure_count > 0 or seen_lines:
                print("[watchdog] Container not running — resetting state", flush=True)
            failure_count = 0
            seen_lines.clear()
            time.sleep(POLL_INTERVAL_SECS)
            continue

        # Fetch logs from last 2x poll interval to handle overlap/missed lines
        logs = fetch_recent_logs(POLL_INTERVAL_SECS * 2)

        for line in logs.splitlines():
            line = line.strip()
            if not line or line in seen_lines:
                continue
            seen_lines.add(line)

            # Detect successful login → reset counter
            if LOGIN_SUCCESS_RE.search(line):
                if failure_count > 0:
                    print(
                        "[watchdog] Login succeeded — resetting failure count",
                        flush=True,
                    )
                failure_count = 0
                continue

            # Detect failure messages
            if LOGIN_FAILED_RE.search(line):
                failure_count += 1
                print(
                    f"[watchdog] Login failure ({failure_count}/{MAX_FAILURES}): "
                    f"{line[:120]}",
                    flush=True,
                )

                if failure_count >= MAX_FAILURES:
                    print(
                        f"[watchdog] Reached {MAX_FAILURES} failures — stopping container",
                        flush=True,
                    )
                    stop_container()
                    send_pushover(
                        "IB Gateway Login Failed",
                        f"IB Gateway failed to log in {MAX_FAILURES} times. "
                        f"Container stopped. Manual intervention needed: VNC in, "
                        f"check credentials, then restart container.",
                    )
                    failure_count = (
                        0  # reset; container_running() check will catch stopped state
                    )
                    seen_lines.clear()
                    break

        # Trim seen_lines to prevent unbounded growth
        if len(seen_lines) > 5000:
            seen_lines = set(list(seen_lines)[-2500:])

        time.sleep(POLL_INTERVAL_SECS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[watchdog] Stopped by user", flush=True)
        sys.exit(0)
    except Exception as e:
        import traceback

        print(f"[watchdog] FATAL: {e}\n{traceback.format_exc()}", flush=True)
        sys.exit(1)
