#!/usr/bin/env python3
"""Memory pressure watchdog — Pushover-pages on real pressure.

Triggers (any one fires the alert):
- MemAvailable < 50 MB sustained over a 60s sample window
- mnq_alerts/main.py RSS > 200 MB (likely leak)
- Swap-in rate > 10 MB/s sustained over a 60s sample window

Rate-limited: at most one Pushover per hour. Designed to be invoked
by systemd timer every ~5 min during market hours.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import requests

STATE_FILE = Path("/tmp/mnq-memory-watch-last-alert")
RATE_LIMIT_SEC = 3600
PAGE_SIZE = 4096
SAMPLE_WINDOW_SEC = 60

# Thresholds
AVAILABLE_MB_MIN = 50
BOT_RSS_MB_MAX = 200
SWAP_IN_MBPS_MAX = 10.0


def read_meminfo() -> dict[str, int]:
    info: dict[str, int] = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            info[k.strip()] = int(v.strip().split()[0])  # KB
    return info


def read_vmstat() -> dict[str, int]:
    info: dict[str, int] = {}
    with open("/proc/vmstat") as f:
        for line in f:
            k, v = line.split()
            info[k] = int(v)
    return info


def find_bot_pid() -> int | None:
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "mnq_alerts/main.py"], text=True
        ).strip()
        return int(out.splitlines()[0]) if out else None
    except subprocess.CalledProcessError:
        return None


def read_rss_kb(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except FileNotFoundError:
        return None
    return None


def rate_limited() -> bool:
    if not STATE_FILE.exists():
        return False
    return (time.time() - STATE_FILE.stat().st_mtime) < RATE_LIMIT_SEC


def mark_alerted() -> None:
    STATE_FILE.touch()


def send_pushover(title: str, message: str) -> None:
    token = os.environ.get("PUSHOVER_TOKEN")
    user = os.environ.get("PUSHOVER_USER_KEY")
    if not token or not user:
        print(f"[memory-watch] No Pushover creds; would send: {title}: {message}")
        return
    try:
        resp = requests.post(
            "https://api.pushover.net/1/messages.json",
            data={"token": token, "user": user, "title": title,
                  "message": message, "priority": 1},
            timeout=10,
        )
        if resp.status_code != 200:
            print(f"[memory-watch] Pushover error {resp.status_code}: {resp.text}")
    except requests.RequestException as exc:
        print(f"[memory-watch] Pushover request failed: {exc}")


def main() -> int:
    mem1 = read_meminfo()
    vm1 = read_vmstat()
    pid = find_bot_pid()
    rss_kb = read_rss_kb(pid) if pid else None

    time.sleep(SAMPLE_WINDOW_SEC)

    mem2 = read_meminfo()
    vm2 = read_vmstat()

    triggers: list[str] = []

    avail_kb = min(mem1["MemAvailable"], mem2["MemAvailable"])
    if avail_kb < AVAILABLE_MB_MIN * 1024:
        triggers.append(f"MemAvailable {avail_kb / 1024:.0f} MB (< {AVAILABLE_MB_MIN})")

    if rss_kb is not None and rss_kb > BOT_RSS_MB_MAX * 1024:
        triggers.append(f"bot RSS {rss_kb / 1024:.0f} MB (> {BOT_RSS_MB_MAX})")

    pswpin_delta = vm2["pswpin"] - vm1["pswpin"]
    swap_in_mbps = pswpin_delta * PAGE_SIZE / (SAMPLE_WINDOW_SEC * 1024 * 1024)
    if swap_in_mbps > SWAP_IN_MBPS_MAX:
        triggers.append(f"swap-in {swap_in_mbps:.1f} MB/s (> {SWAP_IN_MBPS_MAX})")

    if not triggers:
        return 0

    if rate_limited():
        print(f"[memory-watch] Triggers: {triggers} — rate-limited, skipping notify")
        return 0

    msg = "Memory pressure on EC2:\n" + "\n".join(f"• {t}" for t in triggers)
    if pid is None:
        msg += "\n• mnq-alerts process not found"
    send_pushover("MNQ memory pressure", msg)
    mark_alerted()
    print(f"[memory-watch] Alerted: {triggers}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
