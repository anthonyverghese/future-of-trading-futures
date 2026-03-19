"""
notifications.py — Sends push notifications via Pushover (https://pushover.net).
Falls back to stdout if credentials are not configured.
"""

import requests

from config import PUSHOVER_TOKEN, PUSHOVER_USER_KEY

PUSHOVER_API_URL = "https://api.pushover.net/1/messages.json"

PRIORITY_SILENT = -1  # No sound or vibration
PRIORITY_NORMAL =  0  # Default sound
PRIORITY_HIGH   =  1  # Bypasses quiet hours


def send_notification(title: str, message: str, priority: int = PRIORITY_HIGH) -> bool:
    """Send a push notification. Returns True on success."""
    if not PUSHOVER_TOKEN or not PUSHOVER_USER_KEY:
        print(f"\n[NOTIFICATION]\nTitle:   {title}\nMessage: {message}\n")
        return False

    try:
        resp = requests.post(
            PUSHOVER_API_URL,
            data={"token": PUSHOVER_TOKEN, "user": PUSHOVER_USER_KEY,
                  "title": title, "message": message, "priority": priority},
            timeout=10,
        )
        if resp.status_code == 200:
            return True
        print(f"[Pushover] Error {resp.status_code}: {resp.text}")
        return False
    except requests.RequestException as exc:
        print(f"[Pushover] Request failed: {exc}")
        return False
