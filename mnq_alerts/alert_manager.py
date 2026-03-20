"""
alert_manager.py — Tracks per-level alert state to prevent notification spam.

Rule: alert once when price enters the zone (within ALERT_THRESHOLD_POINTS).
Stay silent while price remains in the zone. Reset when price exits, so the
next entry triggers a fresh alert.
"""

from __future__ import annotations

from dataclasses import dataclass

from cache import log_alert
from config import ALERT_THRESHOLD_POINTS
from notifications import send_notification

_TICKER = "MNQ"


@dataclass
class LevelState:
    """Alert state for a single price level (IBH, IBL, or VWAP)."""

    name: str
    price: float
    in_zone: bool = False  # True while price is within the alert threshold

    def update(self, current_price: float) -> bool:
        """Returns True if an alert should fire (price just entered the zone)."""
        within_threshold = abs(current_price - self.price) <= ALERT_THRESHOLD_POINTS

        if within_threshold and not self.in_zone:
            self.in_zone = True
            return True

        if not within_threshold and self.in_zone:
            self.in_zone = False  # Reset — next entry will alert again

        return False


class AlertManager:
    """Coordinates alert state across IBH, IBL, and VWAP for one session."""

    def __init__(self):
        self._levels: dict[str, LevelState] = {}

    def update_levels(self, ibh: float | None, ibl: float | None, vwap: float | None) -> None:
        """Register or update price levels. Pass None to skip a level."""
        for name, price in {"IBH": ibh, "IBL": ibl, "VWAP": vwap}.items():
            if price is None:
                continue
            if name not in self._levels:
                self._levels[name] = LevelState(name=name, price=price)
                print(f"[AlertManager] {name} registered at {price:.2f}")
            else:
                self._levels[name].price = price  # VWAP drifts; always update

    def advance_state(self, current_price: float) -> None:
        """Update in_zone state for all levels without sending notifications.
        Use during historical replay to prime the state machine."""
        for level in self._levels.values():
            level.update(current_price)

    def check_and_notify(self, current_price: float) -> None:
        """Fire a notification for any level whose zone is newly entered."""
        for level in self._levels.values():
            if level.update(current_price):
                title, body = _build_message(level.name, level.price, current_price)
                print(f"[ALERT] {title} | {body}")
                send_notification(title, body)
                log_alert(ticker=_TICKER, line=level.name, line_price=level.price)


def _build_message(level_name: str, level_price: float, current_price: float) -> tuple[str, str]:
    """
    Build the notification title and body with a trading bias.
      IBH: approaching from below = sell/fade; from above = buy (support)
      IBL: approaching from above = buy/bounce; from below = sell (resistance)
      VWAP: above VWAP = buy (support); below VWAP = sell (resistance)
    """
    distance = abs(current_price - level_price)
    side = "above" if current_price > level_price else "below"

    if level_name == "IBH":
        action = "Look to SELL (fade resistance) or watch for LONG breakout" \
                 if current_price < level_price else "Look to BUY (IBH as support)"
    elif level_name == "IBL":
        action = "Look to BUY (bounce off support) or watch for SHORT breakdown" \
                 if current_price > level_price else "Look to SELL (IBL as resistance)"
    elif level_name == "VWAP":
        action = "Look to BUY (VWAP support)" \
                 if current_price > level_price else "Look to SELL (VWAP resistance)"
    else:
        action = "Monitor price action"

    title = f"MNQ Near {level_name}"
    body  = (f"MNQ @ {current_price:.2f}  |  {level_name} @ {level_price:.2f}\n"
             f"{distance:.1f} pts {side}\n{action}")
    return title, body
