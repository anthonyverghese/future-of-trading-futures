"""Debug: verify momentum filter is actually working in v2."""
import datetime, sys, os, pytz
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from collections import deque
from unittest.mock import MagicMock
from mnq_alerts.bot_trader import BotTrader, BotZone

ET = pytz.timezone("America/New_York")

# Create a bot with a mock broker
bt = BotTrader.__new__(BotTrader)
bt._broker = MagicMock()
bt._broker.is_connected = True
bt._broker._position_open = False
bt._broker._consecutive_losses = 0
bt._broker.can_trade.return_value = (True, "")
bt._broker.submit_bracket.return_value = MagicMock(success=False)
bt._zones = {"TEST": BotZone("TEST", 27800.0)}
bt._price_window = deque()
bt._price_window_5m = deque()
bt._price_5m_ago = None
bt._level_trade_counts = {}
bt._active_trade_level = None
bt._level_cooldown_until = {}
bt._global_cooldown_until = 0.0
bt._adaptive_caps_restored = True

base = ET.localize(datetime.datetime(2026, 5, 1, 10, 31, 0))

# Simulate 6 minutes of rising price (price going UP through the level)
print("=== Feeding 6 min of ticks, price rising from 27790 to 27802 ===")
for sec in range(360):
    sim_now = base + datetime.timedelta(seconds=sec)
    price = 27790.0 + sec * (12.0 / 360)  # 27790 to 27802 over 6 min
    bt.on_tick(price, tick_rate=1500, now_et=sim_now.time(), _now_override=sim_now)

print(f"After 6 min: _price_5m_ago = {bt._price_5m_ago}")
print(f"Current price ~= 27802")
print(f"price_window_5m length: {len(bt._price_window_5m)}")

# Now price hits the level from below — this is a BUY with momentum
# (price has been rising for 6 min, momentum = 27800 - ~27793 = +7 > 5)
print()
print("=== Price hits TEST level at 27800.5 (BUY, with momentum) ===")
bt._zones["TEST"] = BotZone("TEST", 27800.0)  # fresh zone
sim_now = base + datetime.timedelta(seconds=361)
bt.on_tick(27800.5, tick_rate=1500, now_et=sim_now.time(), _now_override=sim_now)

if bt._broker.submit_bracket.called:
    print("TRADE FIRED — momentum filter did NOT block")
else:
    print("TRADE BLOCKED — momentum filter working!")
    mom = 27800.5 - bt._price_5m_ago
    print(f"  momentum = {27800.5} - {bt._price_5m_ago:.2f} = {mom:.2f} (> 5.0 threshold)")

# Reset and test against momentum
print()
print("=== Reset: price drops from 27810 to 27800 (BUY, against momentum) ===")
bt._broker.submit_bracket.reset_mock()
bt._broker.submit_bracket.return_value = MagicMock(success=False)
bt._price_window_5m = deque()
bt._price_5m_ago = None
bt._price_window = deque()

base2 = ET.localize(datetime.datetime(2026, 5, 1, 11, 0, 0))
for sec in range(360):
    sim_now = base2 + datetime.timedelta(seconds=sec)
    price = 27810.0 - sec * (10.0 / 360)  # 27810 to 27800 over 6 min
    bt.on_tick(price, tick_rate=1500, now_et=sim_now.time(), _now_override=sim_now)

bt._zones["TEST"] = BotZone("TEST", 27800.0)
sim_now = base2 + datetime.timedelta(seconds=361)
bt.on_tick(27800.5, tick_rate=1500, now_et=sim_now.time(), _now_override=sim_now)

if bt._broker.submit_bracket.called:
    print("TRADE FIRED — correctly allowed (against momentum)")
    mom = 27800.5 - bt._price_5m_ago
    print(f"  momentum = {27800.5} - {bt._price_5m_ago:.2f} = {mom:.2f} (< 5.0)")
else:
    print("TRADE BLOCKED — unexpected!")
