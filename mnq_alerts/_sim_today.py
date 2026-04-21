"""Simulate what the bot would have done today (2026-04-10)."""

import sqlite3
import datetime
from collections import deque

# Bot config (matches production)
BOT_ENTRY_THRESHOLD = 1.0
BOT_EXIT_THRESHOLD = 15.0
BOT_TARGET_POINTS = 12.0
BOT_STOP_POINTS = 25.0
BOT_MIN_SCORE = 1
BOT_TREND_LOOKBACK_MIN = 60
BOT_TREND_THRESHOLD = 50.0
BOT_TREND_PENALTY = -3
BOT_MAX_ENTRIES_PER_LEVEL = 5
BOT_TIMEOUT_SECS = 15 * 60
DAILY_LOSS_LIMIT_USD = 150.0
MNQ_POINT_VALUE = 2.0
FEE_PER_TRADE = 1.24

# Market times (ET)
MARKET_OPEN = datetime.time(9, 30)
IB_END = datetime.time(10, 30)
MARKET_CLOSE = datetime.time(16, 0)
EOD_FLATTEN = datetime.time(15, 58)


class BotZone:
    def __init__(self, name, price, drifts=False):
        self.name = name
        self.price = price
        self.in_zone = False
        self.entry_count = 0
        self.drifts = drifts
        self._ref = None

    def update(self, p):
        if self.in_zone:
            exit_ref = self.price if self.drifts else self._ref
            if exit_ref and abs(p - exit_ref) > BOT_EXIT_THRESHOLD:
                self.in_zone = False
                self._ref = None
            return False
        if abs(p - self.price) <= BOT_ENTRY_THRESHOLD:
            self.in_zone = True
            self._ref = self.price
            self.entry_count += 1
            return True
        return False


def bot_score(level, direction, entry_count, hour, trend_60m):
    score = 0
    if level == "IBL":
        score += 2
    elif level in ("FIB_EXT_LO_1.272", "FIB_EXT_HI_1.272"):
        score += 1
    elif level == "IBH":
        score -= 1
    if level == "IBL" and direction == "up":
        score += 1
    if level == "IBH" and direction == "down":
        score += 1
    if level == "IBH" and direction == "up":
        score -= 1
    if entry_count == 1:
        score -= 2
    elif entry_count >= 3:
        score += 1
    if hour >= 15.0:
        score += 2
    elif 13.0 <= hour < 15.0:
        score += 1
    elif 10.5 <= hour < 11.5:
        score -= 1
    # Trend filter
    if direction == "up" and trend_60m < -BOT_TREND_THRESHOLD:
        score += BOT_TREND_PENALTY
    elif direction == "down" and trend_60m > BOT_TREND_THRESHOLD:
        score += BOT_TREND_PENALTY
    return score


# Load today's trades
conn = sqlite3.connect(".session_cache.db")
rows = conn.execute(
    "SELECT ts_ns, price FROM trades WHERE date = '2026-04-10' ORDER BY ts_ns"
).fetchall()
conn.close()
print(f"Loaded {len(rows)} trades")

# Convert ts_ns to datetime (assume UTC nanoseconds)
import pytz

ET = pytz.timezone("America/New_York")


def ts_to_dt(ns):
    return datetime.datetime.fromtimestamp(
        ns / 1e9, tz=datetime.timezone.utc
    ).astimezone(ET)


# First pass: compute IB H/L from 9:30-10:30 ET
ib_high = -float("inf")
ib_low = float("inf")
first_price = None
for ts_ns, price in rows:
    dt = ts_to_dt(ts_ns)
    if dt.time() < MARKET_OPEN:
        continue
    if dt.time() >= MARKET_CLOSE:
        break
    if first_price is None:
        first_price = price
    if MARKET_OPEN <= dt.time() < IB_END:
        ib_high = max(ib_high, price)
        ib_low = min(ib_low, price)

ibh = ib_high
ibl = ib_low
ib_range = ibh - ibl
fib_lo = ibl - 0.272 * ib_range
fib_hi = ibh + 0.272 * ib_range

print(f"IBH: {ibh:.2f}, IBL: {ibl:.2f}")
print(f"FIB_EXT_LO: {fib_lo:.2f}, FIB_EXT_HI: {fib_hi:.2f}")

# Second pass: simulate bot trades
zones = {
    "IBH": BotZone("IBH", ibh),
    "IBL": BotZone("IBL", ibl),
    "FIB_EXT_LO_1.272": BotZone("FIB_EXT_LO_1.272", fib_lo),
    "FIB_EXT_HI_1.272": BotZone("FIB_EXT_HI_1.272", fib_hi),
}

price_window = deque()  # (dt, price) for trend
level_trade_counts = {}
position = None  # current open position
trades = []
daily_pnl = 0.0
consec_losses = 0
stopped = False

for ts_ns, price in rows:
    dt = ts_to_dt(ts_ns)
    t = dt.time()
    if t < IB_END:
        continue  # Bot waits for IB to lock
    if t >= EOD_FLATTEN:
        # Close any open position at market
        if position:
            exit_price = price
            if position["direction"] == "up":
                pnl_pts = exit_price - position["entry_price"]
            else:
                pnl_pts = position["entry_price"] - exit_price
            pnl_usd = pnl_pts * MNQ_POINT_VALUE - FEE_PER_TRADE
            position["exit_price"] = exit_price
            position["pnl_usd"] = pnl_usd
            position["exit_reason"] = "eod"
            position["outcome"] = "win" if pnl_usd >= 0 else "loss"
            trades.append(position)
            position = None
        break

    # Update 60m price window
    price_window.append((dt, price))
    cutoff = dt - datetime.timedelta(minutes=BOT_TREND_LOOKBACK_MIN)
    while price_window and price_window[0][0] < cutoff:
        price_window.popleft()
    trend_60m = (
        (price_window[-1][1] - price_window[0][1]) if len(price_window) >= 2 else 0.0
    )

    # Check existing position
    if position:
        entry = position["entry_price"]
        direction = position["direction"]
        elapsed = (dt - position["entry_dt"]).total_seconds()

        if direction == "up":
            pnl_pts = price - entry
            target_hit = price >= entry + BOT_TARGET_POINTS
            stop_hit = price <= entry - BOT_STOP_POINTS
        else:
            pnl_pts = entry - price
            target_hit = price <= entry - BOT_TARGET_POINTS
            stop_hit = price >= entry + BOT_STOP_POINTS

        if target_hit:
            pnl_usd = BOT_TARGET_POINTS * MNQ_POINT_VALUE - FEE_PER_TRADE
            position.update(
                exit_price=price, pnl_usd=pnl_usd, exit_reason="target", outcome="win"
            )
            trades.append(position)
            daily_pnl += pnl_usd
            consec_losses = 0
            position = None
        elif stop_hit:
            pnl_usd = -BOT_STOP_POINTS * MNQ_POINT_VALUE - FEE_PER_TRADE
            position.update(
                exit_price=price, pnl_usd=pnl_usd, exit_reason="stop", outcome="loss"
            )
            trades.append(position)
            daily_pnl += pnl_usd
            consec_losses += 1
            position = None
        elif elapsed >= BOT_TIMEOUT_SECS:
            pnl_usd = pnl_pts * MNQ_POINT_VALUE - FEE_PER_TRADE
            outcome = "win" if pnl_usd >= 0 else "loss"
            position.update(
                exit_price=price,
                pnl_usd=pnl_usd,
                exit_reason="timeout",
                outcome=outcome,
            )
            trades.append(position)
            daily_pnl += pnl_usd
            if pnl_usd < 0:
                consec_losses += 1
            else:
                consec_losses = 0
            position = None

        # Check risk limits
        if daily_pnl <= -DAILY_LOSS_LIMIT_USD:
            stopped = True

    if stopped:
        continue

    # Check for new entries
    if position is None:
        for bz in zones.values():
            if bz.update(price):
                direction = "up" if price > bz.price else "down"
                # Check per-level cap
                count = level_trade_counts.get(bz.name, 0)
                if count >= BOT_MAX_ENTRIES_PER_LEVEL:
                    continue
                hour = dt.hour + dt.minute / 60.0
                score = bot_score(bz.name, direction, bz.entry_count, hour, trend_60m)
                if score < BOT_MIN_SCORE:
                    continue
                position = {
                    "entry_time": dt.strftime("%H:%M:%S"),
                    "entry_dt": dt,
                    "level": bz.name,
                    "direction": direction,
                    "entry_price": price,
                    "line_price": bz.price,
                    "score": score,
                }
                level_trade_counts[bz.name] = count + 1
                break
    else:
        for bz in zones.values():
            bz.update(price)

# Report
print(
    f"\n{'#':<3} {'Time':<10} {'Level':<20} {'Dir':<5} {'Entry':>10} {'Exit':>10} {'P&L':>10} {'Result'}"
)
print("-" * 80)
total = 0.0
wins = losses = 0
for i, t in enumerate(trades, 1):
    total += t["pnl_usd"]
    if t["outcome"] == "win":
        wins += 1
    else:
        losses += 1
    print(
        f"{i:<3} {t['entry_time']:<10} {t['level']:<20} {t['direction']:<5} {t['entry_price']:>10.2f} {t['exit_price']:>10.2f} {t['pnl_usd']:>+10.2f} {t['outcome']:<5} ({t['exit_reason']})"
    )
print("-" * 80)
print(f"Total: {len(trades)} trades, {wins}W/{losses}L, P&L: ${total:+.2f}")
if len(trades) > 0:
    print(f"Win rate: {wins/len(trades)*100:.0f}%")
