"""
bot_risk_backtest.py — Risk-aware backtest for automated IBKR bot trading.

Models a $10k account with realistic constraints:
  - One position at a time (no stacking)
  - Daily loss limits (stop trading after N losses or -$X)
  - Tracks equity curve, drawdown, and worst-case scenarios
  - MNQ: $2/point multiplier, ~$1.24 round-trip commission

Usage:
    python bot_risk_backtest.py
"""

from __future__ import annotations

import datetime
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import pytz

sys.path.insert(0, os.path.dirname(__file__))

from bot_backtest import (
    BOT_ENTRY_THRESHOLD,
    BOT_EXIT_THRESHOLD,
    BOT_WINDOW_SECS,
    FEE_PTS,
)
from targeted_backtest import (
    DayCache,
    _run_zone_numpy,
    load_cached_days,
    load_day,
    preprocess_day,
)

ET = pytz.timezone("America/New_York")

# Trade params (best EV/day from bot_backtest.py).
TARGET_PTS = 12.0
STOP_PTS = 25.0
WINDOW_SECS = BOT_WINDOW_SECS  # 15 min

# MNQ contract specs.
MULTIPLIER = 2.0  # $2 per point
FEE_USD = FEE_PTS * MULTIPLIER  # ~$0.54 round-trip

# Account.
STARTING_BALANCE = 10_000.0


@dataclass
class TradeResult:
    """Result of a single bot trade with timing info."""

    date: datetime.date
    entry_time_ns: int
    exit_time_ns: int
    level: str
    direction: str
    entry_price: float
    line_price: float
    outcome: str  # "win", "loss", "timeout"
    pnl_pts: float  # signed P&L in points (fee-adjusted)
    pnl_usd: float  # signed P&L in USD
    entry_count: int
    tick_rate: float
    session_move: float


def evaluate_bot_trade(
    entry_idx: int,
    line_price: float,
    direction: str,
    ts_ns: np.ndarray,
    prices: np.ndarray,
    target_pts: float,
    stop_pts: float,
    window_secs: int,
    eod_cutoff_ns: int | None = None,
) -> tuple[str, int, float]:
    """Evaluate a bot trade and return (outcome, exit_idx, pnl_pts).

    Like evaluate_bot_outcome but also returns the exit index so we know
    when the position closes (needed for 1-position-at-a-time constraint).

    eod_cutoff_ns: if provided, the evaluation window is clipped to this
    wall-clock time (ns since epoch). Matches the live bot's pre-close flatten
    so stalled positions close before market close rather than running past it.
    """
    entry_ns = ts_ns[entry_idx]
    window_ns = np.int64(window_secs * 1_000_000_000)
    eval_end_ns = entry_ns + window_ns
    if eod_cutoff_ns is not None and eod_cutoff_ns < eval_end_ns:
        eval_end_ns = np.int64(eod_cutoff_ns)

    target_idx = -1
    stop_idx = -1
    last_idx = entry_idx

    if direction == "up":
        target_price = line_price + target_pts
        stop_price = line_price - stop_pts
        for i in range(entry_idx + 1, len(prices)):
            if ts_ns[i] > eval_end_ns:
                break
            last_idx = i
            if target_idx < 0 and prices[i] >= target_price:
                target_idx = i
            if stop_idx < 0 and prices[i] <= stop_price:
                stop_idx = i
            if target_idx >= 0 and stop_idx >= 0:
                break
    else:
        target_price = line_price - target_pts
        stop_price = line_price + stop_pts
        for i in range(entry_idx + 1, len(prices)):
            if ts_ns[i] > eval_end_ns:
                break
            last_idx = i
            if target_idx < 0 and prices[i] <= target_price:
                target_idx = i
            if stop_idx < 0 and prices[i] >= stop_price:
                stop_idx = i
            if target_idx >= 0 and stop_idx >= 0:
                break

    if target_idx >= 0 and stop_idx >= 0:
        if target_idx <= stop_idx:
            return "win", target_idx, target_pts - FEE_PTS
        else:
            return "loss", stop_idx, -(stop_pts + FEE_PTS)
    elif target_idx >= 0:
        return "win", target_idx, target_pts - FEE_PTS
    elif stop_idx >= 0:
        return "loss", stop_idx, -(stop_pts + FEE_PTS)
    else:
        # Timeout — close at actual last price within window (not assumed flat).
        # Measured from line_price to stay consistent with how win/loss P&L is
        # computed here (wins = +target_pts from line, losses = -stop_pts from line).
        exit_price = float(prices[last_idx])
        if direction == "up":
            pnl_pts = exit_price - line_price
        else:
            pnl_pts = line_price - exit_price
        return "timeout", last_idx, pnl_pts - FEE_PTS


def simulate_risk_day(
    dc: DayCache,
    daily_loss_limit_usd: float | None = None,
    max_consecutive_losses: int | None = None,
    target_pts: float = TARGET_PTS,
    stop_pts: float = STOP_PTS,
    window_secs: int = WINDOW_SECS,
) -> list[TradeResult]:
    """Simulate one day with risk constraints.

    Key constraint: ONE POSITION AT A TIME. If a trade is open,
    all other signals are skipped until it closes.
    """
    prices = dc.post_ib_prices
    n = len(prices)

    # Collect all zone entries across all levels.
    all_entries: list[tuple[int, str, int, float]] = (
        []
    )  # (global_idx, level, entry_count, ref_price)

    exit_threshold = BOT_EXIT_THRESHOLD
    for level_name, level_arr in [
        ("IBH", np.full(n, dc.ibh)),
        ("IBL", np.full(n, dc.ibl)),
        ("VWAP", dc.post_ib_vwaps),
        ("FIB_EXT_LO_1.272", np.full(n, dc.fib_lo)),
        ("FIB_EXT_HI_1.272", np.full(n, dc.fib_hi)),
    ]:
        entries = _run_zone_numpy(
            prices, level_arr, BOT_ENTRY_THRESHOLD, exit_threshold
        )
        for local_idx, entry_count, ref_price in entries:
            global_idx = dc.post_ib_start_idx + local_idx
            all_entries.append((global_idx, level_name, entry_count, ref_price))

    # Sort by time (global index).
    all_entries.sort(key=lambda x: x[0])

    # Compute RTH start for session move.
    rth_start_idx = int(
        np.searchsorted(
            dc.full_ts_ns,
            dc.post_ib_timestamps[0].replace(hour=9, minute=30).value,
            side="left",
        )
    )
    day_open = float(dc.full_prices[max(0, rth_start_idx)])

    trades: list[TradeResult] = []
    position_exit_ns: int = 0  # when current position closes (0 = no position)
    daily_pnl_usd = 0.0
    consec_losses = 0
    stopped_for_day = False

    for global_idx, level_name, entry_count, ref_price in all_entries:
        if stopped_for_day:
            break

        entry_ns = int(dc.full_ts_ns[global_idx])

        # Skip if position is still open.
        if entry_ns < position_exit_ns:
            continue

        entry_price = float(dc.full_prices[global_idx])
        line_price = ref_price
        direction = "up" if entry_price > line_price else "down"

        # Evaluate the trade.
        outcome, exit_idx, pnl_pts = evaluate_bot_trade(
            global_idx,
            line_price,
            direction,
            dc.full_ts_ns,
            dc.full_prices,
            target_pts,
            stop_pts,
            window_secs,
        )

        pnl_usd = pnl_pts * MULTIPLIER
        exit_ns = int(dc.full_ts_ns[exit_idx])

        # Block next entry until this trade closes.
        position_exit_ns = exit_ns

        # Compute tick rate.
        window_start_ns = dc.full_ts_ns[global_idx] - np.int64(3 * 60 * 1_000_000_000)
        tick_start = int(np.searchsorted(dc.full_ts_ns, window_start_ns, side="left"))
        tick_rate = (global_idx - tick_start) / 3.0

        session_move = entry_price - day_open

        trade = TradeResult(
            date=dc.date,
            entry_time_ns=entry_ns,
            exit_time_ns=exit_ns,
            level=level_name,
            direction=direction,
            entry_price=entry_price,
            line_price=line_price,
            outcome=outcome,
            pnl_pts=pnl_pts,
            pnl_usd=pnl_usd,
            entry_count=entry_count,
            tick_rate=tick_rate,
            session_move=session_move,
        )
        trades.append(trade)
        daily_pnl_usd += pnl_usd

        # Track consecutive losses for risk limit.
        if outcome == "loss":
            consec_losses += 1
        else:
            consec_losses = 0

        # Check daily loss limits.
        if daily_loss_limit_usd is not None and daily_pnl_usd <= -daily_loss_limit_usd:
            stopped_for_day = True
        if (
            max_consecutive_losses is not None
            and consec_losses >= max_consecutive_losses
        ):
            stopped_for_day = True

    return trades


def run_risk_backtest(
    valid_days: list[datetime.date],
    day_caches: dict[datetime.date, DayCache],
    daily_loss_limit_usd: float | None = None,
    max_consecutive_losses: int | None = None,
    target_pts: float = TARGET_PTS,
    stop_pts: float = STOP_PTS,
) -> list[TradeResult]:
    """Run risk-aware backtest across all days."""
    all_trades: list[TradeResult] = []
    for date in valid_days:
        dc = day_caches.get(date)
        if dc is None:
            continue
        day_trades = simulate_risk_day(
            dc,
            daily_loss_limit_usd=daily_loss_limit_usd,
            max_consecutive_losses=max_consecutive_losses,
            target_pts=target_pts,
            stop_pts=stop_pts,
        )
        all_trades.extend(day_trades)
    return all_trades


def print_risk_summary(
    trades: list[TradeResult],
    label: str,
    num_days: int,
    starting_balance: float = STARTING_BALANCE,
) -> None:
    """Print comprehensive risk metrics."""
    if not trades:
        print(f"\n  {label}: No trades")
        return

    wins = [t for t in trades if t.outcome == "win"]
    losses = [t for t in trades if t.outcome == "loss"]
    timeouts = [t for t in trades if t.outcome == "timeout"]
    decided = [t for t in trades if t.outcome in ("win", "loss")]

    total_pnl = sum(t.pnl_usd for t in trades)
    wr = len(wins) / len(decided) * 100 if decided else 0

    print(f"\n  {label}")
    print(f"  {'─' * 60}")
    print(f"  Trades: {len(trades)} ({len(trades)/num_days:.1f}/day)")
    print(f"  Record: {len(wins)}W / {len(losses)}L / {len(timeouts)}T = {wr:.1f}% WR")
    print(f"  Total P&L: ${total_pnl:+,.2f} (${total_pnl/num_days:+,.2f}/day)")

    # Daily P&L breakdown.
    daily_pnl: dict[datetime.date, float] = {}
    daily_trades: dict[datetime.date, int] = {}
    for t in trades:
        daily_pnl[t.date] = daily_pnl.get(t.date, 0) + t.pnl_usd
        daily_trades[t.date] = daily_trades.get(t.date, 0) + 1

    pnl_values = sorted(daily_pnl.values())
    print(f"\n  Daily P&L:")
    print(f"    Worst day:  ${pnl_values[0]:+,.2f}")
    print(f"    10th %ile:  ${pnl_values[int(len(pnl_values)*0.10)]:+,.2f}")
    print(f"    Median:     ${pnl_values[len(pnl_values)//2]:+,.2f}")
    print(f"    90th %ile:  ${pnl_values[int(len(pnl_values)*0.90)]:+,.2f}")
    print(f"    Best day:   ${pnl_values[-1]:+,.2f}")

    # Losing days.
    losing_days = [v for v in pnl_values if v < 0]
    winning_days = [v for v in pnl_values if v > 0]
    print(
        f"    Losing days: {len(losing_days)}/{len(pnl_values)} ({len(losing_days)/len(pnl_values)*100:.0f}%)"
    )

    # Equity curve and max drawdown.
    equity = starting_balance
    peak = equity
    max_dd_usd = 0.0
    max_dd_pct = 0.0
    equity_min = equity

    # Track by trade for granular drawdown.
    for t in trades:
        equity += t.pnl_usd
        if equity > peak:
            peak = equity
        dd = peak - equity
        dd_pct = dd / peak * 100 if peak > 0 else 0
        if dd > max_dd_usd:
            max_dd_usd = dd
            max_dd_pct = dd_pct
        if equity < equity_min:
            equity_min = equity

    print(f"\n  Account (starting ${starting_balance:,.0f}):")
    print(f"    Final balance:  ${equity:,.2f}")
    print(f"    Max drawdown:   ${max_dd_usd:,.2f} ({max_dd_pct:.1f}%)")
    print(f"    Lowest balance: ${equity_min:,.2f}")
    print(
        f"    Return:         {(equity - starting_balance) / starting_balance * 100:+.1f}%"
    )
    print(
        f"    Return/day:     {(equity - starting_balance) / starting_balance / num_days * 100:.3f}%"
    )

    # Longest losing streak.
    max_streak = 0
    current_streak = 0
    streak_pnl = 0.0
    max_streak_pnl = 0.0
    for t in trades:
        if t.outcome == "loss":
            current_streak += 1
            streak_pnl += t.pnl_usd
            if current_streak > max_streak:
                max_streak = current_streak
                max_streak_pnl = streak_pnl
        else:
            current_streak = 0
            streak_pnl = 0.0

    print(
        f"\n  Worst losing streak: {max_streak} trades in a row (${max_streak_pnl:+,.2f})"
    )

    # Consecutive loss probability distribution.
    streaks = []
    current = 0
    for t in decided:
        if t.outcome == "loss":
            current += 1
        else:
            if current > 0:
                streaks.append(current)
            current = 0
    if current > 0:
        streaks.append(current)

    if streaks:
        print(f"  Loss streaks: ", end="")
        for length in range(1, max_streak + 1):
            count = sum(1 for s in streaks if s >= length)
            if count > 0:
                print(f"{length}+={count}  ", end="")
        print()


def main() -> None:
    print("=" * 75)
    print("  BOT RISK BACKTEST — $10k account simulation")
    print(f"  Config: target={TARGET_PTS}, stop={STOP_PTS}")
    print(f"  MNQ: ${MULTIPLIER}/point, fee=${FEE_USD:.2f}/trade")
    print(
        f"  Win = +${(TARGET_PTS - FEE_PTS) * MULTIPLIER:.2f}, "
        f"Loss = -${(STOP_PTS + FEE_PTS) * MULTIPLIER:.2f}"
    )
    print("=" * 75)

    t0 = time.time()

    days = load_cached_days()
    print(f"\n  Loading {len(days)} cached days...")

    day_caches: dict[datetime.date, DayCache] = {}
    for date in days:
        try:
            df = load_day(date)
            dc = preprocess_day(df, date)
            if dc is not None:
                day_caches[date] = dc
        except Exception:
            pass
    print(f"  Loaded {len(day_caches)} days in {time.time() - t0:.1f}s")

    valid_days = sorted(day_caches.keys())
    num_days = len(valid_days)

    # ══════════════════════════════════════════════════════════════════════
    # TEST 1: UNRESTRICTED — no risk limits, 1 position at a time
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 75}")
    print("  TEST 1: UNRESTRICTED — 1 position at a time, no loss limits")
    print(f"{'═' * 75}")

    trades_unrestricted = run_risk_backtest(valid_days, day_caches)
    print_risk_summary(trades_unrestricted, "Unrestricted", num_days)

    # ══════════════════════════════════════════════════════════════════════
    # TEST 2: DAILY LOSS LIMIT SWEEP
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 75}")
    print("  TEST 2: DAILY LOSS LIMIT SWEEP")
    print(f"  Stop trading for the day after losing $X")
    print(f"{'═' * 75}")

    loss_limits = [50, 75, 100, 125, 150, 200, 300, None]
    print(
        f"\n  {'Limit':>8}  {'Trades':>7}  {'/day':>5}  {'W':>5}  {'L':>5}  "
        f"{'WR%':>6}  {'Total P&L':>10}  {'$/day':>8}  {'MaxDD':>8}  {'MaxDD%':>7}  {'MaxStrk':>8}"
    )
    print(
        f"  {'-'*8}  {'-'*7}  {'-'*5}  {'-'*5}  {'-'*5}  "
        f"{'-'*6}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*8}"
    )

    for limit in loss_limits:
        trades = run_risk_backtest(valid_days, day_caches, daily_loss_limit_usd=limit)
        decided = [t for t in trades if t.outcome in ("win", "loss")]
        wins = sum(1 for t in decided if t.outcome == "win")
        losses_count = len(decided) - wins
        wr = wins / len(decided) * 100 if decided else 0
        total_pnl = sum(t.pnl_usd for t in trades)
        per_day = len(trades) / num_days

        # Max drawdown.
        equity = STARTING_BALANCE
        peak = equity
        max_dd = 0.0
        for t in trades:
            equity += t.pnl_usd
            peak = max(peak, equity)
            max_dd = max(max_dd, peak - equity)
        max_dd_pct = max_dd / STARTING_BALANCE * 100

        # Longest losing streak.
        max_streak = 0
        cur = 0
        for t in trades:
            if t.outcome == "loss":
                cur += 1
                max_streak = max(max_streak, cur)
            else:
                cur = 0

        limit_str = f"${limit}" if limit is not None else "None"
        print(
            f"  {limit_str:>8}  {len(trades):>7}  {per_day:>5.1f}  {wins:>5}  "
            f"{losses_count:>5}  {wr:>5.1f}%  ${total_pnl:>+9,.0f}  "
            f"${total_pnl/num_days:>+7,.0f}  ${max_dd:>7,.0f}  "
            f"{max_dd_pct:>6.1f}%  {max_streak:>8}"
        )

    # ══════════════════════════════════════════════════════════════════════
    # TEST 3: CONSECUTIVE LOSS LIMIT SWEEP
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 75}")
    print("  TEST 3: CONSECUTIVE LOSS LIMIT SWEEP")
    print(f"  Stop trading for the day after N straight losses")
    print(f"{'═' * 75}")

    consec_limits = [2, 3, 4, 5, None]
    print(
        f"\n  {'Limit':>8}  {'Trades':>7}  {'/day':>5}  {'W':>5}  {'L':>5}  "
        f"{'WR%':>6}  {'Total P&L':>10}  {'$/day':>8}  {'MaxDD':>8}  {'MaxDD%':>7}"
    )
    print(
        f"  {'-'*8}  {'-'*7}  {'-'*5}  {'-'*5}  {'-'*5}  "
        f"{'-'*6}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*7}"
    )

    for limit in consec_limits:
        trades = run_risk_backtest(valid_days, day_caches, max_consecutive_losses=limit)
        decided = [t for t in trades if t.outcome in ("win", "loss")]
        wins = sum(1 for t in decided if t.outcome == "win")
        losses_count = len(decided) - wins
        wr = wins / len(decided) * 100 if decided else 0
        total_pnl = sum(t.pnl_usd for t in trades)
        per_day = len(trades) / num_days

        equity = STARTING_BALANCE
        peak = equity
        max_dd = 0.0
        for t in trades:
            equity += t.pnl_usd
            peak = max(peak, equity)
            max_dd = max(max_dd, peak - equity)
        max_dd_pct = max_dd / STARTING_BALANCE * 100

        limit_str = str(limit) if limit is not None else "None"
        print(
            f"  {limit_str:>8}  {len(trades):>7}  {per_day:>5.1f}  {wins:>5}  "
            f"{losses_count:>5}  {wr:>5.1f}%  ${total_pnl:>+9,.0f}  "
            f"${total_pnl/num_days:>+7,.0f}  ${max_dd:>7,.0f}  "
            f"{max_dd_pct:>6.1f}%"
        )

    # ══════════════════════════════════════════════════════════════════════
    # TEST 4: COMBINED RISK LIMITS
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 75}")
    print("  TEST 4: COMBINED — daily $ limit + consecutive loss limit")
    print(f"{'═' * 75}")

    combos = [
        (100, 3),
        (100, 4),
        (150, 3),
        (150, 4),
        (200, 3),
        (200, 4),
        (None, None),
    ]
    print(
        f"\n  {'$Limit':>8}  {'Consec':>6}  {'Trades':>7}  {'/day':>5}  {'W':>5}  "
        f"{'L':>5}  {'WR%':>6}  {'Total P&L':>10}  {'$/day':>8}  {'MaxDD':>8}  {'MaxDD%':>7}"
    )
    print(
        f"  {'-'*8}  {'-'*6}  {'-'*7}  {'-'*5}  {'-'*5}  {'-'*5}  "
        f"{'-'*6}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*7}"
    )

    for dollar_limit, consec_limit in combos:
        trades = run_risk_backtest(
            valid_days,
            day_caches,
            daily_loss_limit_usd=dollar_limit,
            max_consecutive_losses=consec_limit,
        )
        decided = [t for t in trades if t.outcome in ("win", "loss")]
        wins = sum(1 for t in decided if t.outcome == "win")
        losses_count = len(decided) - wins
        wr = wins / len(decided) * 100 if decided else 0
        total_pnl = sum(t.pnl_usd for t in trades)
        per_day = len(trades) / num_days

        equity = STARTING_BALANCE
        peak = equity
        max_dd = 0.0
        for t in trades:
            equity += t.pnl_usd
            peak = max(peak, equity)
            max_dd = max(max_dd, peak - equity)
        max_dd_pct = max_dd / STARTING_BALANCE * 100

        dl_str = f"${dollar_limit}" if dollar_limit is not None else "None"
        cl_str = str(consec_limit) if consec_limit is not None else "None"
        print(
            f"  {dl_str:>8}  {cl_str:>6}  {len(trades):>7}  {per_day:>5.1f}  "
            f"{wins:>5}  {losses_count:>5}  {wr:>5.1f}%  "
            f"${total_pnl:>+9,.0f}  ${total_pnl/num_days:>+7,.0f}  "
            f"${max_dd:>7,.0f}  {max_dd_pct:>6.1f}%"
        )

    # ══════════════════════════════════════════════════════════════════════
    # TEST 5: DETAILED — Best risk config equity curve
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 75}")
    print("  TEST 5: DETAILED — Full equity curve with $150 daily limit + 3 consec")
    print(f"{'═' * 75}")

    trades_safe = run_risk_backtest(
        valid_days,
        day_caches,
        daily_loss_limit_usd=150,
        max_consecutive_losses=3,
    )
    print_risk_summary(trades_safe, "$150 limit + 3 consec stop", num_days)

    # Show worst 10 days.
    daily_pnl: dict[datetime.date, float] = {}
    daily_count: dict[datetime.date, int] = {}
    for t in trades_safe:
        daily_pnl[t.date] = daily_pnl.get(t.date, 0) + t.pnl_usd
        daily_count[t.date] = daily_count.get(t.date, 0) + 1

    worst_days = sorted(daily_pnl.items(), key=lambda x: x[1])[:10]
    print(f"\n  Worst 10 days:")
    for date, pnl in worst_days:
        n_trades = daily_count[date]
        print(f"    {date}  {n_trades} trades  ${pnl:+,.2f}")

    # Show best 10 days.
    best_days = sorted(daily_pnl.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\n  Best 10 days:")
    for date, pnl in best_days:
        n_trades = daily_count[date]
        print(f"    {date}  {n_trades} trades  ${pnl:+,.2f}")

    # Monthly equity summary.
    print(f"\n  Monthly P&L:")
    monthly: dict[str, float] = {}
    for date, pnl in daily_pnl.items():
        month = date.strftime("%Y-%m")
        monthly[month] = monthly.get(month, 0) + pnl
    for month in sorted(monthly.keys()):
        bar_len = int(abs(monthly[month]) / 20)
        bar = ("█" * bar_len) if monthly[month] >= 0 else ("░" * bar_len)
        sign = "+" if monthly[month] >= 0 else "-"
        print(f"    {month}  ${monthly[month]:>+8,.0f}  {bar}")

    # ══════════════════════════════════════════════════════════════════════
    # TEST 6: SMALLER TARGETS for risk reduction
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 75}")
    print("  TEST 6: TARGET/STOP vs RISK — Which configs keep drawdown <15%?")
    print(f"  With $150 daily loss limit + 3 consecutive loss stop")
    print(f"{'═' * 75}")

    configs = [
        (4, 8),
        (6, 12),
        (8, 16),
        (8, 20),
        (10, 20),
        (12, 25),
        (16, 25),
    ]
    print(
        f"\n  {'T/S':>7}  {'Trades':>7}  {'/day':>5}  {'WR%':>6}  "
        f"{'Total P&L':>10}  {'$/day':>8}  {'MaxDD':>8}  {'MaxDD%':>7}  {'Low$':>8}"
    )
    print(
        f"  {'-'*7}  {'-'*7}  {'-'*5}  {'-'*6}  "
        f"{'-'*10}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*8}"
    )

    for target, stop in configs:
        trades = run_risk_backtest(
            valid_days,
            day_caches,
            daily_loss_limit_usd=150,
            max_consecutive_losses=3,
            target_pts=target,
            stop_pts=stop,
        )
        decided = [t for t in trades if t.outcome in ("win", "loss")]
        wins = sum(1 for t in decided if t.outcome == "win")
        losses_count = len(decided) - wins
        wr = wins / len(decided) * 100 if decided else 0
        total_pnl = sum(t.pnl_usd for t in trades)
        per_day = len(trades) / num_days

        equity = STARTING_BALANCE
        peak = equity
        max_dd = 0.0
        low = equity
        for t in trades:
            equity += t.pnl_usd
            peak = max(peak, equity)
            max_dd = max(max_dd, peak - equity)
            low = min(low, equity)
        max_dd_pct = max_dd / STARTING_BALANCE * 100

        marker = " ← current" if target == 12 and stop == 25 else ""
        print(
            f"  {target:>2}/{stop:<4}  {len(trades):>7}  {per_day:>5.1f}  {wr:>5.1f}%  "
            f"${total_pnl:>+9,.0f}  ${total_pnl/num_days:>+7,.0f}  "
            f"${max_dd:>7,.0f}  {max_dd_pct:>6.1f}%  ${low:>7,.0f}{marker}"
        )

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
