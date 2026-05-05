"""Re-do losing day analysis with proper metrics.

Fixes two bugs in analyze_losing_days_v2.py:
1. Stopped-trade pnl_usd is fixed at -(stop+fee)*2, so you cannot infer MAE
   from pnl alone. We replay each trade window to compute true MAE/MFE.
2. Per-level $-loss totals confound stop size with selection quality.
   We rank levels by excess WR over breakeven, not total $ loss.

Pipeline: simulate (~12min) once -> pickle -> all analyses run off pickle.
"""

import datetime
import io
import os
import pickle
import sys
import time
from collections import defaultdict
from contextlib import redirect_stdout

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from mnq_alerts.backtest.data import load_all_days, precompute_arrays
from mnq_alerts.backtest.simulate_v2 import simulate_day_v2
from mnq_alerts.bot_risk_backtest import FEE_PTS, MULTIPLIER

# Deployed config (matches reference_baseline_performance.md)
TS = {
    "FIB_EXT_HI_1.272": (6, 20),
    "FIB_EXT_LO_1.272": (6, 20),
    "FIB_0.236": (8, 25),
    "FIB_0.618": (12, 20),
    "FIB_0.764": (10, 25),
    "IBH": (6, 20),
}
BASE_CAPS = {
    "FIB_0.236": 18,
    "FIB_0.618": 3,
    "FIB_0.764": 5,
    "FIB_EXT_HI_1.272": 6,
    "FIB_EXT_LO_1.272": 6,
    "IBH": 7,
}
PICKLE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "results", "variants_v1_trades.pkl"
)


def level_prices_for_day(dc) -> dict[str, float]:
    rng = dc.ibh - dc.ibl
    return {
        "IBH": dc.ibh,
        "IBL": dc.ibl,
        "FIB_EXT_HI_1.272": dc.fib_hi,
        "FIB_EXT_LO_1.272": dc.fib_lo,
        "FIB_0.236": dc.ibl + 0.236 * rng,
        "FIB_0.5": dc.ibl + 0.5 * rng,
        "FIB_0.618": dc.ibl + 0.618 * rng,
        "FIB_0.764": dc.ibl + 0.764 * rng,
    }


def compute_excursions(
    line: float, direction: str, prices: np.ndarray
) -> tuple[float, float]:
    """Return (MAE_pts, MFE_pts) over the trade's price path.

    MAE: peak ADVERSE excursion in points (positive number).
    MFE: peak FAVORABLE excursion in points (positive number).
    """
    if len(prices) == 0:
        return 0.0, 0.0
    if direction == "up":
        # adverse = line - price (positive when below line)
        # favorable = price - line (positive when above line)
        adv = line - np.min(prices)
        fav = np.max(prices) - line
    else:
        adv = np.max(prices) - line
        fav = line - np.min(prices)
    return float(max(0.0, adv)), float(max(0.0, fav))


def simulate_and_collect(force_resimulate: bool = False) -> list[dict]:
    """Run deployed config over all days, compute MAE/MFE per trade.

    Returns list of dicts (one per trade) with: date, level, direction,
    outcome, pnl_usd, stop_pts, target_pts, line_price, entry_count,
    entry_ns, mae_pts, mfe_pts, ttx_secs (time to exit).
    """
    if (not force_resimulate) and os.path.exists(PICKLE_PATH):
        with open(PICKLE_PATH, "rb") as f:
            data = pickle.load(f)
        print(f"Loaded {len(data)} trades from cached pickle.", flush=True)
        return data

    print("Loading day caches...", flush=True)
    t0 = time.time()
    dates, caches = load_all_days()
    print(f"  {len(dates)} days loaded in {time.time()-t0:.1f}s", flush=True)

    print("Simulating (deployed config) and computing MAE/MFE...", flush=True)
    t1 = time.time()
    rows: list[dict] = []
    sink = io.StringIO()  # silence bot_trader debug prints

    for di, date in enumerate(dates):
        dc = caches[date]
        arr = precompute_arrays(dc)
        caps = dict(BASE_CAPS)
        if date.weekday() == 0:
            caps = {k: v * 2 for k, v in caps.items()}

        with redirect_stdout(sink):
            trades = simulate_day_v2(
                dc, arr,
                per_level_ts=TS,
                per_level_caps=caps,
                exclude_levels={"FIB_0.5", "IBL"},
                direction_filter={"IBH": "down"},
                daily_loss=200.0,
                momentum_max=0.0,
            )
        sink.truncate(0); sink.seek(0)

        lp_by_level = level_prices_for_day(dc)
        full_prices = dc.full_prices
        full_ts = dc.full_ts_ns

        for t in trades:
            line = lp_by_level[t.level]
            ei, xi = t.entry_idx, t.exit_idx
            seg = full_prices[ei:xi + 1]
            mae, mfe = compute_excursions(line, t.direction, seg)
            ttx = (int(full_ts[xi]) - int(full_ts[ei])) / 1e9
            tgt, stop = TS[t.level]
            rows.append({
                "date": date,
                "level": t.level,
                "direction": t.direction,
                "outcome": t.outcome,
                "pnl_usd": t.pnl_usd,
                "target_pts": tgt,
                "stop_pts": stop,
                "line_price": line,
                "entry_count": t.entry_count,
                "entry_ns": t.entry_ns,
                "mae_pts": mae,
                "mfe_pts": mfe,
                "ttx_secs": ttx,
            })

        if (di + 1) % 50 == 0:
            elapsed = time.time() - t1
            rate = (di + 1) / elapsed
            eta = (len(dates) - di - 1) / rate
            print(
                f"  {di+1}/{len(dates)} days, {len(rows)} trades, "
                f"{elapsed:.0f}s elapsed, ETA {eta:.0f}s",
                flush=True,
            )

    print(f"  Done. {len(rows)} trades in {time.time()-t1:.0f}s", flush=True)

    os.makedirs(os.path.dirname(PICKLE_PATH), exist_ok=True)
    with open(PICKLE_PATH, "wb") as f:
        pickle.dump(rows, f)
    print(f"Pickled to {PICKLE_PATH}", flush=True)
    return rows


# -- ANALYSES ----------------------------------------------------------------

def section(title: str) -> None:
    print(f"\n{'='*70}\n{title}\n{'='*70}", flush=True)


def per_level_edge(rows: list[dict]) -> None:
    """A. Per-level edge: WR, breakeven WR, excess WR, $/trade, count.

    Across full sample, last 60 trading days, last 30 trading days.
    """
    section("A. PER-LEVEL EDGE (full / last 60 days / last 30 days)")

    dates_sorted = sorted({r["date"] for r in rows})
    cutoff_60 = dates_sorted[-60] if len(dates_sorted) >= 60 else dates_sorted[0]
    cutoff_30 = dates_sorted[-30] if len(dates_sorted) >= 30 else dates_sorted[0]

    windows = [
        ("full", lambda r: True),
        ("60d", lambda r: r["date"] >= cutoff_60),
        ("30d", lambda r: r["date"] >= cutoff_30),
    ]

    header = (
        f"  {'Level':<20} {'T':>3} {'S':>3} {'BE%':>5}  "
        f"{'N':>5} {'WR%':>5} {'XS':>5} {'$/tr':>7} {'$tot':>7}"
    )
    for label, fn in windows:
        print(f"\n  WINDOW: {label}")
        print(header)
        sub = [r for r in rows if fn(r)]
        days_in = len({r["date"] for r in sub})
        by_level: dict[str, list[dict]] = defaultdict(list)
        for r in sub:
            by_level[r["level"]].append(r)
        agg = []
        for lv, lst in by_level.items():
            n = len(lst)
            wins = sum(1 for r in lst if r["pnl_usd"] >= 0)
            wr = wins / n * 100 if n else 0.0
            t_pts = lst[0]["target_pts"]
            s_pts = lst[0]["stop_pts"]
            # Breakeven WR ignoring fees: stop / (target + stop)
            be = s_pts / (t_pts + s_pts) * 100
            xs = wr - be
            pnl_total = sum(r["pnl_usd"] for r in lst)
            pnl_tr = pnl_total / n if n else 0.0
            agg.append((lv, t_pts, s_pts, be, n, wr, xs, pnl_tr, pnl_total))
        # rank by $/trade descending
        agg.sort(key=lambda x: -x[7])
        for row in agg:
            lv, t, s, be, n, wr, xs, pt, pn = row
            print(
                f"  {lv:<20} {t:>3.0f} {s:>3.0f} {be:>5.1f}  "
                f"{n:>5d} {wr:>5.1f} {xs:>+5.1f} {pt:>+7.2f} {pn:>+7.0f}"
            )
        # Day-normalized totals
        total_pnl = sum(r["pnl_usd"] for r in sub)
        print(
            f"  -- {len(sub)} trades over {days_in} days, "
            f"total ${total_pnl:+.0f} = ${total_pnl/days_in:+.2f}/day --"
        )


def mae_distribution(rows: list[dict]) -> None:
    """B. MAE distribution per level, split by outcome.

    For each level x outcome, show MAE percentiles. Tells us whether
    a tighter stop would help (low p70 MAE for losses = stop is too wide,
    losers cluster well below current stop).
    """
    section("B. MAE DISTRIBUTION PER LEVEL (loss / timeout / win)")
    print(
        "  Reading: 'Loss p50=24' on a S25 level means the median losing\n"
        "  trade hit 24pts adverse before the 25pt stop. Most losers needed\n"
        "  the full 25pt stop — tightening to S20 would only convert ~25% of\n"
        "  current losses into 'still-loses-at-tighter-stop'.\n"
    )
    by_level_outcome: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_level_outcome_pnl: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        by_level_outcome[(r["level"], r["outcome"])].append(r["mae_pts"])
        by_level_outcome_pnl[(r["level"], r["outcome"])].append(r["pnl_usd"])

    levels = sorted({r["level"] for r in rows})
    print(
        f"  {'Level':<20} {'S':>3} {'Outcome':<8} {'N':>5} "
        f"{'p25':>5} {'p50':>5} {'p70':>5} {'p90':>5} {'max':>5}"
    )
    for lv in levels:
        s = TS.get(lv, (0, 0))[1]
        for outcome in ["win", "loss", "timeout"]:
            mae = by_level_outcome.get((lv, outcome), [])
            if not mae:
                continue
            p25, p50, p70, p90 = np.percentile(mae, [25, 50, 70, 90])
            mx = max(mae)
            print(
                f"  {lv:<20} {s:>3.0f} {outcome:<8} {len(mae):>5d} "
                f"{p25:>5.1f} {p50:>5.1f} {p70:>5.1f} {p90:>5.1f} {mx:>5.1f}"
            )

    section("B2. WHAT-IF: TIGHTER STOP IMPACT (correctly computed)")
    print(
        "  For each level, count how many CURRENT WINS would have been\n"
        "  killed by a tighter stop (their MAE >= tighter_stop), and the\n"
        "  $ impact (lose stop_pts*2 instead of winning target_pts*2).\n"
        "  Plus: how many CURRENT LOSSES are unchanged (MAE >= current_stop\n"
        "  always; tighter stop catches them earlier but same $ outcome IS\n"
        "  same -- only the few with MAE < tighter_stop would be 'saved',\n"
        "  but those didn't actually hit the current stop, so they aren't\n"
        "  current losses. Tighter stops cannot save current losses.)\n"
        "\n"
        "  Net P&L delta = -(wins_killed * (target+stop)*2)\n"
        "                + 0  (losses unchanged)\n"
        "  -> tighter stops are strictly negative. Listed for sanity.\n"
    )
    print(
        f"  {'Level':<20} {'Cur S':>5} {'Try S':>5} "
        f"{'Wins killed':>11} {'$ delta':>10}"
    )
    for lv in levels:
        cur_s = TS.get(lv, (0, 0))[1]
        cur_t = TS.get(lv, (0, 0))[0]
        if cur_s == 0:
            continue
        for try_s in [15, 18, 20]:
            if try_s >= cur_s:
                continue
            wins = [
                r for r in rows
                if r["level"] == lv and r["outcome"] == "win"
            ]
            killed = [r for r in wins if r["mae_pts"] >= try_s]
            n_killed = len(killed)
            delta = n_killed * (-(try_s + FEE_PTS) - (cur_t - FEE_PTS)) * MULTIPLIER
            print(
                f"  {lv:<20} {cur_s:>5.0f} {try_s:>5.0f} "
                f"{n_killed:>11d} {delta:>+10.0f}"
            )

    section("B3. WHAT-IF: WIDER STOP IMPACT")
    print(
        "  For each level, find current LOSSES whose MAE >= current_stop\n"
        "  (all of them — that's what 'loss' means) but where some COULD\n"
        "  have recovered to a TARGET if the stop had been wider, IF AND\n"
        "  ONLY IF MFE eventually reached target after the adverse move.\n"
        "  Approximation: if a loss had MAE just barely above current stop\n"
        "  (e.g., 21pt on S20), and wider stop S25 would have held, would\n"
        "  the trade have ultimately won? We don't know without re-replay\n"
        "  past the stop tick. Flagged here for follow-up.\n"
    )
    print(
        f"  {'Level':<20} {'S':>3} {'Losses':>7} {'MAE in [S, S+5)':>17} "
        f"{'MAE in [S+5, S+10)':>20} {'MAE >= S+10':>13}"
    )
    for lv in levels:
        s = TS.get(lv, (0, 0))[1]
        if s == 0:
            continue
        losses = [
            r["mae_pts"] for r in rows
            if r["level"] == lv and r["outcome"] == "loss"
        ]
        if not losses:
            continue
        b1 = sum(1 for m in losses if s <= m < s + 5)
        b2 = sum(1 for m in losses if s + 5 <= m < s + 10)
        b3 = sum(1 for m in losses if m >= s + 10)
        print(
            f"  {lv:<20} {s:>3.0f} {len(losses):>7d} "
            f"{b1:>17d} {b2:>20d} {b3:>13d}"
        )


def loss_timing(rows: list[dict]) -> None:
    """C. Loss-timing: when in the day do the losses cluster?"""
    section("C. LOSS TIMING (entry hour-of-day, ET)")
    import pytz
    et = pytz.timezone("America/New_York")
    by_hour_outcome: dict[tuple[int, str], int] = defaultdict(int)
    by_hour_pnl: dict[int, float] = defaultdict(float)
    for r in rows:
        ts = datetime.datetime.fromtimestamp(r["entry_ns"] / 1e9, tz=pytz.utc)
        hr = ts.astimezone(et).hour
        by_hour_outcome[(hr, r["outcome"])] += 1
        by_hour_pnl[hr] += r["pnl_usd"]
    days = len({r["date"] for r in rows})
    print(
        f"  {'Hour ET':>7} {'Trades':>6} {'Wins':>5} {'Losses':>6} "
        f"{'Timeouts':>8} {'WR%':>5} {'$/day':>7}"
    )
    for hr in sorted({h for h, _ in by_hour_outcome.keys()}):
        w = by_hour_outcome.get((hr, "win"), 0)
        l = by_hour_outcome.get((hr, "loss"), 0)
        t = by_hour_outcome.get((hr, "timeout"), 0)
        n = w + l + t
        wr = w / n * 100 if n else 0.0
        print(
            f"  {hr:>7d} {n:>6d} {w:>5d} {l:>6d} {t:>8d} "
            f"{wr:>5.1f} {by_hour_pnl[hr]/days:>+7.2f}"
        )


def loss_streaks(rows: list[dict]) -> None:
    """D. Streak structure on bad days: what does the loss pattern look like?"""
    section("D. LOSS STREAK STRUCTURE (within-day)")
    by_date: dict[datetime.date, list[dict]] = defaultdict(list)
    for r in rows:
        by_date[r["date"]].append(r)
    for trades in by_date.values():
        trades.sort(key=lambda r: r["entry_ns"])

    bad_days = [
        d for d, ts in by_date.items()
        if sum(t["pnl_usd"] for t in ts) <= -100
    ]
    good_days = [
        d for d, ts in by_date.items()
        if sum(t["pnl_usd"] for t in ts) >= 100
    ]
    print(f"  {len(bad_days)} bad days (<= -$100), {len(good_days)} good days (>= +$100)")

    # First-trade outcome
    for label, days in [("BAD", bad_days), ("GOOD", good_days)]:
        first_w = sum(1 for d in days if by_date[d][0]["pnl_usd"] >= 0)
        print(
            f"  {label}: first trade wins {first_w}/{len(days)} "
            f"({first_w/len(days)*100:.0f}%)"
        )

    # WR after N consecutive losses on bad days
    print("\n  Bad days: trade outcomes after N consecutive prior losses")
    after_streak: dict[int, list[int]] = defaultdict(list)
    for d in bad_days:
        streak = 0
        for t in by_date[d]:
            after_streak[streak].append(1 if t["pnl_usd"] >= 0 else 0)
            if t["pnl_usd"] < 0:
                streak += 1
            else:
                streak = 0
    print(f"  {'Prior losses':>12} {'N':>5} {'WR%':>5}")
    for k in sorted(after_streak.keys())[:8]:
        v = after_streak[k]
        wr = sum(v) / len(v) * 100 if v else 0.0
        print(f"  {k:>12d} {len(v):>5d} {wr:>5.1f}")

    # Same for good days for comparison
    print("\n  Good days: trade outcomes after N consecutive prior losses")
    after_streak_g: dict[int, list[int]] = defaultdict(list)
    for d in good_days:
        streak = 0
        for t in by_date[d]:
            after_streak_g[streak].append(1 if t["pnl_usd"] >= 0 else 0)
            if t["pnl_usd"] < 0:
                streak += 1
            else:
                streak = 0
    print(f"  {'Prior losses':>12} {'N':>5} {'WR%':>5}")
    for k in sorted(after_streak_g.keys())[:8]:
        v = after_streak_g[k]
        wr = sum(v) / len(v) * 100 if v else 0.0
        print(f"  {k:>12d} {len(v):>5d} {wr:>5.1f}")


def breakeven_stop_simulation(rows: list[dict]) -> None:
    """Bonus: how would breakeven-stop-after-+X-favorable change outcomes?

    Approximation: if a current LOSS had MFE >= X before going adverse,
    a breakeven stop would have closed at entry (~$0 ex-fee instead of
    -(stop+fee)*2). Note: we don't know the order of MFE vs MAE
    here -- need the trade's path. This is a conservative upper bound.
    Need full path replay for accuracy; flag here.
    """
    section("E. BREAKEVEN STOP UPPER BOUND (after +X favorable)")
    print(
        "  Approximation: if a current LOSS had MFE >= X, *upper bound*\n"
        "  is: that loss is replaced by ~$0 (closed at entry, lose only fee).\n"
        "  This is OPTIMISTIC -- doesn't account for whether MFE happened\n"
        "  BEFORE MAE. For precise numbers, need full-path replay (next step).\n"
    )
    for X in [3, 4, 5, 6, 8]:
        saved_count = 0
        saved_pnl = 0.0
        for r in rows:
            if r["outcome"] != "loss":
                continue
            if r["mfe_pts"] >= X:
                saved_count += 1
                # Replace -(stop+fee)*2 with -fee*2 (closed at entry)
                saved_pnl += -FEE_PTS * MULTIPLIER - r["pnl_usd"]
        days = len({r["date"] for r in rows})
        print(
            f"  X={X}pt: {saved_count} losses 'saved' (upper bound), "
            f"+${saved_pnl:.0f} = +${saved_pnl/days:.2f}/day"
        )


def main() -> None:
    rows = simulate_and_collect(force_resimulate=False)
    print(f"\nTotal trades: {len(rows)} over {len({r['date'] for r in rows})} days")
    total_pnl = sum(r["pnl_usd"] for r in rows)
    days = len({r["date"] for r in rows})
    print(f"Baseline P&L/day: ${total_pnl/days:+.2f}")

    per_level_edge(rows)
    mae_distribution(rows)
    loss_timing(rows)
    loss_streaks(rows)
    breakeven_stop_simulation(rows)


if __name__ == "__main__":
    main()
