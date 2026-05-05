"""Run V1, V2, V3 variants.

V1 — Drop FIB_EXT_LO_1.272 entirely. FULL re-simulation (~12 min).
V2 — Breakeven stop after +X favorable pts; sweep X ∈ {3, 4, 5, 6}.
     POST-PROCESS replay against deployed-config trades. Fast (<1 min).
V3 — First-trade-loss bad-day brake: after the first daily loss, halve
     all caps. POST-PROCESS replay (<1 min).

Baseline (deployed): +$49.60/day, 6787 trades, 333 days.
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

TS = {
    "FIB_EXT_HI_1.272": (6, 20),
    "FIB_EXT_LO_1.272": (6, 20),
    "FIB_0.236": (8, 25),
    "FIB_0.618": (12, 20),
    "FIB_0.764": (10, 25),
    "IBH": (6, 20),
}
BASE_CAPS = {
    "FIB_0.236": 18, "FIB_0.618": 3, "FIB_0.764": 5,
    "FIB_EXT_HI_1.272": 6, "FIB_EXT_LO_1.272": 6, "IBH": 7,
}
PICKLE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "results", "variants_v1_trades.pkl"
)


def section(t: str) -> None:
    print(f"\n{'='*70}\n{t}\n{'='*70}", flush=True)


# ---------------------------------------------------------------------------
# V1 — drop FIB_EXT_LO_1.272 entirely
# ---------------------------------------------------------------------------

def run_v1(dates, caches) -> tuple[float, int, int]:
    """Full re-simulation with FIB_EXT_LO_1.272 excluded."""
    section("V1 — Drop FIB_EXT_LO_1.272 (full re-simulation)")
    sink = io.StringIO()
    total_pnl = 0.0
    total_trades = 0
    days_in = 0
    t0 = time.time()
    for di, date in enumerate(dates):
        dc = caches[date]
        arr = precompute_arrays(dc)
        caps = dict(BASE_CAPS)
        if date.weekday() == 0:
            caps = {k: v * 2 for k, v in caps.items()}
        # Drop the level from caps too (defense in depth)
        caps.pop("FIB_EXT_LO_1.272", None)
        with redirect_stdout(sink):
            trades = simulate_day_v2(
                dc, arr,
                per_level_ts=TS,
                per_level_caps=caps,
                exclude_levels={"FIB_0.5", "IBL", "FIB_EXT_LO_1.272"},
                direction_filter={"IBH": "down"},
                daily_loss=200.0, momentum_max=0.0,
            )
        sink.truncate(0); sink.seek(0)
        total_pnl += sum(t.pnl_usd for t in trades)
        total_trades += len(trades)
        days_in += 1
        if (di + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (di + 1) / elapsed
            eta = (len(dates) - di - 1) / rate
            print(
                f"  {di+1}/{len(dates)} days, {total_trades} trades, "
                f"${total_pnl/days_in:+.2f}/day so far, ETA {eta:.0f}s",
                flush=True,
            )
    pnl_per_day = total_pnl / days_in
    print(f"  V1: ${pnl_per_day:+.2f}/day, {total_trades} trades, {days_in} days", flush=True)
    print(f"  V1 vs baseline (+$49.60): {pnl_per_day - 49.60:+.2f}/day", flush=True)
    return pnl_per_day, total_trades, days_in


# ---------------------------------------------------------------------------
# V2 — breakeven stop replay
# ---------------------------------------------------------------------------

def replay_breakeven(rows: list[dict], caches: dict, X: float) -> tuple[float, dict]:
    """Replay each trade with breakeven-stop-after-+X-favorable.

    For each trade, walk prices[entry_idx+1 : exit_idx+1]:
      - track running max favorable excursion (MFE)
      - once MFE >= X, mark BE armed
      - in BE armed state, if price returns to line, exit at line
        (pnl = -FEE_PTS * 2 = -$0.54)
      - else, keep original outcome (target/stop/timeout)

    Returns (delta_pnl_usd_total, breakdown_dict).

    Limitation: does not model that an earlier exit might unblock a
    later trade (one-position-at-a-time). So this is a CONSERVATIVE
    estimate of V2 impact (real impact >= replay impact, since saving
    trades earlier gives more trade slots to subsequent zone entries).
    """
    delta_total = 0.0
    by_outcome_change = defaultdict(int)
    by_level_delta = defaultdict(float)

    # Group rows by date so we can grab the right cache
    by_date: dict = defaultdict(list)
    for r in rows:
        by_date[r["date"]].append(r)

    fee_usd = FEE_PTS * MULTIPLIER  # ~$0.54

    for date, day_rows in by_date.items():
        dc = caches[date]
        prices = dc.full_prices
        for r in day_rows:
            ei, xi = r["entry_idx"] if "entry_idx" in r else None, None
            # entry_idx not stored in pickle — use entry_ns + ts arrays
            # (we DID store entry_idx implicitly via mae/mfe but didn't
            # save it. Recover from entry_ns lookup.)
            # Simpler: re-derive from entry_ns
            ei = int(np.searchsorted(dc.full_ts_ns, r["entry_ns"], side="left"))
            # We don't have exit_idx in pickle; cap path at +15 min window
            ent_ns = int(dc.full_ts_ns[ei])
            window_end_ns = ent_ns + 900 * 1_000_000_000
            xi = int(np.searchsorted(dc.full_ts_ns, window_end_ns, side="right"))
            xi = min(xi, len(prices) - 1)
            line = r["line_price"]
            direction = r["direction"]
            target_pts = r["target_pts"]
            stop_pts = r["stop_pts"]

            # Walk path
            be_armed = False
            new_pnl = None
            for i in range(ei + 1, xi + 1):
                p = float(prices[i])
                if direction == "up":
                    fav = p - line
                    adv = line - p
                else:
                    fav = line - p
                    adv = p - line
                if not be_armed and fav >= X:
                    be_armed = True
                if be_armed and adv >= 0:
                    # Price returned to line — exit flat ex-fees
                    new_pnl = -fee_usd
                    break
                if fav >= target_pts:
                    new_pnl = (target_pts - FEE_PTS) * MULTIPLIER  # win
                    break
                if adv >= stop_pts:
                    new_pnl = -(stop_pts + FEE_PTS) * MULTIPLIER  # loss
                    break

            if new_pnl is None:
                # No exit hit — original timeout outcome retained
                new_pnl = r["pnl_usd"]

            delta = new_pnl - r["pnl_usd"]
            delta_total += delta
            by_level_delta[r["level"]] += delta

            # Categorize
            orig = r["outcome"]
            if abs(new_pnl - r["pnl_usd"]) < 0.01:
                by_outcome_change[f"{orig}_unchanged"] += 1
            elif new_pnl > r["pnl_usd"]:
                by_outcome_change[f"{orig}_be_saved"] += 1
            else:
                by_outcome_change[f"{orig}_be_killed"] += 1

    return delta_total, {
        "by_outcome_change": dict(by_outcome_change),
        "by_level_delta": dict(by_level_delta),
    }


def run_v2(rows, caches) -> dict:
    section("V2 — Breakeven stop after +X favorable (replay sweep)")
    days = len({r["date"] for r in rows})
    print(
        "  Replay model: BE armed once MFE >= X. After armed, if price\n"
        "  returns to line, exit flat (ex-fees). Conservative (doesn't\n"
        "  model unblocked subsequent trades from earlier exits).\n"
    )
    print(f"  {'X':>3} {'$ delta':>10} {'$/day delta':>12} {'losses saved':>14} {'wins killed':>13}")
    out = {}
    for X in [3, 4, 5, 6]:
        delta, info = replay_breakeven(rows, caches, X)
        per_day = delta / days
        ls = info["by_outcome_change"].get("loss_be_saved", 0)
        wk = info["by_outcome_change"].get("win_be_killed", 0)
        print(f"  {X:>3.0f} {delta:>+10.0f} {per_day:>+12.2f} {ls:>14d} {wk:>13d}")
        out[X] = {"delta_total": delta, "delta_per_day": per_day, "info": info}
    print()
    print("  Per-level $ delta breakdown for X=4 (most promising):")
    info4 = out[4]["info"]["by_level_delta"]
    for lv in sorted(info4.keys(), key=lambda k: -info4[k]):
        print(f"    {lv:<22} ${info4[lv]:+.0f}")
    return out


# ---------------------------------------------------------------------------
# V3 — first-trade-loss bad-day brake
# ---------------------------------------------------------------------------

def run_v3(rows: list[dict]) -> dict:
    """Filter trades that exceed halved caps after first daily loss.

    For each day, walk trades in entry_ns order. Track per-level entry
    counts. Once first loss happens, switch to halved caps (ceil(cap/2)).
    Drop subsequent trades whose entry_count exceeds halved cap.

    Limitation: does not model that dropping a trade may unblock the
    1-position constraint earlier. So new trades that would have fired
    are not added. Conservative estimate.
    """
    section("V3 — First-trade-loss bad-day brake (halve caps after 1st loss)")
    days = len({r["date"] for r in rows})

    # Halve caps
    halved = {k: max(1, (v + 1) // 2) for k, v in BASE_CAPS.items()}
    print(f"  Original caps:  {BASE_CAPS}")
    print(f"  Halved caps:    {halved}")
    print(
        "  Drop trades after the first daily loss whose entry_count exceeds\n"
        "  halved cap. Monday double-caps assumed (so halved caps on Monday\n"
        "  are also doubled).\n"
    )

    by_date: dict = defaultdict(list)
    for r in rows:
        by_date[r["date"]].append(r)

    kept = []
    dropped = []
    for date, day_rows in by_date.items():
        day_rows.sort(key=lambda r: r["entry_ns"])
        is_monday = date.weekday() == 0
        eff_halved = halved.copy()
        if is_monday:
            eff_halved = {k: v * 2 for k, v in eff_halved.items()}

        first_loss_seen = False
        for r in day_rows:
            cap = eff_halved.get(r["level"], 999)
            if first_loss_seen and r["entry_count"] > cap:
                dropped.append(r)
                continue
            kept.append(r)
            if r["pnl_usd"] < 0 and not first_loss_seen:
                first_loss_seen = True

    pnl_orig = sum(r["pnl_usd"] for r in rows)
    pnl_v3 = sum(r["pnl_usd"] for r in kept)
    drop_pnl = sum(r["pnl_usd"] for r in dropped)
    drop_wr = (sum(1 for r in dropped if r["pnl_usd"] >= 0)
               / len(dropped) * 100 if dropped else 0)
    print(f"  Original: {len(rows)} trades, ${pnl_orig:+.0f} (${pnl_orig/days:+.2f}/day)")
    print(f"  V3 kept:  {len(kept)} trades, ${pnl_v3:+.0f} (${pnl_v3/days:+.2f}/day)")
    print(f"  Dropped:  {len(dropped)} trades, ${drop_pnl:+.0f}, dropped WR={drop_wr:.1f}%")
    print(f"  V3 delta: ${pnl_v3 - pnl_orig:+.0f} = ${(pnl_v3-pnl_orig)/days:+.2f}/day")

    # Per-level breakdown
    by_lv = defaultdict(lambda: [0, 0.0])
    for r in dropped:
        by_lv[r["level"]][0] += 1
        by_lv[r["level"]][1] += r["pnl_usd"]
    print("\n  Dropped trades by level:")
    for lv in sorted(by_lv.keys()):
        n, p = by_lv[lv]
        print(f"    {lv:<22} {n:>4d} trades, ${p:+.0f} ({p/n:+.2f}/trade if dropped)")
    return {
        "delta_per_day": (pnl_v3 - pnl_orig) / days,
        "dropped_count": len(dropped),
        "dropped_pnl": drop_pnl,
    }


# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading pickle...", flush=True)
    with open(PICKLE_PATH, "rb") as f:
        rows = pickle.load(f)
    print(f"  {len(rows)} trades over {len({r['date'] for r in rows})} days", flush=True)
    print(f"  Baseline: ${sum(r['pnl_usd'] for r in rows)/len({r['date'] for r in rows}):+.2f}/day")

    print("\nLoading day caches (~3 min)...", flush=True)
    t0 = time.time()
    dates, caches = load_all_days()
    print(f"  {len(dates)} days loaded in {time.time()-t0:.1f}s", flush=True)

    # V3 doesn't need caches and is fastest — run first for sanity
    v3 = run_v3(rows)

    # V2 needs caches for path replay
    v2 = run_v2(rows, caches)

    # V1 is the slowest
    v1 = run_v1(dates, caches)

    section("SUMMARY")
    baseline = 49.60
    print(f"  Baseline (deployed):              ${baseline:+.2f}/day")
    print(f"  V1 (drop FIB_EXT_LO):             ${v1[0]:+.2f}/day  delta {v1[0]-baseline:+.2f}")
    for X in [3, 4, 5, 6]:
        d = v2[X]["delta_per_day"]
        print(f"  V2 X={X} (BE replay, conservative): ${baseline + d:+.2f}/day  delta {d:+.2f}")
    print(f"  V3 (halve caps after 1st loss):   ${baseline + v3['delta_per_day']:+.2f}/day  delta {v3['delta_per_day']:+.2f}")


if __name__ == "__main__":
    main()
