"""multi_bracket_explore_2026_05_06.py — post-process upper bound on multi-bracket exits.

Tests two alternative exit mechanics against the current single-bracket
(target T or stop S) on the existing 5,634-trade pickle:

  1. 50/50 multi-bracket: leg 1 fills at +T/2, leg 2 at +T (or stops at -S)
     Outcome formula:
       if path hit T/2 first (always true on wins, sometimes on losses):
         leg1 = T/2 (locked profit)
         leg2 = target_pts (if path then hit T) or -stop_pts (if path then hit S)
                or 0 (timeout/EOD between)
         multi_pts = 0.5*leg1 + 0.5*leg2
       else: multi_pts = full -S (never hit T/2 to lock leg 1)

  2. BE-after-T/2 (single contract): once price reaches +T/2, stop moves
     to fill price. Outcome:
       if path never hit T/2: same as single-bracket
       if path hit T/2 and never retraced to fill: same as single-bracket
                                                   (BE didn't trigger)
       if path hit T/2 and retraced to fill: BE stop fires at $0
         (saves losses; whipsaws some wins)

Both are upper bounds — no cap-budget effects are modeled. Real-sim
required if a mechanic shows >$3/day potential.

Wall clock: ~10 min (3min caches + ~7min path simulation).
"""

from __future__ import annotations

import os
import pickle
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from mnq_alerts.backtest.data import load_all_days
from mnq_alerts.backtest.experiments.buffer_sweep_v1 import simulate_fill

PICKLE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "results", "variants_v1_trades.pkl"
)
EXCLUDED_LEVELS = {"FIB_EXT_LO_1.272"}
PER_LEVEL_BUFFER = {"IBH": 0.75}
MNQ_PT = 2.0  # $ per point per contract


def simulate_path(prices, ts_ns, fill_idx, fill_price, direction,
                  target_pts, stop_pts, eod_cutoff_ns):
    """Walk fill_idx → exit, returning event flags and indices.

    Returns dict with: target_idx, stop_idx, t_half_idx, be_after_half_idx,
    trail_t4_idx (first time after T/2 that price retraced to fill+T/4),
    trail_t3_idx (same for fill+T/3), mfe_pts.
    """
    sign = 1 if direction == "up" else -1
    target_price = fill_price + sign * target_pts
    stop_price = fill_price - sign * stop_pts
    half_pts = target_pts / 2.0
    trail_t4_pts = target_pts / 4.0
    trail_t3_pts = target_pts / 3.0

    mfe_pts = 0.0
    t_half_idx = None
    target_idx = None
    stop_idx = None
    be_after_half_idx = None
    trail_t4_idx = None
    trail_t3_idx = None

    n = len(prices)
    for i in range(fill_idx, n):
        if int(ts_ns[i]) >= eod_cutoff_ns:
            break
        price = float(prices[i])
        adv = sign * (price - fill_price)
        if adv > mfe_pts:
            mfe_pts = adv
        if t_half_idx is None and adv >= half_pts:
            t_half_idx = i
        if t_half_idx is not None:
            # Price returned to fill (BE)
            if be_after_half_idx is None:
                if (sign == 1 and price <= fill_price) or (sign == -1 and price >= fill_price):
                    be_after_half_idx = i
            # Price returned to fill + T/4 (trail-T4)
            if trail_t4_idx is None and adv <= trail_t4_pts:
                trail_t4_idx = i
            # Price returned to fill + T/3 (trail-T3)
            if trail_t3_idx is None and adv <= trail_t3_pts:
                trail_t3_idx = i
        if target_idx is None and (
            (sign == 1 and price >= target_price)
            or (sign == -1 and price <= target_price)
        ):
            target_idx = i
        if stop_idx is None and (
            (sign == 1 and price <= stop_price)
            or (sign == -1 and price >= stop_price)
        ):
            stop_idx = i
        if target_idx is not None or stop_idx is not None:
            break

    return {
        "target_idx": target_idx,
        "stop_idx": stop_idx,
        "t_half_idx": t_half_idx,
        "be_after_half_idx": be_after_half_idx,
        "trail_t4_idx": trail_t4_idx,
        "trail_t3_idx": trail_t3_idx,
        "mfe_pts": mfe_pts,
    }


def classify(path, target_pts, stop_pts):
    """Return dict of {mechanic: pts} under multiple exit mechanics."""
    target_idx = path["target_idx"]
    stop_idx = path["stop_idx"]
    t_half_idx = path["t_half_idx"]
    be_after_half_idx = path["be_after_half_idx"]
    trail_t4_idx = path["trail_t4_idx"]
    trail_t3_idx = path["trail_t3_idx"]

    # Single-bracket
    if target_idx is not None and (stop_idx is None or target_idx < stop_idx):
        single_pts = float(target_pts)
    elif stop_idx is not None:
        single_pts = -float(stop_pts)
    else:
        single_pts = 0.0  # EOD/timeout

    # 50/50 multi-bracket
    if t_half_idx is not None:
        multi_pts = 0.5 * (target_pts / 2.0) + 0.5 * single_pts
    else:
        multi_pts = single_pts

    # BE-after-T/2 (trail to fill)
    if t_half_idx is not None and be_after_half_idx is not None:
        if target_idx is not None and target_idx <= be_after_half_idx:
            be_pts = float(target_pts)  # target first
        else:
            be_pts = 0.0  # BE fired
    else:
        be_pts = single_pts

    # Trail-to-T/4 (after T/2, stop = fill + T/4 — locks T/4 profit)
    # Trail fires when price returns to fill + T/4 (adv == T/4).
    if t_half_idx is not None and trail_t4_idx is not None:
        if target_idx is not None and target_idx <= trail_t4_idx:
            t4_pts = float(target_pts)  # target first
        else:
            t4_pts = target_pts / 4.0  # trail fires, lock T/4
    else:
        t4_pts = single_pts

    # Trail-to-T/3 (after T/2, stop = fill + T/3 — locks T/3 profit)
    if t_half_idx is not None and trail_t3_idx is not None:
        if target_idx is not None and target_idx <= trail_t3_idx:
            t3_pts = float(target_pts)
        else:
            t3_pts = target_pts / 3.0
    else:
        t3_pts = single_pts

    return {
        "single": single_pts,
        "multi": multi_pts,
        "be": be_pts,
        "trail_t4": t4_pts,
        "trail_t3": t3_pts,
    }


def _eod_cutoff_ns(date) -> int:
    """RTH close at 16:00 ET. Returns ns since epoch in UTC."""
    import datetime
    import pytz
    et = pytz.timezone("America/New_York")
    eod_local = et.localize(
        datetime.datetime(date.year, date.month, date.day, 16, 0, 0)
    )
    return int(eod_local.timestamp() * 1e9)


def main() -> None:
    print("Loading pickle + caches...", flush=True)
    with open(PICKLE_PATH, "rb") as f:
        rows = pickle.load(f)
    t0 = time.time()
    dates_all, caches = load_all_days()
    print(f"  caches in {time.time()-t0:.0f}s", flush=True)

    print("\nSimulating paths + computing 3 exit mechanics...", flush=True)
    results = []
    skipped = 0
    fill_misses = 0
    t1 = time.time()
    for ri, r in enumerate(rows):
        if r["level"] in EXCLUDED_LEVELS:
            continue
        if r["date"] not in caches:
            skipped += 1
            continue
        dc = caches[r["date"]]
        buf = PER_LEVEL_BUFFER.get(r["level"], 1.0)
        fill_idx, fill_price = simulate_fill(
            dc.full_prices, dc.full_ts_ns,
            entry_ns=r["entry_ns"], direction=r["direction"],
            line=r["line_price"], buffer=buf, latency_ms=100.0,
        )
        if fill_idx is None:
            fill_misses += 1
            continue

        eod = _eod_cutoff_ns(r["date"])
        path = simulate_path(
            dc.full_prices, dc.full_ts_ns,
            fill_idx, fill_price, r["direction"],
            r["target_pts"], r["stop_pts"], eod,
        )
        outcomes = classify(path, r["target_pts"], r["stop_pts"])
        results.append({
            "date": r["date"],
            "level": r["level"],
            "direction": r["direction"],
            "target_pts": r["target_pts"],
            "stop_pts": r["stop_pts"],
            "mfe_pts": path["mfe_pts"],
            "single_pts": outcomes["single"],
            "multi_pts": outcomes["multi"],
            "be_pts": outcomes["be"],
            "trail_t4_pts": outcomes["trail_t4"],
            "trail_t3_pts": outcomes["trail_t3"],
            "hit_half": path["t_half_idx"] is not None,
            "be_triggered": path["be_after_half_idx"] is not None,
            "trail_t4_triggered": path["trail_t4_idx"] is not None,
            "trail_t3_triggered": path["trail_t3_idx"] is not None,
            "target_hit": path["target_idx"] is not None,
            "stop_hit": path["stop_idx"] is not None,
        })
        if (ri + 1) % 1000 == 0:
            print(f"  {ri+1}/{len(rows)} ({time.time()-t1:.0f}s)", flush=True)

    n = len(results)
    days = len({r["date"] for r in results})
    print(f"  done: {n} trades over {days} days "
          f"(fill_misses={fill_misses}, skipped={skipped})", flush=True)

    # Aggregate $/day
    total_single = sum(r["single_pts"] for r in results) * MNQ_PT
    total_multi = sum(r["multi_pts"] for r in results) * MNQ_PT
    total_be = sum(r["be_pts"] for r in results) * MNQ_PT
    total_t4 = sum(r["trail_t4_pts"] for r in results) * MNQ_PT
    total_t3 = sum(r["trail_t3_pts"] for r in results) * MNQ_PT

    print(f"\n{'='*94}\nAGGREGATE — exit-mechanic comparison\n{'='*94}")
    print(f"  Single-bracket:    ${total_single/days:+8.2f}/day  ${total_single:+8.0f} total")
    print(f"  50/50 multi:       ${total_multi/days:+8.2f}/day  Δ ${(total_multi-total_single)/days:+6.2f}/day")
    print(f"  BE-after-T/2:      ${total_be/days:+8.2f}/day  Δ ${(total_be-total_single)/days:+6.2f}/day")
    print(f"  Trail-to-T/4:      ${total_t4/days:+8.2f}/day  Δ ${(total_t4-total_single)/days:+6.2f}/day")
    print(f"  Trail-to-T/3:      ${total_t3/days:+8.2f}/day  Δ ${(total_t3-total_single)/days:+6.2f}/day")

    # Composition
    n_target = sum(1 for r in results if r["target_hit"])
    n_stop = sum(1 for r in results if r["stop_hit"] and not r["target_hit"])
    n_timeout = sum(1 for r in results if not r["target_hit"] and not r["stop_hit"])
    n_hit_half = sum(1 for r in results if r["hit_half"])
    n_be_trig = sum(1 for r in results if r["be_triggered"])
    n_loss_with_half = sum(
        1 for r in results
        if r["stop_hit"] and not r["target_hit"] and r["hit_half"]
    )
    n_win_be_trig = sum(
        1 for r in results if r["target_hit"] and r["be_triggered"]
    )
    n_win_be_trig_target_first = sum(
        1 for r in results if r["target_hit"] and r["be_triggered"]
    )

    print(f"\nComposition (n={n}):")
    print(f"  Target hit (wins):           {n_target} ({n_target/n*100:5.1f}%)")
    print(f"  Stop hit (losses):           {n_stop} ({n_stop/n*100:5.1f}%)")
    print(f"  Timeout/EOD:                 {n_timeout} ({n_timeout/n*100:5.1f}%)")
    print(f"  Reached +T/2:                {n_hit_half} ({n_hit_half/n*100:5.1f}%)")
    print(f"  BE-after-T/2 triggered:      {n_be_trig} ({n_be_trig/n*100:5.1f}%)")
    print(f"  Losses that hit T/2 first:   {n_loss_with_half} "
          f"({n_loss_with_half/max(n_stop,1)*100:5.1f}% of losses)")
    print(f"  Wins where BE was retraced:  {n_win_be_trig}")

    # Per-quarter walk-forward
    sorted_dates = sorted({r["date"] for r in results})
    q_size = len(sorted_dates) // 4
    quarters = [
        ("Q1", sorted_dates[:q_size]),
        ("Q2", sorted_dates[q_size:2 * q_size]),
        ("Q3", sorted_dates[2 * q_size:3 * q_size]),
        ("Q4", sorted_dates[3 * q_size:]),
    ]

    print(f"\n{'='*94}\nPER-QUARTER WALK-FORWARD ($/day)\n{'='*94}")
    print(f"  {'Quarter':<8} {'Single':>9} {'Multi':>9} {'BE':>9} {'TrlT/4':>9} {'TrlT/3':>9} "
          f"{'ΔMulti':>8} {'ΔBE':>8} {'ΔT/4':>8} {'ΔT/3':>8}")
    pos = {"multi": 0, "be": 0, "t4": 0, "t3": 0}
    quarter_data = []
    for q_label, qdates in quarters:
        qd = set(qdates)
        in_q = [r for r in results if r["date"] in qd]
        s = sum(r["single_pts"] for r in in_q) * MNQ_PT / len(qdates)
        m = sum(r["multi_pts"] for r in in_q) * MNQ_PT / len(qdates)
        b = sum(r["be_pts"] for r in in_q) * MNQ_PT / len(qdates)
        t4 = sum(r["trail_t4_pts"] for r in in_q) * MNQ_PT / len(qdates)
        t3 = sum(r["trail_t3_pts"] for r in in_q) * MNQ_PT / len(qdates)
        if m - s > 0.5: pos["multi"] += 1
        if b - s > 0.5: pos["be"] += 1
        if t4 - s > 0.5: pos["t4"] += 1
        if t3 - s > 0.5: pos["t3"] += 1
        quarter_data.append((q_label, s, m, b, t4, t3))
        print(f"  {q_label:<8} ${s:>+8.2f} ${m:>+8.2f} ${b:>+8.2f} ${t4:>+8.2f} ${t3:>+8.2f} "
              f"${m-s:>+7.2f} ${b-s:>+7.2f} ${t4-s:>+7.2f} ${t3-s:>+7.2f}")

    print(f"\n  Multi:    {pos['multi']}/4 quarters with >+$0.50/day lift")
    print(f"  BE-T/2:   {pos['be']}/4 quarters")
    print(f"  Trail-T/4: {pos['t4']}/4 quarters")
    print(f"  Trail-T/3: {pos['t3']}/4 quarters")

    # Per-level breakdown
    print(f"\n{'='*94}\nPER-LEVEL — ΔBE-after-T/2 (only mechanic worth investigating further)\n{'='*94}")
    print(f"  {'Level':<22} {'n':>5} {'Single $/day':>13} {'BE $/day':>10} "
          f"{'ΔBE/day':>10} {'%hit T/2':>9} {'%loss-T/2':>11}")
    by_lv = defaultdict(list)
    for r in results:
        by_lv[r["level"]].append(r)
    for lv in sorted(by_lv.keys()):
        lv_rows = by_lv[lv]
        s = sum(r["single_pts"] for r in lv_rows) * MNQ_PT / days
        b = sum(r["be_pts"] for r in lv_rows) * MNQ_PT / days
        n_lv = len(lv_rows)
        n_hit = sum(1 for r in lv_rows if r["hit_half"])
        n_loss = sum(1 for r in lv_rows if r["stop_hit"] and not r["target_hit"])
        n_loss_half = sum(
            1 for r in lv_rows
            if r["stop_hit"] and not r["target_hit"] and r["hit_half"]
        )
        pct_loss_half = n_loss_half / max(n_loss, 1) * 100
        print(f"  {lv:<22} {n_lv:>5} ${s:>+12.2f} ${b:>+9.2f} "
              f"${b-s:>+9.2f} {n_hit/n_lv*100:>8.1f}% {pct_loss_half:>10.1f}%")

    # Save aggregated results
    import json
    import datetime
    out_path = os.path.join(
        os.path.dirname(__file__), "..", "results",
        "multi_bracket_explore_2026_05_06.json"
    )
    summary = {
        "generated_at": datetime.datetime.now().isoformat(),
        "n_trades": n,
        "n_days": days,
        "single_per_day": total_single / days,
        "multi_per_day": total_multi / days,
        "be_per_day": total_be / days,
        "trail_t4_per_day": total_t4 / days,
        "trail_t3_per_day": total_t3 / days,
        "delta_multi_per_day": (total_multi - total_single) / days,
        "delta_be_per_day": (total_be - total_single) / days,
        "delta_t4_per_day": (total_t4 - total_single) / days,
        "delta_t3_per_day": (total_t3 - total_single) / days,
        "composition": {
            "target_hits": n_target,
            "stop_hits": n_stop,
            "timeouts": n_timeout,
            "hit_half": n_hit_half,
            "be_triggered": n_be_trig,
            "loss_with_half": n_loss_with_half,
            "win_be_retraced": n_win_be_trig,
        },
        "walk_forward": {},
    }
    for q_label, qdates in quarters:
        qd = set(qdates)
        in_q = [r for r in results if r["date"] in qd]
        summary["walk_forward"][q_label] = {
            "single_per_day": sum(r["single_pts"] for r in in_q) * MNQ_PT / len(qdates),
            "multi_per_day": sum(r["multi_pts"] for r in in_q) * MNQ_PT / len(qdates),
            "be_per_day": sum(r["be_pts"] for r in in_q) * MNQ_PT / len(qdates),
            "trail_t4_per_day": sum(r["trail_t4_pts"] for r in in_q) * MNQ_PT / len(qdates),
            "trail_t3_per_day": sum(r["trail_t3_pts"] for r in in_q) * MNQ_PT / len(qdates),
        }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
