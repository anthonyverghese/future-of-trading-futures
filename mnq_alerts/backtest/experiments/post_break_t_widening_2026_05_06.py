"""post_break_t_widening_2026_05_06.py — focused test of T widening on
post-break retest trades.

The honest question (after realizing the bot's existing mean-reversion
already fires at most retests in the same direction as a "continuation"
trade would prescribe):

  For trades that occur AFTER a level breaks in the same direction as the
  trade's fill direction, does widening T capture more profit than the
  resulting whipsaw cost?

Classifies every pickle trade into one of three groups based on the level's
break state at the time of trade entry:

  GROUP A (fresh): level NOT broken before this trade today
  GROUP B (post-break-aligned): level broken in trade direction within
          MAX_RETEST_SECS before trade entry. The bot's direction matches
          the break — i.e., bot fires BUY when level was UP-broken, or
          SELL when level was DOWN-broken.
  GROUP C (post-break-mismatch): level broken in opposite direction. Bot
          is fading the break — should be MORE conservative, not less.

For each group, simulate path under T multipliers [1.0, 1.25, 1.5, 2.0]
and compute $/day. If Group B benefits from wider T while Group A doesn't,
that's the lift. Walk-forward by quarter.

Wall clock: ~5 min (3 min cache + ~1 min sim).
"""

from __future__ import annotations

import datetime as _dt
import json
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
MNQ_PT = 2.0
MAX_RETEST_SECS = 3600  # consider trades within 60min of break as post-break

T_MULT_SWEEP = [1.0, 1.25, 1.5, 2.0]


def _eod_cutoff_ns(date) -> int:
    import datetime
    import pytz
    et = pytz.timezone("America/New_York")
    eod_local = et.localize(
        datetime.datetime(date.year, date.month, date.day, 16, 0, 0)
    )
    return int(eod_local.timestamp() * 1e9)


def find_break_events(prices, ts_ns, levels_with_stop, eod_cutoff_ns):
    """For one day, scan ticks once and record first time each level
    broke in each direction. Returns:
      {level_name: {"up_at": int|None, "down_at": int|None}}
    levels_with_stop: {level_name: (line_price, stop_pts)}
    """
    state = {lv: {"up_at": None, "down_at": None} for lv in levels_with_stop}
    n = len(prices)
    for i in range(n):
        ts = int(ts_ns[i])
        if ts >= eod_cutoff_ns:
            break
        p = float(prices[i])
        for lv, (line, sp) in levels_with_stop.items():
            s = state[lv]
            if s["up_at"] is None and p >= line + sp:
                s["up_at"] = ts
            if s["down_at"] is None and p <= line - sp:
                s["down_at"] = ts
    return state


def walk_path(prices, ts_ns, fill_idx, fill_price, direction,
              base_target_pts, stop_pts, eod_cutoff_ns):
    """Walk fill_idx forward, return (target_idxs_per_multiplier, stop_idx).

    base_target_pts: the level's BASE T (e.g., 8 for FIB_0.236).
    Threshold for each multiplier m is base_target_pts * m.

    Returns:
      - target_idxs: for each T multiplier in T_MULT_SWEEP, first idx where
        MFE >= base * m (or None if never reached)
      - stop_idx: first idx where price hit stop (or None)
    """
    sign = 1 if direction == "up" else -1
    stop_price = fill_price - sign * stop_pts
    target_thresholds = [base_target_pts * m for m in T_MULT_SWEEP]
    target_idxs = [None] * len(T_MULT_SWEEP)
    stop_idx = None

    n = len(prices)
    for i in range(fill_idx, n):
        if int(ts_ns[i]) >= eod_cutoff_ns:
            break
        price = float(prices[i])
        adv = sign * (price - fill_price)
        for j, thr in enumerate(target_thresholds):
            if target_idxs[j] is None and adv >= thr:
                target_idxs[j] = i
        if stop_idx is None and (
            (sign == 1 and price <= stop_price)
            or (sign == -1 and price >= stop_price)
        ):
            stop_idx = i
        # All targets resolved? exit early
        if all(idx is not None for idx in target_idxs) or stop_idx is not None:
            # Even if stop hit, continue for higher T thresholds — wait, no:
            # if stop hit, ALL higher T multipliers are losses
            if stop_idx is not None:
                break
            if all(idx is not None for idx in target_idxs):
                break
    return target_idxs, stop_idx


def classify_outcomes(target_idxs, stop_idx, target_pts):
    """For each T multiplier, return outcome pts.

    Outcome rules:
      target_idx is not None AND (stop_idx is None OR target_idx < stop_idx):
        WIN at T * mult
      stop_idx is not None AND (target_idx is None OR stop_idx <= target_idx):
        LOSS at -S
      else: timeout at 0
    """
    outcomes = []
    for j, mult in enumerate(T_MULT_SWEEP):
        ti = target_idxs[j]
        if ti is not None and (stop_idx is None or ti < stop_idx):
            outcomes.append(target_pts * mult)
        elif stop_idx is not None:
            # stop hit before this target threshold
            outcomes.append(None)  # placeholder for stop loss; handled by caller
        else:
            outcomes.append(0.0)
    return outcomes


def main() -> None:
    print("Loading pickle + caches...", flush=True)
    with open(PICKLE_PATH, "rb") as f:
        rows = pickle.load(f)
    t0 = time.time()
    dates_all, caches = load_all_days()
    print(f"  caches in {time.time()-t0:.0f}s", flush=True)

    # Build per-day level dict from pickle (line_price varies daily for fibs/IB)
    levels_per_day: dict = defaultdict(dict)
    for r in rows:
        if r["level"] in EXCLUDED_LEVELS:
            continue
        levels_per_day[r["date"]][r["level"]] = (r["line_price"], r["stop_pts"])

    # Compute break events per day per level
    print("\nComputing break events per day...", flush=True)
    breaks_per_day = {}
    for d, lv_dict in levels_per_day.items():
        if d not in caches:
            continue
        dc = caches[d]
        eod = _eod_cutoff_ns(d)
        breaks_per_day[d] = find_break_events(
            dc.full_prices, dc.full_ts_ns, lv_dict, eod
        )

    # Process trades: classify, walk path, record per-multiplier outcomes
    print("\nProcessing trades + multi-T-multiplier path simulation...", flush=True)
    results = []
    skipped = fill_misses = 0
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

        # Classify based on break state at fill time
        fill_ts = int(dc.full_ts_ns[fill_idx])
        breaks = breaks_per_day.get(r["date"], {}).get(r["level"], {})
        # Trade direction: "up"=BUY (expects price to go up),
        #                  "down"=SELL (expects price to go down)
        # For BUY (direction="up"): aligned with UP break (broken_up_at)
        # For SELL (direction="down"): aligned with DOWN break (broken_down_at)
        if r["direction"] == "up":
            aligned_break_at = breaks.get("up_at")
            mismatch_break_at = breaks.get("down_at")
        else:
            aligned_break_at = breaks.get("down_at")
            mismatch_break_at = breaks.get("up_at")

        # Determine group
        secs_since_aligned = (
            (fill_ts - aligned_break_at) / 1e9
            if aligned_break_at is not None and fill_ts > aligned_break_at
            else None
        )
        secs_since_mismatch = (
            (fill_ts - mismatch_break_at) / 1e9
            if mismatch_break_at is not None and fill_ts > mismatch_break_at
            else None
        )

        if secs_since_aligned is not None and secs_since_aligned <= MAX_RETEST_SECS:
            group = "B_post_break_aligned"
        elif secs_since_mismatch is not None and secs_since_mismatch <= MAX_RETEST_SECS:
            group = "C_post_break_mismatch"
        else:
            group = "A_fresh"

        # Walk path with multiple T multipliers
        eod = _eod_cutoff_ns(r["date"])
        target_idxs, stop_idx = walk_path(
            dc.full_prices, dc.full_ts_ns, fill_idx, fill_price,
            r["direction"], r["target_pts"], r["stop_pts"], eod,
        )

        # Compute pnl pts per T multiplier
        pnls_pts = []
        for j, mult in enumerate(T_MULT_SWEEP):
            ti = target_idxs[j]
            target_threshold = r["target_pts"] * mult
            if ti is not None and (stop_idx is None or ti < stop_idx):
                pnls_pts.append(target_threshold)
            elif stop_idx is not None:
                pnls_pts.append(-r["stop_pts"])
            else:
                pnls_pts.append(0.0)

        results.append({
            "date": r["date"],
            "level": r["level"],
            "direction": r["direction"],
            "target_pts": r["target_pts"],
            "stop_pts": r["stop_pts"],
            "group": group,
            "pnls_pts": pnls_pts,
        })

        if (ri + 1) % 1000 == 0:
            print(f"  {ri+1}/{len(rows)} ({time.time()-t1:.0f}s)", flush=True)

    n = len(results)
    days = len({r["date"] for r in results})
    print(f"  done: {n} trades over {days} days "
          f"(skipped={skipped}, fill_miss={fill_misses})", flush=True)

    # Aggregate by group × T multiplier
    print(f"\n{'='*94}\nAGGREGATE: $/day by group × T multiplier\n{'='*94}")
    print(f"  {'Group':<26} {'n':>5} ", end="")
    for m in T_MULT_SWEEP:
        print(f"T{m}x_$day  ", end="")
    print()
    print("-" * 94)

    by_group = defaultdict(list)
    for r in results:
        by_group[r["group"]].append(r)

    group_summary = {}
    for grp in ["A_fresh", "B_post_break_aligned", "C_post_break_mismatch"]:
        rs = by_group[grp]
        n_grp = len(rs)
        print(f"  {grp:<26} {n_grp:>5} ", end="")
        per_mult = []
        for j, m in enumerate(T_MULT_SWEEP):
            total = sum(r["pnls_pts"][j] for r in rs) * MNQ_PT
            pday = total / days if days else 0
            per_mult.append({"total": total, "per_day": pday})
            print(f"  ${pday:>+7.2f}", end="")
        print()
        group_summary[grp] = per_mult

    # Walk-forward for Group B per T mult
    print(f"\n{'='*94}\nWALK-FORWARD on GROUP B (post-break-aligned)\n{'='*94}")
    sorted_dates = sorted({r["date"] for r in results})
    q_size = len(sorted_dates) // 4
    quarters = [
        ("Q1", set(sorted_dates[:q_size])),
        ("Q2", set(sorted_dates[q_size:2 * q_size])),
        ("Q3", set(sorted_dates[2 * q_size:3 * q_size])),
        ("Q4", set(sorted_dates[3 * q_size:])),
    ]
    print(f"  {'T mult':<8} {'Q1 $/day':>10} {'Q2 $/day':>10} "
          f"{'Q3 $/day':>10} {'Q4 $/day':>10} {'Pos quarters':>14}")
    rs_b = by_group["B_post_break_aligned"]
    for j, m in enumerate(T_MULT_SWEEP):
        per_q = []
        pos_q = 0
        for q_label, qd in quarters:
            in_q = [r for r in rs_b if r["date"] in qd]
            q_total = sum(r["pnls_pts"][j] for r in in_q) * MNQ_PT
            q_dates = [d for d in sorted_dates if d in qd]
            qday = q_total / len(q_dates) if q_dates else 0
            per_q.append(qday)
            if qday > 0.5:
                pos_q += 1
        print(f"  {m}x      ${per_q[0]:>+9.2f} ${per_q[1]:>+9.2f} "
              f"${per_q[2]:>+9.2f} ${per_q[3]:>+9.2f} {pos_q}/4")

    # Marginal effect of widening T on Group B (vs T=1.0)
    print(f"\n{'='*94}\nMARGINAL LIFT on GROUP B from widening T (vs T=1.0x)\n{'='*94}")
    for j, m in enumerate(T_MULT_SWEEP):
        if m == 1.0:
            continue
        delta_total = sum((r["pnls_pts"][j] - r["pnls_pts"][0]) for r in rs_b) * MNQ_PT
        delta_pday = delta_total / days
        print(f"  T={m}x: lift on Group B = ${delta_pday:+.2f}/day "
              f"(${delta_total:+.0f} total over {days} days)")

    # Group A check: does widening hurt fresh trades?
    print(f"\n{'='*94}\nGROUP A (fresh) — confirms T widening hurts on the bulk of trades\n{'='*94}")
    rs_a = by_group["A_fresh"]
    for j, m in enumerate(T_MULT_SWEEP):
        if m == 1.0:
            continue
        delta_total = sum((r["pnls_pts"][j] - r["pnls_pts"][0]) for r in rs_a) * MNQ_PT
        delta_pday = delta_total / days
        print(f"  T={m}x: lift on Group A = ${delta_pday:+.2f}/day "
              f"(${delta_total:+.0f} total)")

    # Combined: keep Group A at T=1.0, apply T widening only to Group B
    print(f"\n{'='*94}\nCOMBINED: T=1.0 on Groups A+C, widened T on Group B\n{'='*94}")
    print(f"  Strategy: post-break-aligned trades use widened T; everything else uses base T")
    base_total = sum(
        sum(r["pnls_pts"][0] for r in by_group[grp])
        for grp in ["A_fresh", "B_post_break_aligned", "C_post_break_mismatch"]
    ) * MNQ_PT
    print(f"  Baseline (all T=1.0): ${base_total/days:+.2f}/day  "
          f"(${base_total:+.0f} total)")
    for j, m in enumerate(T_MULT_SWEEP):
        if m == 1.0:
            continue
        # A: T=1.0, B: T=m, C: T=1.0
        a_total = sum(r["pnls_pts"][0] for r in by_group["A_fresh"]) * MNQ_PT
        b_total = sum(r["pnls_pts"][j] for r in by_group["B_post_break_aligned"]) * MNQ_PT
        c_total = sum(r["pnls_pts"][0] for r in by_group["C_post_break_mismatch"]) * MNQ_PT
        combined = a_total + b_total + c_total
        delta = combined - base_total
        print(f"  Group B T={m}x:  ${combined/days:+.2f}/day  "
              f"(Δ ${delta/days:+.2f}/day)")

    # Save JSON
    summary = {
        "generated_at": _dt.datetime.now().isoformat(),
        "n_trades": n, "n_days": days,
        "max_retest_secs": MAX_RETEST_SECS,
        "group_counts": {grp: len(rs) for grp, rs in by_group.items()},
        "group_pnl_per_day": {
            grp: {f"T{m}x": v["per_day"] for m, v in zip(T_MULT_SWEEP, per_mult)}
            for grp, per_mult in group_summary.items()
        },
    }
    out_path = os.path.join(
        os.path.dirname(__file__), "..", "results",
        "post_break_t_widening_2026_05_06.json"
    )
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
