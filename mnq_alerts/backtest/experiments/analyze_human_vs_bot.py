"""Deep analysis: what makes the human app succeed where the bot fails?

The human and bot apps trade the SAME market but with different rules:

Bot rules (current deployed):
- 1pt entry threshold (very close to level)
- 6 levels: IBH (SELL only), FIB_EXT_HI, FIB_EXT_LO, FIB_0.236, FIB_0.618, FIB_0.764
- Excluded: IBL, VWAP, FIB_0.5
- Per-level caps: 0.236=18, 0.618=3, 0.764=5, EXT_HI=6, EXT_LO=6, IBH=7
- Monday double caps, 30s global cooldown after loss
- Target/stop per level (T6-12/S20-25)
- Unscored (score >= -99, everything passes)

Human rules:
- 7pt entry threshold (wider, catches bounces earlier)
- All levels: IBH, IBL, VWAP, FIB_EXT_HI, FIB_EXT_LO
- Score >= 5 filter with HUMAN_WEIGHTS (proven in production)
- Human scoring includes STREAKS (+3 for 2+ wins, -2 for 2+ losses)
- 13:30-14:00 suppressed
- ~5.6 alerts/day, 82% WR

Key question: which human advantages are APPLICABLE to the bot?
- Streaks: YES — bot could track and use them
- Session move sweet spots: YES — bot has session_move data
- Power hour bonus: MAYBE — bot penalizes power hour, human rewards it
- VWAP/IBL levels: NO — bot excluded them for good reason (backtested)
- 7pt threshold: NO — bot uses 1pt, fundamentally different entry
- Score >= 5: MAYBE — but with HUMAN weights, not bot weights

This analysis computes human scores on bot trades to find actionable patterns.

Usage:
    cd /Users/anthonyverghese/future-of-trading-futures
    PYTHONPATH=. python -u mnq_alerts/backtest/experiments/analyze_human_vs_bot.py
"""
import os, sys, time
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from mnq_alerts.backtest.data import load_all_days, precompute_arrays
from mnq_alerts.backtest.simulate import simulate_day
from mnq_alerts.backtest.zones import BotZoneTradeReset
from mnq_alerts.backtest.scoring import score_entry, HUMAN_WEIGHTS, EntryFactors

BASE_TS = {
    "FIB_EXT_HI_1.272": (6, 20), "FIB_EXT_LO_1.272": (6, 20),
    "FIB_0.236": (8, 25), "FIB_0.618": (12, 20), "FIB_0.764": (10, 25),
    "IBH": (6, 20),
}
BASE_CAPS = {
    "FIB_0.236": 18, "FIB_0.618": 3, "FIB_0.764": 5,
    "FIB_EXT_HI_1.272": 6, "FIB_EXT_LO_1.272": 6, "IBH": 7,
}
BASE_EXCLUDE = {"FIB_0.5", "IBL"}
IB_SET = 630
FIRST_HOUR_END = 690


def main():
    t0 = time.time()
    print("Loading data...", flush=True)
    dates, caches = load_all_days()
    print(f"Loaded {len(dates)} days in {time.time()-t0:.0f}s", flush=True)

    print("Precomputing arrays...", flush=True)
    arrays = {d: precompute_arrays(caches[d]) for d in dates}

    print("Simulating all days...", flush=True)
    day_results = []
    streak = (0, 0)
    for date in dates:
        dc = caches[date]
        caps = dict(BASE_CAPS)
        if date.weekday() == 0:
            caps = {k: v * 2 for k, v in caps.items()}
        trades, streak = simulate_day(
            dc, arrays[date],
            zone_factory=lambda n, p, dr: BotZoneTradeReset(p, dr),
            target_fn=lambda lv: BASE_TS.get(lv, (8, 25))[0],
            stop_fn=lambda lv: BASE_TS.get(lv, (8, 25))[1],
            max_per_level_map=caps,
            exclude_levels=BASE_EXCLUDE,
            include_ibl=False, include_vwap=False,
            global_cooldown_after_loss_secs=30,
            direction_filter={"IBH": "down"},
        )
        day_pnl = sum(t.pnl_usd for t in trades)
        day_results.append((date, trades, day_pnl))

    all_trades = [t for _, dt, _ in day_results for t in dt]
    bad_days = [(d, t, p) for d, t, p in day_results if p <= -100]
    good_days = [(d, t, p) for d, t, p in day_results if p >= 50]
    neutral_days = [(d, t, p) for d, t, p in day_results if -100 < p < 50]

    n_days = len(dates)
    total_pnl = sum(p for _, _, p in day_results)
    print(f"\nSimulated {n_days} days, {len(all_trades)} trades, "
          f"${total_pnl/n_days:+.2f}/day")
    print(f"Bad days: {len(bad_days)}, Good days: {len(good_days)}, "
          f"Neutral: {len(neutral_days)}")

    # =====================================================================
    # 1. STREAK ANALYSIS — The biggest difference between human and bot
    # =====================================================================
    print("\n" + "=" * 80)
    print("1. STREAK ANALYSIS")
    print("    Human app: +3 for 2+ wins, -2 for 2+ losses")
    print("    Bot: ignores streaks entirely")
    print("=" * 80)

    # WR and P&L after N consecutive losses (within each day)
    print("\n  a) WR by consecutive loss count (intraday streaks):")
    print(f"    {'Consec losses':<16} {'Trades':>7} {'WR%':>6} {'P&L':>9} {'$/trade':>8}")
    for cl_min in range(5):
        wins = 0; total = 0; pnl = 0.0
        for _, dt, _ in day_results:
            cw, cl = 0, 0
            for t in dt:
                if cl == cl_min:
                    total += 1
                    if t.pnl_usd >= 0: wins += 1
                    pnl += t.pnl_usd
                if t.pnl_usd >= 0: cw += 1; cl = 0
                else: cl += 1; cw = 0
        wr = wins / total * 100 if total > 0 else 0
        avg = pnl / total if total > 0 else 0
        label = f"after {cl_min}L" if cl_min > 0 else "no prior L"
        print(f"    {label:<16} {total:>7} {wr:>5.1f}% {pnl:>+9.0f} {avg:>+8.2f}")

    # Same but for 2+ losses specifically, broken by bad/good days
    print("\n  b) After 2+ consecutive losses — bad vs good days:")
    for label, days in [("Bad days", bad_days), ("Good days", good_days), ("All days", day_results)]:
        wins = 0; total = 0; pnl = 0.0
        for _, dt, _ in days:
            cw, cl = 0, 0
            for t in dt:
                if cl >= 2:
                    total += 1
                    if t.pnl_usd >= 0: wins += 1
                    pnl += t.pnl_usd
                if t.pnl_usd >= 0: cw += 1; cl = 0
                else: cl += 1; cw = 0
        wr = wins / total * 100 if total > 0 else 0
        print(f"    {label:<12}: {total} trades, {wr:.1f}% WR, ${pnl:+.0f}")

    # What exactly would streak filter remove?
    print("\n  c) Trades that streak filter (skip on 2+L) would remove:")
    removed_by_level = defaultdict(lambda: [0, 0, 0.0])  # wins, losses, pnl
    kept_by_level = defaultdict(lambda: [0, 0, 0.0])
    for _, dt, _ in day_results:
        cw, cl = 0, 0
        for t in dt:
            target = removed_by_level if cl >= 2 else kept_by_level
            w = 1 if t.pnl_usd >= 0 else 0
            target[t.level][0] += w
            target[t.level][1] += (1 - w)
            target[t.level][2] += t.pnl_usd
            if t.pnl_usd >= 0: cw += 1; cl = 0
            else: cl += 1; cw = 0
    print(f"    {'Level':<20} {'Removed':>8} {'Rem WR%':>8} {'Rem P&L':>9} {'Kept':>8} {'Kept WR%':>9}")
    for lv in sorted(set(list(removed_by_level.keys()) + list(kept_by_level.keys()))):
        rw, rl, rp = removed_by_level[lv]
        kw, kl, kp = kept_by_level[lv]
        rt = rw + rl
        kt = kw + kl
        rwr = rw / rt * 100 if rt > 0 else 0
        kwr = kw / kt * 100 if kt > 0 else 0
        print(f"    {lv:<20} {rt:>8} {rwr:>7.1f}% {rp:>+9.0f} {kt:>8} {kwr:>8.1f}%")
    total_removed = sum(v[0]+v[1] for v in removed_by_level.values())
    total_removed_pnl = sum(v[2] for v in removed_by_level.values())
    total_removed_wins = sum(v[0] for v in removed_by_level.values())
    if total_removed > 0:
        print(f"    TOTAL removed: {total_removed} trades, "
              f"{total_removed_wins/total_removed*100:.1f}% WR, ${total_removed_pnl:+.0f}")

    # =====================================================================
    # 2. HUMAN SCORE APPLIED TO BOT TRADES
    # =====================================================================
    print("\n" + "=" * 80)
    print("2. HUMAN SCORE DISTRIBUTION ON BOT TRADES")
    print("    Note: human scores include streak tracking (+3/-2)")
    print("    Bot levels differ from human (no VWAP/IBL), so level/combo")
    print("    weights may not apply. Streak and session move weights DO apply.")
    print("=" * 80)

    # Score distribution with streak tracking
    print("\n  a) Score distribution (with streak tracking):")
    score_buckets = defaultdict(lambda: [0, 0, 0.0])
    for _, dt, _ in day_results:
        cw, cl = 0, 0  # reset per day
        for t in dt:
            if t.factors is None: continue
            hs = score_entry(t.factors, HUMAN_WEIGHTS, cw=cw, cl=cl)
            w = 1 if t.pnl_usd >= 0 else 0
            score_buckets[hs][0] += w
            score_buckets[hs][1] += (1 - w)
            score_buckets[hs][2] += t.pnl_usd
            if t.pnl_usd >= 0: cw += 1; cl = 0
            else: cl += 1; cw = 0

    print(f"    {'Score':<8} {'Trades':>7} {'WR%':>6} {'P&L':>9} {'$/trade':>8} {'cum P&L':>9}")
    cum = 0.0
    for s in sorted(score_buckets.keys()):
        w, l, p = score_buckets[s]
        total = w + l
        wr = w / total * 100 if total > 0 else 0
        cum += p
        print(f"    {s:<8} {total:>7} {wr:>5.1f}% {p:>+9.0f} {p/total:>+8.2f} {cum:>+9.0f}")

    # Score WITHOUT streaks to isolate streak contribution
    print("\n  b) Score distribution (WITHOUT streak — isolate other factors):")
    score_no_streak = defaultdict(lambda: [0, 0, 0.0])
    for t in all_trades:
        if t.factors is None: continue
        hs = score_entry(t.factors, HUMAN_WEIGHTS, cw=0, cl=0)  # no streak
        w = 1 if t.pnl_usd >= 0 else 0
        score_no_streak[hs][0] += w
        score_no_streak[hs][1] += (1 - w)
        score_no_streak[hs][2] += t.pnl_usd
    print(f"    {'Score':<8} {'Trades':>7} {'WR%':>6} {'P&L':>9} {'$/trade':>8}")
    for s in sorted(score_no_streak.keys()):
        w, l, p = score_no_streak[s]
        total = w + l
        wr = w / total * 100 if total > 0 else 0
        print(f"    {s:<8} {total:>7} {wr:>5.1f}% {p:>+9.0f} {p/total:>+8.2f}")

    # =====================================================================
    # 3. INDIVIDUAL FACTOR ANALYSIS — which human factors help the bot?
    # =====================================================================
    print("\n" + "=" * 80)
    print("3. INDIVIDUAL FACTOR ANALYSIS")
    print("    Which human scoring factors actually predict WR on bot trades?")
    print("=" * 80)

    # Session move buckets (human uses specific sweet spots)
    print("\n  a) Session move (human: +2 for 10-20 green/red, -3 for 0-10 green):")
    move_buckets = [
        ("(-inf,-50]", lambda m: m <= -50),
        ("(-50,-20]", lambda m: -50 < m <= -20),
        ("(-20,-10]", lambda m: -20 < m <= -10),
        ("(-10,0]", lambda m: -10 < m <= 0),
        ("(0,10]", lambda m: 0 < m <= 10),
        ("(10,20]", lambda m: 10 < m <= 20),
        ("(20,50]", lambda m: 20 < m <= 50),
        ("(50,inf)", lambda m: m > 50),
    ]
    human_move_wts = {"(-inf,-50]": "+1", "(-50,-20]": "0", "(-20,-10]": "+2",
                      "(-10,0]": "0", "(0,10]": "-3", "(10,20]": "+2",
                      "(20,50]": "0", "(50,inf)": "0"}
    print(f"    {'Bucket':<14} {'Trades':>7} {'WR%':>6} {'P&L':>9} {'$/trade':>8} {'Human wt':>9}")
    for bname, bfn in move_buckets:
        wins = 0; total = 0; pnl = 0.0
        for t in all_trades:
            if t.factors is None: continue
            if bfn(t.factors.session_move):
                total += 1
                if t.pnl_usd >= 0: wins += 1
                pnl += t.pnl_usd
        wr = wins / total * 100 if total > 0 else 0
        avg = pnl / total if total > 0 else 0
        print(f"    {bname:<14} {total:>7} {wr:>5.1f}% {pnl:>+9.0f} {avg:>+8.2f} {human_move_wts.get(bname,'?'):>9}")

    # Power hour (human: +2, bot: -2 — opposite!)
    print("\n  b) Power hour (human: +2 after 3PM, bot: -2 after 3PM):")
    for label, fn in [("Before 3PM", lambda et: et < 900), ("After 3PM", lambda et: et >= 900)]:
        wins = 0; total = 0; pnl = 0.0
        for t in all_trades:
            if t.factors is None: continue
            if fn(t.factors.et_mins):
                total += 1
                if t.pnl_usd >= 0: wins += 1
                pnl += t.pnl_usd
        wr = wins / total * 100 if total > 0 else 0
        print(f"    {label:<14}: {total} trades, {wr:.1f}% WR, ${pnl:+.0f}")

    # Tick rate
    print("\n  c) Tick rate (human: +2 for 1750-2000 only):")
    tick_buckets = [
        ("<500", lambda tr: tr < 500),
        ("500-1000", lambda tr: 500 <= tr < 1000),
        ("1000-1750", lambda tr: 1000 <= tr < 1750),
        ("1750-2000", lambda tr: 1750 <= tr < 2000),
        ("2000-2500", lambda tr: 2000 <= tr < 2500),
        ("2500+", lambda tr: tr >= 2500),
    ]
    print(f"    {'Bucket':<14} {'Trades':>7} {'WR%':>6} {'P&L':>9}")
    for bname, bfn in tick_buckets:
        wins = 0; total = 0; pnl = 0.0
        for t in all_trades:
            if t.factors is None: continue
            if bfn(t.factors.tick_rate):
                total += 1
                if t.pnl_usd >= 0: wins += 1
                pnl += t.pnl_usd
        wr = wins / total * 100 if total > 0 else 0
        print(f"    {bname:<14} {total:>7} {wr:>5.1f}% {pnl:>+9.0f}")

    # Entry count
    print("\n  d) Entry count (human: -1 for #1/#3, +1 for #2/#5):")
    print(f"    {'Entry #':<10} {'Trades':>7} {'WR%':>6} {'P&L':>9} {'$/trade':>8}")
    for ec in range(1, 19):
        wins = 0; total = 0; pnl = 0.0
        for t in all_trades:
            if t.factors is None: continue
            if t.factors.entry_count == ec:
                total += 1
                if t.pnl_usd >= 0: wins += 1
                pnl += t.pnl_usd
        if total < 10: continue
        wr = wins / total * 100 if total > 0 else 0
        avg = pnl / total if total > 0 else 0
        print(f"    #{ec:<9} {total:>7} {wr:>5.1f}% {pnl:>+9.0f} {avg:>+8.2f}")

    # =====================================================================
    # 4. BAD DAY DEEP DIVE — what specifically goes wrong?
    # =====================================================================
    print("\n" + "=" * 80)
    print(f"4. BAD DAY DEEP DIVE ({len(bad_days)} days)")
    print("=" * 80)

    # On bad days, what % of losses come from trades with human score < 5?
    print("\n  a) Human score of losing trades on bad days (with streaks):")
    bad_loss_scores = defaultdict(int)
    bad_win_scores = defaultdict(int)
    for _, dt, _ in bad_days:
        cw, cl = 0, 0
        for t in dt:
            if t.factors is None: continue
            hs = score_entry(t.factors, HUMAN_WEIGHTS, cw=cw, cl=cl)
            if t.pnl_usd < 0:
                bad_loss_scores[hs] += 1
            else:
                bad_win_scores[hs] += 1
            if t.pnl_usd >= 0: cw += 1; cl = 0
            else: cl += 1; cw = 0
    total_bad_losses = sum(bad_loss_scores.values())
    low_score_losses = sum(v for k, v in bad_loss_scores.items() if k < 5)
    print(f"    Losses with score < 5: {low_score_losses}/{total_bad_losses} "
          f"({low_score_losses/max(total_bad_losses,1)*100:.0f}%)")
    print(f"    Score distribution of losses:")
    for s in sorted(set(list(bad_loss_scores.keys()) + list(bad_win_scores.keys()))):
        bl = bad_loss_scores.get(s, 0)
        bw = bad_win_scores.get(s, 0)
        bt = bl + bw
        bwr = bw / bt * 100 if bt > 0 else 0
        print(f"      score={s}: {bt} trades ({bw}W/{bl}L), WR={bwr:.0f}%")

    # Streak state when losses happen on bad days
    print("\n  b) Consecutive loss count when losses happen (bad days):")
    cl_at_loss = defaultdict(int)
    for _, dt, _ in bad_days:
        cw, cl = 0, 0
        for t in dt:
            if t.pnl_usd < 0:
                cl_at_loss[cl] += 1
            if t.pnl_usd >= 0: cw += 1; cl = 0
            else: cl += 1; cw = 0
    for cl_val in sorted(cl_at_loss.keys()):
        count = cl_at_loss[cl_val]
        label = f"after {cl_val} prior losses"
        print(f"    {label}: {count} losses ({count/max(total_bad_losses,1)*100:.0f}%)")

    # Time of losses on bad days by 15-min buckets
    print("\n  c) Time of losses on bad days (15-min buckets after IB):")
    time_losses = defaultdict(lambda: [0, 0])  # [losses, wins]
    for _, dt, _ in bad_days:
        for t in dt:
            if t.factors is None: continue
            bucket = ((t.factors.et_mins - IB_SET) // 15) * 15
            if t.pnl_usd < 0:
                time_losses[bucket][0] += 1
            else:
                time_losses[bucket][1] += 1
    print(f"    {'Minutes after IB':<18} {'Losses':>7} {'Wins':>7} {'WR%':>6}")
    for b in sorted(time_losses.keys()):
        if b < 0: continue
        l, w = time_losses[b]
        wr = w / (w + l) * 100 if (w + l) > 0 else 0
        start = IB_SET + b
        end = start + 15
        h1, m1 = divmod(start, 60)
        h2, m2 = divmod(end, 60)
        print(f"    {h1}:{m1:02d}-{h2}:{m2:02d} ET     {l:>7} {w:>7} {wr:>5.1f}%")

    # Level performance on bad days with entry count detail
    print("\n  d) Level x entry count on bad days:")
    level_ec_bad = defaultdict(lambda: defaultdict(lambda: [0, 0, 0.0]))
    for _, dt, _ in bad_days:
        for t in dt:
            if t.factors is None: continue
            ec = t.factors.entry_count
            bucket = "1-3" if ec <= 3 else "4-6" if ec <= 6 else "7+"
            w = 1 if t.pnl_usd >= 0 else 0
            level_ec_bad[t.level][bucket][0] += w
            level_ec_bad[t.level][bucket][1] += (1 - w)
            level_ec_bad[t.level][bucket][2] += t.pnl_usd
    for lv in sorted(level_ec_bad.keys()):
        print(f"    {lv}:")
        for b in ["1-3", "4-6", "7+"]:
            if b in level_ec_bad[lv]:
                w, l, p = level_ec_bad[lv][b]
                total = w + l
                wr = w / total * 100 if total > 0 else 0
                print(f"      #{b}: {total} trades, {wr:.0f}% WR, ${p:+.0f}")

    # =====================================================================
    # 5. GOOD DAY PATTERNS — what should we preserve?
    # =====================================================================
    print("\n" + "=" * 80)
    print(f"5. GOOD DAY PATTERNS ({len(good_days)} days) — what to preserve")
    print("=" * 80)

    # Recovery after first loss on good days
    print("\n  a) Recovery pattern after first loss on good days:")
    recovery_trades = []
    recovery_time = []
    for _, dt, _ in good_days:
        first_loss_idx = None
        for i, t in enumerate(dt):
            if t.pnl_usd < 0 and first_loss_idx is None:
                first_loss_idx = i
                break
        if first_loss_idx is not None:
            remaining = dt[first_loss_idx + 1:]
            if remaining:
                recovery_trades.append(len(remaining))
                if remaining[-1].factors and dt[first_loss_idx].factors:
                    recovery_time.append(
                        remaining[-1].factors.et_mins - dt[first_loss_idx].factors.et_mins
                    )
    if recovery_trades:
        print(f"    Avg trades after first loss: {np.mean(recovery_trades):.1f}")
        print(f"    Median trades after first loss: {np.median(recovery_trades):.0f}")
    if recovery_time:
        print(f"    Avg time to recover: {np.mean(recovery_time):.0f} min")

    # Consecutive wins streak on good days
    print("\n  b) Consecutive win streaks on good days:")
    max_streaks = []
    for _, dt, _ in good_days:
        cw = 0; max_cw = 0
        for t in dt:
            if t.pnl_usd >= 0: cw += 1; max_cw = max(max_cw, cw)
            else: cw = 0
        max_streaks.append(max_cw)
    print(f"    Avg max win streak: {np.mean(max_streaks):.1f}")
    print(f"    Median max win streak: {np.median(max_streaks):.0f}")

    # Which levels drive good-day P&L?
    print("\n  c) Level contribution on good days:")
    level_good = defaultdict(lambda: [0, 0, 0.0])
    for _, dt, _ in good_days:
        for t in dt:
            w = 1 if t.pnl_usd >= 0 else 0
            level_good[t.level][0] += w
            level_good[t.level][1] += (1 - w)
            level_good[t.level][2] += t.pnl_usd
    print(f"    {'Level':<20} {'Trades':>7} {'WR%':>6} {'P&L':>9}")
    for lv in sorted(level_good.keys(), key=lambda x: -level_good[x][2]):
        w, l, p = level_good[lv]
        total = w + l
        wr = w / total * 100 if total > 0 else 0
        print(f"    {lv:<20} {total:>7} {wr:>5.1f}% {p:>+9.0f}")

    # =====================================================================
    # 6. CROSS-DAY STREAK ANALYSIS
    # =====================================================================
    print("\n" + "=" * 80)
    print("6. CROSS-DAY STREAK — does previous day's outcome matter?")
    print("=" * 80)

    prev_day_pnl = None
    after_bad = [0, 0, 0.0]
    after_good = [0, 0, 0.0]
    for d, dt, p in day_results:
        if prev_day_pnl is not None:
            if prev_day_pnl <= -100:
                after_bad[0] += 1
                if p > 0: after_bad[1] += 1
                after_bad[2] += p
            elif prev_day_pnl >= 50:
                after_good[0] += 1
                if p > 0: after_good[1] += 1
                after_good[2] += p
        prev_day_pnl = p
    if after_bad[0] > 0:
        print(f"  After bad day: {after_bad[0]} days, "
              f"{after_bad[1]/after_bad[0]*100:.0f}% profitable, "
              f"avg ${after_bad[2]/after_bad[0]:+.1f}/day")
    if after_good[0] > 0:
        print(f"  After good day: {after_good[0]} days, "
              f"{after_good[1]/after_good[0]*100:.0f}% profitable, "
              f"avg ${after_good[2]/after_good[0]:+.1f}/day")

    # =====================================================================
    # 7. TREND DAY ANALYSIS — bot sells into rallies
    # =====================================================================
    print("\n" + "=" * 80)
    print("7. TREND DAY ANALYSIS")
    print("    On trend days, bot takes counter-trend entries that get destroyed.")
    print("    Today (Apr 30): bot sold FIB_0.236/FIB_0.618 into a 200pt rally.")
    print("    Human app waited and only traded FIB_EXT_HI with the trend.")
    print("=" * 80)

    # Classify days by post-IB move direction and magnitude
    print("\n  a) Day type classification (post-IB move):")
    day_types = defaultdict(lambda: [0, 0.0])  # [count, total_pnl]
    for date, dt, day_pnl in day_results:
        dc = caches[date]
        # Post-IB move: where does price end up relative to IB midpoint?
        ib_mid = (dc.ibh + dc.ibl) / 2
        ib_range = dc.ibh - dc.ibl
        if ib_range == 0:
            continue
        # Use price 2 hours after IB (et_mins=750, 12:30 PM) to classify
        late_trades = [t for t in dt if t.factors and t.factors.et_mins >= 750]
        if late_trades:
            late_price = float(dc.full_prices[late_trades[0].entry_idx])
        elif dt:
            late_price = float(dc.full_prices[dt[-1].entry_idx])
        else:
            continue
        move_from_mid = (late_price - ib_mid) / ib_range  # in IB range units

        if move_from_mid > 0.5:
            dtype = "Strong up trend"
        elif move_from_mid > 0:
            dtype = "Mild up"
        elif move_from_mid > -0.5:
            dtype = "Mild down"
        else:
            dtype = "Strong down trend"
        day_types[dtype][0] += 1
        day_types[dtype][1] += day_pnl

    print(f"    {'Day type':<20} {'Days':>5} {'Avg P&L':>9} {'Total P&L':>10}")
    for dtype in ["Strong up trend", "Mild up", "Mild down", "Strong down trend"]:
        if dtype in day_types:
            count, total = day_types[dtype]
            print(f"    {dtype:<20} {count:>5} {total/count:>+9.1f} {total:>+10.0f}")

    # On trend days, are SELL trades the problem?
    print("\n  b) Direction performance on strong trend days:")
    for trend_label, trend_fn in [
        ("Strong up trend", lambda mm: mm > 0.5),
        ("Strong down trend", lambda mm: mm < -0.5),
    ]:
        dir_stats = defaultdict(lambda: [0, 0, 0.0])  # wins, losses, pnl
        for date, dt, day_pnl in day_results:
            dc = caches[date]
            ib_mid = (dc.ibh + dc.ibl) / 2
            ib_range = dc.ibh - dc.ibl
            if ib_range == 0:
                continue
            late_trades = [t for t in dt if t.factors and t.factors.et_mins >= 750]
            if late_trades:
                late_price = float(dc.full_prices[late_trades[0].entry_idx])
            elif dt:
                late_price = float(dc.full_prices[dt[-1].entry_idx])
            else:
                continue
            move_from_mid = (late_price - ib_mid) / ib_range
            if not trend_fn(move_from_mid):
                continue
            for t in dt:
                w = 1 if t.pnl_usd >= 0 else 0
                dir_stats[t.direction][0] += w
                dir_stats[t.direction][1] += (1 - w)
                dir_stats[t.direction][2] += t.pnl_usd
        print(f"    {trend_label}:")
        for d in ["up", "down"]:
            if d in dir_stats:
                w, l, p = dir_stats[d]
                total = w + l
                wr = w / total * 100 if total > 0 else 0
                print(f"      {d}: {total} trades, {wr:.1f}% WR, ${p:+.0f}")

    # =====================================================================
    # 8. WHICH LEVELS FIRE FIRST — and does it matter?
    # =====================================================================
    print("\n" + "=" * 80)
    print("8. FIRST LEVEL TO FIRE — does the first trade level predict the day?")
    print("=" * 80)

    first_level_good = defaultdict(int)
    first_level_bad = defaultdict(int)
    for date, dt, day_pnl in day_results:
        if not dt:
            continue
        first_level = dt[0].level
        if day_pnl <= -100:
            first_level_bad[first_level] += 1
        elif day_pnl >= 50:
            first_level_good[first_level] += 1

    all_levels = sorted(set(list(first_level_good.keys()) + list(first_level_bad.keys())))
    print(f"    {'First level':<20} {'Good days':>10} {'Bad days':>10} {'Bad %':>7}")
    for lv in all_levels:
        g = first_level_good.get(lv, 0)
        b = first_level_bad.get(lv, 0)
        total = g + b
        bad_pct = b / total * 100 if total > 0 else 0
        print(f"    {lv:<20} {g:>10} {b:>10} {bad_pct:>6.0f}%")

    # First trade outcome by level
    print(f"\n    First trade WR by level:")
    first_trade_stats = defaultdict(lambda: [0, 0])  # wins, losses
    for date, dt, day_pnl in day_results:
        if not dt:
            continue
        w = 1 if dt[0].pnl_usd >= 0 else 0
        first_trade_stats[dt[0].level][0] += w
        first_trade_stats[dt[0].level][1] += (1 - w)
    print(f"    {'Level':<20} {'Trades':>7} {'WR%':>6}")
    for lv in sorted(first_trade_stats.keys()):
        w, l = first_trade_stats[lv]
        total = w + l
        wr = w / total * 100 if total > 0 else 0
        print(f"    {lv:<20} {total:>7} {wr:>5.1f}%")

    # =====================================================================
    # 9. COUNTER-TREND ENTRIES — the core problem
    # =====================================================================
    print("\n" + "=" * 80)
    print("9. COUNTER-TREND vs WITH-TREND ENTRIES")
    print("    Counter-trend = SELL when price is above IB mid, BUY when below")
    print("    Today's losses were both counter-trend sells into a rally")
    print("=" * 80)

    ct_stats = defaultdict(lambda: [0, 0, 0.0])  # wins, losses, pnl
    wt_stats = defaultdict(lambda: [0, 0, 0.0])
    for date, dt, day_pnl in day_results:
        dc = caches[date]
        ib_mid = (dc.ibh + dc.ibl) / 2
        for t in dt:
            trade_price = float(dc.full_prices[t.entry_idx])
            # Determine if trade is with or against intraday trend
            price_above_mid = trade_price > ib_mid
            if (t.direction == "down" and price_above_mid) or \
               (t.direction == "up" and not price_above_mid):
                # With trend: selling from above, buying from below
                target = wt_stats
            else:
                # Counter trend: selling from below mid, buying from above
                target = ct_stats
            w = 1 if t.pnl_usd >= 0 else 0
            target[t.level][0] += w
            target[t.level][1] += (1 - w)
            target[t.level][2] += t.pnl_usd

    print(f"\n  WITH-TREND entries (sell above mid, buy below mid):")
    print(f"    {'Level':<20} {'Trades':>7} {'WR%':>6} {'P&L':>9}")
    wt_total_w = wt_total_l = 0
    wt_total_p = 0.0
    for lv in sorted(wt_stats.keys()):
        w, l, p = wt_stats[lv]
        total = w + l
        wr = w / total * 100 if total > 0 else 0
        print(f"    {lv:<20} {total:>7} {wr:>5.1f}% {p:>+9.0f}")
        wt_total_w += w; wt_total_l += l; wt_total_p += p
    wt_all = wt_total_w + wt_total_l
    print(f"    {'TOTAL':<20} {wt_all:>7} {wt_total_w/max(wt_all,1)*100:>5.1f}% {wt_total_p:>+9.0f}")

    print(f"\n  COUNTER-TREND entries (sell below mid, buy above mid):")
    print(f"    {'Level':<20} {'Trades':>7} {'WR%':>6} {'P&L':>9}")
    ct_total_w = ct_total_l = 0
    ct_total_p = 0.0
    for lv in sorted(ct_stats.keys()):
        w, l, p = ct_stats[lv]
        total = w + l
        wr = w / total * 100 if total > 0 else 0
        print(f"    {lv:<20} {total:>7} {wr:>5.1f}% {p:>+9.0f}")
        ct_total_w += w; ct_total_l += l; ct_total_p += p
    ct_all = ct_total_w + ct_total_l
    print(f"    {'TOTAL':<20} {ct_all:>7} {ct_total_w/max(ct_all,1)*100:>5.1f}% {ct_total_p:>+9.0f}")

    # Counter-trend on BAD days specifically
    print(f"\n  Counter-trend on BAD DAYS only:")
    ct_bad = [0, 0, 0.0]
    wt_bad = [0, 0, 0.0]
    for date, dt, day_pnl in bad_days:
        dc = caches[date]
        ib_mid = (dc.ibh + dc.ibl) / 2
        for t in dt:
            trade_price = float(dc.full_prices[t.entry_idx])
            price_above_mid = trade_price > ib_mid
            if (t.direction == "down" and price_above_mid) or \
               (t.direction == "up" and not price_above_mid):
                target = wt_bad
            else:
                target = ct_bad
            w = 1 if t.pnl_usd >= 0 else 0
            target[0] += w
            target[1] += (1 - w)
            target[2] += t.pnl_usd
    ct_t = ct_bad[0] + ct_bad[1]
    wt_t = wt_bad[0] + wt_bad[1]
    print(f"    Counter-trend: {ct_t} trades, {ct_bad[0]/max(ct_t,1)*100:.1f}% WR, ${ct_bad[2]:+.0f}")
    print(f"    With-trend:    {wt_t} trades, {wt_bad[0]/max(wt_t,1)*100:.1f}% WR, ${wt_bad[2]:+.0f}")

    elapsed = time.time() - t0
    print(f"\nAnalysis complete in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print("\nNext step: review these results, then design variants in")
    print("test_human_scoring_v1.py based on what actually shows signal.")


if __name__ == "__main__":
    main()
