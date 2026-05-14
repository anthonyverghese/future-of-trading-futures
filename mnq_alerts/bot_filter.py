"""bot_filter.py — secondary V_MULTI filter for the bot.

Modular gate that runs AFTER the bot's existing decision (1pt-zone entry +
level/direction filters). When enabled, decides whether to actually take a
trade based on:
  1. Time-of-day suppression (13:30-14:00 ET) — matches human's algorithm
  2. Human's composite_score >= 5 (computed leak-safe from current state)
  3. Three LightGBM models trained on (8/20, 8/25, 10/20) bot outcome labels
     must ALL agree expected_pnl > 0 at their respective TP/SL geometry

REVERSIBILITY:
  Set config.BOT_FILTER_ENABLED = False → filter passes through every event
  unchanged (`should_take` returns True). No other side effects.

LATENCY:
  Per-decision cost: ~3-5 ms (feature compute + 3 LightGBM single-row predicts).
  Models are loaded once at startup. Negligible vs IBKR's ~50-200 ms order
  round-trip.
"""
from __future__ import annotations

import datetime
import json
import os
from dataclasses import dataclass, field
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# Lazy imports for joblib/lightgbm — only when enabled. Lets disabled-mode work
# even if those packages are missing on the host.
ET = ZoneInfo("America/New_York")

# Sub-zone boundary inside the 7pt zone (4pt). Matches augment_dataset.py.
ZONE_PT = 7.0
SUB_BAND_PT = 4.0
TICK_RATE_WIN_MIN = 3.0


@dataclass
class FilterContext:
    """All state needed to compute the filter decision. Caller builds this
    from the bot's existing in-memory state at the decision moment."""

    tick_buffer: pd.DataFrame
    """DataFrame indexed by tz-aware UTC datetime, columns: price, size.
    Should span at least the last 15 minutes. The filter slices internally."""

    session_open_price: float
    """First trade price of the RTH session (for session_move calc)."""

    levels: dict[str, float]
    """All level prices in scope for the session (e.g.,
    {'IBH': 18000.0, 'IBL': 17850.0, 'FIB_0.236': 17900.0, ...,
     'VWAP': 17920.0}). VWAP is used only for distance_to_vwap;
    not a tradeable level."""

    bot_touches_today: int
    """Count of prior 1pt-zone entries today at this (level, direction).
    Used as the `touches_today` ML feature."""

    resolution_order_cw: int
    """Consecutive wins among prior bot trades RESOLVED before this event_ts.
    Maintained by the broker; leak-safe."""

    resolution_order_cl: int
    """Consecutive losses among prior bot trades RESOLVED before this event_ts."""

    suppressed_window: tuple[int, int] = (810, 840)
    """Minutes-of-day ET range in which trading is suppressed. Default
    13:30-14:00 ET (matches scoring.py SUPPRESSED_WINDOWS)."""


@dataclass
class FilterDecision:
    take: bool
    reason: str
    """One of: 'disabled', 'time_suppressed', 'human_score_low',
    'model_skip', 'model_take', 'error'."""
    human_score: int | None = None
    model_probs: dict[str, float] = field(default_factory=dict)
    """Map of model_name → {p_win, expected_pnl}."""
    votes: int = 0
    """How many of 3 models said take (need 3/3 for V_MULTI)."""
    diagnostics: dict[str, Any] = field(default_factory=dict)


class BotFilter:
    """V_MULTI filter. Loads 3 LGBM models at startup. Use `should_take(...)`
    at each bot decision moment."""

    def __init__(self, model_dir: str | None = None, enabled: bool = False):
        self.enabled = bool(enabled)
        self.model_dir = model_dir
        self.models: list[tuple[str, Any, int, int]] = []  # (name, model, tp_pts, sl_pts)
        self.feature_list: list[str] = []
        self.levels: list[str] = []
        if not self.enabled:
            return
        # Defer imports until enabled — keeps disabled mode dependency-free.
        try:
            import joblib  # noqa: F401
            import lightgbm  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                f"bot_filter: enabled but joblib/lightgbm missing: {e}. "
                "Install with: pip install joblib lightgbm"
            )
        if not model_dir or not os.path.isdir(model_dir):
            raise RuntimeError(f"bot_filter: enabled but model_dir does not exist: {model_dir}")
        self._load_models()

    def _load_models(self) -> None:
        import joblib
        with open(os.path.join(self.model_dir, "feature_list.json")) as f:
            meta = json.load(f)
        self.feature_list = meta["features"]
        self.levels = meta["levels"]
        for fname, tp, sl in [
            ("model_8_20.joblib", 8, 20),
            ("model_8_25.joblib", 8, 25),
            ("model_10_20.joblib", 10, 20),
        ]:
            path = os.path.join(self.model_dir, fname)
            if not os.path.exists(path):
                raise RuntimeError(f"bot_filter: missing model file {path}")
            m = joblib.load(path)
            self.models.append((fname.replace(".joblib", ""), m, tp, sl))
        print(f"[bot_filter] loaded {len(self.models)} models from {self.model_dir}", flush=True)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def should_take(
        self,
        *,
        level_name: str,
        level_price: float,
        event_ts: pd.Timestamp,
        event_price: float,
        approach_direction: int,
        entry_count_7pt: int,
        context: FilterContext,
    ) -> FilterDecision:
        """Decide whether to take the trade.

        approach_direction: +1 if price approached from below (bounce-down/SELL),
                           -1 if from above (bounce-up/BUY).
        entry_count_7pt: 1-indexed count of zone entries for this
                        (level_name, approach_direction) today, INCLUDING this
                        event. Matches the test_count_7pt feature used in
                        training (cumcount + 1 on event_date × level × direction).
        """
        if not self.enabled:
            return FilterDecision(take=True, reason="disabled")

        try:
            # 1. Time-of-day suppression (matches human's algorithm).
            et = event_ts.tz_convert(ET)
            mins = et.hour * 60 + et.minute
            if context.suppressed_window[0] <= mins < context.suppressed_window[1]:
                return FilterDecision(take=False, reason="time_suppressed",
                                      diagnostics={"et_minutes": mins})

            # 2. Find T7 (when price first crossed within 7pt of level).
            ticks = context.tick_buffer
            if ticks.empty:
                return FilterDecision(take=False, reason="error",
                                      diagnostics={"err": "empty_tick_buffer"})
            ts_ns = ticks.index.asi8
            prices = ticks["price"].to_numpy()
            sizes = ticks["size"].to_numpy()
            event_idx = int(np.searchsorted(ts_ns, event_ts.value, side="left"))
            if event_idx >= len(prices):
                event_idx = len(prices) - 1
            # Walk back to find T7
            dist = np.abs(prices[: event_idx + 1] - level_price)
            t7_idx = event_idx
            while t7_idx > 0 and dist[t7_idx - 1] <= ZONE_PT:
                t7_idx -= 1
            t7_ns = ts_ns[t7_idx]
            t7_price = prices[t7_idx]

            # 3. Compute human composite_score at T7.
            from scoring import composite_score
            t7_dt = pd.Timestamp(int(t7_ns)).tz_localize("UTC").tz_convert(ET)
            now_et = datetime.time(t7_dt.hour, t7_dt.minute)
            # 3-min tick rate ending at T7
            tick_rate_win_ns = int(TICK_RATE_WIN_MIN * 60 * 1_000_000_000)
            i_rate_start = int(np.searchsorted(ts_ns, t7_ns - tick_rate_win_ns, side="left"))
            tick_rate = (t7_idx - i_rate_start) / TICK_RATE_WIN_MIN
            session_move = t7_price - context.session_open_price
            human_dir = "down" if approach_direction == 1 else "up"
            score = composite_score(
                level_name=level_name,
                entry_count=int(entry_count_7pt),
                now_et=now_et,
                tick_rate=float(tick_rate),
                session_move_pts=float(session_move),
                direction=human_dir,
                consecutive_wins=int(context.resolution_order_cw),
                consecutive_losses=int(context.resolution_order_cl),
            )
            if score < 5:
                return FilterDecision(take=False, reason="human_score_low",
                                      human_score=int(score))

            # 4. Compute ML features at the 1pt-zone-entry moment.
            features = self._compute_features(
                ticks_prices=prices, ticks_sizes=sizes, ticks_times=ts_ns,
                event_idx=event_idx, t7_idx=t7_idx, level_price=level_price,
                event_price=event_price, event_ts_ns=event_ts.value,
                level_name=level_name, context=context,
                tick_rate=tick_rate, session_move=session_move,
            )

            # 5. Build feature vector in canonical order, predict from 3 models.
            # Pass as a single-row DataFrame so column names match training and
            # sklearn doesn't emit "X does not have valid feature names" warnings.
            feat_vec = pd.DataFrame(
                [[features.get(f, 0.0) for f in self.feature_list]],
                columns=self.feature_list,
            )
            votes = 0
            probs = {}
            for name, model, tp, sl in self.models:
                p = float(model.predict_proba(feat_vec)[0, 1])
                ev = tp * p - sl * (1 - p)
                probs[name] = {"p_win": p, "expected_pnl": ev}
                if ev > 0:
                    votes += 1

            # V_MULTI: require 3-of-3 agreement.
            take = votes >= 3
            return FilterDecision(
                take=take,
                reason="model_take" if take else "model_skip",
                human_score=int(score),
                model_probs=probs,
                votes=votes,
                diagnostics={"t7_idx_offset": event_idx - t7_idx,
                             "tick_rate": float(tick_rate),
                             "session_move_pts": float(session_move),
                             "entry_count_7pt": int(entry_count_7pt)},
            )
        except Exception as e:
            return FilterDecision(take=True, reason="error",
                                  diagnostics={"err": str(e), "fallback": "passthrough"})

    # ------------------------------------------------------------------ #
    # Feature compute — mirrors augment_dataset.py + add_features.py logic.
    # ------------------------------------------------------------------ #
    def _compute_features(
        self,
        *,
        ticks_prices: np.ndarray,
        ticks_sizes: np.ndarray,
        ticks_times: np.ndarray,
        event_idx: int,
        t7_idx: int,
        level_price: float,
        event_price: float,
        event_ts_ns: int,
        level_name: str,
        context: FilterContext,
        tick_rate: float,
        session_move: float,
    ) -> dict[str, float]:
        f: dict[str, float] = {}

        # Helper: slice ticks in last N seconds ending at event_ts.
        def _slice_last_sec(secs: int) -> tuple[int, int]:
            lo_ns = event_ts_ns - int(secs * 1e9)
            i_lo = int(np.searchsorted(ticks_times, lo_ns, side="left"))
            return i_lo, event_idx + 1

        # --- Family 1: approach kinematics (5s/30s/5min velocities) ---
        for win_name, win_sec in [("5s", 5), ("30s", 30), ("5min", 300)]:
            lo, hi = _slice_last_sec(win_sec)
            if hi - lo < 2:
                f[f"velocity_{win_name}"] = 0.0
                continue
            dp = float(ticks_prices[hi - 1] - ticks_prices[lo])
            dt = (ticks_times[hi - 1] - ticks_times[lo]) / 1e9
            f[f"velocity_{win_name}"] = dp / dt if dt > 0 else 0.0
        f["acceleration_30s"] = f.get("velocity_5s", 0) - f.get("velocity_30s", 0)
        lo5, hi5 = _slice_last_sec(300)
        if hi5 - lo5 >= 2:
            p5 = ticks_prices[lo5:hi5]
            disp = abs(p5[-1] - p5[0])
            total = float(np.abs(np.diff(p5)).sum())
            f["path_efficiency_5min"] = disp / total if total > 0 else 0.0
        else:
            f["path_efficiency_5min"] = 0.0

        # --- Family 2: tick-rule aggressor balance ---
        def _classify(prices: np.ndarray) -> np.ndarray:
            n = len(prices)
            side = np.zeros(n, dtype=np.int8)
            last = 0
            for i in range(1, n):
                if prices[i] > prices[i - 1]:
                    last = 1
                elif prices[i] < prices[i - 1]:
                    last = -1
                side[i] = last
            return side

        for win_name, win_sec in [("5s", 5), ("30s", 30), ("5min", 300)]:
            lo, hi = _slice_last_sec(win_sec)
            if hi - lo < 2:
                f[f"aggressor_balance_{win_name}"] = 0.0
                continue
            p_sub = ticks_prices[lo:hi]; s_sub = ticks_sizes[lo:hi]
            sd = _classify(p_sub)
            cls_total = float(s_sub[sd != 0].sum())
            signed = float((s_sub * sd).sum())
            f[f"aggressor_balance_{win_name}"] = signed / cls_total if cls_total > 0 else 0.0
        lo5, hi5 = _slice_last_sec(300)
        if hi5 - lo5 >= 2:
            p5 = ticks_prices[lo5:hi5]; s5 = ticks_sizes[lo5:hi5]
            sd5 = _classify(p5)
            f["net_dollar_flow_5min"] = float((p5 * s5 * sd5).sum())
        else:
            f["net_dollar_flow_5min"] = 0.0

        # --- Family 3: volume profile ---
        for win_name, win_sec in [("5s", 5), ("30s", 30), ("5min", 300)]:
            lo, hi = _slice_last_sec(win_sec)
            f[f"volume_{win_name}"] = float(ticks_sizes[lo:hi].sum()) if hi > lo else 0.0
        lo30, hi30 = _slice_last_sec(30)
        if hi30 > lo30:
            sub30 = ticks_sizes[lo30:hi30].astype(float)
            f["trade_rate_30s"] = (hi30 - lo30) / max(1.0, (ticks_times[hi30 - 1] - ticks_times[lo30]) / 1e9)
            f["max_print_size_30s"] = float(sub30.max())
            total = float(sub30.sum())
            f["volume_concentration_30s"] = float((sub30 ** 2).sum() / total ** 2) if total > 0 else 0.0
        else:
            f["trade_rate_30s"] = 0.0
            f["max_print_size_30s"] = 0.0
            f["volume_concentration_30s"] = 0.0

        # --- Family 4: zone features (7→1pt approach window) ---
        # Use [t7_idx, event_idx] as the window.
        if event_idx > t7_idx:
            win_prices = ticks_prices[t7_idx:event_idx + 1]
            win_sizes = ticks_sizes[t7_idx:event_idx + 1]
            win_times = ticks_times[t7_idx:event_idx + 1]
            win_dist = np.abs(win_prices - level_price)
            duration_sec = (win_times[-1] - win_times[0]) / 1e9
            f["zone_time_sec"] = float(duration_sec) if duration_sec > 0 else 0.0
            f["zone_max_retreat_pts"] = float(win_dist.max())
            if duration_sec > 0:
                f["zone_distance_velocity"] = (win_dist[-1] - win_dist[0]) / duration_sec
            else:
                f["zone_distance_velocity"] = 0.0
            mid = len(win_prices) // 2
            if mid >= 1 and (win_times[mid] - win_times[0]) > 0 and (win_times[-1] - win_times[mid]) > 0:
                v1 = (win_dist[mid] - win_dist[0]) / ((win_times[mid] - win_times[0]) / 1e9)
                v2 = (win_dist[-1] - win_dist[mid]) / ((win_times[-1] - win_times[mid]) / 1e9)
                f["zone_acceleration"] = v2 - v1
            else:
                f["zone_acceleration"] = 0.0
            # Sub-bands
            sd_win = _classify(win_prices)
            mask_outer = win_dist > SUB_BAND_PT
            mask_inner = ~mask_outer
            for mask, suffix in [(mask_outer, "7to4"), (mask_inner, "4to1")]:
                if mask.any():
                    s_m = win_sizes[mask]; sd_m = sd_win[mask]
                    t_m = win_times[mask]
                    cls_total = float(s_m[sd_m != 0].sum())
                    signed = float((s_m * sd_m).sum())
                    f[f"zone_aggressor_{suffix}"] = signed / cls_total if cls_total > 0 else 0.0
                    f[f"zone_volume_{suffix}"] = float(s_m.sum())
                    f[f"zone_time_{suffix}_sec"] = float((t_m.max() - t_m.min()) / 1e9) if mask.sum() > 1 else 0.0
                else:
                    f[f"zone_aggressor_{suffix}"] = 0.0
                    f[f"zone_volume_{suffix}"] = 0.0
                    f[f"zone_time_{suffix}_sec"] = 0.0
            # Re-entry count over 5min lookback before t7
            lb_start = max(0, t7_idx - 300)
            full_dist = np.abs(ticks_prices[lb_start:event_idx + 1] - level_price)
            in_zone = full_dist <= ZONE_PT
            transitions = np.diff(in_zone.astype(np.int8))
            f["zone_n_reentries"] = int((transitions == 1).sum())
        else:
            for k in ["zone_time_sec", "zone_max_retreat_pts", "zone_distance_velocity",
                     "zone_acceleration", "zone_aggressor_7to4", "zone_volume_7to4",
                     "zone_time_7to4_sec", "zone_aggressor_4to1", "zone_volume_4to1",
                     "zone_time_4to1_sec", "zone_n_reentries"]:
                f[k] = 0.0

        # --- Family 5: vol & time ---
        for win_name, win_sec in [("5min", 300), ("30min", 1800)]:
            lo, hi = _slice_last_sec(win_sec)
            if hi - lo >= 2:
                p = ticks_prices[lo:hi]
                ret = np.diff(p) / p[:-1]
                if win_name == "5min":
                    f["realized_vol_5min"] = float(np.std(ret)) if len(ret) > 0 else 0.0
                else:
                    f["realized_vol_30min"] = float(np.std(ret)) if len(ret) > 0 else 0.0
                    f["range_30min"] = float(p.max() - p.min())
            else:
                if win_name == "5min":
                    f["realized_vol_5min"] = 0.0
                else:
                    f["realized_vol_30min"] = 0.0
                    f["range_30min"] = 0.0
        # Time-of-day
        et = pd.Timestamp(int(event_ts_ns)).tz_localize("UTC").tz_convert(ET)
        close_et = et.replace(hour=16, minute=0, second=0, microsecond=0)
        open_et = et.replace(hour=9, minute=30, second=0, microsecond=0)
        f["seconds_to_market_close"] = max(0.0, (close_et - et).total_seconds())
        f["seconds_into_session"] = (et - open_et).total_seconds()

        # --- Add-features v2 ---
        # jerk_5s: change in acceleration over last 5s
        win10_lo, win10_hi = _slice_last_sec(10)
        win5_lo, win5_hi = _slice_last_sec(5)
        if win5_hi > win5_lo and win10_hi - win10_lo > 1 and win5_hi > 1:
            t1 = (ticks_times[win5_lo] - ticks_times[win10_lo]) / 1e9 if win10_lo < win5_lo else 1.0
            t2 = (ticks_times[event_idx] - ticks_times[win5_lo]) / 1e9 if win5_lo < event_idx else 1.0
            if t1 > 0 and t2 > 0 and win10_lo < win5_lo < event_idx:
                v1 = (ticks_prices[win5_lo] - ticks_prices[win10_lo]) / t1
                v2 = (ticks_prices[event_idx] - ticks_prices[win5_lo]) / t2
                f["jerk_5s"] = (v2 - v1) / ((t1 + t2) / 2)
            else:
                f["jerk_5s"] = 0.0
        else:
            f["jerk_5s"] = 0.0
        # Round-number proximity
        ROUND_25, ROUND_50 = 25.0, 50.0
        f["distance_to_round_25"] = abs(event_price - round(event_price / ROUND_25) * ROUND_25)
        f["distance_to_round_50"] = abs(event_price - round(event_price / ROUND_50) * ROUND_50)
        # Large prints in 30s
        lo30, hi30 = _slice_last_sec(30)
        f["large_print_count_30s"] = int((ticks_sizes[lo30:hi30] >= 10).sum()) if hi30 > lo30 else 0
        # Session range + time since extremes
        all_prices = ticks_prices[:event_idx + 1]
        if len(all_prices) > 0:
            high = float(all_prices.max()); low = float(all_prices.min())
            f["session_range_pts"] = high - low
            high_idx = int(np.argmax(all_prices))
            low_idx = int(np.argmin(all_prices))
            f["time_since_session_high_sec"] = (event_ts_ns - ticks_times[high_idx]) / 1e9
            f["time_since_session_low_sec"] = (event_ts_ns - ticks_times[low_idx]) / 1e9
        else:
            f["session_range_pts"] = 0.0
            f["time_since_session_high_sec"] = -1.0
            f["time_since_session_low_sec"] = -1.0
        # Pre-zone volume (5min before T7)
        pre_lo = int(np.searchsorted(ticks_times, ticks_times[t7_idx] - int(5 * 60 * 1e9), side="left"))
        f["pre_zone_volume_5min"] = float(ticks_sizes[pre_lo:t7_idx].sum()) if t7_idx > pre_lo else 0.0
        # Approach consistency
        if event_idx > t7_idx:
            in_zone_dist = np.abs(ticks_prices[t7_idx:event_idx + 1] - level_price)
            diffs = np.diff(in_zone_dist)
            non_zero = diffs[diffs != 0]
            f["approach_consistency"] = float((non_zero < 0).sum() / len(non_zero)) if len(non_zero) > 0 else 0.5
        else:
            f["approach_consistency"] = 0.5
        # Relative volume
        in_zone_vol = float(ticks_sizes[t7_idx:event_idx + 1].sum())
        in_zone_dur = max(1.0, (ticks_times[event_idx] - ticks_times[t7_idx]) / 1e9) if event_idx > t7_idx else 1.0
        in_zone_rate = in_zone_vol / in_zone_dur
        lb_30min_ns = int(30 * 60 * 1e9)
        lb_lo = int(np.searchsorted(ticks_times, event_ts_ns - lb_30min_ns, side="left"))
        lb_vol = float(ticks_sizes[lb_lo:event_idx + 1].sum())
        lb_dur = max(1.0, (event_ts_ns - ticks_times[lb_lo]) / 1e9) if event_idx > lb_lo else 1800.0
        f["relative_volume_zone"] = in_zone_rate / max(0.01, lb_vol / lb_dur)

        # --- Family 6: level context ---
        f["touches_today"] = float(context.bot_touches_today)
        f["distance_to_vwap"] = float(event_price - context.levels.get("VWAP", event_price))
        others = [p for n, p in context.levels.items()
                  if n != level_name and n != "VWAP"]
        f["distance_to_nearest_other_level"] = min(abs(event_price - p) for p in others) if others else 0.0
        # seconds_since_last_touch unused in models (was in training but importance ~0); set 0.
        f["seconds_since_last_touch"] = 0.0
        f["is_post_IB"] = 1.0

        # --- One-hot level ---
        for lvl in self.levels:
            f[f"is_{lvl}"] = 1.0 if level_name == lvl else 0.0

        return f


# ------------------------------------------------------------------ #
# Module-level singleton — instantiated once, used by bot_trader
# ------------------------------------------------------------------ #
_filter_instance: BotFilter | None = None


def get_filter() -> BotFilter:
    """Returns the singleton BotFilter. Constructed lazily from config."""
    global _filter_instance
    if _filter_instance is None:
        try:
            from config import BOT_FILTER_ENABLED, BOT_FILTER_MODEL_DIR
        except ImportError:
            BOT_FILTER_ENABLED = False
            BOT_FILTER_MODEL_DIR = None
        _filter_instance = BotFilter(
            model_dir=BOT_FILTER_MODEL_DIR,
            enabled=BOT_FILTER_ENABLED,
        )
    return _filter_instance


def reset_filter_for_tests() -> None:
    """Tests use this to clear the singleton between cases."""
    global _filter_instance
    _filter_instance = None
