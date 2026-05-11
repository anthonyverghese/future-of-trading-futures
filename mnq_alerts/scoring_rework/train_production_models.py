"""train_production_models.py — train and persist the 3 V_MULTI models for live use.

Trained on the FULL v4 dataset (no walk-forward since this is the live model).
The held-out 30-day test already validated the methodology — for live, we use
all available data to maximize signal.

Outputs (in mnq_alerts/_bot_filter_models/):
  - model_8_20.joblib    (LightGBM for TP=8/SL=20 label)
  - model_8_25.joblib    (TP=8/SL=25)
  - model_10_20.joblib   (TP=10/SL=20)
  - feature_list.json    (canonical feature order, levels list)
  - meta.json            (training date, dataset size, train WR, etc.)
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

AUG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_events_augmented_v4.parquet")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_bot_filter_models")

DECISION_FEATURES = [
    "zone_time_sec", "zone_n_reentries", "zone_max_retreat_pts",
    "zone_distance_velocity", "zone_acceleration",
    "zone_time_7to4_sec", "zone_time_4to1_sec",
    "zone_aggressor_7to4", "zone_aggressor_4to1",
    "zone_volume_7to4", "zone_volume_4to1",
    "velocity_5s", "velocity_30s", "velocity_5min",
    "acceleration_30s", "path_efficiency_5min",
    "aggressor_balance_5s", "aggressor_balance_30s", "aggressor_balance_5min",
    "net_dollar_flow_5min",
    "volume_5s", "volume_30s", "volume_5min",
    "trade_rate_30s", "max_print_size_30s", "volume_concentration_30s",
    "jerk_5s", "large_print_count_30s",
    "approach_consistency", "relative_volume_zone", "pre_zone_volume_5min",
    "realized_vol_5min", "realized_vol_30min", "range_30min",
    "session_range_pts",
    "distance_to_round_25", "distance_to_round_50",
    "distance_to_vwap", "distance_to_nearest_other_level",
    "seconds_into_session", "seconds_to_market_close",
    "time_since_session_high_sec", "time_since_session_low_sec",
    "touches_today",
]

LEVELS = ["IBH", "IBL", "FIB_0.236", "FIB_0.618", "FIB_0.764",
          "FIB_EXT_HI_1.272", "FIB_EXT_LO_1.272"]

LABEL_VARIANTS = [
    ("label_b_8_20", "model_8_20.joblib", 8, 20),
    ("label_b_8_25", "model_8_25.joblib", 8, 25),
    ("label_b_10_20", "model_10_20.joblib", 10, 20),
]


def is_suppressed_time(event_ts):
    et = event_ts.tz_convert("America/New_York")
    mins = et.hour * 60 + et.minute
    return 810 <= mins < 840


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_parquet(AUG_PATH)
    print(f"Loaded {len(df):,} events from {AUG_PATH}")

    df["suppressed"] = df["event_ts"].apply(is_suppressed_time)
    universe = df[(df["human_score"] >= 5) & (~df["suppressed"])].copy().reset_index(drop=True)
    print(f"Training universe (score>=5, no lunch): {len(universe):,} events")

    for lvl in LEVELS:
        universe[f"is_{lvl}"] = (universe["level_name"] == lvl).astype(int)
    feat_cols = [c for c in DECISION_FEATURES if c in universe.columns] + [f"is_{lvl}" for lvl in LEVELS]
    print(f"Features: {len(feat_cols)}")

    X = universe[feat_cols].fillna(0).values
    n_val = max(200, len(X) // 10)

    models_meta = {}
    for label_col, fname, tp, sl in LABEL_VARIANTS:
        if label_col not in universe.columns:
            print(f"[skip] {label_col} not in dataset")
            continue
        y = universe[label_col].astype(int).values
        X_fit, X_val = X[:-n_val], X[-n_val:]
        y_fit, y_val = y[:-n_val], y[-n_val:]
        model = lgb.LGBMClassifier(
            num_leaves=15, max_depth=4, min_data_in_leaf=30,
            learning_rate=0.05, n_estimators=300,
            feature_fraction=0.8, bagging_fraction=0.8,
            objective="binary", verbosity=-1, random_state=42,
        )
        model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
        out_path = os.path.join(OUT_DIR, fname)
        joblib.dump(model, out_path)
        # Self-check WR on val
        p_val = model.predict_proba(X_val)[:, 1]
        ev_val = tp * p_val - sl * (1 - p_val)
        take = ev_val > 0
        wr_taken = float(y_val[take].mean()) if take.sum() > 0 else 0.0
        models_meta[fname] = {
            "label_col": label_col, "tp": tp, "sl": sl, "be": sl / (tp + sl),
            "train_n": int(len(X_fit)), "val_n": int(len(X_val)),
            "val_wr_overall": float(y_val.mean()),
            "val_wr_when_taken": wr_taken,
            "val_take_rate": float(take.mean()),
        }
        print(f"  Wrote {out_path}  (val WR taken={wr_taken:.3f}, take rate={take.mean():.0%})")

    with open(os.path.join(OUT_DIR, "feature_list.json"), "w") as f:
        json.dump({"features": feat_cols, "levels": LEVELS}, f, indent=2)
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump({
            "trained_at": datetime.now().isoformat(),
            "dataset": AUG_PATH,
            "universe_size": len(universe),
            "models": models_meta,
        }, f, indent=2)
    print(f"\nDone. Models saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
