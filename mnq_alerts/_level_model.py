"""LightGBM training and inference for both Architecture A (per-level/variant)
and Architecture B (single pooled model)."""
from __future__ import annotations

import os
from typing import Iterable

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

# Final feature column list — must match outputs of compute_all_features.
FEATURE_COLUMNS = [
    "velocity_5s", "velocity_30s", "velocity_5min",
    "acceleration_30s", "path_efficiency_5min",
    "aggressor_balance_5s", "aggressor_balance_30s", "aggressor_balance_5min",
    "net_dollar_flow_5min",
    "volume_5s", "volume_30s", "volume_5min",
    "trade_rate_30s", "max_print_size_30s", "volume_concentration_30s",
    "touches_today", "seconds_since_last_touch",
    "distance_to_vwap", "distance_to_nearest_other_level", "is_post_IB",
    "realized_vol_5min", "realized_vol_30min", "range_30min",
    "seconds_to_market_close", "seconds_into_session",
    "day_of_week_Mon", "day_of_week_Tue", "day_of_week_Wed", "day_of_week_Thu", "day_of_week_Fri",
]

# prior_touch_outcome is categorical — encoded separately.
CATEGORICAL_FEATURES = ["prior_touch_outcome"]

# Conditioning features added to Architecture B only.
CONDITIONING_FEATURES = ["level_id", "direction_id", "tp", "sl"]

LGBM_PARAMS = dict(
    num_leaves=31, max_depth=6, min_data_in_leaf=50,
    learning_rate=0.05, n_estimators=500,
    feature_fraction=0.8, bagging_fraction=0.8,
    objective="binary", metric="auc", verbosity=-1,
)


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "prior_touch_outcome" in df.columns:
        df["prior_touch_outcome"] = df["prior_touch_outcome"].astype("category")
    return df


def _effective_feat_cols(df: pd.DataFrame) -> list[str]:
    """Return feature columns that are actually present in df.

    CATEGORICAL_FEATURES are optional — if the column is absent from the
    dataset (e.g. synthetic test data) we drop it silently so that
    slice_train[feat_cols] never raises a KeyError.
    """
    base = list(FEATURE_COLUMNS)
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            base.append(col)
    return base


def train_architecture_a(
    train: pd.DataFrame, val: pd.DataFrame,
) -> dict[tuple, lgb.LGBMClassifier]:
    """One model per (level_name, direction, tp, sl). Returns dict keyed by tuple."""
    train = _encode_categoricals(train)
    val = _encode_categoricals(val)
    feat_cols = _effective_feat_cols(train)
    cat_cols_present = [c for c in CATEGORICAL_FEATURES if c in feat_cols]
    models: dict[tuple, lgb.LGBMClassifier] = {}
    keys = train[["level_name", "direction", "tp", "sl"]].drop_duplicates().values.tolist()
    for level, direction, tp, sl in keys:
        slice_train = train[(train["level_name"] == level) & (train["direction"] == direction)
                            & (train["tp"] == tp) & (train["sl"] == sl)]
        slice_val = val[(val["level_name"] == level) & (val["direction"] == direction)
                        & (val["tp"] == tp) & (val["sl"] == sl)]
        if len(slice_train) < 100 or len(slice_val) < 20:
            continue  # too small to train reliably; skip this variant.
        X_train = slice_train[feat_cols]
        y_train = slice_train["label"].astype(int)
        X_val = slice_val[feat_cols]
        y_val = slice_val["label"].astype(int)
        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(
            X_train, y_train, eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
            categorical_feature=cat_cols_present if cat_cols_present else "auto",
        )
        models[(level, direction, int(tp), int(sl))] = model
    return models


def predict_architecture_a(
    models: dict[tuple, lgb.LGBMClassifier], event: dict,
) -> dict[tuple, float]:
    """For an event, query all 8 (direction, tp, sl) variants for this level.

    Returns dict keyed by (direction, tp, sl) -> P(win).

    Note: only returns entries for variants that were successfully trained.
    If fewer than 8 variants were trained (e.g. due to insufficient training
    data), the returned dict will contain fewer than 8 entries.
    """
    out: dict[tuple, float] = {}
    # Build feat_cols from whatever columns are actually in the models.
    # Determine which categorical features are present from the event dict.
    cat_present = [c for c in CATEGORICAL_FEATURES if c in event]
    feat_cols = FEATURE_COLUMNS + cat_present
    for (level, direction, tp, sl), m in models.items():
        if level != event["level_name"]:
            continue
        x = pd.DataFrame([{c: event.get(c) for c in feat_cols}])
        for col in cat_present:
            x[col] = x[col].astype("category")
        p = float(m.predict_proba(x)[0, 1])
        out[(direction, int(tp), int(sl))] = p
    return out


def save_models(models: dict, dir_path: str) -> None:
    os.makedirs(dir_path, exist_ok=True)
    for key, m in models.items():
        fn = "_".join(str(k) for k in key) + ".joblib"
        joblib.dump(m, os.path.join(dir_path, fn))


LEVEL_ID_MAP = {
    "IBH": 0, "IBL": 1, "FIB_0.236": 2, "FIB_0.618": 3, "FIB_0.764": 4,
    "FIB_EXT_HI_1.272": 5, "FIB_EXT_LO_1.272": 6,
}
DIRECTION_ID_MAP = {"bounce": 0, "breakthrough": 1}


def _add_conditioning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["level_id"] = df["level_name"].map(LEVEL_ID_MAP).astype(int)
    df["direction_id"] = df["direction"].map(DIRECTION_ID_MAP).astype(int)
    return df


def train_architecture_b(
    train: pd.DataFrame, val: pd.DataFrame,
) -> lgb.LGBMClassifier:
    train = _add_conditioning(_encode_categoricals(train))
    val = _add_conditioning(_encode_categoricals(val))
    feat_cols = FEATURE_COLUMNS + CATEGORICAL_FEATURES + CONDITIONING_FEATURES
    X_train, y_train = train[feat_cols], train["label"].astype(int)
    X_val, y_val = val[feat_cols], val["label"].astype(int)
    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(
        X_train, y_train, eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
        categorical_feature=CATEGORICAL_FEATURES + ["level_id", "direction_id"],
    )
    return model


def predict_architecture_b(
    model: lgb.LGBMClassifier, event: dict,
) -> dict[tuple, float]:
    """Query 8 variants for this event by replicating the row with each conditioning."""
    feat_cols = FEATURE_COLUMNS + CATEGORICAL_FEATURES + CONDITIONING_FEATURES
    rows = []
    keys = []
    for direction in ("bounce", "breakthrough"):
        for tp, sl in [(8, 25), (8, 20), (10, 25), (10, 20)]:
            r = {c: event.get(c) for c in FEATURE_COLUMNS + CATEGORICAL_FEATURES}
            r["level_id"] = LEVEL_ID_MAP[event["level_name"]]
            r["direction_id"] = DIRECTION_ID_MAP[direction]
            r["tp"] = tp
            r["sl"] = sl
            rows.append(r)
            keys.append((direction, tp, sl))
    df = pd.DataFrame(rows)
    if "prior_touch_outcome" in df.columns:
        df["prior_touch_outcome"] = df["prior_touch_outcome"].astype("category")
    probs = model.predict_proba(df[feat_cols])[:, 1]
    return dict(zip(keys, probs.astype(float)))
