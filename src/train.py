"""
train.py — trains the freight rate model and saves it to disk.

Usage:
    python src/train.py
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from features import build_features, RouteEncoder, FEATURE_COLS

DATA_DIR = Path("data")
MODEL_OUT = Path("model.pkl")

# ── Time-based split ──────────────────────────────────────────────────────────
# Train: Jan 2025 – Sep 2025  |  Holdout: Oct 2025
HOLDOUT_START = "2025-10-01"


def load_and_split(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path, parse_dates=["date"])
    train = df[df["date"] < HOLDOUT_START].copy()
    holdout = df[df["date"] >= HOLDOUT_START].copy()
    print(f"Train rows : {len(train):,}")
    print(f"Holdout rows: {len(holdout):,}")
    return train, holdout


def train_model(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Train LightGBM. Falls back to XGBoost if lgbm is unavailable.
    Returns a fitted model object.
    """
    try:
        import lightgbm as lgb
        model = lgb.LGBMRegressor(
            n_estimators=800,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        print("LightGBM not found — using GradientBoostingRegressor as fallback.")
        model = GradientBoostingRegressor(n_estimators=300, random_state=42)

    model.fit(X_train, y_train)
    return model


def evaluate(model, X: pd.DataFrame, y: pd.Series, label: str) -> None:
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    rmse = mean_squared_error(y, preds, squared=False)
    r2 = r2_score(y, preds)
    print(f"\n── {label} ──")
    print(f"  MAE  : ${mae:.2f}  (avg error in dollars)")
    print(f"  RMSE : ${rmse:.2f}")
    print(f"  R²   : {r2:.4f}")


def main() -> None:
    print("Loading data …")
    train_raw, holdout_raw = load_and_split(DATA_DIR / "train-test.csv")

    # Fit route encoder on training data only — no leakage
    route_encoder = RouteEncoder().fit(train_raw)

    print("Building features …")
    train_feat = build_features(train_raw, route_encoder)
    holdout_feat = build_features(holdout_raw, route_encoder)

    # Only use columns that exist
    cols = [c for c in FEATURE_COLS if c in train_feat.columns]
    X_train = train_feat[cols]
    y_train = train_raw["posted_rate"]
    X_holdout = holdout_feat[cols]
    y_holdout = holdout_raw["posted_rate"]

    print("\nTraining model …")
    model = train_model(X_train, y_train)

    evaluate(model, X_train, y_train, "Train")
    evaluate(model, X_holdout, y_holdout, "Holdout (Oct 2025)")

    # Save model + encoder together
    artifact = {"model": model, "route_encoder": route_encoder, "feature_cols": cols}
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(artifact, f)
    print(f"\nModel saved → {MODEL_OUT}")


if __name__ == "__main__":
    main()
