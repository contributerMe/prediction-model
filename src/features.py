from __future__ import annotations

import numpy as np
import pandas as pd


# ── Haversine distance ────────────────────────────────────────────────────────

def haversine_km(lat1: pd.Series, lon1: pd.Series,
                 lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    """Great-circle distance in km."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# ── Route statistics (fit on train, apply to all) ─────────────────────────────

class RouteEncoder:
    """Stores per-route mean rate and frequency from training data."""

    def __init__(self) -> None:
        self._stats: pd.DataFrame | None = None

    def fit(self, df: pd.DataFrame) -> "RouteEncoder":
        route = df["pickup"] + "→" + df["delivery"]
        self._stats = (
            df.assign(route=route)
            .groupby("route")["posted_rate"]
            .agg(route_mean_rate="mean", route_count="count")
            .reset_index()
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self._stats is not None, "Call fit() before transform()"
        route = (df["pickup"] + "→" + df["delivery"]).rename("route")
        out = df.copy()
        merged = out.join(
            self._stats.set_index("route")[["route_mean_rate", "route_count"]],
            on=route,
        )
        # Fill unseen routes with overall mean
        global_mean = self._stats["route_mean_rate"].mean()
        out["route_mean_rate"] = merged["route_mean_rate"].fillna(global_mean)
        out["route_count"] = merged["route_count"].fillna(0)
        return out


# ── Main feature builder ──────────────────────────────────────────────────────

EQUIPMENT_TYPES = ["Dry Van", "Reefer", "Flatbed"]


def build_features(df: pd.DataFrame, route_encoder: RouteEncoder | None = None) -> pd.DataFrame:
    """
    Takes a raw dataframe (train or predict) and returns a feature matrix.
    Pass route_encoder=None for training (it will be created and returned separately).
    """
    out = df.copy()

    # ── Date features
    out["date"] = pd.to_datetime(out["date"])
    out["month"] = out["date"].dt.month
    out["day_of_week"] = out["date"].dt.dayofweek   # 0=Mon, 6=Sun
    out["week_of_year"] = out["date"].dt.isocalendar().week.astype(int)
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["day_of_month"] = out["date"].dt.day

    # ── Equipment one-hot
    for eq in EQUIPMENT_TYPES:
        out[f"eq_{eq.lower().replace(' ', '_')}"] = (out["equipment"] == eq).astype(int)

    # ── Market interaction (may be absent in december CSV)
    if "market_index" in out.columns and "quote_signal" in out.columns:
        out["market_signal"] = out["market_index"] * out["quote_signal"]
    else:
        out["market_index"] = np.nan
        out["quote_signal"] = np.nan
        out["market_signal"] = np.nan

    # ── Haversine distance (km), compare with given distance
    if all(c in out.columns for c in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]):
        out["haversine_km"] = haversine_km(
            out["pickup_lat"], out["pickup_lon"],
            out["delivery_lat"], out["delivery_lon"],
        )
        out["distance_ratio"] = out["distance"] / (out["haversine_km"] * 0.621371 + 1e-6)
    else:
        out["haversine_km"] = np.nan
        out["distance_ratio"] = np.nan

    # ── Route statistics
    if route_encoder is not None:
        out = route_encoder.transform(out)

    return out


FEATURE_COLS = [
    "distance", "weight",
    "month", "day_of_week", "week_of_year", "is_weekend", "day_of_month",
    "eq_dry_van", "eq_reefer", "eq_flatbed",
    "market_index", "quote_signal", "market_signal",
    "haversine_km", "distance_ratio",
    "route_mean_rate", "route_count",
]
