"""
predict.py — loads the trained model and generates predictions.

Usage:
    # Validation predictions (12,000 rows)
    python src/predict.py --input data/validation.csv --output validation_predictions.csv

    # December chart predictions (31 rows)
    python src/predict.py --input data/december-chart-inputs.csv --output data/december-chart-inputs.csv --december
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

from features import build_features

MODEL_PATH = Path("model.pkl")


def load_model() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No model found at {MODEL_PATH}. Run src/train.py first.")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_validation(artifact: dict, input_path: Path, output_path: Path) -> None:
    df = pd.read_csv(input_path, parse_dates=["date"])
    feat = build_features(df, artifact["route_encoder"])
    cols = [c for c in artifact["feature_cols"] if c in feat.columns]

    preds = artifact["model"].predict(feat[cols])
    preds = preds.clip(min=0.01)   # rates must be positive

    out = pd.DataFrame({"load_id": df["load_id"], "predicted_rate": preds.round(2)})
    out.to_csv(output_path, index=False)
    print(f"Saved {len(out):,} predictions → {output_path}")


def predict_december(artifact: dict, input_path: Path) -> None:
    """Fill the predicted_rate column in-place in the december-chart-inputs.csv."""
    df = pd.read_csv(input_path, parse_dates=["date"])
    feat = build_features(df, artifact["route_encoder"])
    cols = [c for c in artifact["feature_cols"] if c in feat.columns]

    preds = artifact["model"].predict(feat[cols])
    preds = preds.clip(min=0.01)

    df["predicted_rate"] = preds.round(2)
    df.to_csv(input_path, index=False)
    print(f"Saved December predictions in-place → {input_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate freight rate predictions.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--december", action="store_true",
                        help="Fill predicted_rate in the december CSV in-place")
    args = parser.parse_args()

    artifact = load_model()

    if args.december:
        predict_december(artifact, Path(args.input))
    else:
        predict_validation(artifact, Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
