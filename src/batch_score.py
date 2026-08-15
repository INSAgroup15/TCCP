"""Score a CSV file in batches using the promoted production model."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .model_registry import ModelRegistry


def prepare_features(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Drop non-feature columns and enforce the training schema before scoring."""
    candidates = frame.drop(columns=["Churn", "customerID"], errors="ignore")
    missing = sorted(set(feature_columns) - set(candidates.columns))
    if missing:
        raise ValueError(f"Input data is missing required features: {missing}")
    return candidates.loc[:, feature_columns]


def score_frame(frame: pd.DataFrame, model, feature_columns: list[str], batch_size: int) -> pd.DataFrame:
    features = prepare_features(frame, feature_columns)
    probabilities = []
    for start in range(0, len(features), batch_size):
        batch = features.iloc[start : start + batch_size]
        probabilities.extend(model.predict_proba(batch)[:, 1])

    output = frame.copy()
    output["predicted_churn_probability"] = probabilities
    output["predicted_churn"] = (output["predicted_churn_probability"] >= 0.5).astype(int)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV file containing model features")
    parser.add_argument("--output", required=True, help="CSV path for scored customers")
    parser.add_argument("--registry", default="models/registry")
    parser.add_argument("--name", default="telco-churn")
    parser.add_argument("--stage", default="production")
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args()

    model, metadata = ModelRegistry(args.registry).load(args.name, args.stage)
    scored = score_frame(pd.read_csv(args.input), model, metadata["feature_columns"], args.batch_size)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output, index=False)
    print(f"Scored {len(scored):,} customers -> {output}")


if __name__ == "__main__":
    main()
