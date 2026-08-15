"""Register the notebook champion model and optionally promote it to production."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from .model_registry import ModelRegistry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="artifacts/production_churn_model.joblib")
    parser.add_argument("--metadata", default="artifacts/production_metadata.json")
    parser.add_argument("--registry", default="models/registry")
    parser.add_argument("--name", default="telco-churn")
    parser.add_argument("--version", required=True, help="Immutable version, for example v1.0.0")
    parser.add_argument("--promote", action="store_true", help="Promote this version to production")
    args = parser.parse_args()

    model = joblib.load(Path(args.model))
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    registry = ModelRegistry(args.registry)
    artifact_dir = registry.register(model, args.name, args.version, metadata)
    if args.promote:
        registry.promote(args.name, args.version)
    print(f"Registered model at {artifact_dir}")


if __name__ == "__main__":
    main()
