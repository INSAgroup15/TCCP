"""FastAPI service that serves the production churn model from the local registry."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .batch_score import prepare_features
from .model_registry import ModelRegistry

REGISTRY_ROOT = os.getenv("REGISTRY_ROOT", "models/registry")
MODEL_NAME = os.getenv("MODEL_NAME", "telco-churn")
MODEL_STAGE = os.getenv("MODEL_STAGE", "production")

app = FastAPI(title="Telco Churn Prediction API", version="1.0.0")


class ScoreRequest(BaseModel):
    records: list[dict[str, Any]] = Field(min_length=1, max_length=10_000)


@lru_cache(maxsize=1)
def get_production_model():
    return ModelRegistry(REGISTRY_ROOT).load(MODEL_NAME, MODEL_STAGE)


@app.get("/health")
def health():
    try:
        _, metadata = get_production_model()
        return {"status": "ok", "model": metadata["model_name"], "version": metadata["version"]}
    except Exception as error:
        return {"status": "unavailable", "detail": str(error)}


@app.get("/metadata")
def metadata():
    try:
        _, record = get_production_model()
        return record
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/predict")
def predict(request: ScoreRequest):
    try:
        model, metadata = get_production_model()
        raw = pd.DataFrame(request.records)
        features = prepare_features(raw, metadata["feature_columns"])
        probability = model.predict_proba(features)[:, 1]
        return {
            "model": metadata["model_name"],
            "version": metadata["version"],
            "predictions": [
                {"predicted_churn_probability": float(value), "predicted_churn": int(value >= 0.5)}
                for value in probability
            ],
        }
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
