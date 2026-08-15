"""A lightweight local model registry for versioned churn-model promotion."""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib


class ModelRegistry:
    """Store versioned model artifacts and designate one version as production."""

    def __init__(self, root: str | Path = "models/registry"):
        self.root = Path(root)

    @staticmethod
    def _validate_name(value: str, label: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
            raise ValueError(f"{label} may contain only letters, numbers, dot, underscore, and hyphen")
        return value

    def _model_dir(self, model_name: str, version: str) -> Path:
        return self.root / self._validate_name(model_name, "model_name") / self._validate_name(version, "version")

    def _stages_path(self, model_name: str) -> Path:
        return self.root / self._validate_name(model_name, "model_name") / "stages.json"

    def register(self, model, model_name: str, version: str, metadata: dict) -> Path:
        """Save a model plus immutable metadata under a named version."""
        target = self._model_dir(model_name, version)
        if target.exists():
            raise FileExistsError(f"Model version already exists: {target}")

        target.mkdir(parents=True, exist_ok=False)
        try:
            joblib.dump(model, target / "model.joblib")
            record = {
                **metadata,
                "model_name": model_name,
                "version": version,
                "registered_at": datetime.now(timezone.utc).isoformat(),
            }
            (target / "metadata.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        except Exception:
            shutil.rmtree(target)
            raise
        return target

    def promote(self, model_name: str, version: str, stage: str = "production") -> Path:
        """Point a deployment stage, usually production, to a registered version."""
        artifact_dir = self._model_dir(model_name, version)
        if not (artifact_dir / "model.joblib").exists():
            raise FileNotFoundError(f"No registered artifact at {artifact_dir}")

        stages_path = self._stages_path(model_name)
        stages_path.parent.mkdir(parents=True, exist_ok=True)
        stages = json.loads(stages_path.read_text(encoding="utf-8")) if stages_path.exists() else {}
        stages[stage] = {"version": version, "promoted_at": datetime.now(timezone.utc).isoformat()}
        stages_path.write_text(json.dumps(stages, indent=2), encoding="utf-8")
        return artifact_dir

    def load(self, model_name: str, stage: str = "production"):
        """Load the model and metadata currently promoted to a stage."""
        stages_path = self._stages_path(model_name)
        if not stages_path.exists():
            raise FileNotFoundError(f"No stages registered for model: {model_name}")
        stages = json.loads(stages_path.read_text(encoding="utf-8"))
        if stage not in stages:
            raise KeyError(f"No {stage!r} version is promoted for {model_name}")

        version = stages[stage]["version"]
        artifact_dir = self._model_dir(model_name, version)
        model = joblib.load(artifact_dir / "model.joblib")
        metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
        return model, metadata
