"""Train and save a calibrated churn model."""
import argparse
import os
from pathlib import Path
import joblib
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from .preprocessing import load_data, split_xy

def train(data_path, output_path, seed=42, mlflow_experiment="telco-churn", mlflow_model_name=None):
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError as error:
        raise RuntimeError("MLflow is required for training. Install requirements.txt first.") from error
    df = load_data(data_path); X, y, prep = split_xy(df)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=seed)
    base = Pipeline([("preprocess", prep), ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))])
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    if os.getenv("MLFLOW_REGISTRY_URI"):
        mlflow.set_registry_uri(os.environ["MLFLOW_REGISTRY_URI"])
    mlflow.set_experiment(mlflow_experiment)
    with mlflow.start_run() as run:
        model = CalibratedClassifierCV(base, method="sigmoid", cv=5); model.fit(Xtr, ytr)
        probabilities = model.predict_proba(Xte)[:, 1]
        mlflow.log_params({"seed": seed, "test_size": 0.2, "calibration": "sigmoid", "cv_folds": 5})
        mlflow.log_metrics({
            "test_roc_auc": roc_auc_score(yte, probabilities),
            "test_brier_score": brier_score_loss(yte, probabilities),
            "test_accuracy": accuracy_score(yte, probabilities >= 0.5),
        })
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=mlflow_model_name,
        )
        print(f"MLflow run: {run.info.run_id} ({tracking_uri})")
    out = Path(output_path); out.parent.mkdir(parents=True, exist_ok=True); joblib.dump({"model": model, "X_test": Xte, "y_test": yte, "seed": seed}, out)
    return model, Xte, yte

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--output", default="artifacts/model.joblib")
    p.add_argument("--mlflow-experiment", default=os.getenv("MLFLOW_EXPERIMENT", "telco-churn"))
    p.add_argument("--mlflow-model-name", default=os.getenv("MLFLOW_MODEL_NAME"))
    a = p.parse_args(); train(a.data, a.output, mlflow_experiment=a.mlflow_experiment, mlflow_model_name=a.mlflow_model_name)
