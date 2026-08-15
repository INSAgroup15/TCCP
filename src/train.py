"""Train and save a calibrated churn model."""
import argparse
from pathlib import Path
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from .preprocessing import load_data, split_xy

def train(data_path, output_path, seed=42):
    df = load_data(data_path); X, y, prep = split_xy(df)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=seed)
    base = Pipeline([("preprocess", prep), ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))])
    model = CalibratedClassifierCV(base, method="sigmoid", cv=5); model.fit(Xtr, ytr)
    out = Path(output_path); out.parent.mkdir(parents=True, exist_ok=True); joblib.dump({"model": model, "X_test": Xte, "y_test": yte, "seed": seed}, out)
    return model, Xte, yte

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--data", required=True); p.add_argument("--output", default="artifacts/model.joblib"); a = p.parse_args(); train(a.data, a.output)
