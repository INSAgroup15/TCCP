"""Evaluate discrimination and probability calibration."""
import argparse
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss

def evaluate(model_path, output_dir="figures"):
    b = joblib.load(model_path); p = b["model"].predict_proba(b["X_test"])[:, 1]; y = b["y_test"]
    metrics = {"brier_score": brier_score_loss(y, p), "roc_auc": roc_auc_score(y, p), "log_loss": log_loss(y, p)}
    frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile"); Path(output_dir).mkdir(exist_ok=True)
    plt.figure(figsize=(6, 5)); plt.plot(mean, frac, "o-", label="model"); plt.plot([0, 1], [0, 1], "--", label="perfect"); plt.xlabel("Mean predicted probability"); plt.ylabel("Observed churn rate"); plt.legend(); plt.tight_layout(); plt.savefig(Path(output_dir) / "calibration_curve.png", dpi=160); plt.close(); print(metrics); return metrics

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--model", default="artifacts/model.joblib"); p.add_argument("--data", required=False); a = p.parse_args(); evaluate(a.model)
