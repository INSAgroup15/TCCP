"""Generate a SHAP summary plot for the fitted logistic model."""
import argparse
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import shap

def explain(model_path, output_dir="figures"):
    b = joblib.load(model_path); est = b["model"].calibrated_classifiers_[0].estimator; X = b["X_test"].head(1000)
    Xt = est.named_steps["preprocess"].transform(X); names = est.named_steps["preprocess"].get_feature_names_out(); values = shap.Explainer(est.named_steps["model"], Xt, feature_names=names)(Xt)
    Path(output_dir).mkdir(exist_ok=True); shap.summary_plot(values, Xt, feature_names=names, show=False, max_display=20); plt.tight_layout(); plt.savefig(Path(output_dir) / "shap_summary.png", dpi=160, bbox_inches="tight"); plt.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--model", default="artifacts/model.joblib"); p.add_argument("--data", required=False); a = p.parse_args(); explain(a.model)
