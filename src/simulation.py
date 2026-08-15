"""Top-k retention targeting under explicit business assumptions."""
import argparse
import joblib
import pandas as pd

def simulate(model_path, budget=.20, contact_cost=5., save_rate=.25, customer_value=200.):
    b = joblib.load(model_path); X, y = b["X_test"], b["y_test"]; s = X.copy(); s["actual_churn"] = y.to_numpy(); s["p_churn"] = b["model"].predict_proba(X)[:, 1]; s = s.sort_values("p_churn", ascending=False); k = max(1, int(len(s) * budget)); t = s.head(k)
    result = {"test_customers": len(s), "targeted": k, "coverage": k / len(s), "expected_saved_customers": t["p_churn"].sum() * save_rate, "expected_net_value": t["p_churn"].sum() * save_rate * customer_value - k * contact_cost, "assumptions": {"save_rate": save_rate, "customer_value": customer_value, "contact_cost": contact_cost}}; print(pd.Series(result)); return result

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--model", default="artifacts/model.joblib"); p.add_argument("--data", required=False); p.add_argument("--budget", type=float, default=.20); a = p.parse_args(); simulate(a.model, budget=a.budget)
