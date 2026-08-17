"""Create clean, report-ready prediction and retention-action visuals."""

from pathlib import Path

import joblib
import matplotlib
import warnings
import seaborn as sns
import shap

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"


def main() -> None:
    bundle = joblib.load(ROOT / "artifacts" / "model.joblib")
    model = bundle["model"]
    X_test = bundle["X_test"]
    y_test = pd.Series(bundle["y_test"]).astype(int).to_numpy()
    probabilities = model.predict_proba(X_test)[:, 1]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 220,
        }
    )
    sns.set_theme(style="whitegrid", context="notebook")
    OUT.mkdir(exist_ok=True)

    # Data-quality view: show the concrete effect of the cleaning contract.
    raw = pd.read_csv(ROOT / "data" / "telco.csv")
    clean = pd.read_csv(ROOT / "data" / "telco_clean_32col.csv")
    raw_missing = raw.isna().sum()
    # The source file stores missing TotalCharges as whitespace rather than NA.
    if "TotalCharges" in raw:
        raw_missing["TotalCharges"] += raw["TotalCharges"].astype(str).str.strip().eq("").sum()
    raw_missing = raw_missing.sort_values(ascending=False)
    raw_missing = raw_missing[raw_missing > 0]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    labels = list(raw_missing.index) if len(raw_missing) else ["No missing values"]
    before = list(raw_missing.values) if len(raw_missing) else [0]
    x = np.arange(len(labels))
    width = 0.36
    ax.bar(x - width / 2, before, width, label="Raw data", color="#d1495b")
    ax.bar(x + width / 2, [int(clean[col].isna().sum()) if col in clean else 0 for col in labels], width, label="Cleaned data", color="#2f6690")
    ax.set(xticks=x, xticklabels=labels, ylabel="Missing cells", title="Data-cleaning check: missing values before and after preparation")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    for i, value in enumerate(before):
        ax.text(i - width / 2, value + 0.25, str(value), ha="center", fontsize=9)
    ax.text(0.99, 0.95, f"Rows retained: {len(clean):,}\nFeatures: {clean.shape[1] - 1}", transform=ax.transAxes, ha="right", va="top", fontsize=9, bbox={"boxstyle": "round,pad=0.35", "fc": "#f7f7f7", "ec": "#dddddd"})
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.22, top=0.78)
    fig.savefig(OUT / "data_quality.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # 2. Target balance.
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    counts = clean["Churn"].map({0: "Stayed", 1: "Churned"}).value_counts().reindex(["Stayed", "Churned"])
    bars = ax.bar(counts.index, counts.values, color=["#2f6690", "#d1495b"])
    ax.set(title="Observed churn balance", ylabel="Customers")
    ax.grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 50, f"{value:,}\n{value / len(clean):.1%}", ha="center")
    fig.tight_layout(); fig.savefig(OUT / "churn_balance.png", bbox_inches="tight", facecolor="white"); plt.close(fig)

    # 3. Tenure distribution by observed outcome.
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for label, color in [(0, "#2f6690"), (1, "#d1495b")]:
        sns.histplot(clean.loc[clean.Churn == label, "tenure"], bins=24, alpha=.62, label={0: "Stayed", 1: "Churned"}[label], color=color, ax=ax)
    ax.set(title="Tenure distribution by observed outcome", xlabel="Tenure (months)", ylabel="Customers")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.22)
    fig.tight_layout(); fig.savefig(OUT / "tenure_distribution.png", bbox_inches="tight", facecolor="white"); plt.close(fig)

    # 4. Monthly-charge distribution by observed outcome.
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    charge_plot = clean[["MonthlyCharges", "Churn"]].copy()
    charge_plot["Outcome"] = charge_plot["Churn"].map({0: "Stayed", 1: "Churned"})
    sns.boxplot(data=charge_plot, x="Outcome", y="MonthlyCharges", hue="Outcome", order=["Stayed", "Churned"], palette=["#2f6690", "#d1495b"], legend=False, width=.55, ax=ax)
    ax.set(title="Monthly charges by observed outcome", ylabel="Monthly charges")
    ax.grid(axis="y", alpha=.22)
    fig.tight_layout(); fig.savefig(OUT / "monthly_charges.png", bbox_inches="tight", facecolor="white"); plt.close(fig)

    # 5. Correlation heatmap for the core numeric fields.
    numeric = clean[["tenure", "MonthlyCharges", "TotalCharges", "NumServices", "Churn"]].corr()
    fig, ax = plt.subplots(figsize=(5.8, 4.7))
    sns.heatmap(numeric, cmap="RdBu_r", vmin=-1, vmax=1, annot=True, fmt=".2f", square=True, ax=ax, cbar_kws={"label": "Pearson correlation"})
    ax.set_title("Correlation of core numeric variables")
    fig.tight_layout(); fig.savefig(OUT / "numeric_correlation.png", bbox_inches="tight", facecolor="white"); plt.close(fig)

    # Prediction view: clear risk separation and the operational threshold.
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bins = np.linspace(0, 1, 26)
    sns.histplot(probabilities[y_test == 0], bins=bins, alpha=.62, label="Stayed", color="#2f6690", ax=ax)
    sns.histplot(probabilities[y_test == 1], bins=bins, alpha=.62, label="Churned", color="#d1495b", ax=ax)
    ax.axvline(0.30, color="#f4a261", lw=2, ls="--", label="Targeting threshold (0.30)")
    ax.axvline(0.50, color="#333333", lw=1.4, ls=":", label="Binary classification threshold (0.50)")
    ax.set(
        title="Predicted churn risk on the independent test set",
        xlabel="Predicted probability of churn",
        ylabel="Number of customers",
        xlim=(0, 1),
    )
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    fig.savefig(OUT / "prediction_risk.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # 7. ROC and precision-recall curves from the independent test set.
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.3, 3.6))
    ax1.plot(fpr, tpr, color="#264653", lw=2, label=f"ROC-AUC = {roc_auc_score(y_test, probabilities):.3f}")
    ax1.plot([0, 1], [0, 1], "--", color="#999999")
    ax1.set(title="ROC curve", xlabel="False-positive rate", ylabel="True-positive rate"); ax1.legend(frameon=False); ax1.grid(alpha=.2)
    ax2.plot(recall, precision, color="#d1495b", lw=2, label=f"PR-AUC = {average_precision_score(y_test, probabilities):.3f}")
    ax2.axhline(y_test.mean(), ls="--", color="#999999", label=f"Churn prevalence = {y_test.mean():.3f}")
    ax2.set(title="Precision-recall curve", xlabel="Recall", ylabel="Precision"); ax2.legend(frameon=False); ax2.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(OUT / "roc_pr_curves.png", bbox_inches="tight", facecolor="white"); plt.close(fig)

    # 8. Empirical calibration curve.
    frame = pd.DataFrame({"probability": probabilities, "actual": y_test})
    frame["bin"] = pd.qcut(frame["probability"], q=10, duplicates="drop")
    calibration = frame.groupby("bin", observed=True).agg(predicted=("probability", "mean"), observed=("actual", "mean"))
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.plot(calibration.predicted, calibration.observed, "o-", color="#264653", label="Champion model")
    ax.plot([0, 1], [0, 1], "--", color="#999999", label="Perfect calibration")
    ax.set(title="Calibration on the independent test set", xlabel="Mean predicted probability", ylabel="Observed churn rate", xlim=(0, 1), ylim=(0, 1))
    ax.legend(frameon=False); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(OUT / "calibration_test.png", bbox_inches="tight", facecolor="white"); plt.close(fig)

    # 9. Model-agnostic feature importance on the real test set.
    importance_sample = X_test.sample(n=min(300, len(X_test)), random_state=42)
    importance_labels = pd.Series(y_test, index=X_test.index).loc[importance_sample.index].to_numpy()
    importance = permutation_importance(model, importance_sample, importance_labels, scoring="roc_auc", n_repeats=1, random_state=42, n_jobs=1)
    names = np.asarray(list(importance_sample.columns))
    order = np.argsort(importance.importances_mean)[-12:]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.barh(names[order], importance.importances_mean[order], xerr=importance.importances_std[order], color="#457b9d", alpha=.9)
    ax.set(title="Permutation importance of the champion model", xlabel="Decrease in ROC-AUC after permutation")
    ax.grid(axis="x", alpha=.22)
    fig.tight_layout(); fig.savefig(OUT / "permutation_importance.png", bbox_inches="tight", facecolor="white"); plt.close(fig)

    # 10. SHAP beeswarm from real held-out test rows.
    # The SVC is model-agnostic, so the modern SHAP Explainer API uses a
    # permutation explainer with a bounded evaluation budget.
    background = X_test.sample(n=min(5, len(X_test)), random_state=42)
    shap_rows = X_test.sample(n=min(8, len(X_test)), random_state=7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.Explainer(model.predict_proba, background, feature_names=list(X_test.columns))
        shap_result = explainer(shap_rows, max_evals=65)
    shap_explanation = shap_result[..., 1] if shap_result.values.ndim == 3 else shap_result
    shap_values = shap_explanation.values
    shap.plots.beeswarm(
        shap_explanation,
        max_display=18,
        show=False,
        color_bar=True,
        s=18,
        plot_size=(9, 7),
        color_bar_label="Feature value",
    )
    plt.title("SHAP beeswarm: drivers of predicted churn", pad=16)
    plt.tight_layout()
    plt.savefig(OUT / "shap_summary.png", bbox_inches="tight", facecolor="white", dpi=220)
    plt.close()

    # 11. SHAP dependence and partial-dependence panels, similar to the demo layout.
    def dependence_panel(feature: str, axes: tuple, color_feature: str) -> None:
        feature_index = list(X_test.columns).index(feature)
        x_values = shap_rows[feature].to_numpy()
        axes[0].scatter(x_values, shap_values[:, feature_index], c=shap_rows[color_feature], cmap="coolwarm", s=24, alpha=.85, edgecolors="none")
        axes[0].axhline(0, color="#888888", lw=.8)
        axes[0].set(title=f"SHAP dependence: {feature}", xlabel=feature, ylabel="SHAP value")

        grid = np.linspace(X_test[feature].quantile(.02), X_test[feature].quantile(.98), 18)
        pd_frame = X_test.sample(n=min(80, len(X_test)), random_state=11).copy()
        ice = []
        for value in grid:
            changed = pd_frame.copy()
            changed[feature] = value
            ice.append(model.predict_proba(changed)[:, 1])
        ice = np.asarray(ice).T
        axes[1].plot(grid, ice[:12].T, color="#9ecae1", alpha=.35, lw=.8)
        axes[1].plot(grid, ice.mean(axis=0), color="#1479b8", lw=2.2, label="Average effect")
        axes[1].set(title=f"Partial dependence: {feature}", xlabel=feature, ylabel="Predicted churn probability")
        axes[1].legend(frameon=False, fontsize=8)

    fig, axes = plt.subplots(2, 2, figsize=(9, 7.2))
    dependence_panel("tenure", (axes[0, 0], axes[0, 1]), "MonthlyCharges")
    dependence_panel("MonthlyCharges", (axes[1, 0], axes[1, 1]), "tenure")
    fig.suptitle("SHAP dependence and partial dependence for key churn features", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(OUT / "shap_pdp_dependence.png", bbox_inches="tight", facecolor="white", dpi=220)
    plt.close(fig)

    # Action view: rank customers, apply an explicit budget, and show net value.
    ranked = pd.DataFrame({"p_churn": probabilities}).sort_values("p_churn", ascending=False)
    save_rate, customer_value, contact_cost = 0.25, 200.0, 5.0
    budgets = np.linspace(0.01, 0.80, 80)
    net_values = []
    expected_saved = []
    for budget in budgets:
        k = max(1, int(len(ranked) * budget))
        risk_sum = ranked.head(k)["p_churn"].sum()
        expected_saved.append(risk_sum * save_rate)
        net_values.append(risk_sum * save_rate * customer_value - k * contact_cost)
    chosen_budget = 0.20
    chosen_k = max(1, int(len(ranked) * chosen_budget))
    chosen_value = net_values[np.abs(budgets - chosen_budget).argmin()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.9), gridspec_kw={"width_ratios": [1.35, 1]})
    ax1.plot(budgets * 100, net_values, color="#264653", lw=2.5)
    ax1.axhline(0, color="#777777", lw=1)
    ax1.axvline(chosen_budget * 100, color="#e76f51", lw=1.8, ls="--")
    ax1.scatter([chosen_budget * 100], [chosen_value], color="#e76f51", zorder=3)
    ax1.annotate(
        f"20% budget\n{k_label(chosen_k)} targeted\nnet value ${chosen_value:,.0f}",
        xy=(chosen_budget * 100, chosen_value), xytext=(33, chosen_value * 0.62),
        arrowprops={"arrowstyle": "->", "color": "#e76f51"}, fontsize=9,
    )
    ax1.set(title="Retention value by budget", xlabel="Customers contacted (% of test set)", ylabel="Expected net value ($)")
    ax1.grid(alpha=0.22)

    tiers = ["Immediate\nintervention", "Nurture\nsequence", "Monitor"]
    counts = [int((probabilities >= 0.50).sum()), int(((probabilities >= 0.30) & (probabilities < 0.50)).sum()), int((probabilities < 0.30).sum())]
    ax2.bar(tiers, counts, color=["#d1495b", "#f4a261", "#2f6690"], width=0.66)
    ax2.set(title="Prediction-to-action tiers", ylabel="Customers")
    ax2.grid(axis="y", alpha=0.22)
    for i, count in enumerate(counts):
        ax2.text(i, count + max(counts) * 0.02, f"{count:,}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("From calibrated prediction to retention action", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "retention_action.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def k_label(value: int) -> str:
    return f"{value:,} customers"


if __name__ == "__main__":
    main()
