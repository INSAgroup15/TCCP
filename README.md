# Trustworthy Customer Churn Prediction

Group 15 assignment: build a churn model whose predicted probabilities are reliable enough to support a retention budget. The project compares five models, calibrates probabilities, explains churn drivers with SHAP, and simulates top-k customer targeting.

## Project goal

The important output is a decision, not only an AUC score: which customers should receive a retention offer under a fixed budget?

Expected net value is calculated as:

`predicted churn probability x save rate x customer value - contact cost`

## Workflow

Run the notebooks in this order.

1. `notebooks/00_project_guide.ipynb` - assignment overview and exploratory charts.
2. `notebooks/01_data_cleaning.ipynb` - cleaning and feature engineering.
3. `notebooks/02_model_training.ipynb` - five-model benchmark, calibration, and production artifact.
4. `notebooks/03_calibration.ipynb` - Brier score and reliability curve.
5. `notebooks/04_shap_analysis.ipynb` - global and local churn explanations.
6. `notebooks/05_retention_simulation.ipynb` - budget and expected-value targeting.
7. `notebooks/06_threshold_optimization.ipynb` - out-of-fold threshold selection and accuracy-recall trade-off.
8. `notebooks/07_model_evaluation_visuals.ipynb` - train/test loss, ROC, PR, calibration, and Seaborn charts.

## Models

- Elastic Net Logistic Regression
- RBF Support Vector Classifier
- Random Forest
- XGBoost
- PyTorch Multilayer Perceptron

The benchmark uses a fixed seed (`42`), an 80/20 stratified holdout test split, five-fold validation for tuning, and five-fold sigmoid calibration. It reports validation PR-AUC mean +/- standard deviation and test metrics with bootstrap uncertainty.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The raw IBM Telco file is stored at `data/telco.csv`. The engineered version is `data/telco_clean_32col.csv`.

## Outputs

After running notebook 02, the `artifacts/` folder contains:

- `production_churn_model.joblib` - calibrated champion model.
- `production_metadata.json` - seed, folds, features, and champion name.
- `benchmark_summary.csv` - comparison of all five models.
- `model.joblib` and `benchmark_models.pkl` - shared evaluation artifacts for later notebooks.

Figures are written to `figures/` and can be used in `reports/telco_churn_report.html`.

## Production note

The notebook includes batch scoring (`BATCH_SIZE = 1000`) for production-style inference. A live web application, cloud deployment, and managed model registry are optional extensions; they are not required for the assignment. Before deployment, validate the champion on a fresh holdout period and monitor calibration drift.

## MLOps workflow

This repository includes a lightweight local MLOps setup: a versioned model registry, batch scoring CLI, FastAPI prediction service, Docker image, and GitHub Actions validation workflow. See [MODEL_REGISTRY.md](MODEL_REGISTRY.md) for the commands to register, promote, score, and serve a model.

## License

MIT License. Dataset rights remain with the original Kaggle provider.
