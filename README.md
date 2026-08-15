# Group 15 — Trustworthy Telco Churn Model

Assignment due in 4 days. We build a churn model whose probabilities are reliable enough to allocate a retention budget, explain churn drivers with SHAP, and simulate a top-k targeting policy.

## Deliverables

- `notebooks/01_data_cleaning.ipynb` — audit and clean IBM Telco Customer Churn data
- `notebooks/02_model_training.ipynb` — train a categorical-encoding baseline
- `notebooks/03_calibration.ipynb` — calibration curves and Brier score
- `notebooks/04_shap_analysis.ipynb` — global and local churn explanations
- `notebooks/05_retention_simulation.ipynb` — fixed-budget targeting and expected value
- `reports/churn_report.html` — report scaffold
- `data/telco_clean_32col.csv` — supplied engineered dataset
- `reports/telco_churn_report.html` — supplied full report

Start with `notebooks/00_project_guide.ipynb`. It explains every step and creates the main exploratory visualizations before the modeling notebooks are run.

## Quick start

Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from Kaggle, save it as `data/telco.csv`, then run:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python -m src.train --data data/telco.csv --output artifacts/model.joblib
python -m src.calibration --model artifacts/model.joblib
python -m src.explain --model artifacts/model.joblib
python -m src.simulation --model artifacts/model.joblib
```

The target is `Churn` (`Yes`/`No`). `customerID` is excluded as an identifier. Missing `TotalCharges` values are imputed inside the modeling pipeline.

For the supplied engineered-data benchmark, run from the repository root:

```bash
Run the notebooks in order: `01_data_cleaning.ipynb`, `02_model_training.ipynb`, `03_calibration.ipynb`, `04_shap_analysis.ipynb`, and `05_retention_simulation.ipynb`.
```

The cleaning notebook requires `data/telco.csv`; the supplied `data/telco_clean_32col.csv` can be used directly for model training.

## Decision framing

The primary probability metric is the Brier score; calibration plots compare predicted risk with observed churn frequency. Targeting ranks customers by:

`expected_value = p(churn) × save_rate × customer_value − contact_cost`

The default simulation targets the top 20%. Replace the assumptions in `src/simulation.py` with the team’s agreed retention cost, save rate, and customer value. This is scenario analysis: the observational dataset cannot identify the true causal save rate.

## Limitations and extension

The IBM dataset is small. The harder WSDM KKBox version requires joining transaction and user-log tables and defining a time-based prediction window to prevent temporal leakage.

## Group and license

Group 15. Code is released under the MIT License; dataset rights remain with the original Kaggle provider.
