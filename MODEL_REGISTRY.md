# Local model registry

Notebook 02 creates a calibrated champion model and metadata in `artifacts/`. Register it as an immutable version:

```bash
python -m src.register_model --version v1.0.0 --promote
```

The registry stores the model in `models/registry/telco-churn/v1.0.0/` and points the `production` stage to that version. A later version can be registered without replacing the current production model:

```bash
python -m src.register_model --version v1.1.0
python -m src.register_model --version v1.1.0 --promote
```

Score a CSV file in batches:

```bash
python -m src.batch_score --input data/telco_clean_32col.csv --output artifacts/scored_customers.csv
```

Run the API locally after registering a production model:

```bash
uvicorn src.serve:app --reload
```

Open `http://127.0.0.1:8000/docs` for interactive API documentation.

## MLflow in production

Training logs parameters, test metrics, and the calibrated sklearn model to MLflow. By default the tracking data is written to `./mlruns`; in a shared environment set `MLFLOW_TRACKING_URI` to the MLflow tracking server and pass `--mlflow-model-name telco-churn` to register the model:

```bash
set MLFLOW_TRACKING_URI=http://mlflow:5000
python -m src.train --data data/telco.csv --output artifacts/model.joblib --mlflow-model-name telco-churn
```

After validating a registered version, assign it a production alias such as `champion`. The API loads that model when `MLFLOW_MODEL_URI` is set, for example `models:/telco-churn@champion`. Set `MODEL_FEATURE_COLUMNS` to the comma-separated feature schema used by training. This is the production path; the local `ModelRegistry` remains a fallback for offline development.

For Databricks production, use a Unity Catalog model name with three parts and the Databricks MLflow backends:

```bash
set MLFLOW_TRACKING_URI=databricks
set MLFLOW_REGISTRY_URI=databricks-uc
set MLFLOW_MODEL_NAME=prod.tccp-cicd.telco_churn
set MLFLOW_MODEL_URI=models:/prod.tccp-cicd.telco_churn@Champion
```

Use Delta tables in Unity Catalog for training and scoring data, and grant the API identity `USE CATALOG`, `USE SCHEMA`, and `EXECUTE` on the registered model. Databricks Model Serving is the preferred production serving option; keep this FastAPI service when the model must be served by your own application infrastructure.
