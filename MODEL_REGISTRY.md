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
