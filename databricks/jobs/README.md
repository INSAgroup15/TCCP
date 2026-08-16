# Jobs

Store Databricks Workflow job definitions here. Recommended production sequence:

1. Load the approved Delta input table.
2. Train and evaluate the model.
3. Register the model in Unity Catalog.
4. Assign the `Champion` alias only after validation.
5. Run batch scoring or update Model Serving.
