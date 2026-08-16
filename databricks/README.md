# Databricks production layout

This folder contains Databricks-specific deployment assets. The reusable model code remains in the repository `src/` folder.

```text
databricks/
├── notebooks/       Training and validation notebooks
├── jobs/            Databricks Workflow job definitions
├── deployment/      Model Serving and endpoint configuration
└── config/          Environment-specific settings
```

Production identifiers:

```text
Experiment: /Shared/telco-churn
Model:      prod.tccp-cicd.telco_churn
Alias:      Champion
Endpoint:   telco-churn-production
```
