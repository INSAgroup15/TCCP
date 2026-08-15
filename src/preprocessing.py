"""Data loading and leakage-safe feature preparation."""
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Churn"
ID_COLUMNS = ["customerID"]

def load_data(path):
    df = pd.read_csv(Path(path)); df.columns = df.columns.str.strip()
    if TARGET not in df: raise ValueError(f"Expected {TARGET!r}; found {list(df.columns)}")
    if "TotalCharges" in df: df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df[TARGET] = df[TARGET].astype(str).str.strip().map({"Yes": 1, "No": 0})
    if df[TARGET].isna().any(): raise ValueError("Churn must contain only Yes/No values")
    return df

def split_xy(df):
    X = df.drop(columns=[TARGET] + [c for c in ID_COLUMNS if c in df]); y = df[TARGET].astype(int)
    categorical = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist(); numeric = [c for c in X if c not in categorical]
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)])
    return X, y, prep
