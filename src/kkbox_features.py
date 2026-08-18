"""Build one leakage-safe modelling row per KKBox subscriber.

The KKBox release is event-oriented: labels are in a train table, while
transactions and listening activity contain many rows per subscriber. This
module reduces those tables to subscriber-level features before the common
training pipeline is called.
"""
from pathlib import Path
import argparse
import pandas as pd


def _find(data_dir, names):
    root = Path(data_dir)
    for name in names:
        path = root / name
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find any of {names} in {root}")


def _date_features(frame, columns):
    out = frame.copy()
    for column in columns:
        if column in out:
            out[column] = pd.to_datetime(out[column], format="%Y%m%d", errors="coerce")
    return out


def _aggregate_transactions(path, chunksize):
    sums = counts = None
    frames = []
    columns = [
        "msno", "payment_method_id", "payment_plan_days", "plan_list_price",
        "actual_amount_paid", "is_auto_renew", "transaction_date",
        "membership_expire_date", "is_cancel",
    ]
    for chunk in pd.read_csv(path, usecols=lambda c: c in columns, chunksize=chunksize):
        chunk = _date_features(chunk, ["transaction_date", "membership_expire_date"])
        numeric = [c for c in columns if c not in {"msno", "transaction_date", "membership_expire_date"} and c in chunk]
        for col in numeric:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        grouped = chunk.groupby("msno", sort=False).agg(
            transaction_count=("msno", "size"),
            payment_method_nunique=("payment_method_id", "nunique"),
            payment_plan_days_mean=("payment_plan_days", "mean"),
            plan_list_price_mean=("plan_list_price", "mean"),
            actual_amount_paid_mean=("actual_amount_paid", "mean"),
            auto_renew_rate=("is_auto_renew", "mean"),
            cancel_rate=("is_cancel", "mean"),
            first_transaction_date=("transaction_date", "min"),
            last_transaction_date=("transaction_date", "max"),
            latest_membership_expire_date=("membership_expire_date", "max"),
        )
        frames.append(grouped)
    if not frames:
        return pd.DataFrame(columns=["msno"])
    result = pd.concat(frames).groupby(level=0).agg({
        "transaction_count": "sum",
        "payment_method_nunique": "sum",
        "payment_plan_days_mean": "mean",
        "plan_list_price_mean": "mean",
        "actual_amount_paid_mean": "mean",
        "auto_renew_rate": "mean",
        "cancel_rate": "mean",
        "first_transaction_date": "min",
        "last_transaction_date": "max",
        "latest_membership_expire_date": "max",
    })
    return result.reset_index()


def _aggregate_logs(path, chunksize):
    frames = []
    columns = ["msno", "date", "num_25", "num_50", "num_75", "num_985", "num_100", "num_unq", "num_uniq", "total_secs"]
    for chunk in pd.read_csv(path, usecols=lambda c: c in columns, chunksize=chunksize):
        chunk = _date_features(chunk, ["date"])
        if "num_uniq" in chunk and "num_unq" not in chunk:
            chunk = chunk.rename(columns={"num_uniq": "num_unq"})
        numeric = [c for c in columns if c not in {"msno", "date"} and c in chunk]
        for col in numeric:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        grouped = chunk.groupby("msno", sort=False).agg(
            log_row_count=("msno", "size"),
            log_days=("date", "nunique"),
            num_25_sum=("num_25", "sum"),
            num_50_sum=("num_50", "sum"),
            num_75_sum=("num_75", "sum"),
            num_985_sum=("num_985", "sum"),
            num_100_sum=("num_100", "sum"),
            num_unq_sum=("num_unq", "sum"),
            total_secs_sum=("total_secs", "sum"),
            total_secs_mean=("total_secs", "mean"),
            last_log_date=("date", "max"),
        )
        frames.append(grouped)
    if not frames:
        return pd.DataFrame(columns=["msno"])
    result = pd.concat(frames).groupby(level=0).agg({
        "log_row_count": "sum", "log_days": "sum", "num_25_sum": "sum",
        "num_50_sum": "sum", "num_75_sum": "sum", "num_985_sum": "sum",
        "num_100_sum": "sum", "num_unq_sum": "sum", "total_secs_sum": "sum",
        "total_secs_mean": "mean", "last_log_date": "max",
    })
    return result.reset_index()


def build_kkbox_features(data_dir, output_path=None, chunksize=250_000):
    """Build a flat, one-row-per-user KKBox dataset.

    Expected files are train/train_v2, members/members_v3, transactions/
    transactions_v2, and user_logs/user_logs_v2 in the same directory.
    """
    train_path = _find(data_dir, ["train_v2.csv", "train.csv"])
    members_path = _find(data_dir, ["members_v3.csv", "members.csv"])
    transactions_path = _find(data_dir, ["transactions_v2.csv", "transactions.csv"])
    logs_path = _find(data_dir, ["user_logs_v2.csv", "user_logs.csv"])

    labels = pd.read_csv(train_path)
    if "is_churn" not in labels:
        raise ValueError("KKBox train file must contain is_churn")
    labels = labels[["msno", "is_churn"]].rename(columns={"is_churn": "Churn"})
    labels["Churn"] = pd.to_numeric(labels["Churn"], errors="raise").astype(int)

    members = pd.read_csv(members_path).drop_duplicates("msno")
    members = members.drop(columns=[c for c in ["registration_init_time"] if c in members], errors="ignore")
    for col in ["bd", "city", "registered_via"]:
        if col in members:
            members[col] = pd.to_numeric(members[col], errors="coerce")
    if "bd" in members:
        members.loc[~members["bd"].between(1, 100), "bd"] = pd.NA

    result = labels.merge(members, on="msno", how="left")
    result = result.merge(_aggregate_transactions(transactions_path, chunksize), on="msno", how="left")
    result = result.merge(_aggregate_logs(logs_path, chunksize), on="msno", how="left")

    date_columns = [c for c in result if c.endswith("_date")]
    for col in date_columns:
        dates = pd.to_datetime(result[col], errors="coerce")
        result[col + "_ordinal"] = (dates - pd.Timestamp("2017-01-01")).dt.days
    result = result.drop(columns=date_columns, errors="ignore")
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate KKBox event tables into modelling rows")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="data/kkbox_features.csv")
    parser.add_argument("--chunksize", type=int, default=250_000)
    args = parser.parse_args()
    frame = build_kkbox_features(args.data_dir, args.output, args.chunksize)
    print(f"Wrote {len(frame):,} rows and {len(frame.columns):,} columns to {args.output}")
