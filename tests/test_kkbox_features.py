import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.kkbox_features import build_kkbox_features
from src.preprocessing import load_data, split_xy


class KKBoxFeatureTest(unittest.TestCase):
    def test_event_tables_become_one_row_per_subscriber(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({"msno": ["a", "b"], "is_churn": [1, 0]}).to_csv(root / "train_v2.csv", index=False)
            pd.DataFrame({
                "msno": ["a", "b"], "city": [1, 2], "bd": [30, 40],
                "gender": ["male", "female"], "registered_via": [7, 9],
            }).to_csv(root / "members_v3.csv", index=False)
            pd.DataFrame({
                "msno": ["a", "a", "b"], "payment_method_id": [1, 1, 2],
                "payment_plan_days": [30, 30, 30], "plan_list_price": [100, 100, 80],
                "actual_amount_paid": [100, 100, 80], "is_auto_renew": [1, 1, 0],
                "transaction_date": [20170101, 20170201, 20170101],
                "membership_expire_date": [20170131, 20170303, 20170131],
                "is_cancel": [0, 0, 1],
            }).to_csv(root / "transactions_v2.csv", index=False)
            pd.DataFrame({
                "msno": ["a", "a", "b"], "date": [20170201, 20170202, 20170201],
                "num_25": [1, 2, 0], "num_50": [0, 1, 0], "num_75": [0, 0, 1],
                "num_985": [0, 0, 0], "num_100": [5, 6, 2], "num_unq": [4, 5, 2],
                "total_secs": [100.0, 200.0, 50.0],
            }).to_csv(root / "user_logs_v2.csv", index=False)

            output = root / "features.csv"
            result = build_kkbox_features(root, output, chunksize=1)
            self.assertEqual(len(result), 2)
            self.assertEqual(result["msno"].nunique(), 2)
            self.assertIn("transaction_count", result)
            self.assertIn("total_secs_sum", result)

            loaded = load_data(output)
            _, y, _ = split_xy(loaded)
            self.assertEqual(y.tolist(), [1, 0])


if __name__ == "__main__":
    unittest.main()
