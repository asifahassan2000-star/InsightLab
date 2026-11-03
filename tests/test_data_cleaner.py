"""
Unit Tests for Data Cleaner (`utils/data_cleaner.py`)
===================================================
Tests all Phase 2 cleaning functions:
- Health metrics computation
- Duplicate detection & removal
- Missing value dropping
- Mean and Median numerical imputation
- Mode categorical imputation
- CSV and Excel binary exports
- Immutability checks (original DataFrame must never be mutated)
"""

import unittest
import io
import pandas as pd
import numpy as np
from utils.data_cleaner import (
    get_dataset_health,
    remove_duplicates,
    drop_missing_values,
    fill_numeric_missing,
    fill_categorical_missing,
    export_dataframe_to_bytes
)


class TestDataCleaner(unittest.TestCase):

    def setUp(self):
        """Create a standard test dataset with known duplicates and missing values."""
        self.df = pd.DataFrame({
            "id": [1, 2, 2, 3, 4],
            "city": ["New York", "Chicago", "Chicago", None, "New York"],
            "score": [10.0, 20.0, 20.0, np.nan, 30.0],
            "category": ["A", "B", "B", "A", None]
        })

    def test_immutability(self):
        """Ensure that cleaning functions return copies and do NOT modify the original df."""
        df_copy = self.df.copy()
        cleaned, _ = remove_duplicates(self.df)
        pd.testing.assert_frame_equal(self.df, df_copy)
        self.assertIsNot(cleaned, self.df)

    def test_get_dataset_health(self):
        """Test the dataset health report dictionary."""
        health = get_dataset_health(self.df)
        self.assertEqual(health["total_rows"], 5)
        self.assertEqual(health["total_cols"], 4)
        self.assertEqual(health["duplicate_rows"], 1)  # row 2 is identical to row 1
        self.assertEqual(health["total_missing"], 3)   # 1 in city, 1 in score, 1 in category
        self.assertIn("score", health["columns_with_missing"])
        self.assertIn("city", health["columns_with_missing"])
        self.assertIn("category", health["columns_with_missing"])

    def test_remove_duplicates(self):
        """Test removing duplicate rows."""
        cleaned, removed = remove_duplicates(self.df)
        self.assertEqual(removed, 1)
        self.assertEqual(len(cleaned), 4)
        self.assertEqual(cleaned.duplicated().sum(), 0)

    def test_drop_missing_values(self):
        """Test dropping rows with any missing values."""
        cleaned, dropped = drop_missing_values(self.df)
        # Rows 0, 1, 2 have no nulls. Row 3 has null in city & score. Row 4 has null in category.
        # So 2 rows should be dropped (indices 3 and 4).
        self.assertEqual(dropped, 2)
        self.assertEqual(len(cleaned), 3)
        self.assertEqual(cleaned.isnull().sum().sum(), 0)

    def test_drop_missing_values_subset(self):
        """Test dropping missing values on a specific column subset."""
        # Only drop if 'category' is missing (row 4)
        cleaned, dropped = drop_missing_values(self.df, subset=["category"])
        self.assertEqual(dropped, 1)
        self.assertEqual(len(cleaned), 4)

    def test_fill_numeric_missing_mean(self):
        """Test filling numerical missing values with Mean."""
        # Non-null scores are 10, 20, 20, 30 -> Mean = 80 / 4 = 20.0
        cleaned, report = fill_numeric_missing(self.df, strategy="mean", columns=["score"])
        self.assertEqual(report["score"]["fill_value"], 20.0)
        self.assertEqual(report["score"]["rows_filled"], 1)
        self.assertEqual(cleaned["score"].isnull().sum(), 0)
        self.assertEqual(cleaned.loc[3, "score"], 20.0)

    def test_fill_numeric_missing_median(self):
        """Test filling numerical missing values with Median."""
        # Non-null scores are 10, 20, 20, 30 -> Median = 20.0
        cleaned, report = fill_numeric_missing(self.df, strategy="median", columns=["score"])
        self.assertEqual(report["score"]["fill_value"], 20.0)
        self.assertEqual(cleaned["score"].isnull().sum(), 0)

    def test_fill_numeric_invalid_strategy(self):
        """Ensure invalid strategy raises a ValueError."""
        with self.assertRaises(ValueError):
            fill_numeric_missing(self.df, strategy="invalid_strategy")

    def test_fill_categorical_missing_mode(self):
        """Test filling categorical missing values with Mode."""
        # 'city' values are 'New York' (2), 'Chicago' (2), None (1)
        # 'category' values are 'A' (2), 'B' (2), None (1)
        cleaned, report = fill_categorical_missing(self.df, columns=["category"])
        self.assertIn("category", report)
        self.assertEqual(report["category"]["rows_filled"], 1)
        self.assertEqual(cleaned["category"].isnull().sum(), 0)

    def test_export_dataframe_to_csv_bytes(self):
        """Test exporting DataFrame to CSV bytes and reading back."""
        csv_bytes = export_dataframe_to_bytes(self.df, "csv")
        self.assertIsInstance(csv_bytes, bytes)
        df_reconstructed = pd.read_csv(io.BytesIO(csv_bytes))
        self.assertEqual(df_reconstructed.shape, self.df.shape)

    def test_export_dataframe_to_excel_bytes(self):
        """Test exporting DataFrame to Excel bytes and reading back."""
        excel_bytes = export_dataframe_to_bytes(self.df, "excel")
        self.assertIsInstance(excel_bytes, bytes)
        df_reconstructed = pd.read_excel(io.BytesIO(excel_bytes), engine="openpyxl")
        self.assertEqual(df_reconstructed.shape, self.df.shape)


if __name__ == "__main__":
    unittest.main()
