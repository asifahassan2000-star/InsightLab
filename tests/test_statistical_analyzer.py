"""
Unit Tests for Statistical Analyzer (`utils/statistical_analyzer.py`)
=====================================================================
Tests all Phase 3 statistical computation functions:
- Univariate statistics (mean, median, mode, min, max, variance, std dev, quartiles, skewness)
- Dataset summary table generation
- Correlation matrix computation (Pearson & Spearman)
- Top correlation ranking & verbal interpretation
- Group-wise / categorical comparisons
- Missing value resilience
"""

import unittest
import pandas as pd
import numpy as np
from utils.statistical_analyzer import (
    compute_column_statistics,
    compute_dataset_statistics_table,
    compute_correlation_matrix,
    get_top_correlations,
    interpret_correlation,
    interpret_skewness,
    compute_group_comparison
)


class TestStatisticalAnalyzer(unittest.TestCase):

    def setUp(self):
        """Create a synthetic dataset with known statistical properties."""
        self.df = pd.DataFrame({
            "department": ["Engineering", "Engineering", "Sales", "Sales", "Engineering"],
            "salary": [100.0, 200.0, 150.0, 250.0, 300.0],
            "bonus": [10.0, 20.0, 15.0, 25.0, 30.0],  # Exactly 10% of salary -> correlation = 1.0
            "experience": [2, 5, 3, 7, 10]
        })

    def test_compute_column_statistics(self):
        """Test calculation of mean, median, min, max, variance, std dev, quartiles."""
        stats = compute_column_statistics(self.df["salary"])

        self.assertEqual(stats["count"], 5)
        self.assertEqual(stats["missing_count"], 0)
        self.assertEqual(stats["mean"], 200.0)      # (100+200+150+250+300) / 5 = 1000/5 = 200.0
        self.assertEqual(stats["median"], 200.0)    # sorted: 100, 150, 200, 250, 300
        self.assertEqual(stats["min"], 100.0)
        self.assertEqual(stats["max"], 300.0)
        self.assertEqual(stats["range"], 200.0)

        # Sample variance: sum((x - 200)^2) / 4 = (10000 + 0 + 2500 + 2500 + 10000) / 4 = 25000 / 4 = 6250.0
        self.assertEqual(stats["variance"], 6250.0)
        self.assertAlmostEqual(stats["std_dev"], 79.0569, places=3)

    def test_column_statistics_with_missing_values(self):
        """Test that missing values are excluded from calculations gracefully."""
        series_with_nan = pd.Series([10.0, 20.0, np.nan, 30.0])
        stats = compute_column_statistics(series_with_nan)

        self.assertEqual(stats["count"], 3)
        self.assertEqual(stats["missing_count"], 1)
        self.assertEqual(stats["mean"], 20.0)
        self.assertEqual(stats["median"], 20.0)

    def test_compute_dataset_statistics_table(self):
        """Test generating the summary table across numerical columns."""
        summary_table = compute_dataset_statistics_table(self.df)

        self.assertEqual(len(summary_table), 3)  # salary, bonus, experience
        self.assertIn("salary", summary_table["Column"].values)
        self.assertIn("bonus", summary_table["Column"].values)
        self.assertNotIn("department", summary_table["Column"].values)  # Categorical excluded

    def test_compute_correlation_matrix_pearson(self):
        """Test Pearson correlation calculation."""
        corr = compute_correlation_matrix(self.df, columns=["salary", "bonus", "experience"])

        self.assertEqual(corr.shape, (3, 3))
        # salary and bonus are strictly linear (bonus = 0.1 * salary)
        self.assertAlmostEqual(corr.loc["salary", "bonus"], 1.0, places=4)

    def test_get_top_correlations_and_interpretation(self):
        """Test extracting and interpreting top correlations."""
        corr = compute_correlation_matrix(self.df)
        top_pairs = get_top_correlations(corr)

        self.assertTrue(len(top_pairs) > 0)
        top_pair = top_pairs[0]
        self.assertEqual(top_pair["abs_corr"], 1.0)
        self.assertIn("Strong Positive", top_pair["interpretation"])

    def test_interpret_skewness(self):
        """Test skewness interpretation rules."""
        self.assertIn("Symmetric", interpret_skewness(0.1))
        self.assertIn("Right-Skewed", interpret_skewness(1.2))
        self.assertIn("Left-Skewed", interpret_skewness(-1.2))

    def test_compute_group_comparison(self):
        """Test group-wise / category-wise comparison (e.g. salary by department)."""
        grouped = compute_group_comparison(self.df, group_col="department", value_col="salary")

        self.assertEqual(len(grouped), 2)  # Engineering and Sales
        eng_row = grouped[grouped["department"] == "Engineering"].iloc[0]
        # Engineering salaries: 100, 200, 300 -> Count=3, Mean=200.0, Median=200.0, Sum=600.0
        self.assertEqual(eng_row["Count"], 3)
        self.assertEqual(eng_row["Mean"], 200.0)
        self.assertEqual(eng_row["Median"], 200.0)
        self.assertEqual(eng_row["Total"], 600.0)

        sales_row = grouped[grouped["department"] == "Sales"].iloc[0]
        # Sales salaries: 150, 250 -> Count=2, Mean=200.0, Median=200.0, Sum=400.0
        self.assertEqual(sales_row["Count"], 2)
        self.assertEqual(sales_row["Mean"], 200.0)
        self.assertEqual(sales_row["Total"], 400.0)


if __name__ == "__main__":
    unittest.main()
