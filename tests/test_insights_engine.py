"""
Unit Tests for Insights Engine (`utils/insights_engine.py`)
===========================================================
Tests all Phase 5 automated analytical functions:
- Tukey's IQR Outlier detection
- Dataset quality index and grading
- Category extreme performers detection
- Complete automated insight generation and insight cards
- Markdown report compilation
"""

import unittest
import pandas as pd
import numpy as np
from utils.insights_engine import (
    detect_outliers_iqr,
    compute_dataset_quality_score,
    analyze_category_performers,
    generate_automated_insights,
    generate_insights_markdown_report
)


class TestInsightsEngine(unittest.TestCase):

    def setUp(self):
        """Create a synthetic dataset with known statistical anomalies and patterns."""
        self.df = pd.DataFrame({
            "department": ["Engineering", "Engineering", "Marketing", "Marketing", "Sales", "Engineering"],
            "salary": [80000.0, 85000.0, 70000.0, 72000.0, 65000.0, 350000.0],  # 350k is an extreme outlier
            "bonus": [8000.0, 8500.0, 7000.0, 7200.0, 6500.0, 35000.0],         # Strictly correlated with salary
            "experience": [2, 3, 2, 4, 1, 15]
        })

    def test_detect_outliers_iqr(self):
        """Test detection of outliers using 1.5 * IQR."""
        outlier_data = detect_outliers_iqr(self.df["salary"])

        self.assertTrue(outlier_data["has_outliers"])
        self.assertEqual(outlier_data["outlier_count"], 1)
        self.assertIn(350000.0, outlier_data["outlier_values"])
        self.assertLess(outlier_data["upper_bound"], 350000.0)

    def test_detect_outliers_none(self):
        """Test on uniformly spaced numbers with no outliers."""
        uniform_series = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        outlier_data = detect_outliers_iqr(uniform_series)
        self.assertFalse(outlier_data["has_outliers"])
        self.assertEqual(outlier_data["outlier_count"], 0)

    def test_compute_dataset_quality_score_clean(self):
        """Test quality score on a pristine dataset."""
        clean_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        quality = compute_dataset_quality_score(clean_df)

        self.assertEqual(quality["score"], 100)
        self.assertIn("A", quality["grade"])
        self.assertEqual(len(quality["issues"]), 0)

    def test_compute_dataset_quality_score_with_issues(self):
        """Test quality score detects duplicates and missing values."""
        dirty_df = pd.DataFrame({
            "a": [1, 1, None],
            "b": [2, 2, 4]
        })
        quality = compute_dataset_quality_score(dirty_df)

        self.assertLess(quality["score"], 100)
        self.assertEqual(quality["duplicate_count"], 1)
        self.assertEqual(quality["missing_count"], 1)
        self.assertTrue(len(quality["issues"]) >= 2)

    def test_analyze_category_performers(self):
        """Test grouping and finding top category share."""
        performer = analyze_category_performers(self.df, "department", "salary")

        self.assertIsNotNone(performer)
        self.assertEqual(performer["top_category"], "Engineering")
        self.assertGreater(performer["top_share_pct"], 50.0)

    def test_generate_automated_insights(self):
        """Test full automated insight generation pipeline."""
        insights = generate_automated_insights(self.df)

        self.assertNotIn("error", insights)
        self.assertEqual(insights["num_rows"], 6)
        self.assertEqual(insights["num_cols"], 4)
        self.assertTrue(len(insights["cards"]) > 0)

        # Check that outlier card for salary was created
        anomaly_cards = [c for c in insights["cards"] if c["type"] == "anomaly"]
        self.assertTrue(len(anomaly_cards) > 0)

        # Check strong correlation was identified (salary & bonus)
        corr_cards = [c for c in insights["cards"] if c["type"] == "relationship"]
        self.assertTrue(len(corr_cards) > 0)

    def test_generate_insights_markdown_report(self):
        """Test report generation in markdown format."""
        insights = generate_automated_insights(self.df)
        report_md = generate_insights_markdown_report(insights, "employee_test.csv")

        self.assertIsInstance(report_md, str)
        self.assertIn("# 🔬 InsightLab — Automated Insights Report", report_md)
        self.assertIn("Executive Summary", report_md)
        self.assertIn("Smart Recommendations", report_md)


if __name__ == "__main__":
    unittest.main()
