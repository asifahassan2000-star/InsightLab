"""
Unit Tests for Machine Learning Engine (`utils/ml_engine.py`)
=============================================================
Tests all Phase 6 Machine Learning functions:
- Pipeline preprocessing (imputation + scaling + one-hot encoding)
- Regression model training (Linear Regression & Random Forest)
- Classification model training (Logistic Regression & Random Forest)
- Clustering model training (K-Means)
- Single instance prediction for Prediction Studio
- Input validation & edge cases (small data, single-class target)
"""

import unittest
import pandas as pd
import numpy as np
from utils.ml_engine import (
    train_regression_model,
    train_classification_model,
    train_clustering_model,
    predict_single_instance
)


class TestMLEngine(unittest.TestCase):

    def setUp(self):
        """Create standard synthetic datasets for testing ML tasks."""
        np.random.seed(42)
        n = 30
        x1 = np.random.uniform(10, 50, n)
        x2 = np.random.uniform(1, 10, n)
        category = np.random.choice(["TypeA", "TypeB"], n)

        # Continuous target: y = 2*x1 + 3*x2 + noise
        y_reg = 2 * x1 + 3 * x2 + np.random.normal(0, 1, n)

        # Categorical target: based on x1 threshold
        y_clf = np.where(x1 > 30, "High", "Low")

        self.df = pd.DataFrame({
            "feature1": x1,
            "feature2": x2,
            "category": category,
            "target_num": y_reg,
            "target_cat": y_clf
        })

    def test_regression_linear(self):
        """Test training Linear Regression."""
        result = train_regression_model(
            self.df,
            target_col="target_num",
            feature_cols=["feature1", "feature2"],
            algorithm="Linear Regression",
            test_size=0.2
        )
        self.assertIn("pipeline", result)
        self.assertIn("metrics", result)
        self.assertGreater(result["metrics"]["r2"], 0.8)
        self.assertLess(result["metrics"]["mae"], 5.0)

    def test_regression_random_forest(self):
        """Test training Random Forest Regressor with mixed features."""
        result = train_regression_model(
            self.df,
            target_col="target_num",
            feature_cols=["feature1", "feature2", "category"],
            algorithm="Random Forest",
            test_size=0.2
        )
        self.assertIn("metrics", result)
        self.assertFalse(result["feature_importance"].empty)

    def test_classification_logistic(self):
        """Test training Logistic Regression."""
        result = train_classification_model(
            self.df,
            target_col="target_cat",
            feature_cols=["feature1", "feature2"],
            algorithm="Logistic Regression",
            test_size=0.2
        )
        self.assertIn("metrics", result)
        self.assertGreaterEqual(result["metrics"]["accuracy"], 0.7)
        self.assertEqual(len(result["classes"]), 2)
        self.assertEqual(result["confusion_matrix"].shape, (2, 2))

    def test_classification_random_forest(self):
        """Test training Random Forest Classifier."""
        result = train_classification_model(
            self.df,
            target_col="target_cat",
            feature_cols=["feature1", "feature2", "category"],
            algorithm="Random Forest",
            test_size=0.2
        )
        self.assertIn("metrics", result)
        self.assertFalse(result["feature_importance"].empty)

    def test_clustering_kmeans(self):
        """Test K-Means clustering."""
        result = train_clustering_model(
            self.df,
            feature_cols=["feature1", "feature2"],
            n_clusters=2
        )
        self.assertIn("clustered_df", result)
        self.assertIn("Cluster", result["clustered_df"].columns)
        self.assertEqual(len(result["cluster_distribution"]), 2)

    def test_predict_single_instance(self):
        """Test generating prediction on new user input."""
        reg_result = train_regression_model(
            self.df,
            target_col="target_num",
            feature_cols=["feature1", "feature2", "category"],
            algorithm="Linear Regression"
        )
        input_data = {"feature1": 25.0, "feature2": 5.0, "category": "TypeA"}
        pred = predict_single_instance(reg_result, input_data)
        self.assertIn("prediction", pred)
        self.assertIsInstance(float(pred["prediction"]), float)

    def test_validation_insufficient_rows(self):
        """Test that datasets with < 5 rows raise an error."""
        small_df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        with self.assertRaises(ValueError):
            train_regression_model(small_df, target_col="y", feature_cols=["x"])

    def test_validation_single_class_target(self):
        """Test that single-class target raises an error in classification."""
        bad_df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5, 6],
            "y": ["A", "A", "A", "A", "A", "A"]
        })
        with self.assertRaises(ValueError):
            train_classification_model(bad_df, target_col="y", feature_cols=["x"])


if __name__ == "__main__":
    unittest.main()
