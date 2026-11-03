"""
Unit Tests for AI Assistant (`utils/ai_assistant.py`)
=====================================================
Tests all Phase 7 AI Assistant functions:
- Privacy-preserving grounded context digest generation
- Deterministic analytical question answering (offline fallback)
- Correlation, missing value, and category leader natural language parsing
- API Key secure discovery
"""

import unittest
import pandas as pd
import numpy as np
from utils.ai_assistant import (
    build_dataset_context_summary,
    answer_query_deterministically,
    get_gemini_api_key
)


class TestAIAssistant(unittest.TestCase):

    def setUp(self):
        """Create sample data with clear categories, missing cells, and correlations."""
        self.df = pd.DataFrame({
            "department": ["Engineering", "Sales", "Engineering", "Sales", "Marketing"],
            "salary": [100000.0, 70000.0, 110000.0, 75000.0, 80000.0],
            "bonus": [10000.0, 7000.0, 11000.0, 7500.0, 8000.0],
            "experience": [5, 2, 6, 3, None]  # 1 missing value
        })

    def test_build_dataset_context_summary(self):
        """Test compiling grounded statistical context."""
        context = build_dataset_context_summary(self.df, "test_dataset.csv")

        self.assertIn("DATASET PROFILE: test_dataset.csv", context)
        self.assertIn("5 rows by 4 columns", context)
        self.assertIn("salary", context)
        self.assertIn("Cleanliness Score", context)
        self.assertIn("NUMERICAL STATISTICS", context)

    def test_answer_missing_values_query(self):
        """Test natural language answer for missing values."""
        ans = answer_query_deterministically("Are there any missing values?", self.df, "test.csv")
        self.assertIn("experience", ans)
        self.assertIn("1 missing", ans.lower())

    def test_answer_correlation_query(self):
        """Test natural language answer for correlations."""
        ans = answer_query_deterministically("Which columns are strongly correlated?", self.df, "test.csv")
        self.assertIn("salary", ans)
        self.assertIn("bonus", ans)
        self.assertIn("Correlation describes association, not causality", ans)

    def test_answer_highest_category_query(self):
        """Test natural language answer for highest average department."""
        ans = answer_query_deterministically("Which department has the highest average salary?", self.df, "test.csv")
        self.assertIn("Engineering", ans)
        self.assertIn("Highest Average", ans)

    def test_get_gemini_api_key_safe(self):
        """Ensure get_gemini_api_key does not crash if no key is set."""
        key = get_gemini_api_key()
        self.assertTrue(key is None or isinstance(key, str))


if __name__ == "__main__":
    unittest.main()
