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
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from utils.ai_assistant import (
    DEFAULT_GEMINI_MODEL,
    build_dataset_context_summary,
    answer_query_deterministically,
    get_gemini_api_key,
    query_gemini_assistant
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

    def test_default_gemini_model_is_3_8_flash(self):
        """Verify the default model constant is gemini-3.8-flash."""
        self.assertEqual(DEFAULT_GEMINI_MODEL, "gemini-3.8-flash")

    def test_query_gemini_assistant_uses_gemini_3_8_flash(self):
        """Verify query_gemini_assistant calls generate_content with gemini-3.8-flash."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Here is an analysis of salary."
        mock_client.models.generate_content.return_value = mock_response

        with patch("google.genai.Client", return_value=mock_client):
            ans = query_gemini_assistant(
                prompt="What is the average salary?",
                context="Salary mean is 90000",
                api_key="test-api-key"
            )
            self.assertEqual(ans, "Here is an analysis of salary.")
            call_kwargs = mock_client.models.generate_content.call_args.kwargs
            self.assertEqual(call_kwargs.get("model"), "gemini-3.8-flash")
            self.assertIn("Salary mean is 90000", call_kwargs.get("contents"))

    def test_query_gemini_assistant_with_chat_history(self):
        """Verify conversational history is included in the full prompt."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Continuing from earlier..."
        mock_client.models.generate_content.return_value = mock_response

        history = [
            {"role": "user", "content": "What columns exist?"},
            {"role": "assistant", "content": "You have salary and bonus columns."},
            {"role": "user", "content": "Which is higher?"}
        ]

        with patch("google.genai.Client", return_value=mock_client):
            ans = query_gemini_assistant(
                prompt="Which is higher?",
                context="Summary context",
                api_key="test-api-key",
                chat_history=history
            )
            self.assertEqual(ans, "Continuing from earlier...")
            call_kwargs = mock_client.models.generate_content.call_args.kwargs
            contents = call_kwargs.get("contents")
            self.assertIn("What columns exist?", contents)
            self.assertIn("You have salary and bonus columns.", contents)

    def test_query_gemini_assistant_error_handling(self):
        """Verify user-friendly error messages for invalid key, quota, and connection errors."""
        # 1. Invalid key error
        with patch("google.genai.Client", side_effect=Exception("API_KEY_INVALID: API key not valid")):
            ans = query_gemini_assistant("test", "context", "bad-key")
            self.assertIn("API Key Error", ans)

        # 2. Quota exceeded error
        with patch("google.genai.Client", side_effect=Exception("RESOURCE_EXHAUSTED: rate limit exceeded")):
            ans = query_gemini_assistant("test", "context", "key")
            self.assertIn("Quota Exceeded", ans)

        # 3. Service disabled error
        with patch("google.genai.Client", side_effect=Exception("SERVICE_DISABLED: Gemini API has not been used")):
            ans = query_gemini_assistant("test", "context", "key")
            self.assertIn("API Not Enabled", ans)

        # 4. General connection error
        with patch("google.genai.Client", side_effect=Exception("Timeout connecting to server")):
            ans = query_gemini_assistant("test", "context", "key")
            self.assertIn("Connection Error", ans)
            self.assertIn("Timeout connecting to server", ans)


if __name__ == "__main__":
    unittest.main()
