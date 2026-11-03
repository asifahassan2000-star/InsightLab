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
    MAX_GEMINI_RETRIES,
    sanitize_gemini_model,
    get_gemini_model,
    _is_retryable_error,
    _sanitize_log_message,
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

    def test_sanitize_gemini_model_intercepts_2_5_flash(self):
        """Verify any attempt to use gemini-2.5-flash is intercepted and upgraded to gemini-3.8-flash."""
        self.assertEqual(sanitize_gemini_model("gemini-2.5-flash"), "gemini-3.8-flash")
        self.assertEqual(sanitize_gemini_model("models/gemini-2.5-flash"), "gemini-3.8-flash")
        self.assertEqual(sanitize_gemini_model("gemini-2.5-pro"), "gemini-3.8-flash")
        self.assertEqual(sanitize_gemini_model(""), "gemini-3.8-flash")
        self.assertEqual(sanitize_gemini_model(None), "gemini-3.8-flash")
        self.assertEqual(sanitize_gemini_model("gemini-3.8-flash"), "gemini-3.8-flash")
        self.assertEqual(sanitize_gemini_model("models/gemini-3.8-flash"), "gemini-3.8-flash")

    def test_get_gemini_model_with_env_override(self):
        """Verify environment variables with deprecated models are safely sanitized."""
        with patch.dict("os.environ", {"GEMINI_MODEL": "gemini-2.5-flash"}):
            self.assertEqual(get_gemini_model(), "gemini-3.8-flash")

        with patch.dict("os.environ", {"GEMINI_MODEL": "gemini-3.8-flash"}):
            self.assertEqual(get_gemini_model(), "gemini-3.8-flash")

    def test_query_gemini_assistant_intercepts_explicit_2_5_param(self):
        """Even if model='gemini-2.5-flash' is explicitly passed, it must be upgraded to gemini-3.8-flash."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Safe response"
        mock_client.models.generate_content.return_value = mock_response

        with patch("google.genai.Client", return_value=mock_client):
            ans = query_gemini_assistant(
                prompt="test",
                context="context",
                api_key="key",
                model="gemini-2.5-flash"
            )
            self.assertEqual(ans, "Safe response")
            call_kwargs = mock_client.models.generate_content.call_args.kwargs
            self.assertEqual(call_kwargs.get("model"), "gemini-3.8-flash")

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
        """Verify user-friendly error messages distinguishing model, key, quota, and network errors."""
        # 1. Deprecated / unavailable model error (404)
        with patch("google.genai.Client", side_effect=Exception("404 NOT_FOUND: models/gemini-2.5-flash is no longer available to new users.")):
            ans = query_gemini_assistant("test", "context", "key")
            self.assertIn("Model Error", ans)
            self.assertIn("gemini-3.8-flash", ans)

        # 2. Invalid key error
        with patch("google.genai.Client", side_effect=Exception("API_KEY_INVALID: API key not valid")):
            ans = query_gemini_assistant("test", "context", "bad-key")
            self.assertIn("API Key Error", ans)

        # 3. Quota exceeded error (permanent or exhausted)
        with patch("google.genai.Client", side_effect=Exception("RESOURCE_EXHAUSTED: 429 rate limit exceeded")):
            with patch("utils.ai_assistant._sleep_fn"):
                ans = query_gemini_assistant("test", "context", "key")
                self.assertIn("Quota Exceeded", ans)

        # 4. Service disabled error
        with patch("google.genai.Client", side_effect=Exception("SERVICE_DISABLED: Gemini API has not been used")):
            ans = query_gemini_assistant("test", "context", "key")
            self.assertIn("API Not Enabled", ans)

        # 5. Network / timeout error
        with patch("google.genai.Client", side_effect=Exception("ConnectionError: Failed to establish a connection to host")):
            with patch("utils.ai_assistant._sleep_fn"):
                ans = query_gemini_assistant("test", "context", "key")
                self.assertIn("Network Connection Error", ans)

        # 6. General fallback error
        with patch("google.genai.Client", side_effect=Exception("Unexpected internal server state")):
            ans = query_gemini_assistant("test", "context", "key")
            self.assertIn("Connection Error", ans)

    def test_is_retryable_error_classification(self):
        """Test distinction between temporary and permanent errors."""
        # Temporary
        self.assertTrue(_is_retryable_error(Exception("503 UNAVAILABLE: high demand"))[0])
        self.assertTrue(_is_retryable_error(Exception("429 RESOURCE_EXHAUSTED: quota exceeded"))[0])
        self.assertTrue(_is_retryable_error(Exception("Connection reset by peer"))[0])
        self.assertTrue(_is_retryable_error(Exception("502 Bad Gateway"))[0])

        # Permanent (must not retry)
        self.assertFalse(_is_retryable_error(Exception("API_KEY_INVALID: bad key"))[0])
        self.assertFalse(_is_retryable_error(Exception("404 NOT_FOUND: model not found"))[0])
        self.assertFalse(_is_retryable_error(Exception("403 PERMISSION_DENIED: service disabled"))[0])

    def test_sanitize_log_message_redacts_credentials(self):
        """Ensure API keys are redacted from logs and errors."""
        msg = "Failed with key AIzaSyDg8IglpO26e1zGVaAWt1gpgyBTxug9BTc at endpoint"
        sanitized = _sanitize_log_message(msg)
        self.assertNotIn("AIzaSy", sanitized)
        self.assertIn("[REDACTED_KEY]", sanitized)

    @patch("utils.ai_assistant._sleep_fn")
    def test_query_gemini_assistant_retries_503_max_3_times(self, mock_sleep):
        """Verify 503 errors trigger up to 3 retries with exponential backoff and friendly message."""
        mock_client = MagicMock()
        err_503 = Exception("503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is currently experiencing high demand.'}}")
        mock_client.models.generate_content.side_effect = err_503

        with patch("google.genai.Client", return_value=mock_client):
            ans = query_gemini_assistant("Explain R2", "context", "key")

            # Initial attempt + 3 retries = 4 calls total
            self.assertEqual(mock_client.models.generate_content.call_count, 4)
            # Sleep called 3 times (between attempts)
            self.assertEqual(mock_sleep.call_count, 3)

            # Friendly message returned
            self.assertIn("Gemini Service Temporarily Unavailable (503)", ans)
            self.assertIn("high demand", ans)
            self.assertIn("automatically retried 3 times", ans)

    @patch("utils.ai_assistant._sleep_fn")
    def test_query_gemini_assistant_recovers_on_retry(self, mock_sleep):
        """Verify successful generation after initial 503 failure."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "R-squared measures goodness of fit."

        err_503 = Exception("503 UNAVAILABLE")
        # Fail once with 503, succeed on 2nd attempt
        mock_client.models.generate_content.side_effect = [err_503, mock_response]

        with patch("google.genai.Client", return_value=mock_client):
            ans = query_gemini_assistant("Explain R2", "context", "key")

            self.assertEqual(ans, "R-squared measures goodness of fit.")
            self.assertEqual(mock_client.models.generate_content.call_count, 2)
            self.assertEqual(mock_sleep.call_count, 1)

    @patch("utils.ai_assistant._sleep_fn")
    def test_permanent_errors_do_not_retry(self, mock_sleep):
        """Verify permanent errors (e.g. invalid key) exit immediately without retries."""
        mock_client = MagicMock()
        err_key = Exception("API_KEY_INVALID: The key is bad")
        mock_client.models.generate_content.side_effect = err_key

        with patch("google.genai.Client", return_value=mock_client):
            ans = query_gemini_assistant("test", "context", "key")

            self.assertEqual(mock_client.models.generate_content.call_count, 1)
            self.assertEqual(mock_sleep.call_count, 0)
            self.assertIn("API Key Error", ans)


if __name__ == "__main__":
    unittest.main()
