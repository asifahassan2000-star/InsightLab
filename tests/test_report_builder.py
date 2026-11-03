"""
test_report_builder.py
======================
Unit Tests for InsightLab Phase 8 — Report Builder Engine.

What We're Testing:
-------------------
1. collect_report_data() — correctly extracts session state data.
2. build_pdf_report()    — produces non-empty valid PDF bytes.
3. build_excel_workbook()— produces non-empty XLSX bytes with correct sheets.
4. export_df_to_csv_bytes() — exports a DataFrame to valid CSV bytes.
5. export_insights_to_markdown_bytes() — produces markdown bytes.
6. Graceful handling when no dataset, no ML model, or no insights are present.

Testing Strategy:
-----------------
We build a minimal fake "session_state" dict for each test instead of
starting a real Streamlit server. This is called "unit testing" — isolating
one function at a time to verify its behaviour independently.
"""

import io
import pytest
import pandas as pd
import numpy as np


# =============================================================================
# FIXTURES — Reusable test data
# =============================================================================

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """A small representative dataset for testing."""
    return pd.DataFrame({
        "category": ["A", "B", "A", "C", "B", "A"],
        "sales": [100.0, 200.0, 150.0, 80.0, np.nan, 120.0],
        "profit": [20.0, 45.0, 30.0, 15.0, 40.0, 25.0],
        "quantity": [5, 10, 7, 4, 8, 6],
    })


@pytest.fixture
def empty_session_state() -> dict:
    """Session state with no data uploaded."""
    return {}


@pytest.fixture
def minimal_session_state(sample_df) -> dict:
    """Session state with only raw data uploaded."""
    return {
        "raw_df": sample_df,
        "dataset_name": "test_data.csv",
    }


@pytest.fixture
def full_session_state(sample_df) -> dict:
    """Session state with all analysis results present."""
    cleaned = sample_df.dropna().copy()
    return {
        "raw_df": sample_df,
        "cleaned_df": cleaned,
        "dataset_name": "sales_data.csv",
        "cleaning_history": [
            "Removed 1 duplicate row(s).",
            "Filled missing values in 'sales' with column mean (130.0).",
        ],
        "ml_model_bundle": {
            "algorithm": "Random Forest Regressor",
            "target_col": "sales",
            "feature_cols": ["profit", "quantity"],
            "metrics": {"MAE": 12.5, "RMSE": 18.3, "R²": 0.87},
            "feature_importance": {"profit": 0.65, "quantity": 0.35},
            "test_results": pd.DataFrame({
                "profit": [20.0, 45.0],
                "quantity": [5, 10],
                "Actual_sales": [100.0, 200.0],
                "Predicted_sales": [108.0, 192.0],
            }),
        },
        "ml_task": "regression",
        "chat_history": [
            {"role": "user", "content": "What are the main findings?"},
            {"role": "assistant", "content": "The dataset has 6 rows and 4 columns."},
        ],
    }


# =============================================================================
# TESTS — collect_report_data
# =============================================================================

class TestCollectReportData:
    """Tests for the data collection function."""

    def test_empty_session_state_returns_no_dataset(self, empty_session_state):
        """With no session state, has_dataset should be False."""
        from utils.report_builder import collect_report_data
        result = collect_report_data(empty_session_state)
        assert result["has_dataset"] is False
        assert result["raw_df"] is None
        assert result["active_df"] is None

    def test_minimal_state_returns_dataset(self, minimal_session_state, sample_df):
        """With raw_df present, has_dataset should be True."""
        from utils.report_builder import collect_report_data
        result = collect_report_data(minimal_session_state)
        assert result["has_dataset"] is True
        assert result["raw_df"] is not None
        assert result["dataset_name"] == "test_data.csv"

    def test_full_state_extracts_ml_bundle(self, full_session_state):
        """ML bundle and task should be correctly extracted."""
        from utils.report_builder import collect_report_data
        result = collect_report_data(full_session_state)
        assert result["ml_bundle"] is not None
        assert result["ml_task"] == "regression"
        assert result["ml_bundle"]["algorithm"] == "Random Forest Regressor"

    def test_full_state_extracts_cleaning_history(self, full_session_state):
        """Cleaning history should be a list of strings."""
        from utils.report_builder import collect_report_data
        result = collect_report_data(full_session_state)
        assert isinstance(result["cleaning_history"], list)
        assert len(result["cleaning_history"]) == 2

    def test_prefers_cleaned_df_as_active(self, full_session_state, sample_df):
        """active_df should be cleaned_df when it exists and is non-empty."""
        from utils.report_builder import collect_report_data
        result = collect_report_data(full_session_state)
        # cleaned_df has fewer rows (dropna removes 1 row with NaN sales)
        assert len(result["active_df"]) == len(sample_df.dropna())

    def test_generated_at_is_string(self, minimal_session_state):
        """generated_at should be a non-empty string."""
        from utils.report_builder import collect_report_data
        result = collect_report_data(minimal_session_state)
        assert isinstance(result["generated_at"], str)
        assert len(result["generated_at"]) > 0

    def test_quality_score_computed(self, minimal_session_state):
        """quality_score should be a dict with 'score' key when dataset present."""
        from utils.report_builder import collect_report_data
        result = collect_report_data(minimal_session_state)
        assert result["quality_score"] is not None
        assert "score" in result["quality_score"]
        assert 0 <= result["quality_score"]["score"] <= 100

    def test_stats_table_computed(self, minimal_session_state):
        """stats_table should be a DataFrame with numerical column stats."""
        from utils.report_builder import collect_report_data
        result = collect_report_data(minimal_session_state)
        assert result["stats_table"] is not None
        assert isinstance(result["stats_table"], pd.DataFrame)
        assert not result["stats_table"].empty


# =============================================================================
# TESTS — build_pdf_report
# =============================================================================

class TestBuildPdfReport:
    """Tests for the PDF generation function."""

    def test_returns_bytes(self, minimal_session_state):
        """build_pdf_report must return bytes."""
        from utils.report_builder import collect_report_data, build_pdf_report
        report_data = collect_report_data(minimal_session_state)
        config = {
            "title": "Test Report",
            "intro_text": "",
            "include_overview": True,
            "include_statistics": True,
            "include_cleaning": True,
            "include_insights": True,
            "include_ml": True,
            "include_ai_summary": False,
            "ai_summary_text": "",
        }
        result = build_pdf_report(report_data, config)
        assert isinstance(result, bytes)

    def test_pdf_starts_with_magic_bytes(self, minimal_session_state):
        """Valid PDFs always start with '%PDF'."""
        from utils.report_builder import collect_report_data, build_pdf_report
        report_data = collect_report_data(minimal_session_state)
        config = {
            "title": "Test PDF",
            "intro_text": "",
            "include_overview": True,
            "include_statistics": False,
            "include_cleaning": False,
            "include_insights": False,
            "include_ml": False,
            "include_ai_summary": False,
            "ai_summary_text": "",
        }
        result = build_pdf_report(report_data, config)
        assert result[:4] == b"%PDF", "PDF must start with '%PDF' magic bytes"

    def test_pdf_with_no_sections_still_generates(self, minimal_session_state):
        """Even with all sections unchecked, a cover page PDF should be produced."""
        from utils.report_builder import collect_report_data, build_pdf_report
        report_data = collect_report_data(minimal_session_state)
        config = {
            "title": "Empty Report",
            "intro_text": "This report intentionally has no sections.",
            "include_overview": False,
            "include_statistics": False,
            "include_cleaning": False,
            "include_insights": False,
            "include_ml": False,
            "include_ai_summary": False,
            "ai_summary_text": "",
        }
        result = build_pdf_report(report_data, config)
        assert isinstance(result, bytes)
        assert len(result) > 100  # Must produce something meaningful

    def test_pdf_with_full_state(self, full_session_state):
        """PDF should build without error when all data is present."""
        from utils.report_builder import collect_report_data, build_pdf_report
        report_data = collect_report_data(full_session_state)
        config = {
            "title": "Full InsightLab Report",
            "intro_text": "Comprehensive analysis.",
            "include_overview": True,
            "include_statistics": True,
            "include_cleaning": True,
            "include_insights": True,
            "include_ml": True,
            "include_ai_summary": True,
            "ai_summary_text": "This is a test AI summary paragraph.",
        }
        result = build_pdf_report(report_data, config)
        assert result[:4] == b"%PDF"

    def test_pdf_with_empty_session_state_still_generates(self, empty_session_state):
        """Even with no dataset, generating a report should not crash."""
        from utils.report_builder import collect_report_data, build_pdf_report
        report_data = collect_report_data(empty_session_state)
        config = {
            "title": "Empty State Report",
            "intro_text": "",
            "include_overview": True,
            "include_statistics": True,
            "include_cleaning": True,
            "include_insights": True,
            "include_ml": True,
            "include_ai_summary": False,
            "ai_summary_text": "",
        }
        result = build_pdf_report(report_data, config)
        assert isinstance(result, bytes)


# =============================================================================
# TESTS — build_excel_workbook
# =============================================================================

class TestBuildExcelWorkbook:
    """Tests for the Excel workbook generation function."""

    def test_returns_bytes(self, minimal_session_state):
        """build_excel_workbook must return bytes."""
        from utils.report_builder import collect_report_data, build_excel_workbook
        report_data = collect_report_data(minimal_session_state)
        result = build_excel_workbook(report_data)
        assert isinstance(result, bytes)

    def test_excel_magic_bytes(self, minimal_session_state):
        """
        Valid XLSX files are ZIP archives. They start with PK (0x50 0x4B).
        XLSX is actually a ZIP file containing XML documents internally.
        """
        from utils.report_builder import collect_report_data, build_excel_workbook
        report_data = collect_report_data(minimal_session_state)
        result = build_excel_workbook(report_data)
        assert result[:2] == b"PK", "XLSX must start with PK (ZIP magic bytes)"

    def test_excel_has_correct_sheets(self, minimal_session_state):
        """The workbook must contain all 6 expected sheet names."""
        from utils.report_builder import collect_report_data, build_excel_workbook
        import openpyxl
        report_data = collect_report_data(minimal_session_state)
        result = build_excel_workbook(report_data)

        # Load the bytes as an openpyxl workbook for inspection
        wb = openpyxl.load_workbook(io.BytesIO(result))
        expected_sheets = {
            "Dataset Overview",
            "Raw Data",
            "Cleaned Data",
            "Statistical Summary",
            "Insights Summary",
            "ML Results",
        }
        actual_sheets = set(wb.sheetnames)
        assert expected_sheets == actual_sheets, (
            f"Expected sheets: {expected_sheets}\nActual sheets: {actual_sheets}"
        )

    def test_excel_raw_data_sheet_contains_data(self, minimal_session_state, sample_df):
        """Raw Data sheet should contain the actual dataset rows."""
        from utils.report_builder import collect_report_data, build_excel_workbook
        import openpyxl
        report_data = collect_report_data(minimal_session_state)
        result = build_excel_workbook(report_data)

        wb = openpyxl.load_workbook(io.BytesIO(result))
        ws = wb["Raw Data"]
        # Row 1 = headers; rows 2+ = data — so max_row should be len(df) + 1
        assert ws.max_row == len(sample_df) + 1

    def test_excel_with_full_state(self, full_session_state):
        """Full session state should produce a valid multi-sheet workbook."""
        from utils.report_builder import collect_report_data, build_excel_workbook
        report_data = collect_report_data(full_session_state)
        result = build_excel_workbook(report_data)
        assert result[:2] == b"PK"

    def test_excel_with_empty_state(self, empty_session_state):
        """Empty session state should still produce a valid workbook."""
        from utils.report_builder import collect_report_data, build_excel_workbook
        report_data = collect_report_data(empty_session_state)
        result = build_excel_workbook(report_data)
        assert isinstance(result, bytes)
        assert result[:2] == b"PK"


# =============================================================================
# TESTS — CSV and Markdown export helpers
# =============================================================================

class TestExportHelpers:
    """Tests for individual file export helpers."""

    def test_csv_export_returns_bytes(self, sample_df):
        """export_df_to_csv_bytes must return bytes."""
        from utils.report_builder import export_df_to_csv_bytes
        result = export_df_to_csv_bytes(sample_df)
        assert isinstance(result, bytes)

    def test_csv_export_contains_column_names(self, sample_df):
        """CSV bytes should contain the column headers as text."""
        from utils.report_builder import export_df_to_csv_bytes
        result = export_df_to_csv_bytes(sample_df)
        text = result.decode("utf-8-sig")
        for col in sample_df.columns:
            assert col in text, f"Column '{col}' not found in CSV output"

    def test_csv_export_row_count(self, sample_df):
        """CSV should have 1 header row + len(df) data rows."""
        from utils.report_builder import export_df_to_csv_bytes
        result = export_df_to_csv_bytes(sample_df)
        text = result.decode("utf-8-sig")
        lines = [l for l in text.strip().split("\n") if l.strip()]
        assert len(lines) == len(sample_df) + 1  # +1 for header

    def test_markdown_export_returns_bytes(self, minimal_session_state):
        """export_insights_to_markdown_bytes must return bytes."""
        from utils.report_builder import collect_report_data, export_insights_to_markdown_bytes
        report_data = collect_report_data(minimal_session_state)
        result = export_insights_to_markdown_bytes(report_data)
        assert isinstance(result, bytes)

    def test_markdown_export_empty_state(self, empty_session_state):
        """Empty state should return a fallback markdown string, not crash."""
        from utils.report_builder import collect_report_data, export_insights_to_markdown_bytes
        report_data = collect_report_data(empty_session_state)
        result = export_insights_to_markdown_bytes(report_data)
        assert isinstance(result, bytes)
        assert len(result) > 0


# =============================================================================
# TESTS — AI Executive Summary (local fallback only — no real API call)
# =============================================================================

class TestAIExecutiveSummary:
    """Tests for the AI Executive Summary function using local fallback."""

    def test_no_dataset_returns_message(self):
        """With no dataset, returns a descriptive message instead of crashing."""
        from utils.report_builder import get_ai_executive_summary
        result = get_ai_executive_summary(None, "test.csv", api_key=None)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_with_dataset_no_key_returns_string(self, sample_df):
        """With dataset but no API key, uses deterministic local fallback."""
        from utils.report_builder import get_ai_executive_summary
        result = get_ai_executive_summary(sample_df, "sales_data.csv", api_key=None)
        assert isinstance(result, str)
        assert len(result) > 50  # Should produce a meaningful response

    def test_fallback_contains_dataset_name(self, sample_df):
        """The fallback report should mention the dataset name."""
        from utils.report_builder import get_ai_executive_summary
        result = get_ai_executive_summary(sample_df, "my_report.csv", api_key=None)
        assert "my_report" in result or "InsightLab" in result
