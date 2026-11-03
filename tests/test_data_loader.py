"""
Unit Tests for Data Loader (`utils/data_loader.py`)
===================================================
Testing is an essential skill in software development.
These tests verify that our file loading and validation functions
behave exactly as expected under both normal and error conditions.
"""

import io
import unittest
import pandas as pd
from utils.data_loader import load_dataset, get_column_summary


class MockUploadedFile(io.BytesIO):
    """
    Mocks Streamlit's UploadedFile object.
    Streamlit uploaded files inherit from BytesIO and provide a `.name` attribute.
    """
    def __init__(self, buffer: bytes, name: str):
        super().__init__(buffer)
        self.name = name


class TestDataLoader(unittest.TestCase):

    def test_load_valid_csv(self):
        """Test loading a standard valid CSV file."""
        csv_content = b"name,age,score\nAlice,25,88.5\nBob,30,92.0\nCharlie,35,79.5\n"
        mock_file = MockUploadedFile(csv_content, "students.csv")
        
        df, error = load_dataset(mock_file)
        
        self.assertIsNone(error)
        self.assertIsNotNone(df)
        self.assertEqual(df.shape, (3, 3))
        self.assertListEqual(list(df.columns), ["name", "age", "score"])

    def test_load_valid_excel(self):
        """Test loading a standard valid Excel (.xlsx) file."""
        df_original = pd.DataFrame({
            "product": ["Laptop", "Mouse", "Keyboard"],
            "price": [1200.0, 25.5, 75.0]
        })
        buffer = io.BytesIO()
        df_original.to_excel(buffer, index=False, engine="openpyxl")
        mock_file = MockUploadedFile(buffer.getvalue(), "products.xlsx")

        df, error = load_dataset(mock_file)

        self.assertIsNone(error)
        self.assertIsNotNone(df)
        self.assertEqual(df.shape, (3, 2))
        self.assertListEqual(list(df.columns), ["product", "price"])

    def test_load_empty_file(self):
        """Test that completely empty files return a graceful error."""
        mock_file = MockUploadedFile(b"", "empty.csv")
        df, error = load_dataset(mock_file)

        self.assertIsNone(df)
        self.assertIsNotNone(error)
        self.assertIn("empty", error.lower())

    def test_load_headers_only_csv(self):
        """Test that a CSV with headers but no data rows returns a clear message."""
        mock_file = MockUploadedFile(b"col1,col2,col3\n", "no_rows.csv")
        df, error = load_dataset(mock_file)

        self.assertIsNone(df)
        self.assertIsNotNone(error)
        self.assertIn("0 data rows", error)

    def test_unsupported_file_extension(self):
        """Test that unsupported file formats (like .txt, .pdf) are rejected."""
        mock_file = MockUploadedFile(b"some text", "notes.txt")
        df, error = load_dataset(mock_file)

        self.assertIsNone(df)
        self.assertIsNotNone(error)
        self.assertIn("unsupported file format", error.lower())

    def test_get_column_summary(self):
        """Test the column summary generation for accuracy."""
        test_df = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "city": ["New York", "London", None, "Tokyo"],
            "temperature": [22.5, None, None, 18.2]
        })

        summary = get_column_summary(test_df)

        self.assertEqual(len(summary), 3)
        self.assertListEqual(
            list(summary.columns),
            ["Column Name", "Data Type", "Non-Null Count", "Missing Count", "Missing (%)"]
        )

        # Check 'city' column stats (1 null out of 4 -> 25%)
        city_row = summary[summary["Column Name"] == "city"].iloc[0]
        self.assertEqual(city_row["Non-Null Count"], 3)
        self.assertEqual(city_row["Missing Count"], 1)
        self.assertEqual(city_row["Missing (%)"], "25.0%")

        # Check 'temperature' stats (2 nulls out of 4 -> 50%)
        temp_row = summary[summary["Column Name"] == "temperature"].iloc[0]
        self.assertEqual(temp_row["Non-Null Count"], 2)
        self.assertEqual(temp_row["Missing Count"], 2)
        self.assertEqual(temp_row["Missing (%)"], "50.0%")


if __name__ == "__main__":
    unittest.main()
