"""
data_loader.py
==============
This module handles all data loading and validation tasks for InsightLab.

Why separate data loading into its own file?
-------------------------------------------
In software development, this follows the 'Single Responsibility Principle' (SRP).
By keeping the data loading logic separate from the user interface (Streamlit):
1. The code is easier to read, test, and maintain.
2. If we ever want to switch from Streamlit to a web framework like FastAPI or Flask,
   we can reuse this exact same data loader without rewriting it!
"""

from typing import Tuple, Optional
import pandas as pd


def load_dataset(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Reads an uploaded file (CSV or Excel) and converts it into a Pandas DataFrame.

    Parameters:
    -----------
    uploaded_file : UploadedFile
        The file object provided by Streamlit's `st.file_uploader`.

    Returns:
    --------
    Tuple[Optional[pd.DataFrame], Optional[str]]:
        A tuple containing:
        - The loaded Pandas DataFrame (or None if an error occurred).
        - An error message string (or None if loading succeeded).

    How it works:
    -------------
    1. Inspects the file name extension (.csv, .xlsx, or .xls).
    2. Calls the appropriate Pandas reader function (`pd.read_csv` or `pd.read_excel`).
    3. Validates whether the dataset is empty.
    4. Catches file-reading errors gracefully so the app does not crash.
    """
    if uploaded_file is None:
        return None, "No file was uploaded."

    filename = uploaded_file.name.lower()

    try:
        # Determine file type based on extension
        if filename.endswith(".csv"):
            # pd.read_csv reads comma-separated values into a DataFrame.
            # We also handle potential encoding issues (e.g., UTF-8 vs Latin-1).
            try:
                df = pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                # If UTF-8 fails, reset file pointer and try 'latin1' encoding
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding="latin1")

        elif filename.endswith((".xlsx", ".xls")):
            # pd.read_excel requires the 'openpyxl' engine for modern Excel files
            df = pd.read_excel(uploaded_file)
        else:
            return None, f"Unsupported file format: '{uploaded_file.name}'. Please upload a .csv, .xlsx, or .xls file."

        # Check if the dataset has no columns and no rows
        if len(df.columns) == 0:
            return None, "The uploaded dataset has no columns or is empty."

        # Check if there are columns but 0 rows
        if len(df) == 0:
            return None, "The uploaded dataset has headers but contains 0 data rows."

        # Check if the dataframe is empty
        if df.empty:
            return None, "The uploaded file is empty (contains no data)."

        return df, None

    except pd.errors.EmptyDataError:
        return None, "The uploaded file is empty and contains no readable data."
    except pd.errors.ParserError as e:
        return None, f"Parsing error: The file is malformed or corrupted. Details: {str(e)}"
    except Exception as e:
        return None, f"An unexpected error occurred while reading the file: {str(e)}"


def get_column_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a summary table describing every column in the DataFrame.

    Why this is helpful:
    --------------------
    Before doing any data analysis or modeling, a data scientist needs to know:
    - What type of data is in each column (numbers, text, dates)?
    - Are there missing values (null / NaN)?
    - What percentage of the data is missing?

    Returns:
    --------
    pd.DataFrame:
        A new DataFrame with columns:
        ['Column Name', 'Data Type', 'Non-Null Count', 'Missing Count', 'Missing (%)']
    """
    total_rows = len(df)

    summary_data = []
    for col in df.columns:
        non_null = int(df[col].notnull().sum())
        missing = int(df[col].isnull().sum())
        missing_pct = round((missing / total_rows) * 100, 2) if total_rows > 0 else 0.0
        dtype_str = str(df[col].dtype)

        summary_data.append({
            "Column Name": col,
            "Data Type": dtype_str,
            "Non-Null Count": non_null,
            "Missing Count": missing,
            "Missing (%)": f"{missing_pct}%"
        })

    return pd.DataFrame(summary_data)
