"""
data_cleaner.py
===============
This module contains pure data-cleaning functions for InsightLab (Phase 2).

Core Design Principles:
-----------------------
1. Immutability / Non-Destructive:
   Functions never modify the input DataFrame directly in-place.
   Instead, they always create and return a modified copy (`df.copy()`).
   This ensures the user's original data remains completely untouched.

2. Beginner-Friendly Explanations:
   - Duplicate Rows: Rows where every column matches another row in the dataset.
   - Missing Values (NaN / None): Cells with no recorded data.
   - Imputation: Replacing missing values with a calculated statistic (mean, median, mode).
"""

from typing import Tuple, Dict, Any, List, Optional
import io
import pandas as pd
import numpy as np


def get_dataset_health(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes a comprehensive health report for a DataFrame.

    Returns:
    --------
    Dict[str, Any]:
        - 'total_rows': Total number of rows.
        - 'total_cols': Total number of columns.
        - 'duplicate_rows': Count of identical duplicate rows.
        - 'total_missing': Total missing (null) cells across the entire dataset.
        - 'missing_percentage': Overall percentage of null cells.
        - 'columns_with_missing': List of column names that have at least one null value.
        - 'numeric_cols': List of column names with numeric types.
        - 'categorical_cols': List of column names with text/categorical types.
    """
    if df is None or df.empty:
        return {
            "total_rows": 0,
            "total_cols": 0,
            "duplicate_rows": 0,
            "total_missing": 0,
            "missing_percentage": 0.0,
            "columns_with_missing": [],
            "numeric_cols": [],
            "categorical_cols": []
        }

    total_rows = len(df)
    total_cols = len(df.columns)
    total_cells = total_rows * total_cols
    total_missing = int(df.isnull().sum().sum())
    missing_pct = round((total_missing / total_cells * 100), 2) if total_cells > 0 else 0.0

    duplicate_rows = int(df.duplicated().sum())

    # Find columns with at least one missing value
    null_counts = df.isnull().sum()
    cols_with_missing = null_counts[null_counts > 0].index.tolist()

    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    return {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "duplicate_rows": duplicate_rows,
        "total_missing": total_missing,
        "missing_percentage": missing_pct,
        "columns_with_missing": cols_with_missing,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols
    }


def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> Tuple[pd.DataFrame, int]:
    """
    Removes duplicate rows from the dataset.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame.
    subset : Optional[List[str]]
        Specific column(s) to check for duplicates. If None, checks across all columns.

    Returns:
    --------
    Tuple[pd.DataFrame, int]:
        - A new DataFrame without duplicate rows.
        - The count of rows removed.

    How it works:
    -------------
    Pandas `df.duplicated()` returns a boolean Series (True for duplicate rows).
    `df.drop_duplicates()` removes them, keeping the first occurrence.
    """
    if df is None or df.empty:
        return df, 0

    cleaned_df = df.copy()
    duplicates_count = int(cleaned_df.duplicated(subset=subset).sum())

    if duplicates_count > 0:
        cleaned_df = cleaned_df.drop_duplicates(subset=subset, keep="first")

    return cleaned_df, duplicates_count


def drop_missing_values(df: pd.DataFrame, subset: Optional[List[str]] = None) -> Tuple[pd.DataFrame, int]:
    """
    Drops rows that contain missing values (NaN / null).

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame.
    subset : Optional[List[str]]
        List of columns to look for missing values in. If None, checks all columns.

    Returns:
    --------
    Tuple[pd.DataFrame, int]:
        - Cleaned DataFrame with missing rows removed.
        - Number of rows dropped.
    """
    if df is None or df.empty:
        return df, 0

    cleaned_df = df.copy()
    initial_rows = len(cleaned_df)
    cleaned_df = cleaned_df.dropna(subset=subset)
    rows_dropped = initial_rows - len(cleaned_df)

    return cleaned_df, rows_dropped


def fill_numeric_missing(
    df: pd.DataFrame,
    strategy: str = "mean",
    columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Imputes (fills) missing values in numerical columns using Mean or Median.

    What is Mean vs Median?
    -----------------------
    - Mean (Average): Sum of all values divided by the count. Best for normal data.
      Formula: sum(x) / N
    - Median (Middle Value): The value separating the higher half from the lower half.
      Best for skewed data with extreme outliers (e.g., salaries or house prices).

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame.
    strategy : str
        Either 'mean' or 'median'.
    columns : Optional[List[str]]
        Target numeric columns to fill. If None, fills all numeric columns with missing values.

    Returns:
    --------
    Tuple[pd.DataFrame, Dict[str, Any]]:
        - Cleaned DataFrame with filled values.
        - Summary dict detailing the filled columns, strategies, and fill values.
    """
    if df is None or df.empty:
        return df, {}

    if strategy.lower() not in ["mean", "median"]:
        raise ValueError(f"Invalid strategy '{strategy}'. Must be 'mean' or 'median'.")

    cleaned_df = df.copy()
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()

    if columns is not None:
        target_cols = [c for c in columns if c in numeric_cols]
    else:
        target_cols = [c for c in numeric_cols if cleaned_df[c].isnull().any()]

    report = {}

    for col in target_cols:
        missing_count = int(cleaned_df[col].isnull().sum())
        if missing_count == 0:
            continue

        if strategy.lower() == "mean":
            fill_val = cleaned_df[col].mean()
        else:
            fill_val = cleaned_df[col].median()

        # Handle the rare case where an entire column is NaN
        if pd.isna(fill_val):
            continue

        cleaned_df[col] = cleaned_df[col].fillna(fill_val)
        report[col] = {
            "strategy": strategy.lower(),
            "fill_value": round(float(fill_val), 4),
            "rows_filled": missing_count
        }

    return cleaned_df, report


def fill_categorical_missing(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Imputes (fills) missing values in categorical / text columns using the Mode.

    What is the Mode?
    -----------------
    The Mode is the most frequently occurring value in a column.
    For example, if 'Country' contains ['USA', 'Canada', 'USA', None], the mode is 'USA'.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame.
    columns : Optional[List[str]]
        Target categorical columns to fill. If None, fills all categorical columns with missing values.

    Returns:
    --------
    Tuple[pd.DataFrame, Dict[str, Any]]:
        - Cleaned DataFrame.
        - Summary dict detailing columns, mode values, and rows filled.
    """
    if df is None or df.empty:
        return df, {}

    cleaned_df = df.copy()
    cat_cols = cleaned_df.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    if columns is not None:
        target_cols = [c for c in columns if c in cat_cols]
    else:
        target_cols = [c for c in cat_cols if cleaned_df[c].isnull().any()]

    report = {}

    for col in target_cols:
        missing_count = int(cleaned_df[col].isnull().sum())
        if missing_count == 0:
            continue

        # pd.Series.mode() returns a Series of the most frequent values
        mode_series = cleaned_df[col].mode(dropna=True)

        if not mode_series.empty:
            mode_val = mode_series.iloc[0]
            cleaned_df[col] = cleaned_df[col].fillna(mode_val)
            report[col] = {
                "strategy": "mode",
                "fill_value": mode_val,
                "rows_filled": missing_count
            }

    return cleaned_df, report


def export_dataframe_to_bytes(df: pd.DataFrame, file_format: str = "csv") -> bytes:
    """
    Converts a DataFrame into downloadable bytes for CSV or Excel formats.

    Why in-memory buffer (`io.BytesIO`)?
    ------------------------------------
    Rather than writing temporary files to the server's hard drive,
    in-memory buffers keep the export in RAM. This is faster and avoids cluttering disk storage.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to export.
    file_format : str
        Either 'csv' or 'excel'.

    Returns:
    --------
    bytes:
        The binary content ready to be served via `st.download_button`.
    """
    if file_format.lower() == "csv":
        # encode to utf-8 bytes
        return df.to_csv(index=False).encode("utf-8")

    elif file_format.lower() in ["excel", "xlsx"]:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Cleaned Data")
        return buffer.getvalue()

    else:
        raise ValueError(f"Unsupported format '{file_format}'. Choose 'csv' or 'excel'.")
