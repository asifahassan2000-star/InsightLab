"""
statistical_analyzer.py
=======================
This module provides pure statistical analysis computations for InsightLab (Phase 3).

Why a dedicated statistical module?
-----------------------------------
Keeps statistical math completely separate from Streamlit UI code.
Uses Pandas and NumPy for vectorized, fast, and numerically stable computations.

Statistical Concepts Covered:
-----------------------------
1. Measures of Central Tendency: Mean, Median, Mode.
2. Measures of Dispersion / Spread: Minimum, Maximum, Range, Variance, Standard Deviation, IQR.
3. Shape of Distribution: Skewness and Kurtosis.
4. Bivariate Association: Pearson & Spearman Correlation.
5. Group-Level Aggregations: Comparing metric distributions across categories or departments.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


def compute_column_statistics(series: pd.Series) -> Dict[str, Any]:
    """
    Computes univariate summary statistics for a single numerical column.

    Parameters:
    -----------
    series : pd.Series
        A pandas Series containing numerical values.

    Returns:
    --------
    Dict[str, Any]:
        Dictionary with count, mean, median, mode, min, max, variance, std dev,
        quartiles, skewness, and kurtosis.
    """
    clean_series = series.dropna()
    total_count = len(series)
    valid_count = len(clean_series)
    missing_count = total_count - valid_count

    if valid_count == 0:
        return {
            "count": 0,
            "missing_count": missing_count,
            "mean": None,
            "median": None,
            "mode": None,
            "min": None,
            "max": None,
            "range": None,
            "variance": None,
            "std_dev": None,
            "q25": None,
            "q75": None,
            "iqr": None,
            "skewness": None,
            "kurtosis": None
        }

    # Central Tendency
    mean_val = float(clean_series.mean())
    median_val = float(clean_series.median())
    
    # Mode can have multiple values or none
    mode_series = clean_series.mode()
    mode_val = float(mode_series.iloc[0]) if not mode_series.empty else None

    # Dispersion / Spread
    min_val = float(clean_series.min())
    max_val = float(clean_series.max())
    range_val = max_val - min_val

    # Sample variance (ddof=1) and Sample Standard Deviation
    var_val = float(clean_series.var(ddof=1)) if valid_count > 1 else 0.0
    std_val = float(clean_series.std(ddof=1)) if valid_count > 1 else 0.0

    # Percentiles / Quartiles
    q25 = float(clean_series.quantile(0.25))
    q75 = float(clean_series.quantile(0.75))
    iqr_val = q75 - q25

    # Shape
    skew_val = float(clean_series.skew()) if valid_count > 2 else 0.0
    kurt_val = float(clean_series.kurt()) if valid_count > 3 else 0.0

    return {
        "count": valid_count,
        "missing_count": missing_count,
        "mean": round(mean_val, 4),
        "median": round(median_val, 4),
        "mode": round(mode_val, 4) if mode_val is not None else None,
        "min": round(min_val, 4),
        "max": round(max_val, 4),
        "range": round(range_val, 4),
        "variance": round(var_val, 4),
        "std_dev": round(std_val, 4),
        "q25": round(q25, 4),
        "q75": round(q75, 4),
        "iqr": round(iqr_val, 4),
        "skewness": round(skew_val, 4),
        "kurtosis": round(kurt_val, 4)
    }


def compute_dataset_statistics_table(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Builds a clean, comprehensive statistical summary table across numerical columns.

    Returns:
    --------
    pd.DataFrame:
        Table containing:
        ['Column', 'Non-Null', 'Mean', 'Median', 'Mode', 'Min', 'Max', 'Variance', 'Std Dev', 'Q1 (25%)', 'Q3 (75%)', 'Skewness']
    """
    if df is None or df.empty:
        return pd.DataFrame()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if columns is not None:
        target_cols = [c for c in columns if c in numeric_cols]
    else:
        target_cols = numeric_cols

    rows = []
    for col in target_cols:
        stats = compute_column_statistics(df[col])
        rows.append({
            "Column": col,
            "Non-Null Count": stats["count"],
            "Mean": stats["mean"],
            "Median": stats["median"],
            "Mode": stats["mode"],
            "Min": stats["min"],
            "Max": stats["max"],
            "Variance": stats["variance"],
            "Std Dev": stats["std_dev"],
            "Q1 (25%)": stats["q25"],
            "Q3 (75%)": stats["q75"],
            "Skewness": stats["skewness"]
        })

    return pd.DataFrame(rows)


def compute_correlation_matrix(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "pearson"
) -> pd.DataFrame:
    """
    Computes the correlation matrix between numerical columns.

    Methods:
    --------
    - 'pearson': Measures linear relationship. Best for continuous, normally distributed data.
    - 'spearman': Measures monotonic ranking relationship. Best for non-linear or ordinal data.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if columns is not None:
        target_cols = [c for c in columns if c in numeric_cols]
    else:
        target_cols = numeric_cols

    if len(target_cols) < 2:
        return pd.DataFrame()

    corr_df = df[target_cols].corr(method=method)
    return corr_df.round(4)


def get_top_correlations(corr_matrix: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Extracts and ranks pairs of features from a correlation matrix,
    filtering out diagonal self-correlations (1.0).

    Returns:
    --------
    List[Dict[str, Any]]:
        List of dictionaries with keys:
        ['feature_1', 'feature_2', 'correlation', 'strength']
    """
    if corr_matrix.empty or len(corr_matrix.columns) < 2:
        return []

    pairs = []
    cols = corr_matrix.columns.tolist()

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col1 = cols[i]
            col2 = cols[j]
            val = corr_matrix.loc[col1, col2]
            if pd.notna(val):
                pairs.append({
                    "feature_1": col1,
                    "feature_2": col2,
                    "correlation": float(val),
                    "abs_corr": abs(float(val)),
                    "interpretation": interpret_correlation(float(val))
                })

    # Sort pairs by absolute correlation in descending order
    pairs.sort(key=lambda x: x["abs_corr"], reverse=True)
    return pairs


def interpret_correlation(r: float) -> str:
    """
    Provides an educational interpretation of Pearson's correlation coefficient (r).
    """
    if r >= 0.8:
        return "Strong Positive: As one increases, the other increases strongly."
    elif r >= 0.4:
        return "Moderate Positive: General upward trend together."
    elif r > 0.1:
        return "Weak Positive: Slight upward relationship, largely independent."
    elif r >= -0.1:
        return "No Correlation: Variables appear independent."
    elif r > -0.4:
        return "Weak Negative: Slight downward relationship."
    elif r > -0.8:
        return "Moderate Negative: General downward trend together."
    else:
        return "Strong Negative: As one increases, the other decreases sharply."


def interpret_skewness(skew: float) -> str:
    """
    Provides an educational explanation of distribution skewness.
    """
    if abs(skew) < 0.5:
        return "Approximately Symmetric: Balanced distribution around the mean."
    elif skew >= 0.5:
        return "Right-Skewed (Positive): Tail stretches right; mean > median."
    else:
        return "Left-Skewed (Negative): Tail stretches left; mean < median."


def compute_group_comparison(
    df: pd.DataFrame,
    group_col: str,
    value_col: str
) -> pd.DataFrame:
    """
    Calculates aggregation statistics for a numerical column grouped by a categorical column.

    Example:
    --------
    Grouping 'salary' by 'department', or 'profit' by 'category'.

    Returns:
    --------
    pd.DataFrame:
        Table with columns:
        [group_col, 'Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Sum']
    """
    if df is None or df.empty or group_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame()

    # Drop rows where group or value is NaN for clean comparison
    valid_df = df[[group_col, value_col]].dropna()

    if valid_df.empty:
        return pd.DataFrame()

    grouped = valid_df.groupby(group_col)[value_col].agg(
        Count="count",
        Mean="mean",
        Median="median",
        Std_Dev=lambda s: s.std(ddof=1) if len(s) > 1 else 0.0,
        Min="min",
        Max="max",
        Total="sum"
    ).reset_index()

    # Round numeric columns for clean presentation
    numeric_result_cols = ["Mean", "Median", "Std_Dev", "Min", "Max", "Total"]
    grouped[numeric_result_cols] = grouped[numeric_result_cols].round(2)
    grouped = grouped.rename(columns={"Std_Dev": "Std Dev"})

    return grouped
