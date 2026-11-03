"""
smart_save.py
=============
Smart Save Helper Module for InsightLab.

What This Module Does:
----------------------
Assembles complete, standardized save payloads from Streamlit session state
for all 7 save operations required by the project:
  1. Save Project (snapshot of dataset, cleaning history, quality, ML model, etc.)
  2. Save Dataset (raw uploaded dataset metadata & storage)
  3. Save Cleaned Dataset (cleaned working copy & cleaning steps applied)
  4. Save Analysis (statistical analysis & correlations)
  5. Save Automated Insights (findings, anomalies, recommendations)
  6. Save ML Model Information & Predictions
  7. Save Generated Reports (PDF / Excel metadata & storage)

Beginner Concept:
-----------------
Instead of duplicating the code that extracts dataset stats, column names,
and model metrics in every single studio component, this helper extracts
everything consistently and packages it into clean dictionaries ready for
Firestore and Firebase Storage.
"""

from typing import Any, Dict, Optional, Tuple
import datetime
import pandas as pd
import numpy as np


def build_project_payload(
    project_name: Optional[str] = None,
    notes: str = ""
) -> Dict[str, Any]:
    """
    Builds a full project snapshot dictionary from current session state.
    """
    import streamlit as st

    raw_df: Optional[pd.DataFrame] = st.session_state.get("raw_df")
    cleaned_df: Optional[pd.DataFrame] = st.session_state.get("cleaned_df")
    active_df = cleaned_df if (cleaned_df is not None and not cleaned_df.empty) else raw_df
    dataset_name = st.session_state.get("dataset_name", "Untitled Dataset")

    default_name = project_name or f"Project - {dataset_name}"

    payload: Dict[str, Any] = {
        "name": default_name,
        "dataset_name": dataset_name,
        "notes": notes,
        "has_cleaned_data": bool(cleaned_df is not None and not cleaned_df.empty),
        "cleaning_history": list(st.session_state.get("cleaning_history", [])),
        "ml_task": st.session_state.get("ml_task"),
    }

    if active_df is not None and not active_df.empty:
        payload["rows"] = int(len(active_df))
        payload["columns"] = int(len(active_df.columns))
        payload["column_names"] = [str(c) for c in active_df.columns.tolist()]
        payload["missing_cells"] = int(active_df.isnull().sum().sum())
    else:
        payload["rows"] = 0
        payload["columns"] = 0
        payload["column_names"] = []
        payload["missing_cells"] = 0

    # Add ML summary if available
    ml_bundle = st.session_state.get("ml_model_bundle")
    if ml_bundle:
        payload["ml_algorithm"] = ml_bundle.get("algorithm", "")
        payload["ml_target"] = ml_bundle.get("target_col", "")
        payload["ml_metrics"] = {
            k: float(v) if isinstance(v, (int, float, np.number)) else str(v)
            for k, v in ml_bundle.get("metrics", {}).items()
        }

    return payload


def build_dataset_payload(is_cleaned: bool = False) -> Dict[str, Any]:
    """
    Builds a dataset metadata payload for Firestore.
    """
    import streamlit as st

    if is_cleaned:
        df: Optional[pd.DataFrame] = st.session_state.get("cleaned_df")
        name = f"Cleaned_{st.session_state.get('dataset_name', 'dataset.csv')}"
    else:
        df = st.session_state.get("raw_df")
        name = st.session_state.get("dataset_name", "dataset.csv")

    if df is None or df.empty:
        return {"name": name, "rows": 0, "columns": 0, "error": "No data"}

    col_types = {str(col): str(dtype) for col, dtype in df.dtypes.items()}
    numeric_cols = [str(c) for c in df.select_dtypes(include=[np.number]).columns.tolist()]
    categorical_cols = [str(c) for c in df.select_dtypes(include=["object", "category", "string"]).columns.tolist()]

    return {
        "name": name,
        "is_cleaned": is_cleaned,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": [str(c) for c in df.columns.tolist()],
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "column_types": col_types,
        "missing_cells": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def build_analysis_payload(analysis_type: str = "statistical_analysis") -> Dict[str, Any]:
    """
    Builds an analysis record payload from the current session.
    """
    import streamlit as st

    raw_df = st.session_state.get("raw_df")
    cleaned_df = st.session_state.get("cleaned_df")
    active_df = cleaned_df if (cleaned_df is not None and not cleaned_df.empty) else raw_df
    dataset_name = st.session_state.get("dataset_name", "dataset")

    payload: Dict[str, Any] = {
        "name": f"Analysis - {dataset_name} ({datetime.datetime.now().strftime('%b %d, %H:%M')})",
        "analysis_type": analysis_type,
        "dataset_name": dataset_name,
    }

    if active_df is not None and not active_df.empty:
        try:
            from utils.statistical_analyzer import compute_dataset_statistics_table, get_top_correlations, compute_correlation_matrix
            stats_table = compute_dataset_statistics_table(active_df)
            payload["analyzed_columns_count"] = int(len(stats_table)) if stats_table is not None else 0

            corr_matrix = compute_correlation_matrix(active_df)
            if corr_matrix is not None and not corr_matrix.empty:
                top_corr = get_top_correlations(corr_matrix, top_n=5)
                payload["top_correlations"] = [
                    f"{c['feature_1']} & {c['feature_2']}: r={c['correlation']}"
                    for c in top_corr
                ]
        except Exception:
            pass

    return payload


def build_ml_model_payload() -> Dict[str, Any]:
    """
    Builds a machine learning model metadata payload.
    """
    import streamlit as st

    ml_bundle = st.session_state.get("ml_model_bundle")
    ml_task = st.session_state.get("ml_task", "unknown")
    dataset_name = st.session_state.get("dataset_name", "dataset")

    if not ml_bundle:
        return {"name": "No Model Trained", "task": ml_task, "error": "No model in session"}

    metrics = ml_bundle.get("metrics", {})
    clean_metrics = {
        str(k): float(v) if isinstance(v, (int, float, np.number)) else str(v)
        for k, v in metrics.items()
    }

    fi = ml_bundle.get("feature_importance", {})
    clean_fi = {
        str(k): float(v) if isinstance(v, (int, float, np.number)) else str(v)
        for k, v in fi.items()
    } if fi else {}

    algorithm = ml_bundle.get("algorithm", "Model")
    name = f"{algorithm} ({ml_task.capitalize()}) - {dataset_name}"

    return {
        "name": name,
        "task": ml_task,
        "algorithm": algorithm,
        "target_col": str(ml_bundle.get("target_col", "")),
        "feature_cols": [str(c) for c in ml_bundle.get("feature_cols", [])],
        "metrics": clean_metrics,
        "feature_importance": clean_fi,
        "dataset_name": dataset_name,
    }


def build_report_payload(report_name: str, report_type: str = "pdf") -> Dict[str, Any]:
    """
    Builds a report metadata payload.
    """
    import streamlit as st

    dataset_name = st.session_state.get("dataset_name", "dataset")
    return {
        "name": report_name,
        "report_type": report_type,
        "dataset_name": dataset_name,
        "has_ml": bool(st.session_state.get("ml_model_bundle")),
        "has_cleaned": bool(st.session_state.get("cleaned_df") is not None),
    }
