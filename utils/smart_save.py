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


def _safe_float(val: Any) -> float:
    """
    Safely converts a scalar to a JSON-serializable float, defaulting to 0.0 for NaN/Inf/invalid.
    Never attempts to convert non-scalar or iterable objects.
    """
    if val is None or pd.isna(val):
        return 0.0
    try:
        if hasattr(val, "__iter__") and not isinstance(val, (str, bytes)):
            return 0.0
        num = float(val)
        if np.isnan(num) or np.isinf(num):
            return 0.0
        return float(num)
    except (ValueError, TypeError):
        return 0.0


def clean_feature_importance(fi: Any) -> Dict[str, float]:
    """
    Safely converts feature importance from DataFrame, Series, dict, None, or empty value
    into a clean, JSON-serializable Python dictionary mapping feature names to floats.
    Handles NaN, Inf, and invalid numeric values safely without ambiguous truth-value checks.
    """
    if fi is None:
        return {}

    raw_pairs = []

    if isinstance(fi, pd.DataFrame):
        if fi.empty:
            return {}

        cols = list(fi.columns)
        feat_col = None
        imp_col = None

        # Look for explicit feature name column
        for c in cols:
            c_str = str(c).lower().strip()
            if c_str in ("feature", "features", "name", "variable", "var", "column", "col"):
                feat_col = c
                break
        if feat_col is None:
            for c in cols:
                c_str = str(c).lower().strip()
                if any(term in c_str for term in ("feature", "var", "col")):
                    feat_col = c
                    break

        # Look for explicit importance column
        for c in cols:
            if c == feat_col:
                continue
            c_str = str(c).lower().strip()
            if c_str in ("importance", "importances", "weight", "weights", "score", "scores", "coef", "coefficient", "value", "val"):
                imp_col = c
                break
        if imp_col is None:
            for c in cols:
                if c == feat_col:
                    continue
                c_str = str(c).lower().strip()
                if any(term in c_str for term in ("importance", "weight", "score", "coef")):
                    imp_col = c
                    break

        # Fallback if columns not identified by name
        if feat_col is None or imp_col is None:
            if len(cols) == 1:
                imp_col = cols[0]
                feat_col = None
            elif len(cols) >= 2:
                remaining = [c for c in cols if c not in (feat_col, imp_col)]
                if imp_col is None:
                    for c in remaining:
                        if pd.api.types.is_numeric_dtype(fi[c]):
                            imp_col = c
                            remaining.remove(c)
                            break
                    if imp_col is None and remaining:
                        imp_col = remaining.pop(-1)
                if feat_col is None:
                    for c in remaining:
                        if not pd.api.types.is_numeric_dtype(fi[c]):
                            feat_col = c
                            remaining.remove(c)
                            break
                    if feat_col is None and remaining:
                        feat_col = remaining.pop(0)

        features = fi[feat_col].tolist() if feat_col is not None else [str(idx) for idx in fi.index]
        importances = fi[imp_col].tolist() if imp_col is not None else []
        raw_pairs = list(zip(features, importances))

    elif isinstance(fi, pd.Series):
        if fi.empty:
            return {}
        raw_pairs = list(zip(fi.index.tolist(), fi.tolist()))

    elif isinstance(fi, dict):
        if not fi:
            return {}
        raw_pairs = list(fi.items())

    elif isinstance(fi, (list, tuple)):
        valid_items = []
        for item in fi:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                valid_items.append((item[0], item[1]))
        raw_pairs = valid_items

    else:
        return {}

    clean_dict: Dict[str, float] = {}
    for k, v in raw_pairs:
        key_str = str(k).strip()
        if not key_str:
            continue
        clean_dict[key_str] = _safe_float(v)

    return clean_dict


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
    clean_metrics = {}
    if isinstance(metrics, dict):
        for k, v in metrics.items():
            if isinstance(v, (int, float, np.number)):
                clean_metrics[str(k)] = _safe_float(v)
            else:
                clean_metrics[str(k)] = str(v)

    fi = ml_bundle.get("feature_importance")
    clean_fi = clean_feature_importance(fi)

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
