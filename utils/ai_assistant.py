"""
ai_assistant.py
===============
AI Data Assistant Engine for InsightLab (Phase 7).

Core Principles:
----------------
1. Grounded & Factual (No Hallucination):
   The AI assistant is provided with a rigorous statistical metadata digest
   calculated by Pandas and NumPy. All analytical conclusions are strictly grounded.

2. Privacy-Preserving Context Injection:
   Never transmits entire raw tabular records or sensitive individual rows.
   Instead, compiles statistical summaries, value counts, schemas, and correlation matrices.

3. Graceful Key & Connection Handling:
   - Supports Streamlit secrets (`st.secrets["GEMINI_API_KEY"]`), environment variables,
     or UI-based key configuration.
   - Provides a built-in deterministic local intelligence fallback so users can ask common
     analytical questions even without an API key!
"""

from typing import Dict, Any, List, Optional
import os
import re
import pandas as pd
import numpy as np

from utils.statistical_analyzer import compute_dataset_statistics_table, compute_correlation_matrix, get_top_correlations
from utils.insights_engine import detect_outliers_iqr, compute_dataset_quality_score

# Default supported Gemini model for InsightLab
DEFAULT_GEMINI_MODEL: str = "gemini-3.8-flash"


def get_gemini_api_key() -> Optional[str]:
    """
    Retrieves the Gemini API key securely from:
    1. Streamlit session state (user-entered in UI).
    2. Streamlit secrets (`.streamlit/secrets.toml`).
    3. OS environment variables (`GEMINI_API_KEY` or `GOOGLE_API_KEY`).
    """
    # 1. Check session state
    try:
        import streamlit as st
        if "user_gemini_api_key" in st.session_state and st.session_state["user_gemini_api_key"]:
            return st.session_state["user_gemini_api_key"].strip()
        if hasattr(st, "secrets"):
            if "GEMINI_API_KEY" in st.secrets:
                return str(st.secrets["GEMINI_API_KEY"]).strip()
            if "GOOGLE_API_KEY" in st.secrets:
                return str(st.secrets["GOOGLE_API_KEY"]).strip()
            if "gemini" in st.secrets and isinstance(st.secrets["gemini"], dict) and "api_key" in st.secrets["gemini"]:
                return str(st.secrets["gemini"]["api_key"]).strip()
    except Exception:
        pass

    # 2. Check environment variables
    env_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if env_key:
        return env_key.strip()

    return None


def build_dataset_context_summary(
    df: pd.DataFrame,
    dataset_name: str,
    extra_state: Optional[Dict[str, Any]] = None
) -> str:
    """
    Compiles a comprehensive, privacy-preserving statistical context digest of the dataset.

    Transmits:
    - Dimensions and schema (column types, nulls)
    - Key summary stats (mean, median, min, max, std dev)
    - Categorical value frequencies
    - Top correlations and anomalies
    - Machine learning metrics (if a model was trained)
    """
    if df is None or df.empty:
        return "No dataset is currently loaded."

    total_rows, total_cols = df.shape
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist()

    lines = [
        f"=== DATASET PROFILE: {dataset_name} ===",
        f"Dimensions: {total_rows} rows by {total_cols} columns",
        f"Numerical Columns: {', '.join(numeric_cols) if numeric_cols else 'None'}",
        f"Categorical Columns: {', '.join(categorical_cols) if categorical_cols else 'None'}",
        ""
    ]

    # Data Quality & Missingness
    quality = compute_dataset_quality_score(df)
    lines.append(f"Data Cleanliness Score: {quality['score']}/100 ({quality['grade']})")
    lines.append(f"Total Missing Cells: {quality['missing_count']} ({quality['missing_pct']}%)")
    lines.append(f"Total Duplicate Rows: {quality['duplicate_count']}")
    lines.append("")

    # Summary Statistics for Numerical Columns
    if numeric_cols:
        lines.append("=== NUMERICAL STATISTICS ===")
        stats_df = compute_dataset_statistics_table(df, numeric_cols)
        for _, row in stats_df.iterrows():
            lines.append(
                f"- {row['Column']}: Mean={row['Mean']}, Median={row['Median']}, "
                f"Min={row['Min']}, Max={row['Max']}, StdDev={row['Std Dev']}, Skewness={row['Skewness']}"
            )
        lines.append("")

    # Categorical Frequencies
    if categorical_cols:
        lines.append("=== CATEGORICAL BREAKDOWN (TOP VALUES) ===")
        for col in categorical_cols[:4]:
            val_counts = df[col].value_counts(dropna=True).head(5).to_dict()
            lines.append(f"- {col}: {val_counts}")
        lines.append("")

    # Top Correlations
    if len(numeric_cols) >= 2:
        corr_matrix = compute_correlation_matrix(df, numeric_cols)
        top_pairs = get_top_correlations(corr_matrix)
        if top_pairs:
            lines.append("=== CORRELATIONS (ASSOCIATIONS) ===")
            for p in top_pairs[:4]:
                lines.append(f"- {p['feature_1']} & {p['feature_2']}: r = {p['correlation']} ({p['interpretation']})")
            lines.append("")

    # Outlier Scan
    outlier_cols = []
    for col in numeric_cols:
        res = detect_outliers_iqr(df[col])
        if res["has_outliers"]:
            outlier_cols.append(f"{col} ({res['outlier_count']} outliers, range: {res['lower_bound']} to {res['upper_bound']})")
    if outlier_cols:
        lines.append(f"=== STATISTICAL ANOMALIES (IQR) ===")
        lines.append(f"- Outliers detected in: {'; '.join(outlier_cols)}")
        lines.append("")

    # Machine Learning Context (if user trained a model)
    if extra_state and extra_state.get("ml_model_bundle"):
        ml = extra_state["ml_model_bundle"]
        task = extra_state.get("ml_task", "unknown")
        lines.append("=== TRAINED MACHINE LEARNING MODEL ===")
        lines.append(f"Task: {task.capitalize()}")
        lines.append(f"Algorithm: {ml.get('algorithm')}")
        lines.append(f"Target: {ml.get('target_col')}")
        lines.append(f"Features: {ml.get('feature_cols')}")
        lines.append(f"Metrics: {ml.get('metrics')}")
        lines.append("")

    return "\n".join(lines)


def query_gemini_assistant(
    prompt: str,
    context: str,
    api_key: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    model: str = DEFAULT_GEMINI_MODEL
) -> str:
    """
    Sends the user query and grounded statistical context to Google Gemini.
    """
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        system_instruction = (
            "You are the InsightLab AI Data Assistant, an expert, encouraging, and beginner-friendly data science tutor.\n"
            "Your role is to help users analyze, understand, and interpret their currently loaded dataset.\n\n"
            "STRICT RULES:\n"
            "1. Ground all answers strictly in the verified dataset facts, metrics, and summary statistics provided in the context.\n"
            "2. Never fabricate numbers, fake records, or hallucinate metrics not present in the data profile.\n"
            "3. Never claim that correlation implies causation. Remind the user that correlation describes association.\n"
            "4. If a question cannot be answered from the provided dataset metadata, clearly explain why and state what information is missing.\n"
            "5. Structure responses cleanly using Markdown headings, bold metrics, bullet points, and beginner-friendly explanations.\n"
            "6. When explaining complex terms (like R2, standard deviation, skewness, or IQR), provide a brief, intuitive analogy."
        )

        history_context = ""
        if chat_history and len(chat_history) > 1:
            prev_turns = [m for m in chat_history[:-1] if m.get("content")]
            if prev_turns:
                history_lines = ["=== RECENT CONVERSATION CONTEXT ==="]
                for msg in prev_turns[-6:]:
                    role_label = "User" if msg.get("role") == "user" else "Assistant"
                    history_lines.append(f"{role_label}: {msg.get('content')}")
                history_context = "\n".join(history_lines) + "\n\n"

        full_prompt = (
            f"Here is the verified profile of the user's currently active dataset:\n\n"
            f"{context}\n\n"
            f"{history_context}"
            f"User Question: {prompt}\n\n"
            f"Please provide an accurate, structured, and pedagogical answer:"
        )

        # Call Gemini (gemini-3.8-flash is the modern, fast, and supported model for tabular analysis)
        response = client.models.generate_content(
            model=model,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,  # Low temperature for strict factual grounding
                max_output_tokens=1500
            )
        )

        return response.text

    except Exception as e:
        error_msg = str(e)
        if "API_KEY_INVALID" in error_msg or "invalid api key" in error_msg.lower():
            return "❌ **API Key Error:** The provided Gemini API key is invalid. Please verify your API key in Streamlit secrets or in the API key settings."
        elif "RESOURCE_EXHAUSTED" in error_msg:
            return "⏳ **Quota Exceeded:** You have reached your Gemini API request limit. Please wait a moment before trying again."
        elif "SERVICE_DISABLED" in error_msg or "has not been used in project" in error_msg:
            return f"❌ **API Not Enabled:** Gemini API is disabled on this Google Cloud project. Enable it at https://console.developers.google.com/apis/api/generativelanguage.googleapis.com/overview"
        else:
            return f"⚠️ **Connection Error:** Could not reach Gemini API: {error_msg}. Using local analytical engine fallback."


def answer_query_deterministically(
    prompt: str,
    df: pd.DataFrame,
    dataset_name: str,
    extra_state: Optional[Dict[str, Any]] = None
) -> str:
    """
    A robust local analytical engine that answers common questions using pure Pandas & NumPy
    when a Gemini API key is not configured.
    """
    p = prompt.lower().strip()
    total_rows, total_cols = df.shape
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist()

    # 1. Missing Values Query
    if any(k in p for k in ["missing", "null", "empty", "na"]):
        null_counts = df.isnull().sum()
        missing_cols = null_counts[null_counts > 0]
        if missing_cols.empty:
            return f"✅ **Missing Values Report:**\n\nThere are **zero missing values** in `{dataset_name}`! Every column is 100% complete across all {total_rows:,} records."
        else:
            lines = [f"⚠️ **Missing Values Report for `{dataset_name}`:**\n"]
            for col, count in missing_cols.items():
                pct = (count / total_rows) * 100
                lines.append(f"- **`{col}`**: {count:,} missing cells ({pct:.1f}%)")
            lines.append("\n💡 *Tip: You can use the **Data Cleaning Studio** to drop or impute these values.*")
            return "\n".join(lines)

    # 2. Correlation Query
    if any(k in p for k in ["correlat", "relationship", "association"]):
        if len(numeric_cols) < 2:
            return "ℹ️ Correlation analysis requires at least two numerical columns in your dataset."
        corr_matrix = compute_correlation_matrix(df, numeric_cols)
        top_pairs = get_top_correlations(corr_matrix)
        if not top_pairs:
            return "No strong linear correlations were identified between numerical columns."
        lines = [f"🔗 **Correlation Analysis for `{dataset_name}`:**\n"]
        for pair in top_pairs[:4]:
            lines.append(f"- **`{pair['feature_1']}` & `{pair['feature_2']}`:** r = **{pair['correlation']}** ({pair['interpretation']})")
        lines.append("\n> ⚠️ *Reminder: Correlation describes association, not causality.*")
        return "\n".join(lines)

    # 3. Highest / Average / Department / Category Queries
    if any(k in p for k in ["highest", "best", "top", "lowest", "department", "category", "average"]):
        # Look for category and numeric combinations
        if categorical_cols and numeric_cols:
            cat_target = categorical_cols[0]
            for c in categorical_cols:
                if c.lower() in p:
                    cat_target = c
                    break

            num_target = numeric_cols[0]
            for n in numeric_cols:
                if n.lower() in p:
                    num_target = n
                    break

            grouped = df.dropna(subset=[cat_target, num_target]).groupby(cat_target)[num_target].agg(["mean", "sum", "count"]).reset_index()
            grouped_sorted = grouped.sort_values(by="mean", ascending=False)
            top_row = grouped_sorted.iloc[0]
            bot_row = grouped_sorted.iloc[-1]

            lines = [
                f"📊 **Category Comparison: `{num_target}` across `{cat_target}`**\n",
                f"- **Highest Average:** **{top_row[cat_target]}** with an average `{num_target}` of **{top_row['mean']:,.2f}** (Total: {top_row['sum']:,.2f} across {int(top_row['count'])} records).",
                f"- **Lowest Average:** **{bot_row[cat_target]}** with an average `{num_target}` of **{bot_row['mean']:,.2f}**.",
                "\nFull breakdown:"
            ]
            for _, r in grouped_sorted.iterrows():
                lines.append(f"  • {r[cat_target]}: Average = {r['mean']:,.2f}")
            return "\n".join(lines)

    # 4. Outlier / Anomaly Query
    if any(k in p for k in ["outlier", "anomaly", "unusual", "strange"]):
        outlier_cols = []
        for col in numeric_cols:
            res = detect_outliers_iqr(df[col])
            if res["has_outliers"]:
                outlier_cols.append(f"- **`{col}`**: {res['outlier_count']} outliers outside [{res['lower_bound']} to {res['upper_bound']}]")
        if not outlier_cols:
            return "✅ **No statistical outliers detected!** All numerical values fall within the standard 1.5 × IQR fences."
        return "⚠️ **Outlier Detection Summary (1.5 × IQR):**\n\n" + "\n".join(outlier_cols)

    # 5. ML Model Query
    if any(k in p for k in ["machine learning", "model", "prediction", "accuracy", "r2"]):
        if extra_state and extra_state.get("ml_model_bundle"):
            ml = extra_state["ml_model_bundle"]
            task = extra_state.get("ml_task", "")
            metrics = ml.get("metrics", {})
            return (
                f"🤖 **Current Trained Machine Learning Model:**\n\n"
                f"- **Task:** {task.capitalize()}\n"
                f"- **Algorithm:** {ml.get('algorithm')}\n"
                f"- **Target:** `{ml.get('target_col')}`\n"
                f"- **Evaluation Metrics:** {metrics}\n"
                f"- **Input Features:** {', '.join(ml.get('feature_cols', []))}\n\n"
                f"💡 *You can test predictions interactively in the **Machine Learning Studio**!*"
            )
        else:
            return "ℹ️ No machine learning model has been trained yet in this session. Head to the **Machine Learning Studio** in the sidebar to train a model!"

    # 6. Default General Summary & Educational Overview
    quality = compute_dataset_quality_score(df)
    return (
        f"### 📋 Overview of `{dataset_name}`\n\n"
        f"Here is a comprehensive breakdown of your active dataset:\n\n"
        f"- **Scale:** **{total_rows:,} records** and **{total_cols} columns**.\n"
        f"- **Cleanliness Score:** **{quality['score']}/100** ({quality['grade']}).\n"
        f"- **Numerical Columns ({len(numeric_cols)}):** `{', '.join(numeric_cols) if numeric_cols else 'None'}`.\n"
        f"- **Categorical Columns ({len(categorical_cols)}):** `{', '.join(categorical_cols) if categorical_cols else 'None'}`.\n"
        f"- **Data Completeness:** {quality['missing_count']} missing cells ({quality['missing_pct']}%).\n\n"
        f"💡 *Tip: To ask freeform questions with full generative intelligence, add your Gemini API key in Streamlit secrets (`.streamlit/secrets.toml`) or enter it in the API settings box above!*"
    )
