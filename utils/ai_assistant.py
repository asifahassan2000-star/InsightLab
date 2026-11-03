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

from typing import Dict, Any, List, Optional, Tuple
import os
import re
import time
import random
import logging
import pandas as pd
import numpy as np

from utils.statistical_analyzer import compute_dataset_statistics_table, compute_correlation_matrix, get_top_correlations
from utils.insights_engine import detect_outliers_iqr, compute_dataset_quality_score

logger = logging.getLogger("insightlab.ai_assistant")

# Default supported Gemini model for InsightLab
DEFAULT_GEMINI_MODEL: str = "gemini-3.8-flash"

# Retry configuration for temporary failures (503, 429, Network)
MAX_GEMINI_RETRIES: int = 3
INITIAL_RETRY_DELAY: float = 1.0
MAX_RETRY_DELAY: float = 8.0
_sleep_fn = time.sleep


def _sanitize_log_message(msg: str) -> str:
    """
    Removes any API keys or credential patterns from error messages before logging or display.
    """
    if not msg:
        return ""
    sanitized = re.sub(r"AIzaSy[A-Za-z0-9_-]{33}", "[REDACTED_KEY]", str(msg))
    sanitized = re.sub(r"([?&]key=)[^&]+", r"\1[REDACTED_KEY]", sanitized)
    return sanitized


def _is_retryable_error(e: Exception) -> Tuple[bool, str]:
    """
    Determines if an error is temporary and eligible for retry.
    Returns (is_retryable, error_category).

    Retryable errors:
    - 503 UNAVAILABLE (high demand, temporary outage)
    - 429 RESOURCE_EXHAUSTED / Rate limit / Quota exceeded
    - 500 INTERNAL / 502 BAD_GATEWAY / 504 GATEWAY_TIMEOUT
    - Network/Connection/Timeout errors

    Permanent errors (DO NOT retry):
    - 400 INVALID_ARGUMENT (malformed request)
    - 401 UNAUTHENTICATED / API_KEY_INVALID (invalid key)
    - 403 PERMISSION_DENIED / SERVICE_DISABLED (API disabled on project)
    - 404 NOT_FOUND (invalid/deprecated model)
    """
    error_msg = str(e)
    error_lower = error_msg.lower()

    # 1. Permanent errors (never retry)
    if (
        "not_found" in error_lower
        or "404" in error_msg
        or "no longer available" in error_lower
        or "unsupported model" in error_lower
        or ("models/" in error_lower and "not found" in error_lower)
    ):
        return False, "model_not_found"

    if (
        "api_key_invalid" in error_lower
        or "invalid api key" in error_lower
        or "api key not valid" in error_lower
        or ("invalid_argument" in error_lower and "key" in error_lower)
    ):
        return False, "invalid_api_key"

    if (
        "permission_denied" in error_lower
        or "403" in error_msg
        or "service_disabled" in error_lower
        or "has not been used in project" in error_lower
    ):
        return False, "permission_denied"

    # 2. Temporary / retryable errors
    if (
        "503" in error_msg
        or "unavailable" in error_lower
        or "high demand" in error_lower
        or "overloaded" in error_lower
    ):
        return True, "503_unavailable"

    if (
        "429" in error_msg
        or "resource_exhausted" in error_lower
        or "rate limit" in error_lower
        or "quota" in error_lower
    ):
        return True, "429_rate_limit"

    if (
        "500" in error_msg
        or "502" in error_msg
        or "504" in error_msg
        or "bad gateway" in error_lower
        or "gateway timeout" in error_lower
        or "internal error" in error_lower
    ):
        return True, "5xx_server_error"

    if (
        "timeout" in error_lower
        or "connection" in error_lower
        or "network" in error_lower
        or "getaddrinfo" in error_lower
        or "disconnected" in error_lower
        or "unreachable" in error_lower
        or "reset by peer" in error_lower
    ):
        return True, "network_error"

    # Default to non-retryable
    return False, "other"


def sanitize_gemini_model(model: Optional[str]) -> str:
    """
    Sanitizes and normalizes the Gemini model identifier.
    Guarantees that deprecated models (such as gemini-2.5-flash) or empty/invalid
    values are strictly intercepted and upgraded to 'gemini-3.8-flash'.
    Prevents any old default or fallback configuration from silently switching back to gemini-2.5-flash.
    """
    if not model or not isinstance(model, str):
        return DEFAULT_GEMINI_MODEL
    cleaned = model.strip()
    if not cleaned or "2.5" in cleaned:
        return DEFAULT_GEMINI_MODEL
    if cleaned.startswith("models/"):
        cleaned = cleaned[len("models/"):]
    return cleaned


def get_gemini_model() -> str:
    """
    Resolves the active Gemini model name from:
    1. Streamlit session state (`user_gemini_model`).
    2. Streamlit secrets (`GEMINI_MODEL` or `st.secrets["gemini"]["model"]`).
    3. OS environment variables (`GEMINI_MODEL` or `GOOGLE_GENAI_MODEL`).
    4. Defaults strictly to DEFAULT_GEMINI_MODEL ('gemini-3.8-flash').
    """
    try:
        import streamlit as st
        if "user_gemini_model" in st.session_state and st.session_state["user_gemini_model"]:
            return sanitize_gemini_model(st.session_state["user_gemini_model"])
        if hasattr(st, "secrets"):
            if "GEMINI_MODEL" in st.secrets:
                return sanitize_gemini_model(str(st.secrets["GEMINI_MODEL"]))
            if "gemini" in st.secrets and isinstance(st.secrets["gemini"], dict) and "model" in st.secrets["gemini"]:
                return sanitize_gemini_model(str(st.secrets["gemini"]["model"]))
    except Exception:
        pass

    env_model = os.environ.get("GEMINI_MODEL") or os.environ.get("GOOGLE_GENAI_MODEL")
    if env_model:
        return sanitize_gemini_model(env_model)

    return DEFAULT_GEMINI_MODEL


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
    model: Optional[str] = None,
    max_retries: int = MAX_GEMINI_RETRIES
) -> str:
    """
    Sends the user query and grounded statistical context to Google Gemini with
    automatic retry for temporary errors (503 UNAVAILABLE, 429 RATE LIMIT, Network).
    Uses exponential backoff with random jitter to prevent thundering herds.
    """
    active_model = sanitize_gemini_model(model or get_gemini_model())
    last_error: Optional[Exception] = None
    last_category: str = "other"

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

        # Retry loop: attempt 0 is initial try, attempts 1..max_retries are retries
        for attempt in range(max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=active_model,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.2,  # Low temperature for strict factual grounding
                        max_output_tokens=1500
                    )
                )
                return response.text

            except Exception as e:
                last_error = e
                is_retryable, category = _is_retryable_error(e)
                last_category = category

                clean_err = _sanitize_log_message(str(e))

                # Do not retry permanent errors or if all retries are exhausted
                if not is_retryable or attempt >= max_retries:
                    logger.error(
                        "Gemini API request failed (category=%s, attempt=%d/%d, retryable=%s): %s",
                        category, attempt + 1, max_retries + 1, is_retryable, clean_err
                    )
                    break

                # Exponential backoff with random jitter
                backoff = min(MAX_RETRY_DELAY, INITIAL_RETRY_DELAY * (2 ** attempt))
                jitter = random.uniform(0.1, 0.5)
                delay = backoff + jitter

                logger.warning(
                    "Gemini API temporary error (category=%s, attempt=%d/%d). Retrying in %.2fs... Error: %s",
                    category, attempt + 1, max_retries + 1, delay, clean_err
                )
                _sleep_fn(delay)

    except Exception as e:
        last_error = e
        last_category = "client_init_error"
        logger.error("Gemini client initialization error: %s", _sanitize_log_message(str(e)))

    error_msg = str(last_error) if last_error else "Unknown error"
    error_lower = error_msg.lower()

    # Friendly, actionable messages according to error category
    if last_category == "503_unavailable" or "503" in error_msg or "unavailable" in error_lower:
        return (
            "⏳ **Gemini Service Temporarily Unavailable (503):** Google Gemini is currently experiencing "
            f"temporary high demand. We automatically retried {max_retries} times with exponential backoff, "
            "but the model remains busy. Please try your question again in a moment, or use the Local Analytical Engine."
        )

    elif last_category == "429_rate_limit" or "429" in error_msg or "resource_exhausted" in error_lower or "quota" in error_lower:
        return (
            "⏳ **Quota Exceeded (429):** You have temporarily reached your Gemini API request limit. "
            f"We retried {max_retries} times with exponential backoff, but the quota limit is still active. "
            "Please wait a moment before sending another question."
        )

    elif (
        last_category == "model_not_found"
        or "not_found" in error_lower
        or "404" in error_msg
        or "is not found" in error_lower
        or "no longer available" in error_lower
    ):
        return (
            f"❌ **Model Error (404):** The requested model (`{active_model}`) is unavailable or deprecated. "
            f"InsightLab is configured to use `{DEFAULT_GEMINI_MODEL}`."
        )

    elif (
        last_category == "invalid_api_key"
        or "api_key_invalid" in error_lower
        or "invalid api key" in error_lower
        or "api key not valid" in error_lower
    ):
        return (
            "❌ **API Key Error:** The provided Gemini API key is invalid. "
            "Please verify your API key in Streamlit secrets or in the API key settings."
        )

    elif (
        last_category == "permission_denied"
        or "permission_denied" in error_lower
        or "403" in error_msg
        or "service_disabled" in error_lower
        or "has not been used in project" in error_lower
    ):
        return (
            "❌ **API Not Enabled (403):** Gemini API is disabled on this Google Cloud project. "
            "Enable it at https://console.developers.google.com/apis/api/generativelanguage.googleapis.com/overview"
        )

    elif (
        last_category == "network_error"
        or "timeout" in error_lower
        or "connection" in error_lower
        or "network" in error_lower
    ):
        return (
            f"🌐 **Network Connection Error:** Could not reach Gemini API after {max_retries} retry attempts "
            f"({_sanitize_log_message(error_msg)}). Please check your internet connection."
        )

    else:
        return f"⚠️ **Connection Error:** Could not reach Gemini API: {_sanitize_log_message(error_msg)}. Using local analytical engine fallback."


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
