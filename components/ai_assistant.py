"""
ai_assistant.py
===============
This module renders the 'AI Data Assistant' in InsightLab (Phase 7).

Features:
---------
1. Natural Language Data Q&A:
   Chat directly with your active dataset to get explanations, summaries,
   category comparisons, correlation insights, and ML explanations.

2. Factual Grounding & Privacy:
   Answers are grounded in mathematical facts, summary statistics, and schemas
   compiled by Pandas and NumPy. Raw personal rows are never transmitted.

3. Graceful Multi-Mode Operation:
   - Full Generative Power: Uses official Google GenAI SDK (Gemini 3.8 Flash)
     when an API key is provided via Streamlit secrets or settings.
   - Deterministic Fallback: Automatically handles analytical queries locally
     even if no API key is configured.
"""

from typing import Optional, List, Dict, Any
import streamlit as st
import pandas as pd
import numpy as np

from utils.data_loader import load_dataset
from utils.ai_assistant import (
    get_gemini_api_key,
    build_dataset_context_summary,
    query_gemini_assistant,
    answer_query_deterministically
)


def render_ai_assistant():
    """Renders the AI Data Assistant Studio UI."""
    st.markdown("## 💬 AI Data Assistant")
    st.markdown(
        "Chat with your data in plain English. Ask questions about columns, statistical patterns, "
        "correlations, or model performance, and get grounded, beginner-friendly answers."
    )

    # --------------------------------------------------------------------------
    # Step 1: Dataset Version & Loading
    # --------------------------------------------------------------------------
    has_raw = "raw_df" in st.session_state and st.session_state["raw_df"] is not None
    has_cleaned = "cleaned_df" in st.session_state and st.session_state["cleaned_df"] is not None

    if not has_raw and not has_cleaned:
        st.info("ℹ️ No dataset is currently loaded. Please upload a dataset or toggle the sample dataset below to start chatting.")
        uploaded_file = st.file_uploader(
            label="Upload dataset for AI Assistant",
            type=["csv", "xlsx", "xls"],
            key="chat_uploader"
        )
        col_sample, _ = st.columns([2, 5])
        with col_sample:
            use_sample = st.checkbox("🧪 Use sample sales dataset", value=False)

        if use_sample:
            sample_df = pd.DataFrame({
                "date": ["2026-01-15", "2026-01-16", "2026-01-17", "2026-01-18", "2026-01-19", "2026-01-20", "2026-01-21", "2026-01-22"],
                "category": ["Electronics", "Furniture", "Office Supplies", "Electronics", "Furniture", "Office Supplies", "Electronics", "Furniture"],
                "quantity": [2, 1, 10, 1, 4, 15, 2, 2],
                "unit_price": [499.99, 150.00, 12.50, 899.99, 85.00, 4.99, 650.00, 220.00],
                "discount": [0.05, 0.00, 0.10, 0.15, 0.00, 0.00, 0.10, 0.05],
                "profit": [120.50, 35.00, 25.00, 180.00, 50.00, 18.50, 110.00, 60.00]
            })
            st.session_state["raw_df"] = sample_df.copy()
            st.session_state["cleaned_df"] = sample_df.copy()
            st.session_state["dataset_name"] = "sample_sales_data.csv"
            st.rerun()

        elif uploaded_file is not None:
            loaded_df, err = load_dataset(uploaded_file)
            if err:
                st.error(f"❌ Failed to load file: {err}")
                return
            st.session_state["raw_df"] = loaded_df.copy()
            st.session_state["cleaned_df"] = loaded_df.copy()
            st.session_state["dataset_name"] = uploaded_file.name
            st.rerun()

        return

    # Choose active dataset version
    dataset_name = st.session_state.get("dataset_name", "dataset")
    col_info, col_ver = st.columns([3, 2])
    with col_info:
        st.markdown(f"**Active Dataset:** `{dataset_name}`")
    with col_ver:
        if has_cleaned and has_raw:
            data_version = st.radio(
                "Chat Source:",
                options=["Cleaned Working Dataset (Phase 2)", "Original Raw Dataset (Phase 1)"],
                horizontal=True
            )
            df = st.session_state["cleaned_df"] if "Cleaned" in data_version else st.session_state["raw_df"]
        else:
            df = st.session_state.get("cleaned_df", st.session_state.get("raw_df"))

    # --------------------------------------------------------------------------
    # Step 2: API Key Configuration & Status Banner
    # --------------------------------------------------------------------------
    api_key = get_gemini_api_key()

    with st.expander("🔑 Gemini API Key Settings & Privacy", expanded=False):
        st.markdown(
            "InsightLab protects your privacy by passing **only verified statistical summaries and schema metadata** "
            "to Google Gemini. Individual personal records or sensitive row-level data are never transmitted."
        )
        if api_key:
            st.success("✅ **Gemini API Key Active:** Connected to Google Gemini (gemini-3.8-flash).")
        else:
            st.info(
                "💡 **No API Key configured:** InsightLab is currently running in **Local Analytical Mode**. "
                "You can still ask common analytical questions using local Pandas calculations!\n\n"
                "To unlock full conversational reasoning, add `GEMINI_API_KEY` to `.streamlit/secrets.toml` "
                "or enter your key below:"
            )

        entered_key = st.text_input(
            "Enter Gemini API Key (Optional for current session):",
            type="password",
            value=st.session_state.get("user_gemini_api_key", ""),
            help="Get an API key from Google AI Studio (https://aistudio.google.com/)."
        )
        if entered_key:
            st.session_state["user_gemini_api_key"] = entered_key
            api_key = entered_key

    # --------------------------------------------------------------------------
    # Step 3: Initialize Conversation State & Quick Clear
    # --------------------------------------------------------------------------
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [
            {
                "role": "assistant",
                "content": (
                    f"👋 Hello! I am your **InsightLab AI Data Assistant**.\n\n"
                    f"I have inspected `{dataset_name}` ({len(df):,} rows × {len(df.columns)} columns). "
                    f"You can ask me anything about your dataset's columns, missing values, "
                    f"trends, or machine learning results!"
                )
            }
        ]

    # Quick action bar
    col_clear, col_status = st.columns([1, 4])
    with col_clear:
        if st.button("🗑️ Clear Chat", help="Reset conversation history"):
            st.session_state["chat_history"] = []
            st.rerun()
    with col_status:
        mode_badge = "🟢 Mode: Gemini AI Online" if api_key else "🔵 Mode: Local Python Analytical Engine"
        st.caption(f"**{mode_badge}** | Active records: **{len(df):,}**")

    # --------------------------------------------------------------------------
    # Step 4: Suggested Quick Questions
    # --------------------------------------------------------------------------
    st.markdown("##### 💡 Suggested Questions (Click to Ask):")
    q1, q2, q3, q4 = st.columns(4)
    selected_suggested = None
    with q1:
        if st.button("📌 Summarize dataset", use_container_width=True):
            selected_suggested = "What are the main findings and statistical summary of this dataset?"
    with q2:
        if st.button("❓ Check missing values", use_container_width=True):
            selected_suggested = "Are there any missing values or duplicates in this dataset?"
    with q3:
        if st.button("🏆 Top category leader", use_container_width=True):
            selected_suggested = "Which category or department has the highest average metric?"
    with q4:
        if st.button("🔗 Find correlations", use_container_width=True):
            selected_suggested = "Which numerical columns are strongly correlated?"

    st.markdown("---")

    # --------------------------------------------------------------------------
    # Step 5: Render Chat History
    # --------------------------------------------------------------------------
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # --------------------------------------------------------------------------
    # Step 6: User Input & Response Generation
    # --------------------------------------------------------------------------
    user_prompt = st.chat_input("Ask a question about your dataset (e.g., 'What is the average profit by category?')...")

    # Use clicked suggested question if available
    active_prompt = selected_suggested if selected_suggested else user_prompt

    if active_prompt:
        # 1. Append user message
        st.session_state["chat_history"].append({"role": "user", "content": active_prompt})
        with st.chat_message("user"):
            st.markdown(active_prompt)

        # 2. Compile grounded factual context
        extra_state = {
            "ml_model_bundle": st.session_state.get("ml_model_bundle"),
            "ml_task": st.session_state.get("ml_task")
        }
        context = build_dataset_context_summary(df, dataset_name, extra_state=extra_state)

        # 3. Generate response with spinner
        with st.chat_message("assistant"):
            with st.spinner("Analyzing dataset & formulating response..."):
                if api_key:
                    response_text = query_gemini_assistant(
                        prompt=active_prompt,
                        context=context,
                        api_key=api_key,
                        chat_history=st.session_state["chat_history"]
                    )
                else:
                    response_text = answer_query_deterministically(
                        prompt=active_prompt,
                        df=df,
                        dataset_name=dataset_name,
                        extra_state=extra_state
                    )

            st.markdown(response_text)

        # 4. Append assistant response to state
        st.session_state["chat_history"].append({"role": "assistant", "content": response_text})
        st.rerun()
