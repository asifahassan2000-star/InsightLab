"""
data_cleaning.py
================
This module renders the 'Data Cleaning' page in InsightLab (Phase 2).

Key Principles:
---------------
1. Immutability:
   The original dataset (`st.session_state['raw_df']`) is NEVER modified.
   All cleaning operations are applied to `st.session_state['cleaned_df']`.

2. User-Driven Decisions:
   InsightLab never makes assumptions or cleans data automatically.
   The user explicitly chooses their cleaning operations.

3. Complete Auditability:
   Maintains a `cleaning_history` log so users know exactly what steps were performed,
   accompanied by before-and-after statistics.
"""

from typing import Optional, List
import streamlit as st
import pandas as pd
from utils.data_loader import load_dataset, get_column_summary
from utils.data_cleaner import (
    get_dataset_health,
    remove_duplicates,
    drop_missing_values,
    fill_numeric_missing,
    fill_categorical_missing,
    export_dataframe_to_bytes
)


def initialize_cleaning_state(df: pd.DataFrame, source_name: str):
    """
    Initializes or updates session state with a fresh copy of the dataset.
    """
    st.session_state["raw_df"] = df.copy()
    st.session_state["cleaned_df"] = df.copy()
    st.session_state["dataset_name"] = source_name
    st.session_state["cleaning_history"] = []


def render_data_cleaning():
    """
    Renders the Data Cleaning Studio UI.
    """
    st.markdown("## 🧹 Data Cleaning Studio")
    st.markdown(
        "Inspect data health, handle missing values, eliminate duplicate rows, "
        "and export clean datasets without modifying your original files."
    )

    # --------------------------------------------------------------------------
    # Step 1: Check if a dataset is loaded in session_state
    # --------------------------------------------------------------------------
    if "raw_df" not in st.session_state or st.session_state["raw_df"] is None:
        st.info("ℹ️ No dataset is currently loaded. You can upload one below or select a sample dataset.")
        
        uploaded_file = st.file_uploader(
            label="Upload CSV or Excel dataset to clean",
            type=["csv", "xlsx", "xls"],
            key="cleaner_uploader"
        )
        col_sample, _ = st.columns([2, 5])
        with col_sample:
            use_sample = st.checkbox("🧪 Use sample dataset with missing values & duplicates", value=False)

        if use_sample:
            sample_df = pd.DataFrame({
                "order_id": [101, 102, 102, 103, 104, 105, 106],
                "customer": ["Alice", "Bob", "Bob", "Charlie", "David", "Eve", "Frank"],
                "category": ["Electronics", "Furniture", "Furniture", None, "Electronics", "Office Supplies", "Furniture"],
                "amount": [250.0, 120.0, 120.0, None, 450.0, 75.0, 310.0],
                "quantity": [2, 1, 1, 3, None, 5, 2]
            })
            initialize_cleaning_state(sample_df, "sample_sales_dirty.csv")
            st.rerun()

        elif uploaded_file is not None:
            loaded_df, err = load_dataset(uploaded_file)
            if err:
                st.error(f"❌ Failed to load file: {err}")
                return
            initialize_cleaning_state(loaded_df, uploaded_file.name)
            st.rerun()

        return

    # Ensure working copy and history exist in session_state
    raw_df: pd.DataFrame = st.session_state["raw_df"]
    if "cleaned_df" not in st.session_state or st.session_state["cleaned_df"] is None:
        st.session_state["cleaned_df"] = raw_df.copy()
    cleaned_df: pd.DataFrame = st.session_state["cleaned_df"]

    if "cleaning_history" not in st.session_state:
        st.session_state["cleaning_history"] = []

    dataset_name = st.session_state.get("dataset_name", "dataset")

    # --------------------------------------------------------------------------
    # Step 2: Top Status Bar & Reset Controls
    # --------------------------------------------------------------------------
    col_name, col_reset = st.columns([4, 1])
    with col_name:
        st.markdown(f"**Active Dataset:** `{dataset_name}`")
    with col_reset:
        if st.button("↺ Reset All", help="Discard all cleaning changes and revert to the original dataset"):
            st.session_state["cleaned_df"] = raw_df.copy()
            st.session_state["cleaning_history"] = []
            st.toast("Reverted dataset back to original state!", icon="↺")
            st.rerun()

    # --------------------------------------------------------------------------
    # Step 3: Before-and-After Comparison Metrics (Requirement 6)
    # --------------------------------------------------------------------------
    raw_health = get_dataset_health(raw_df)
    clean_health = get_dataset_health(cleaned_df)

    # Compute deltas
    row_delta = clean_health["total_rows"] - raw_health["total_rows"]
    missing_delta = clean_health["total_missing"] - raw_health["total_missing"]
    dup_delta = clean_health["duplicate_rows"] - raw_health["duplicate_rows"]

    st.markdown("### 📊 Dataset Health: Before vs. After")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            label="Total Rows",
            value=f"{clean_health['total_rows']:,}",
            delta=f"{row_delta} rows" if row_delta != 0 else "No change",
            delta_color="normal"
        )
    with m2:
        st.metric(
            label="Total Columns",
            value=f"{clean_health['total_cols']:,}"
        )
    with m3:
        st.metric(
            label="Missing Values",
            value=f"{clean_health['total_missing']:,}",
            delta=f"{missing_delta} values" if missing_delta != 0 else "Clean (0 missing)" if clean_health['total_missing'] == 0 else "No change",
            delta_color="inverse"
        )
    with m4:
        st.metric(
            label="Duplicate Rows",
            value=f"{clean_health['duplicate_rows']:,}",
            delta=f"{dup_delta} dups" if dup_delta != 0 else "0 duplicates" if clean_health['duplicate_rows'] == 0 else "No change",
            delta_color="inverse"
        )

    # --------------------------------------------------------------------------
    # Step 4: Interactive Tabs
    # --------------------------------------------------------------------------
    tab_inspect, tab_actions, tab_preview, tab_learn = st.tabs([
        "🔎 1. Inspect Quality",
        "🛠️ 2. Apply Cleaning Actions",
        "👁️ 3. Preview & Download",
        "💡 4. Learning Zone"
    ])

    # --------------------------------------------------------------------------
    # TAB 1: INSPECT QUALITY (Requirements 1, 2, 3)
    # --------------------------------------------------------------------------
    with tab_inspect:
        st.subheader("Data Health & Quality Report")
        st.caption("Review missing value frequencies, detected duplicates, and inferred data types.")

        # Duplicate Rows Alert
        if clean_health["duplicate_rows"] > 0:
            st.warning(
                f"⚠️ **Duplicate Rows Detected:** Found `{clean_health['duplicate_rows']}` duplicate rows in the current working dataset. "
                "You can remove them in the **Apply Cleaning Actions** tab."
            )
            with st.expander("👀 View Duplicate Rows"):
                dups_view = cleaned_df[cleaned_df.duplicated(keep=False)]
                st.dataframe(dups_view, use_container_width=True)
        else:
            st.success("✅ **No Duplicate Rows:** Every row in this dataset is distinct.")

        # Column Quality Table (Requirements 1 & 3)
        st.markdown("#### Column Breakdown: Missing Values & Data Types")
        col_summary = get_column_summary(cleaned_df)
        st.dataframe(col_summary, use_container_width=True)

        if len(clean_health["columns_with_missing"]) == 0:
            st.success("🎉 **Clean Data:** No missing values exist in any column!")
        else:
            st.info(
                f"ℹ️ Columns with missing values: `{', '.join(clean_health['columns_with_missing'])}`. "
                "Switch to the **Apply Cleaning Actions** tab to address them."
            )

    # --------------------------------------------------------------------------
    # TAB 2: APPLY CLEANING ACTIONS (Requirements 4 & 5)
    # --------------------------------------------------------------------------
    with tab_actions:
        st.subheader("Interactive Cleaning Operations")
        st.caption("Select and execute cleaning operations. Changes update the working copy immediately.")

        # ACTION 1: Remove Duplicate Rows (Requirement 4)
        with st.container(border=True):
            st.markdown("#### 1. Duplicate Rows Removal")
            st.markdown(
                "Removes duplicate rows from the dataset. By default, it checks across all columns "
                "and keeps the first occurrence."
            )
            dup_count = clean_health["duplicate_rows"]

            if dup_count == 0:
                st.info("No duplicates detected to remove.")
            else:
                col_d1, col_d2 = st.columns([3, 1])
                with col_d1:
                    st.write(f"Identified **{dup_count}** duplicate rows ready for removal.")
                with col_d2:
                    if st.button("🗑️ Remove Duplicates", key="btn_remove_dups"):
                        updated_df, removed = remove_duplicates(cleaned_df)
                        st.session_state["cleaned_df"] = updated_df
                        st.session_state["cleaning_history"].append(f"Removed {removed} duplicate rows.")
                        st.toast(f"Successfully removed {removed} duplicate rows!", icon="✅")
                        st.rerun()

        # ACTION 2: Drop Missing Rows (Requirement 5a)
        with st.container(border=True):
            st.markdown("#### 2. Remove Rows with Missing Values")
            st.markdown(
                "Drops entire rows where null values are present (`pd.DataFrame.dropna()`). "
                "Use caution: dropping rows reduces sample size."
            )
            drop_mode = st.radio(
                "Where should missing values trigger row removal?",
                options=["Any column with missing values", "Specific selected columns only"],
                key="drop_missing_mode"
            )

            target_drop_cols = None
            if drop_mode == "Specific selected columns only":
                target_drop_cols = st.multiselect(
                    "Select column(s) to check:",
                    options=cleaned_df.columns.tolist(),
                    default=clean_health["columns_with_missing"],
                    key="drop_cols_select"
                )

            if st.button("❌ Drop Rows with Missing Values", key="btn_drop_missing"):
                if drop_mode == "Specific selected columns only" and not target_drop_cols:
                    st.warning("Please select at least one column to check.")
                else:
                    updated_df, dropped = drop_missing_values(cleaned_df, subset=target_drop_cols)
                    if len(updated_df) == 0:
                        st.error("⚠️ Dropping these rows would result in an empty dataset! Operation aborted.")
                    else:
                        st.session_state["cleaned_df"] = updated_df
                        cols_desc = f"in {target_drop_cols}" if target_drop_cols else "in any column"
                        st.session_state["cleaning_history"].append(f"Dropped {dropped} rows with missing values ({cols_desc}).")
                        st.toast(f"Dropped {dropped} rows containing missing values.", icon="✅")
                        st.rerun()

        # ACTION 3: Impute Numerical Values (Requirement 5b)
        with st.container(border=True):
            st.markdown("#### 3. Fill Numerical Missing Values (Imputation)")
            st.markdown(
                "Replaces empty numerical cells with either the **Mean** (average) or **Median** (middle value). "
                "This preserves row count without introducing empty cells."
            )

            numeric_missing_cols = [c for c in clean_health["numeric_cols"] if cleaned_df[c].isnull().any()]

            if not numeric_missing_cols:
                st.info("No numerical columns currently have missing values.")
            else:
                col_strat, col_cols = st.columns(2)
                with col_strat:
                    num_strategy = st.selectbox(
                        "Imputation Strategy:",
                        options=["Mean", "Median"],
                        help="Mean is best for normally distributed data. Median is best when data has outliers."
                    )
                with col_cols:
                    selected_num_cols = st.multiselect(
                        "Target Numerical Columns:",
                        options=numeric_missing_cols,
                        default=numeric_missing_cols,
                        key="selected_num_cols"
                    )

                if st.button(f"🔢 Fill Selected with {num_strategy}", key="btn_fill_numeric"):
                    if not selected_num_cols:
                        st.warning("Please select at least one numerical column.")
                    else:
                        updated_df, report = fill_numeric_missing(
                            cleaned_df,
                            strategy=num_strategy.lower(),
                            columns=selected_num_cols
                        )
                        st.session_state["cleaned_df"] = updated_df
                        for col, details in report.items():
                            st.session_state["cleaning_history"].append(
                                f"Filled {details['rows_filled']} missing value(s) in '{col}' with {details['strategy']} ({details['fill_value']})."
                            )
                        st.toast(f"Filled numerical missing values using {num_strategy}!", icon="✅")
                        st.rerun()

        # ACTION 4: Impute Categorical Values (Requirement 5c)
        with st.container(border=True):
            st.markdown("#### 4. Fill Categorical Missing Values (Mode Imputation)")
            st.markdown(
                "Replaces missing text/categorical cells with the **Mode** (the most frequent value). "
                "For example, if 'Country' is missing, it is filled with the country that appears most often."
            )

            cat_missing_cols = [c for c in clean_health["categorical_cols"] if cleaned_df[c].isnull().any()]

            if not cat_missing_cols:
                st.info("No categorical/text columns currently have missing values.")
            else:
                selected_cat_cols = st.multiselect(
                    "Target Categorical Columns:",
                    options=cat_missing_cols,
                    default=cat_missing_cols,
                    key="selected_cat_cols"
                )

                if st.button("🏷️ Fill Selected with Mode", key="btn_fill_categorical"):
                    if not selected_cat_cols:
                        st.warning("Please select at least one categorical column.")
                    else:
                        updated_df, report = fill_categorical_missing(
                            cleaned_df,
                            columns=selected_cat_cols
                        )
                        st.session_state["cleaned_df"] = updated_df
                        for col, details in report.items():
                            st.session_state["cleaning_history"].append(
                                f"Filled {details['rows_filled']} missing value(s) in '{col}' with mode ('{details['fill_value']}')."
                            )
                        st.toast("Filled categorical missing values using Mode!", icon="✅")
                        st.rerun()

    # --------------------------------------------------------------------------
    # TAB 3: PREVIEW & DOWNLOAD (Requirements 7 & 8)
    # --------------------------------------------------------------------------
    with tab_preview:
        st.subheader("Cleaned Dataset Preview & Export")
        st.caption("Inspect the final cleaned data and download it as CSV or Excel.")

        # Audit Log / History
        st.markdown("#### 📋 Cleaning Audit Log")
        history = st.session_state.get("cleaning_history", [])
        if not history:
            st.info("No cleaning operations have been applied yet. The dataset is identical to the original.")
        else:
            for idx, entry in enumerate(history, 1):
                st.markdown(f"**{idx}.** {entry}")

        st.markdown("---")

        # Side-by-side or Tabbed preview (Requirement 7)
        st.markdown("#### 🔍 Dataset Preview")
        preview_mode = st.radio("Choose view:", ["Cleaned Dataset (Working Copy)", "Original Dataset (Unmodified)"], horizontal=True)

        slider_max = max(10, min(100, len(cleaned_df) if preview_mode.startswith("Cleaned") else len(raw_df)))
        row_limit = st.slider("Rows to preview:", min_value=5, max_value=slider_max, value=min(10, slider_max), step=5)

        if preview_mode.startswith("Cleaned"):
            st.dataframe(cleaned_df.head(row_limit), use_container_width=True)
            st.caption(f"Displaying top {min(row_limit, len(cleaned_df))} of {len(cleaned_df)} rows.")
        else:
            st.dataframe(raw_df.head(row_limit), use_container_width=True)
            st.caption(f"Displaying top {min(row_limit, len(raw_df))} of {len(raw_df)} rows.")

        st.markdown("---")

        # Download Options (Requirement 8)
        st.markdown("#### 💾 Download Cleaned Dataset")
        st.caption("Download the cleaned dataset in your preferred format:")

        base_name = dataset_name.rsplit(".", 1)[0] if "." in dataset_name else dataset_name
        col_dl_csv, col_dl_excel = st.columns(2)

        with col_dl_csv:
            csv_bytes = export_dataframe_to_bytes(cleaned_df, "csv")
            st.download_button(
                label="📥 Download as CSV (.csv)",
                data=csv_bytes,
                file_name=f"{base_name}_cleaned.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col_dl_excel:
            excel_bytes = export_dataframe_to_bytes(cleaned_df, "excel")
            st.download_button(
                label="📊 Download as Excel (.xlsx)",
                data=excel_bytes,
                file_name=f"{base_name}_cleaned.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        # Cloud Save Cleaned Dataset (Guest-friendly optional save)
        from utils.smart_save import build_dataset_payload
        from components.auth_ui import render_cloud_save_button
        cleaned_payload = build_dataset_payload(is_cleaned=True)
        cleaned_payload["file_bytes"] = csv_bytes
        st.markdown("---")
        st.markdown("##### ☁️ Cloud Backup")
        render_cloud_save_button("dataset", cleaned_payload, label="☁️ Save Cleaned Dataset to Cloud", key_suffix="cleaning")

    # --------------------------------------------------------------------------
    # TAB 4: LEARNING ZONE
    # --------------------------------------------------------------------------
    with tab_learn:
        st.markdown("#### 🎓 Beginner's Guide: Data Cleaning Concepts")
        st.markdown("""
        ### Why is Data Cleaning Important?
        In machine learning and data science, there is a golden rule: **"Garbage In, Garbage Out"**.
        If you feed dirty, duplicated, or missing data into charts or predictive models, the results will be misleading or incorrect.

        ---

        ### 1. Handling Missing Data: Drop vs. Impute
        When data is missing (`NaN`), you have two primary options:
        
        - **Option A: Dropping Rows (`df.dropna()`)**
          - *Pros:* Clean and natural; no artificial data is created.
          - *Cons:* Throws away real observations. If 30% of rows have one missing cell, you lose 30% of your data!
        
        - **Option B: Imputation (`df.fillna()`)**
          - *Pros:* Keeps all rows intact and preserves dataset size.
          - *Cons:* Injects estimated values which can slightly reduce variability.

        ---

        ### 2. Choosing Between Mean, Median, and Mode
        - **Mean (Average)**:
          - Use for symmetric, bell-shaped numerical distributions (e.g., student test scores, heights).
          - *Warning:* Heavily influenced by extreme outliers!
        - **Median (50th Percentile)**:
          - Use for skewed numerical data (e.g., salaries, house prices) because it ignores extreme outliers.
        - **Mode (Most Frequent)**:
          - The only valid mathematical average for categorical/text data (e.g., country names, payment methods).

        ---

        ### 3. The Power of Immutability (`df.copy()`)
        Notice that in Python, doing `df2 = df` does **not** create a copy—it merely creates a second pointer to the *same memory*.
        Modifying `df2` would silently alter `df`.
        By using `df.copy()`, we allocate fresh memory so the user's original raw data is never destroyed!
        """)
