"""
dataset_explorer.py
===================
This module renders the 'Dataset Explorer' page in InsightLab.

What is Streamlit?
------------------
Streamlit is an open-source Python framework that allows data scientists and developers
to build interactive web applications with simple Python scripts.
Instead of writing HTML, CSS, and JavaScript, you write Python functions like:
- `st.title()` to render a main heading
- `st.file_uploader()` to let users upload files
- `st.metric()` to display key performance indicators (KPIs)
- `st.dataframe()` to show interactive tables
"""

from typing import Optional
import streamlit as st
import pandas as pd
from utils.data_loader import load_dataset, get_column_summary


def render_dataset_explorer():
    """
    Renders the Dataset Explorer view inside the Streamlit app.
    Guides the user through file upload, validation, and exploratory data inspection.
    """
    st.markdown("## 📊 Dataset Explorer")
    st.markdown(
        "Upload a dataset in **CSV** or **Excel** format to inspect its structure, "
        "examine column data types, and review summary statistics."
    )

    # --------------------------------------------------------------------------
    # Step 1: File Uploader Component
    # --------------------------------------------------------------------------
    # st.file_uploader creates a drag-and-drop file upload widget in the browser.
    # The 'type' parameter restricts allowed file extensions.
    uploaded_file = st.file_uploader(
        label="Choose a CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="Supported formats: .csv, .xlsx, .xls. Maximum default size is 200MB."
    )

    # Provide a sample dataset option for easy testing and learning
    col_sample, _ = st.columns([2, 5])
    with col_sample:
        use_sample = st.checkbox("🧪 Use sample dataset (Iris Flower Dataset)", value=False)

    df: Optional[pd.DataFrame] = None
    file_source_name = ""

    if use_sample:
        # Load a built-in clean sample dataset so beginners can test instantly
        file_source_name = "sample_iris_dataset.csv"
        sample_data = {
            "sepal_length": [5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.4, 4.9, 5.4, 4.8],
            "sepal_width": [3.5, 3.0, 3.2, 3.1, 3.6, 3.9, 3.4, 3.4, 2.9, 3.1, 3.7, 3.4],
            "petal_length": [1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 1.5, 1.4, 1.5, 1.5, 1.6],
            "petal_width": [0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1, 0.2, 0.2],
            "species": ["setosa", "setosa", "setosa", "setosa", "setosa", "setosa",
                        "setosa", "setosa", "setosa", "setosa", "setosa", "setosa"]
        }
        df = pd.DataFrame(sample_data)
        st.info("ℹ️ Using built-in sample dataset for demonstration.")

    elif uploaded_file is not None:
        file_source_name = uploaded_file.name
        # Use our modular data loader function
        with st.spinner(f"Loading '{file_source_name}'..."):
            df, error = load_dataset(uploaded_file)

        # Handle errors gracefully as requested in Requirement 8
        if error:
            st.error(f"❌ **Failed to load file:** {error}")
            st.info("💡 **Tips to resolve this issue:**\n"
                    "- Ensure the file is not empty.\n"
                    "- Ensure the file is a valid CSV or Excel spreadsheet.\n"
                    "- Check that the file isn't corrupted or password-protected.")
            return

    if df is not None:
        # Store in session state for cross-module usage (Phase 2 Data Cleaning)
        if st.session_state.get("dataset_name") != file_source_name or "raw_df" not in st.session_state:
            st.session_state["raw_df"] = df.copy()
            st.session_state["cleaned_df"] = df.copy()
            st.session_state["dataset_name"] = file_source_name
            st.session_state["cleaning_history"] = []
    elif "raw_df" in st.session_state and st.session_state["raw_df"] is not None:
        # Use existing dataset from session state if user navigated between tabs
        df = st.session_state["raw_df"]
        file_source_name = st.session_state.get("dataset_name", "dataset")

    # If no file is selected yet, display a helpful welcoming card
    if df is None:
        st.markdown("---")
        st.info("👆 **Get started:** Upload a CSV/Excel file above or toggle the sample dataset checkbox.")
        return

    # --------------------------------------------------------------------------
    # Step 2: High-Level Overview Metrics (Requirements 5 & 7)
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.subheader(f"📁 Dataset Overview: `{file_source_name}`")

    # Calculate key dataset metrics
    num_rows, num_cols = df.shape
    total_cells = num_rows * num_cols
    total_missing = int(df.isnull().sum().sum())
    missing_ratio = (total_missing / total_cells * 100) if total_cells > 0 else 0
    duplicate_rows = int(df.duplicated().sum())

    # Format memory usage
    memory_bytes = df.memory_usage(deep=True).sum()
    if memory_bytes < 1024 * 1024:
        memory_str = f"{memory_bytes / 1024:.2f} KB"
    else:
        memory_str = f"{memory_bytes / (1024 * 1024):.2f} MB"

    # Display key metrics in neat cards using st.columns and st.metric
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(label="Total Rows", value=f"{num_rows:,}")
    with col2:
        st.metric(label="Total Columns", value=f"{num_cols:,}")
    with col3:
        st.metric(label="Missing Values", value=f"{total_missing:,}", delta=f"{missing_ratio:.1f}%", delta_color="inverse")
    with col4:
        st.metric(label="Duplicate Rows", value=f"{duplicate_rows:,}")
    with col5:
        st.metric(label="Memory Usage", value=memory_str)

    # --------------------------------------------------------------------------
    # Step 3: Interactive Tabs for Detailed Exploration (Requirements 4, 6, 7)
    # --------------------------------------------------------------------------
    tab_preview, tab_columns, tab_stats, tab_learn = st.tabs([
        "🔍 Data Preview (First 10 Rows)",
        "📋 Column Types & Missing Values",
        "📈 Statistical Summary",
        "💡 Learning Zone"
    ])

    # TAB 1: Preview Data
    with tab_preview:
        st.markdown("#### First 10 Rows of the Dataset")
        st.caption("Here is a quick snapshot of the top of your dataset (`df.head(10)`).")
        
        # Display the first 10 rows
        preview_rows = df.head(10)
        st.dataframe(preview_rows, use_container_width=True)

        with st.expander("⚙️ Customize Preview Row Count"):
            max_slider = min(100, max(10, num_rows))
            row_count = st.slider("Select number of rows to display:", min_value=5, max_value=max_slider, value=10, step=5)
            if row_count != 10:
                st.dataframe(df.head(row_count), use_container_width=True)

    # TAB 2: Column Names & Data Types
    with tab_columns:
        st.markdown("#### Column Information & Data Types")
        st.caption("Detailed breakdown of each column's name, inferred Pandas data type, and completeness.")
        
        col_summary_df = get_column_summary(df)
        st.dataframe(col_summary_df, use_container_width=True)

        # Quick breakdown of column data types
        st.markdown("##### Data Type Distribution")
        dtype_counts = df.dtypes.astype(str).value_counts().reset_index()
        dtype_counts.columns = ["Data Type", "Count"]
        st.dataframe(dtype_counts, use_container_width=True)

    # TAB 3: Basic Dataset Information & Summary Statistics
    with tab_stats:
        st.markdown("#### Basic Dataset Information")

        # Numerical summary using df.describe()
        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            st.markdown("##### Numerical Features Summary (`df.describe()`)")
            st.caption("Shows count, mean, standard deviation, minimum, quartiles (25%, 50%, 75%), and maximum.")
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        else:
            st.info("No numerical columns found in this dataset.")

        # Categorical / Text summary
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        if len(categorical_cols) > 0:
            st.markdown("##### Categorical / Text Features Summary")
            st.caption("Shows count, unique categories, top (most frequent) value, and frequency.")
            st.dataframe(df[categorical_cols].describe(), use_container_width=True)

    # TAB 4: Educational / Beginner Learning Zone
    with tab_learn:
        st.markdown("#### 🎓 Beginner's Guide: Understanding Data Concepts")
        st.markdown("""
        Here is what each key concept means in Python and Pandas:

        1. **DataFrame (`pd.DataFrame`)**:
           - A 2-dimensional labeled data structure with columns of potentially different types (like an Excel spreadsheet or SQL table).
        
        2. **`df.shape`**:
           - A tuple containing `(number_of_rows, number_of_columns)`. For example, `(150, 5)` means 150 rows and 5 columns.
        
        3. **`df.head(10)`**:
           - Returns the first 10 rows of the dataset. This allows you to quickly inspect column contents without loading all data.
        
        4. **Common Pandas Data Types (`dtypes`)**:
           - `int64`: Whole numbers (e.g., 10, 42, -5).
           - `float64`: Numbers with decimals (e.g., 3.14, 0.05).
           - `object`: Text strings or mixed data types (e.g., 'setosa', 'John Doe').
           - `datetime64`: Dates and timestamps.
           - `bool`: True or False flags.
        
        5. **Missing Values (`NaN` / `null`)**:
           - Represents missing or unrecorded information. Knowing missing counts helps decide whether to drop rows or fill them with averages (imputation).
        """)

    # --------------------------------------------------------------------------
    # Cloud Save Section (Guest-friendly optional permanent cloud save)
    # --------------------------------------------------------------------------
    if df is not None and not df.empty:
        from utils.smart_save import build_dataset_payload
        from components.auth_ui import render_save_section
        payload = build_dataset_payload(is_cleaned=False)
        render_save_section("dataset", payload, key_suffix="explorer")
