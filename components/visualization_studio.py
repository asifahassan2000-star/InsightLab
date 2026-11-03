"""
visualization_studio.py
=======================
This module renders the 'Interactive Visualization Studio' in InsightLab (Phase 4).

Features:
---------
1. Six Core Chart Types:
   - Bar Charts (category comparisons & frequency counts)
   - Line Charts (trends over time or sequence)
   - Pie & Donut Charts (composition & proportions)
   - Histograms (continuous numerical distributions)
   - Scatter Plots (relationships between two numerical variables + trendline)
   - Correlation Heatmaps (multivariate matrix)
2. Interactive Filtering:
   - Filter by categorical subsets or numerical ranges before plotting.
3. Customization & Styling:
   - Custom chart titles, X/Y axis labels, Plotly themes, and color palettes.
4. Export & Download:
   - Download as self-contained Interactive HTML or underlying plotted data as CSV.
5. Defensive Data Type Validation:
   - Prevents nonsensical charts (e.g. text on scatter plot axes).
"""

from typing import Optional, List
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import load_dataset
from utils.visualizer import (
    validate_chart_inputs,
    build_bar_chart,
    build_line_chart,
    build_pie_chart,
    build_histogram,
    build_scatter_plot,
    build_correlation_heatmap,
    export_figure_to_html,
    PLOTLY_THEMES,
    COLOR_PALETTES
)


def render_visualization_studio():
    """Renders the Visualization Studio UI."""
    st.markdown("## 🎨 Interactive Visualization Studio")
    st.markdown(
        "Generate interactive charts dynamically from your data. "
        "Customize axes, apply filters, examine relationships, and export standalone visualizations."
    )

    # --------------------------------------------------------------------------
    # Step 1: Dataset Version & Loading
    # --------------------------------------------------------------------------
    has_raw = "raw_df" in st.session_state and st.session_state["raw_df"] is not None
    has_cleaned = "cleaned_df" in st.session_state and st.session_state["cleaned_df"] is not None

    if not has_raw and not has_cleaned:
        st.info("ℹ️ No dataset is currently loaded. Please upload a dataset or use the sample dataset below.")
        uploaded_file = st.file_uploader(
            label="Upload dataset for visualization",
            type=["csv", "xlsx", "xls"],
            key="vis_uploader"
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

    # Choose dataset version
    dataset_name = st.session_state.get("dataset_name", "dataset")
    col_name, col_ver = st.columns([3, 2])
    with col_name:
        st.markdown(f"**Active Dataset:** `{dataset_name}`")
    with col_ver:
        if has_cleaned and has_raw:
            data_version = st.radio(
                "Data Source:",
                options=["Cleaned Working Dataset (Phase 2)", "Original Raw Dataset (Phase 1)"],
                horizontal=True
            )
            df = st.session_state["cleaned_df"] if "Cleaned" in data_version else st.session_state["raw_df"]
        else:
            df = st.session_state.get("cleaned_df", st.session_state.get("raw_df"))

    # Extract column groups
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    all_cols = df.columns.tolist()

    # --------------------------------------------------------------------------
    # Step 2: Interactive Data Filter (Requirement 9)
    # --------------------------------------------------------------------------
    working_df = df.copy()
    with st.expander("🔍 Interactive Data Filters (Optional)", expanded=False):
        st.caption("Filter data before charting. Charts will automatically reflect your filtered subset.")
        enable_filter = st.checkbox("Enable filtering on a column", value=False)

        if enable_filter:
            filter_col = st.selectbox("Choose column to filter by:", options=all_cols, key="flt_col")
            if filter_col in categorical_cols:
                unique_vals = sorted(df[filter_col].dropna().unique().tolist())
                selected_vals = st.multiselect(
                    f"Select values of '{filter_col}':",
                    options=unique_vals,
                    default=unique_vals,
                    key="flt_cat_vals"
                )
                if selected_vals:
                    working_df = working_df[working_df[filter_col].isin(selected_vals)]
                else:
                    st.warning("No categories selected. Filter ignored.")
            elif filter_col in numeric_cols:
                min_val = float(df[filter_col].min())
                max_val = float(df[filter_col].max())
                if min_val < max_val:
                    num_range = st.slider(
                        f"Range for '{filter_col}':",
                        min_value=min_val,
                        max_value=max_val,
                        value=(min_val, max_val),
                        key="flt_num_range"
                    )
                    working_df = working_df[(working_df[filter_col] >= num_range[0]) & (working_df[filter_col] <= num_range[1])]

            st.info(f"📊 Filter active: Plotting **{len(working_df)}** of **{len(df)}** rows.")

    if working_df.empty:
        st.warning("⚠️ No data matches your active filter. Please adjust your filter criteria.")
        return

    # --------------------------------------------------------------------------
    # Step 3: Chart Builder Controls
    # --------------------------------------------------------------------------
    st.markdown("---")
    col_config, col_canvas = st.columns([1, 2], gap="large")

    with col_config:
        st.subheader("⚙️ Chart Configuration")

        # 1. Select Chart Type
        chart_type = st.selectbox(
            "1. Select Chart Type:",
            options=[
                "Bar Chart",
                "Line Chart",
                "Pie Chart",
                "Histogram",
                "Scatter Plot",
                "Correlation Heatmap"
            ],
            help="Choose the visual representation that best matches your analytical question."
        )

        st.markdown("---")
        st.markdown("##### 2. Map Columns")

        x_col = None
        y_col = None
        color_col = None
        size_col = None
        aggregation = "Sum"
        is_donut = False
        nbins = 20
        marginal = "box"
        add_trendline = False
        selected_corr_cols = []

        # DYNAMIC COLUMN SELECTION BASED ON CHART TYPE
        if chart_type == "Bar Chart":
            x_col = st.selectbox("Category (X-Axis):", options=all_cols, index=0)
            agg_mode = st.radio("Calculation:", ["Metric Aggregation (Sum / Mean)", "Row Count (Frequency)"], horizontal=True)

            if "Aggregation" in agg_mode:
                if numeric_cols:
                    y_col = st.selectbox("Metric Value (Y-Axis):", options=numeric_cols, index=0)
                    aggregation = st.selectbox("Aggregation Function:", options=["Sum", "Mean", "Median"])
                else:
                    st.warning("No numerical columns available. Defaulting to Row Count.")
                    aggregation = "Count"
            else:
                aggregation = "Count"

            color_choice = st.selectbox("Group by Color (Optional):", options=["None"] + categorical_cols)
            color_col = color_choice if color_choice != "None" else None

        elif chart_type == "Line Chart":
            x_col = st.selectbox("X-Axis (Date or Ordered Sequence):", options=all_cols, index=0)
            if numeric_cols:
                y_col = st.selectbox("Y-Axis (Numerical Metric):", options=numeric_cols, index=0)
            color_choice = st.selectbox("Group by Line (Optional):", options=["None"] + categorical_cols)
            color_col = color_choice if color_choice != "None" else None

        elif chart_type == "Pie Chart":
            x_col = st.selectbox("Category Slices:", options=categorical_cols if categorical_cols else all_cols)
            val_choice = st.radio("Slice Size Based On:", ["Metric Sum", "Row Count (Frequency)"], horizontal=True)
            if val_choice == "Metric Sum" and numeric_cols:
                y_col = st.selectbox("Value Column:", options=numeric_cols)
            is_donut = st.checkbox("Render as Donut Chart (Hollow Center)", value=False)

        elif chart_type == "Histogram":
            if numeric_cols:
                x_col = st.selectbox("Numerical Column (X-Axis):", options=numeric_cols, index=0)
            nbins = st.slider("Number of Bins:", min_value=5, max_value=50, value=20, step=5)
            marginal = st.selectbox("Marginal Distribution Plot:", options=["box", "violin", "rug", "none"])
            color_choice = st.selectbox("Group by Color (Optional):", options=["None"] + categorical_cols)
            color_col = color_choice if color_choice != "None" else None

        elif chart_type == "Scatter Plot":
            if len(numeric_cols) >= 2:
                x_col = st.selectbox("X-Axis Variable (Independent):", options=numeric_cols, index=0)
                y_col = st.selectbox("Y-Axis Variable (Dependent):", options=numeric_cols, index=min(1, len(numeric_cols) - 1))
            elif len(numeric_cols) == 1:
                x_col = numeric_cols[0]
                y_col = numeric_cols[0]

            color_choice = st.selectbox("Color Dimension (Optional):", options=["None"] + categorical_cols)
            color_col = color_choice if color_choice != "None" else None

            size_choice = st.selectbox("Bubble Size Dimension (Optional):", options=["None"] + numeric_cols)
            size_col = size_choice if size_choice != "None" else None

            add_trendline = st.checkbox("Overlay Linear Trendline (OLS)", value=True)

        elif chart_type == "Correlation Heatmap":
            selected_corr_cols = st.multiselect(
                "Select Numerical Columns:",
                options=numeric_cols,
                default=numeric_cols,
                key="vis_corr_cols"
            )
            corr_method = st.selectbox("Correlation Method:", options=["pearson", "spearman"])
            corr_colorscale = st.selectbox("Color Scale:", options=["RdBu_r", "Viridis", "Cividis", "Blues"])

        st.markdown("---")
        st.markdown("##### 3. Styling & Labels")
        custom_title = st.text_input("Chart Title (Optional):", value="")
        custom_xlabel = st.text_input("X-Axis Label (Optional):", value="")
        custom_ylabel = st.text_input("Y-Axis Label (Optional):", value="")

        theme_col, pal_col = st.columns(2)
        with theme_col:
            selected_theme = st.selectbox("Theme:", options=PLOTLY_THEMES, index=0)
        with pal_col:
            selected_palette_name = st.selectbox("Palette:", options=list(COLOR_PALETTES.keys()), index=0)
            selected_palette = COLOR_PALETTES[selected_palette_name]

    # --------------------------------------------------------------------------
    # Step 4: Canvas & Rendering (Right Column)
    # --------------------------------------------------------------------------
    with col_canvas:
        st.subheader("🖼️ Interactive Canvas")

        # Validate inputs before generating chart
        is_valid, validation_msg = validate_chart_inputs(
            chart_type=chart_type,
            df=working_df,
            x_col=x_col,
            y_col=y_col,
            numeric_cols=numeric_cols
        )

        if not is_valid:
            st.error(f"⚠️ **Incompatible Selection:** {validation_msg}")
            st.info("💡 **Tip:** Adjust your column choices in the left panel to match the requirements for this chart type.")
            return

        # Generate the figure
        fig: Optional[go.Figure] = None

        if chart_type == "Bar Chart":
            fig = build_bar_chart(
                df=working_df,
                x_col=x_col,
                y_col=y_col,
                color_col=color_col,
                aggregation=aggregation,
                title=custom_title,
                x_label=custom_xlabel,
                y_label=custom_ylabel,
                theme=selected_theme,
                color_sequence=selected_palette
            )

        elif chart_type == "Line Chart":
            fig = build_line_chart(
                df=working_df,
                x_col=x_col,
                y_col=y_col,
                color_col=color_col,
                title=custom_title,
                x_label=custom_xlabel,
                y_label=custom_ylabel,
                theme=selected_theme,
                color_sequence=selected_palette
            )

        elif chart_type == "Pie Chart":
            fig = build_pie_chart(
                df=working_df,
                names_col=x_col,
                values_col=y_col,
                is_donut=is_donut,
                title=custom_title,
                theme=selected_theme,
                color_sequence=selected_palette
            )

        elif chart_type == "Histogram":
            fig = build_histogram(
                df=working_df,
                x_col=x_col,
                color_col=color_col,
                nbins=nbins,
                marginal=marginal,
                title=custom_title,
                x_label=custom_xlabel,
                y_label=custom_ylabel or "Frequency",
                theme=selected_theme,
                color_sequence=selected_palette
            )

        elif chart_type == "Scatter Plot":
            fig = build_scatter_plot(
                df=working_df,
                x_col=x_col,
                y_col=y_col,
                color_col=color_col,
                size_col=size_col,
                add_trendline=add_trendline,
                title=custom_title,
                x_label=custom_xlabel,
                y_label=custom_ylabel,
                theme=selected_theme,
                color_sequence=selected_palette
            )

        elif chart_type == "Correlation Heatmap":
            fig = build_correlation_heatmap(
                df=working_df,
                columns=selected_corr_cols,
                method=corr_method,
                colorscale=corr_colorscale,
                title=custom_title,
                theme=selected_theme
            )

        if fig is not None:
            # Render interactive Plotly chart
            st.plotly_chart(fig, use_container_width=True)

            # ------------------------------------------------------------------
            # Step 5: Export & Download Options (Requirement 10)
            # ------------------------------------------------------------------
            st.markdown("#### 💾 Export & Download")
            st.caption(
                "Export your visualization. You can also hover over the top-right of the chart "
                "and click the 📷 camera icon to save a high-resolution PNG image."
            )

            col_exp_html, col_exp_data = st.columns(2)

            with col_exp_html:
                html_bytes = export_figure_to_html(fig).encode("utf-8")
                safe_title = (custom_title or f"{chart_type.lower()}_chart").replace(" ", "_")
                st.download_button(
                    label="🌐 Download Interactive Chart (.html)",
                    data=html_bytes,
                    file_name=f"{safe_title}.html",
                    mime="text/html",
                    use_container_width=True,
                    help="Downloads a standalone HTML file. Double-click to open in any web browser with full interactive tooltips and zoom!"
                )

            with col_exp_data:
                # Prepare plotted data download
                csv_data = working_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📄 Download Plotted Data (.csv)",
                    data=csv_data,
                    file_name=f"{safe_title}_data.csv",
                    mime="text/csv",
                    use_container_width=True,
                    help="Downloads the exact filtered dataset used to generate this visualization."
                )

    # --------------------------------------------------------------------------
    # Step 6: Learning Zone (Beginner Teacher Guide)
    # --------------------------------------------------------------------------
    st.markdown("---")
    with st.expander("🎓 Visualization Guide: Choosing the Right Chart Type"):
        st.markdown("""
        ### When Should You Use Each Chart Type?

        | Chart Type | Best Used For | Example Question It Answers | Common Mistake to Avoid |
        | :--- | :--- | :--- | :--- |
        | **Bar Chart** | Comparing quantities across discrete categories | *"Which department has the highest total payroll?"* | Truncating the Y-axis baseline away from zero |
        | **Line Chart** | Tracking continuous metrics over time or sequence | *"How have company profits trended month by month?"* | Using for unordered categories (e.g. connecting different products with a line) |
        | **Pie / Donut Chart** | Showing proportions of a whole (must sum to 100%) | *"What percentage of total revenue came from Electronics?"* | Using more than 7 slices; pie slices become impossible to compare accurately |
        | **Histogram** | Viewing the spread and frequency of continuous numbers | *"Are employee salaries normally distributed or right-skewed?"* | Confusing histograms with bar charts (histograms bin continuous data) |
        | **Scatter Plot** | Detecting relationships or correlations between two numbers | *"Does higher advertising spending correlate with higher sales?"* | Plotting categorical text values on continuous axes |
        | **Correlation Heatmap** | Scanning an entire dataset for multivariate relationships | *"Which metrics move in tandem across our business?"* | Assuming correlation implies causation |
        """)
