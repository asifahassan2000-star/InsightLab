"""
statistical_analysis.py
=======================
This module renders the 'Statistical Analysis Studio' page in InsightLab (Phase 3).

Key Features:
-------------
1. Descriptive Statistics:
   - Central Tendency: Mean, Median, Mode
   - Spread & Dispersion: Minimum, Maximum, Variance, Standard Deviation, IQR
2. Distribution Analysis:
   - Interactive Plotly Histograms with marginal Box plots
   - Skewness and Kurtosis metrics with verbal interpretations
3. Correlation Analysis:
   - Interactive Plotly Heatmap (Pearson & Spearman)
   - Bivariate scatter plot with trendlines
   - Ranked pairwise correlation interpretations
4. Category & Department Comparisons:
   - GroupBy aggregations comparing metrics across categorical segments
   - Interactive grouped bar charts and box plots
"""

from typing import Optional, List
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import load_dataset
from utils.statistical_analyzer import (
    compute_column_statistics,
    compute_dataset_statistics_table,
    compute_correlation_matrix,
    get_top_correlations,
    interpret_skewness,
    compute_group_comparison
)


def render_statistical_analysis():
    """
    Renders the Statistical Analysis Studio view.
    """
    st.markdown("## 📈 Statistical Analysis Studio")
    st.markdown(
        "Analyze central tendencies, explore dispersion and distribution shapes, "
        "measure correlations between variables, and compare metrics across categories."
    )

    # --------------------------------------------------------------------------
    # Step 1: Dataset Selection (Session State or Inline Upload)
    # --------------------------------------------------------------------------
    has_raw = "raw_df" in st.session_state and st.session_state["raw_df"] is not None
    has_cleaned = "cleaned_df" in st.session_state and st.session_state["cleaned_df"] is not None

    if not has_raw and not has_cleaned:
        st.info("ℹ️ No dataset is currently loaded. Upload a CSV/Excel file or use the sample dataset below to begin.")
        uploaded_file = st.file_uploader(
            label="Upload dataset for statistical analysis",
            type=["csv", "xlsx", "xls"],
            key="stats_uploader"
        )
        col_sample, _ = st.columns([2, 5])
        with col_sample:
            use_sample = st.checkbox("🧪 Use sample sales dataset", value=False)

        if use_sample:
            sample_df = pd.DataFrame({
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

    # Choose dataset version (Cleaned copy or Raw original)
    dataset_name = st.session_state.get("dataset_name", "dataset")
    col_info, col_version = st.columns([3, 2])
    with col_info:
        st.markdown(f"**Active Dataset:** `{dataset_name}`")

    with col_version:
        if has_cleaned and has_raw:
            data_version = st.radio(
                "Select Data Version:",
                options=["Cleaned Working Dataset (Phase 2)", "Original Raw Dataset (Phase 1)"],
                horizontal=True
            )
            df = st.session_state["cleaned_df"] if "Cleaned" in data_version else st.session_state["raw_df"]
        else:
            df = st.session_state.get("cleaned_df", st.session_state.get("raw_df"))

    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    if not numeric_cols:
        st.warning("⚠️ No numerical columns found in this dataset. Statistical analysis requires numerical data.")
        return

    # --------------------------------------------------------------------------
    # Step 2: Interactive Tabs
    # --------------------------------------------------------------------------
    tab_summary, tab_dist, tab_corr, tab_groups, tab_learn = st.tabs([
        "📋 1. Summary Table",
        "📊 2. Distributions & Dispersion",
        "🔗 3. Correlation Matrix",
        "👥 4. Group Comparisons",
        "💡 5. Learning Zone"
    ])

    # --------------------------------------------------------------------------
    # TAB 1: SUMMARY TABLE (Requirements 1, 2, 3, 7)
    # --------------------------------------------------------------------------
    with tab_summary:
        st.subheader("Statistical Summary Table")
        st.caption("A centralized overview of central tendencies, dispersion, and distribution symmetry for all numerical features.")

        # Multi-select to filter columns
        selected_cols = st.multiselect(
            "Filter numerical columns to display:",
            options=numeric_cols,
            default=numeric_cols,
            key="summary_table_cols"
        )

        if not selected_cols:
            st.info("Select at least one column to view the summary table.")
        else:
            summary_table = compute_dataset_statistics_table(df, columns=selected_cols)
            st.dataframe(summary_table, use_container_width=True)

            with st.expander("📖 Metric Definitions & Formulas"):
                st.markdown(r"""
                - **Mean**: The arithmetic average ($\sum x / n$).
                - **Median**: The middle value separating the upper and lower halves.
                - **Mode**: The most frequently occurring value in the column.
                - **Variance**: Average squared deviation from the mean ($s^2$).
                - **Std Dev**: Square root of variance ($s = \sqrt{s^2}$), expressed in original units.
                - **Q1 / Q3**: 25th and 75th percentiles (first and third quartiles).
                - **Skewness**: Measure of distribution asymmetry (0 = symmetric, >0 = right tail, <0 = left tail).
                """)

    # --------------------------------------------------------------------------
    # TAB 2: DISTRIBUTIONS & DISPERSION (Requirements 1, 2, 3, 5)
    # --------------------------------------------------------------------------
    with tab_dist:
        st.subheader("Univariate Distribution & Dispersion Analysis")
        st.caption("Select a column to inspect its distribution curve, spread, quartiles, and outliers.")

        col_target = st.selectbox("Select numerical column to inspect:", options=numeric_cols, key="dist_col")
        stats = compute_column_statistics(df[col_target])

        # KPI Metrics Cards (Requirements 1, 2, 3)
        st.markdown(f"#### Key Statistics for `{col_target}`")
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Mean", f"{stats['mean']:,}")
            st.metric("Min Value", f"{stats['min']:,}")
        with k2:
            st.metric("Median", f"{stats['median']:,}")
            st.metric("Max Value", f"{stats['max']:,}")
        with k3:
            st.metric("Mode", f"{stats['mode']}" if stats['mode'] is not None else "N/A")
            st.metric("Range (Max - Min)", f"{stats['range']:,}")
        with k4:
            st.metric("Std Deviation (s)", f"{stats['std_dev']:,}")
            st.metric("Variance (s²)", f"{stats['variance']:,}")

        # Skewness Alert
        skew_val = stats["skewness"]
        skew_msg = interpret_skewness(skew_val)
        if abs(skew_val) < 0.5:
            st.success(f"⚖️ **Skewness = {skew_val}:** {skew_msg}")
        else:
            st.info(f"📊 **Skewness = {skew_val}:** {skew_msg}")

        st.markdown("---")

        # Interactive Plotly Histogram and Box Plot (Requirement 5)
        st.markdown("#### Interactive Distribution Visualizations")
        col_plot1, col_plot2 = st.columns(2)

        # Drop NaN for plotting
        plot_series = df[col_target].dropna()

        with col_plot1:
            st.markdown("##### Histogram & Box Marginal")
            fig_hist = px.histogram(
                df,
                x=col_target,
                nbins=20,
                marginal="box",
                title=f"Frequency Distribution: {col_target}",
                color_discrete_sequence=["#1f77b4"]
            )
            fig_hist.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_plot2:
            st.markdown("##### Box Plot (Quartiles & Outliers)")
            fig_box = px.box(
                df,
                y=col_target,
                points="all",
                title=f"Box & Whisker Plot: {col_target}",
                color_discrete_sequence=["#2ca02c"]
            )
            fig_box.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_box, use_container_width=True)

    # --------------------------------------------------------------------------
    # TAB 3: CORRELATION ANALYSIS (Requirement 4)
    # --------------------------------------------------------------------------
    with tab_corr:
        st.subheader("Correlation Analysis")
        st.caption("Measure the strength and direction of relationships between numerical variables.")

        if len(numeric_cols) < 2:
            st.info("At least two numerical columns are required for correlation analysis.")
        else:
            col_method, col_corr_cols = st.columns([1, 2])
            with col_method:
                corr_method = st.selectbox(
                    "Correlation Method:",
                    options=["pearson", "spearman"],
                    format_func=lambda x: "Pearson (Linear)" if x == "pearson" else "Spearman (Rank Monotonic)"
                )
            with col_corr_cols:
                corr_selected = st.multiselect(
                    "Select columns for correlation matrix:",
                    options=numeric_cols,
                    default=numeric_cols,
                    key="corr_multiselect"
                )

            if len(corr_selected) >= 2:
                corr_matrix = compute_correlation_matrix(df, columns=corr_selected, method=corr_method)

                # Interactive Plotly Heatmap
                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    zmin=-1,
                    zmax=1,
                    title=f"{corr_method.capitalize()} Correlation Heatmap"
                )
                fig_corr.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_corr, use_container_width=True)

                # Top Correlated Pairs Breakdown Table
                st.markdown("#### Ranked Variable Relationships")
                top_pairs = get_top_correlations(corr_matrix)

                if top_pairs:
                    pairs_df = pd.DataFrame(top_pairs)[["feature_1", "feature_2", "correlation", "interpretation"]]
                    pairs_df.columns = ["Variable 1", "Variable 2", "Correlation (r)", "Interpretation"]
                    st.dataframe(pairs_df, use_container_width=True)

                st.markdown("---")

                # Bivariate Scatter Plot with Trendline
                st.markdown("#### Bivariate Scatter Plot & Trendline")
                sc_col1, sc_col2 = st.columns(2)
                with sc_col1:
                    x_col = st.selectbox("X-axis variable:", options=corr_selected, index=0)
                with sc_col2:
                    y_index = 1 if len(corr_selected) > 1 else 0
                    y_col = st.selectbox("Y-axis variable:", options=corr_selected, index=y_index)

                if x_col != y_col:
                    valid_scatter = df.dropna(subset=[x_col, y_col])
                    fig_scatter = px.scatter(
                        valid_scatter,
                        x=x_col,
                        y=y_col,
                        title=f"{y_col} vs. {x_col}",
                        color_discrete_sequence=["#ff7f0e"]
                    )
                    # Add linear regression trendline using NumPy (pure, no statsmodels required)
                    if len(valid_scatter) > 1 and valid_scatter[x_col].nunique() > 1:
                        slope, intercept = np.polyfit(valid_scatter[x_col], valid_scatter[y_col], 1)
                        x_range = np.linspace(valid_scatter[x_col].min(), valid_scatter[x_col].max(), 50)
                        y_trend = slope * x_range + intercept
                        fig_scatter.add_trace(
                            go.Scatter(
                                x=x_range,
                                y=y_trend,
                                mode="lines",
                                name=f"Fit: y = {slope:.2f}x + {intercept:.2f}",
                                line=dict(color="#d62728", dash="dash")
                            )
                        )
                    fig_scatter.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("Please select at least 2 numerical columns.")

    # --------------------------------------------------------------------------
    # TAB 4: GROUP & CATEGORY COMPARISONS (Requirement 6)
    # --------------------------------------------------------------------------
    with tab_groups:
        st.subheader("Category-Wise & Department-Wise Comparisons")
        st.caption("Compare how a numerical metric behaves across different categorical groups or departments.")

        if not categorical_cols:
            st.warning("⚠️ No categorical or text columns found in this dataset to group by.")
        else:
            col_grp, col_val = st.columns(2)
            with col_grp:
                selected_group = st.selectbox(
                    "Group By (Category / Department):",
                    options=categorical_cols,
                    key="grp_by_col"
                )
            with col_val:
                selected_val = st.selectbox(
                    "Numerical Metric to Compare:",
                    options=numeric_cols,
                    key="grp_val_col"
                )

            # Compute grouped table
            grouped_df = compute_group_comparison(df, group_col=selected_group, value_col=selected_val)

            if grouped_df.empty:
                st.info("No valid data available for this comparison.")
            else:
                st.markdown(f"#### Grouped Summary: `{selected_val}` by `{selected_group}`")
                st.dataframe(grouped_df, use_container_width=True)

                # Visual comparisons using Plotly
                st.markdown("#### Interactive Comparison Visualizations")
                col_gplot1, col_gplot2 = st.columns(2)

                with col_gplot1:
                    st.markdown("##### Mean Comparison (Bar Chart)")
                    fig_bar = px.bar(
                        grouped_df,
                        x=selected_group,
                        y="Mean",
                        text_auto=True,
                        title=f"Average {selected_val} by {selected_group}",
                        color=selected_group
                    )
                    fig_bar.update_layout(template="plotly_white", showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_bar, use_container_width=True)

                with col_gplot2:
                    st.markdown("##### Full Distribution by Group (Box Plot)")
                    fig_grp_box = px.box(
                        df.dropna(subset=[selected_group, selected_val]),
                        x=selected_group,
                        y=selected_val,
                        color=selected_group,
                        title=f"{selected_val} Distribution Across {selected_group}"
                    )
                    fig_grp_box.update_layout(template="plotly_white", showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_grp_box, use_container_width=True)

    # --------------------------------------------------------------------------
    # TAB 5: LEARNING ZONE (Formulas & Concepts)
    # --------------------------------------------------------------------------
    with tab_learn:
        st.markdown("#### 🎓 Beginner's Guide: Statistical Formulas & Intuition")
        st.markdown(r"""
        ### 1. Measures of Central Tendency
        Where does the "center" of your data lie?

        - **Arithmetic Mean ($\bar{x}$)**:
          $$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$$
          The mathematical balance point. Sensitive to extreme values.

        - **Median**:
          The middle value when data is sorted in ascending order.
          If $n$ is odd, it's the element at position $\frac{n+1}{2}$. If $n$ is even, it's the average of the two middle elements.

        - **Mode**:
          The most frequent observation. A dataset can be unimodal, bimodal, or multimodal.

        ---

        ### 2. Measures of Dispersion (Spread)
        How far do individual data points spread out from the center?

        - **Sample Variance ($s^2$)**:
          $$s^2 = \frac{1}{n - 1}\sum_{i=1}^{n} (x_i - \bar{x})^2$$
          *Note on $n-1$ (Bessel's Correction):* We divide by $n-1$ instead of $n$ to provide an unbiased estimate of the true population variance from a sample.

        - **Sample Standard Deviation ($s$)**:
          $$s = \sqrt{s^2}$$
          Brings the variance back into the original units of measurement.

        ---

        ### 3. Pearson Correlation Coefficient ($r$)
        Measures the linear association between two variables $X$ and $Y$:
        $$r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}$$
        - $r = +1.0$: Perfect positive linear correlation.
        - $r = 0.0$: No linear correlation.
        - $r = -1.0$: Perfect negative linear correlation.

        ---

        ### 4. Skewness
        Measures the asymmetry of the probability distribution:
        - **Symmetric ($\approx 0$)**: Mean $\approx$ Median.
        - **Right-Skewed ($> 0$)**: Long right tail. Mean $>$ Median (e.g., incomes, wealth).
        - **Left-Skewed ($< 0$)**: Long left tail. Mean $<$ Median (e.g., age at death, retirement age).
        """)

    # --------------------------------------------------------------------------
    # Cloud Save Section (Guest-friendly optional permanent cloud save)
    # --------------------------------------------------------------------------
    from utils.smart_save import build_analysis_payload
    from components.auth_ui import render_save_section
    analysis_payload = build_analysis_payload("statistical_analysis")
    render_save_section("analysis", analysis_payload, key_suffix="stats")
