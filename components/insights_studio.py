"""
insights_studio.py
==================
This module renders the 'Automated Insights Studio' in InsightLab (Phase 5).

Key Principles:
---------------
1. Deterministic Statistical Analysis:
   Insights are computed strictly using Pandas and NumPy, avoiding fabricated AI claims.
2. Insight Cards:
   Structured cards displaying:
   - Badge & Title
   - Key Quantitative Finding
   - Simple Plain-English Explanation
   - Relevant Column Names
   - Embedded Plotly Visualizations
3. Non-Destructive:
   Flags anomalies and missing patterns without modifying the active dataset.
"""

from typing import Optional, List, Dict, Any
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import load_dataset
from utils.visualizer import build_correlation_heatmap
from utils.insights_engine import (
    generate_automated_insights,
    generate_insights_markdown_report
)


def render_insights_studio():
    """Renders the Automated Insights Studio UI."""
    st.markdown("## 💡 Automated Insights Studio")
    st.markdown(
        "Automatically scan your dataset to uncover hidden patterns, detect statistical anomalies, "
        "measure relationships between variables, and receive actionable data recommendations."
    )

    # --------------------------------------------------------------------------
    # Step 1: Dataset Version & Loading
    # --------------------------------------------------------------------------
    has_raw = "raw_df" in st.session_state and st.session_state["raw_df"] is not None
    has_cleaned = "cleaned_df" in st.session_state and st.session_state["cleaned_df"] is not None

    if not has_raw and not has_cleaned:
        st.info("ℹ️ No dataset is currently loaded. Please upload a dataset or use the sample dataset below to generate insights.")
        uploaded_file = st.file_uploader(
            label="Upload dataset for automated insights",
            type=["csv", "xlsx", "xls"],
            key="insights_uploader"
        )
        col_sample, _ = st.columns([2, 5])
        with col_sample:
            use_sample = st.checkbox("🧪 Use sample sales dataset", value=False)

        if use_sample:
            sample_df = pd.DataFrame({
                "date": ["2026-01-15", "2026-01-16", "2026-01-17", "2026-01-18", "2026-01-19", "2026-01-20", "2026-01-21", "2026-01-22", "2026-01-23", "2026-01-24"],
                "category": ["Electronics", "Furniture", "Office Supplies", "Electronics", "Furniture", "Office Supplies", "Electronics", "Furniture", "Office Supplies", "Electronics"],
                "quantity": [2, 1, 10, 1, 4, 15, 2, 2, 8, 3],
                "unit_price": [499.99, 150.00, 12.50, 899.99, 85.00, 4.99, 650.00, 220.00, 15.00, 120.00],
                "discount": [0.05, 0.00, 0.10, 0.15, 0.00, 0.00, 0.10, 0.05, 0.05, 0.10],
                "profit": [120.50, 35.00, 25.00, 180.00, 50.00, 18.50, 110.00, 60.00, 19.50, 55.00]
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
    col_info, col_version = st.columns([3, 2])
    with col_info:
        st.markdown(f"**Active Dataset:** `{dataset_name}`")
    with col_version:
        if has_cleaned and has_raw:
            data_version = st.radio(
                "Data Version for Insights:",
                options=["Cleaned Working Dataset (Phase 2)", "Original Raw Dataset (Phase 1)"],
                horizontal=True
            )
            df = st.session_state["cleaned_df"] if "Cleaned" in data_version else st.session_state["raw_df"]
        else:
            df = st.session_state.get("cleaned_df", st.session_state.get("raw_df"))

    st.markdown("---")

    # --------------------------------------------------------------------------
    # Step 2: Generate Insights Button
    # --------------------------------------------------------------------------
    col_btn, col_desc = st.columns([1, 3])
    with col_btn:
        generate_clicked = st.button("🚀 Generate Insights", type="primary", use_container_width=True)

    with col_desc:
        st.caption("Click to perform an automated statistical audit, scan for outliers, discover correlations, and generate recommendations.")

    # Cache insights in session state so user interactions don't force recalculations
    cache_key = f"insights_{dataset_name}_{len(df)}"
    if generate_clicked or cache_key in st.session_state:
        if generate_clicked or cache_key not in st.session_state:
            with st.spinner("🔬 Performing statistical scans, outlier detection, and relationship analysis..."):
                st.session_state[cache_key] = generate_automated_insights(df)

        insights = st.session_state[cache_key]
    else:
        st.info("👆 Click **'🚀 Generate Insights'** above to run the automated statistical engine on this dataset.")
        return

    quality = insights.get("quality", {})
    cards = insights.get("cards", [])
    recommendations = insights.get("recommendations", [])
    strong_corrs = insights.get("strong_correlations", [])
    outlier_reports = insights.get("outlier_reports", {})

    # --------------------------------------------------------------------------
    # Step 3: Executive Scorecard & Dataset Summary (Feature 1)
    # --------------------------------------------------------------------------
    st.markdown("### 🏆 Executive Data Scorecard")
    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    with sc1:
        st.metric("Cleanliness Score", f"{quality.get('score', 0)} / 100", delta=quality.get('grade', 'N/A'))
    with sc2:
        st.metric("Total Records", f"{insights['num_rows']:,}")
    with sc3:
        st.metric("Total Features", f"{insights['num_cols']:,}", f"{len(insights['numeric_cols'])} Num, {len(insights['categorical_cols'])} Cat")
    with sc4:
        st.metric("Missing Cells", f"{quality.get('missing_count', 0):,}", delta=f"{quality.get('missing_pct', 0.0)}%", delta_color="inverse")
    with sc5:
        st.metric("Duplicate Rows", f"{quality.get('duplicate_count', 0):,}", delta=f"{quality.get('duplicate_pct', 0.0)}%", delta_color="inverse")

    if quality.get("issues"):
        with st.expander("⚠️ Flagged Data Quality Issues", expanded=False):
            for issue in quality["issues"]:
                st.write(f"- {issue}")

    st.markdown("---")

    # --------------------------------------------------------------------------
    # Step 4: Organized Tabs
    # --------------------------------------------------------------------------
    tab_cards, tab_corr, tab_anomalies, tab_recs, tab_export = st.tabs([
        "🌟 Key Insight Cards",
        "🔗 Relationships & Heatmap",
        "⚠️ Anomaly & Outlier Radar",
        "💡 Smart Recommendations",
        "📄 Export Report"
    ])

    # --------------------------------------------------------------------------
    # TAB 1: STRUCTURED INSIGHT CARDS (Features 2, 5, 7)
    # --------------------------------------------------------------------------
    with tab_cards:
        st.subheader("Automated Discovery Cards")
        st.caption("Key patterns, category leaders, distribution shifts, and statistical outliers automatically discovered from your data.")

        if not cards:
            st.info("No extreme anomalies or distribution imbalances were detected. The dataset is well-balanced.")
        else:
            for idx, card in enumerate(cards, 1):
                with st.container(border=True):
                    head_col, col_tags = st.columns([3, 1])
                    with head_col:
                        st.markdown(f"#### {idx}. {card['title']}")
                    with col_tags:
                        st.caption(f"**{card['badge']}**")

                    # Calculated Finding
                    st.markdown(f"**Finding:** {card['finding']}")
                    # Simple Explanation
                    st.markdown(f"_{card['explanation']}_")
                    st.caption(f"**Columns Analyzed:** `{', '.join(card['columns'])}`")

                    # Embedded Plotly Visualization (Feature 5)
                    rec_chart = card.get("chart_recommendation")
                    if rec_chart:
                        chart_type = rec_chart["chart_type"]
                        if chart_type == "box":
                            target_c = rec_chart["col"]
                            fig_box = px.box(
                                df,
                                y=target_c,
                                points="all",
                                title=f"Distribution & Outlier Plot: {target_c}",
                                color_discrete_sequence=["#e74c3c"]
                            )
                            fig_box.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=35, b=20), height=300)
                            st.plotly_chart(fig_box, use_container_width=True)

                        elif chart_type == "bar":
                            x_col = rec_chart["x_col"]
                            y_col = rec_chart["y_col"]
                            grouped_sum = df.dropna(subset=[x_col, y_col]).groupby(x_col, as_index=False)[y_col].sum()
                            fig_bar = px.bar(
                                grouped_sum,
                                x=x_col,
                                y=y_col,
                                text_auto=True,
                                title=f"Total {y_col} by {x_col}",
                                color=x_col
                            )
                            fig_bar.update_layout(template="plotly_white", showlegend=False, margin=dict(l=20, r=20, t=35, b=20), height=300)
                            st.plotly_chart(fig_bar, use_container_width=True)

                        elif chart_type == "scatter":
                            x_col = rec_chart["x_col"]
                            y_col = rec_chart["y_col"]
                            clean_scatter = df.dropna(subset=[x_col, y_col])
                            fig_sc = px.scatter(
                                clean_scatter,
                                x=x_col,
                                y=y_col,
                                title=f"Association: {y_col} vs. {x_col}",
                                color_discrete_sequence=["#3498db"]
                            )
                            # Add linear trendline
                            if len(clean_scatter) > 1 and clean_scatter[x_col].nunique() > 1:
                                slope, intercept = np.polyfit(clean_scatter[x_col].values, clean_scatter[y_col].values, 1)
                                x_line = np.linspace(clean_scatter[x_col].min(), clean_scatter[x_col].max(), 50)
                                y_line = slope * x_line + intercept
                                fig_sc.add_trace(
                                    go.Scatter(x=x_line, y=y_line, mode="lines", name="Linear Trend", line=dict(color="#e74c3c", dash="dash"))
                                )
                            fig_sc.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=35, b=20), height=300)
                            st.plotly_chart(fig_sc, use_container_width=True)

                        elif chart_type == "histogram":
                            h_col = rec_chart["col"]
                            fig_hist = px.histogram(
                                df,
                                x=h_col,
                                marginal="box",
                                title=f"Skewed Distribution: {h_col}",
                                color_discrete_sequence=["#9b59b6"]
                            )
                            fig_hist.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=35, b=20), height=300)
                            st.plotly_chart(fig_hist, use_container_width=True)

    # --------------------------------------------------------------------------
    # TAB 2: RELATIONSHIPS & HEATMAP (Feature 3)
    # --------------------------------------------------------------------------
    with tab_corr:
        st.subheader("Relationship & Correlation Discovery")
        st.caption("Scans for co-movement between numerical variables. Correlation measures linear strength from -1.0 to +1.0.")

        # Mandatory Statistical Warning
        st.warning(
            "⚠️ **Important Statistical Reminder:** *Correlation does not imply causation.* "
            "A strong correlation indicates that two variables move together, but it does NOT prove that one causes the other. "
            "Always consider confounding variables and domain context."
        )

        if len(insights["numeric_cols"]) < 2:
            st.info("At least two numerical columns are required for correlation analysis.")
        else:
            fig_heatmap = build_correlation_heatmap(df, columns=insights["numeric_cols"], colorscale="RdBu_r")
            st.plotly_chart(fig_heatmap, use_container_width=True)

            st.markdown("#### Identified Strong Relationships (|r| ≥ 0.5)")
            if not strong_corrs:
                st.info("No strong linear relationships (|r| ≥ 0.5) were detected between numerical columns.")
            else:
                for item in strong_corrs:
                    with st.container(border=True):
                        st.markdown(f"**`{item['feature_1']}` & `{item['feature_2']}`** — Correlation: `r = {item['correlation']}`")
                        st.write(item["interpretation"])

    # --------------------------------------------------------------------------
    # TAB 3: ANOMALY & OUTLIER RADAR (Feature 4)
    # --------------------------------------------------------------------------
    with tab_anomalies:
        st.subheader("Statistical Outlier & Anomaly Radar")
        st.markdown(
            "Anomalies are detected using **Tukey's Interquartile Range (IQR) method**. "
            "Values falling outside $Q_1 - 1.5 \\times \\text{IQR}$ or $Q_3 + 1.5 \\times \\text{IQR}$ "
            "are statistically unusual compared to the rest of your data."
        )

        if not outlier_reports:
            st.success("✅ **No Outliers Detected:** Numerical features fall comfortably within standard statistical fences.")
        else:
            outlier_summary = []
            for col, res in outlier_reports.items():
                outlier_summary.append({
                    "Column": col,
                    "Outlier Count": res["outlier_count"],
                    "Outlier (%)": f"{res['outlier_pct']}%",
                    "Lower Bound": res["lower_bound"],
                    "Upper Bound": res["upper_bound"],
                    "Sample Outliers": ", ".join([str(v) for v in res["outlier_values"][:3]])
                })
            st.dataframe(pd.DataFrame(outlier_summary), use_container_width=True)

            with st.expander("🔍 How to handle statistical outliers"):
                st.markdown("""
                1. **Never delete outliers blindly**: First verify whether they are recording errors (e.g., negative age, typo in price) or legitimate high-impact observations (e.g., bulk wholesale order).
                2. **Use robust metrics**: If outliers are legitimate, report the **Median** rather than the **Mean**, as the median is resistant to extreme values.
                3. **Clean if necessary**: If an outlier is a confirmed typo, switch to the **Data Cleaning Studio** to handle it.
                """)

    # --------------------------------------------------------------------------
    # TAB 4: SMART RECOMMENDATIONS (Feature 6)
    # --------------------------------------------------------------------------
    with tab_recs:
        st.subheader("Smart Recommendations & Next Investigative Questions")
        st.caption("Actionable recommendations generated dynamically based on your dataset's specific statistical profile.")

        for idx, rec in enumerate(recommendations, 1):
            with st.container(border=True):
                st.markdown(f"##### {idx}. {rec['action']}")
                st.markdown(f"👉 **Recommendation:** {rec['text']}")
                st.markdown(f"💡 **Why this matters:** _{rec['reason']}_")

        st.markdown("#### 🤔 Useful Investigative Questions to Explore")
        st.markdown("""
        - *Does the top-performing category drive revenue due to higher transaction volumes, or higher average prices?*
        - *Are the identified outliers concentrated within a single region or department, or scattered randomly?*
        - *If two variables are strongly correlated, is there a hidden third variable (a confounding factor) driving both?*
        """)

    # --------------------------------------------------------------------------
    # TAB 5: EXPORT REPORT
    # --------------------------------------------------------------------------
    with tab_export:
        st.subheader("Export Automated Insights Report")
        st.caption("Download a clean, formatted Markdown document summarizing all findings, quality scores, anomalies, and recommendations.")

        report_markdown = generate_insights_markdown_report(insights, dataset_name)

        safe_name = dataset_name.rsplit(".", 1)[0]
        st.download_button(
            label="📥 Download Insights Report (.md)",
            data=report_markdown.encode("utf-8"),
            file_name=f"{safe_name}_insights_report.md",
            mime="text/markdown",
            use_container_width=True,
            help="Download as a standard Markdown file suitable for sharing with teammates or documentation."
        )

        with st.expander("👀 Preview Generated Markdown Report"):
            st.code(report_markdown, language="markdown")
