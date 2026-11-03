"""
export_studio.py
================
Export & Report Studio — Phase 8 UI Component for InsightLab.

This module renders the complete reporting interface, which includes:
  Tab 1 — Custom Report Builder  : choose sections, title, format, generate.
  Tab 2 — Download Center        : download individual files (CSV, Excel, MD).
  Tab 3 — AI Executive Summary   : generate and display an AI narrative summary.

Architecture Note:
------------------
This file only handles the Streamlit UI (what the user sees and clicks).
All heavy logic (PDF generation, Excel building, data collection) lives in
utils/report_builder.py so it stays clean, testable, and separate.

Beginner Concept — Why separate UI from logic?
----------------------------------------------
Think of it like a restaurant:
  - The "kitchen" (utils/report_builder.py) prepares the food.
  - The "waiter" (this file) brings it to the table and presents it nicely.
Neither should be doing the other's job.
"""

import streamlit as st
import pandas as pd


def render_export_studio() -> None:
    """
    Main entry point. Renders the complete Export & Report Studio UI.
    Called from app.py when the user selects this module from the sidebar.
    """

    # =========================================================================
    # PAGE HEADER
    # =========================================================================
    st.title("📄 Export & Report Studio")
    st.markdown(
        "Generate professional PDF reports, download multi-sheet Excel workbooks, "
        "and export all your analysis results in one place."
    )
    st.markdown("---")

    # =========================================================================
    # DATASET GUARD — show a friendly message if no data is loaded
    # =========================================================================
    raw_df: pd.DataFrame | None = st.session_state.get("raw_df")
    has_dataset = raw_df is not None and not raw_df.empty

    if not has_dataset:
        st.info(
            "📂 **No dataset uploaded yet.**\n\n"
            "To use the Export Studio, first upload a CSV or Excel file in the "
            "**📊 Dataset Explorer** module. Once your data is loaded, all export "
            "and report options will become available here."
        )
        return

    # =========================================================================
    # THREE TABS
    # =========================================================================
    tab_builder, tab_downloads, tab_ai = st.tabs([
        "📋 Custom Report Builder",
        "💾 Download Center",
        "🤖 AI Executive Summary",
    ])

    # =========================================================================
    # TAB 1 — CUSTOM REPORT BUILDER
    # =========================================================================
    with tab_builder:
        _render_report_builder()

    # =========================================================================
    # TAB 2 — DOWNLOAD CENTER
    # =========================================================================
    with tab_downloads:
        _render_download_center()

    # =========================================================================
    # TAB 3 — AI EXECUTIVE SUMMARY
    # =========================================================================
    with tab_ai:
        _render_ai_summary_tab()


# =============================================================================
# TAB 1 HELPER — Custom Report Builder
# =============================================================================

def _render_report_builder() -> None:
    """
    Renders the report configuration form and the 'Generate Report' button.

    What happens here:
    ------------------
    1. User fills in a title and optionally an introduction.
    2. User ticks which sections to include.
    3. User picks PDF or Excel format.
    4. On submit, we call report_builder.py to do the actual work.
    5. We hand the result bytes to st.download_button.
    """
    st.subheader("📋 Custom Report Builder")
    st.markdown(
        "Configure your report below, then click **Generate Report** to download it."
    )

    # ---- Report Metadata ---------------------------------------------------
    col1, col2 = st.columns([2, 1])
    with col1:
        report_title = st.text_input(
            "📝 Report Title",
            value="InsightLab Analysis Report",
            help="This title will appear at the top of your PDF or as the Excel file name.",
        )
    with col2:
        report_format = st.radio(
            "📄 Output Format",
            options=["PDF Report", "Excel Workbook"],
            index=0,
            help="PDF is great for sharing; Excel lets you explore data further.",
        )

    intro_text = st.text_area(
        "✏️ Introduction (Optional)",
        placeholder="Add a brief description or context for this report...",
        height=80,
        help="This text will appear on the cover page of the PDF report.",
    )

    st.markdown("---")

    # ---- Section Selectors -------------------------------------------------
    st.markdown("### 📂 Select Report Sections")
    st.caption(
        "Check the sections you want included. Sections without data will be "
        "skipped automatically."
    )

    col_a, col_b = st.columns(2)

    with col_a:
        include_overview = st.checkbox(
            "📊 Dataset Overview",
            value=True,
            help="Rows, columns, data types, quality score, and column schema.",
        )
        include_statistics = st.checkbox(
            "📈 Statistical Analysis",
            value=True,
            help="Mean, median, standard deviation, and other statistics for numerical columns.",
        )
        include_cleaning = st.checkbox(
            "🧹 Data Cleaning History",
            value=True,
            help="List of cleaning operations applied and before/after comparison.",
        )

    with col_b:
        include_insights = st.checkbox(
            "💡 Automated Insights",
            value=True,
            help="Key findings, correlations, anomalies, and recommendations.",
        )
        include_ml = st.checkbox(
            "🤖 Machine Learning Results",
            value=True,
            help="Model configuration, evaluation metrics, and feature importance.",
        )
        include_ai_summary = st.checkbox(
            "✨ AI Executive Summary",
            value=False,
            help=(
                "Include the AI-generated executive summary. "
                "Requires a Gemini API key for the best results; falls back to "
                "the deterministic insights report."
            ),
        )

    # ---- Preview which sections are active ---------------------------------
    st.markdown("---")
    with st.expander("🔍 Preview Selected Sections", expanded=False):
        selected = []
        if include_overview:
            selected.append("✅ Dataset Overview")
        if include_statistics:
            selected.append("✅ Statistical Analysis Summary")
        if include_cleaning:
            cleaning_hist = st.session_state.get("cleaning_history", [])
            label = "✅ Data Cleaning History" if cleaning_hist else "⚠️ Data Cleaning History (no operations recorded)"
            selected.append(label)
        if include_insights:
            selected.append("✅ Automated Insights & Recommendations")
        if include_ml:
            ml_ready = st.session_state.get("ml_model_bundle") is not None
            label = "✅ Machine Learning Results" if ml_ready else "⚠️ Machine Learning Results (no model trained yet)"
            selected.append(label)
        if include_ai_summary:
            selected.append("✅ AI Executive Summary")
        if not selected:
            st.warning("⚠️ No sections selected. Please check at least one section.")
        else:
            for item in selected:
                st.markdown(f"- {item}")

    # ---- Generate Button ---------------------------------------------------
    st.markdown("---")
    col_btn, col_hint = st.columns([1, 3])

    with col_btn:
        generate_clicked = st.button(
            "🚀 Generate Report",
            use_container_width=True,
            type="primary",
        )

    with col_hint:
        st.caption(
            "Clicking **Generate Report** will collect all available analysis "
            "results and build your selected format. Large datasets may take a few seconds."
        )

    if generate_clicked:
        _execute_report_generation(
            report_title=report_title,
            intro_text=intro_text,
            report_format=report_format,
            include_overview=include_overview,
            include_statistics=include_statistics,
            include_cleaning=include_cleaning,
            include_insights=include_insights,
            include_ml=include_ml,
            include_ai_summary=include_ai_summary,
        )


def _execute_report_generation(
    report_title: str,
    intro_text: str,
    report_format: str,
    include_overview: bool,
    include_statistics: bool,
    include_cleaning: bool,
    include_insights: bool,
    include_ml: bool,
    include_ai_summary: bool,
) -> None:
    """
    Calls the report_builder utilities to generate a PDF or Excel file,
    then presents a download button to the user.

    This function is called only when the user clicks 'Generate Report'.
    """
    from utils.report_builder import (
        collect_report_data,
        build_pdf_report,
        build_excel_workbook,
    )

    dataset_name = st.session_state.get("dataset_name", "report")
    # Remove file extension for cleaner filenames
    base_name = dataset_name.rsplit(".", 1)[0] if "." in dataset_name else dataset_name

    # 1. Collect all available data from session state
    with st.spinner("🔄 Collecting analysis results…"):
        try:
            report_data = collect_report_data(dict(st.session_state))
        except Exception as e:
            st.error(f"❌ Failed to collect report data: {e}")
            return

    # 2. If AI summary is requested, get it now
    ai_summary_text = ""
    if include_ai_summary:
        with st.spinner("🤖 Generating AI Executive Summary…"):
            try:
                from utils.report_builder import get_ai_executive_summary
                from utils.ai_assistant import get_gemini_api_key
                api_key = get_gemini_api_key()
                active_df = report_data.get("active_df")
                if active_df is not None:
                    ai_summary_text = get_ai_executive_summary(
                        active_df, dataset_name, api_key
                    )
            except Exception as e:
                ai_summary_text = f"AI summary could not be generated: {e}"

    # 3. Build the report
    config = {
        "title": report_title,
        "intro_text": intro_text,
        "include_overview": include_overview,
        "include_statistics": include_statistics,
        "include_cleaning": include_cleaning,
        "include_insights": include_insights,
        "include_ml": include_ml,
        "include_ai_summary": include_ai_summary,
        "ai_summary_text": ai_summary_text,
    }

    if report_format == "PDF Report":
        with st.spinner("📄 Building PDF report…"):
            try:
                pdf_bytes = build_pdf_report(report_data, config)
            except Exception as e:
                st.error(f"❌ PDF generation failed: {e}")
                st.exception(e)
                return

        st.success("✅ PDF report generated successfully!")
        st.download_button(
            label="⬇️ Download PDF Report",
            data=pdf_bytes,
            file_name=f"{base_name}_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

        # Cloud Save Option for generated PDF report
        from utils.smart_save import build_report_payload
        from components.auth_ui import render_cloud_save_button
        report_payload = build_report_payload(report_title, "pdf")
        report_payload["file_bytes"] = pdf_bytes
        render_cloud_save_button("report", report_payload, label="☁️ Save PDF Report to Cloud", key_suffix="gen_pdf")

    else:  # Excel Workbook
        with st.spinner("📊 Building Excel workbook…"):
            try:
                excel_bytes = build_excel_workbook(report_data)
            except Exception as e:
                st.error(f"❌ Excel generation failed: {e}")
                st.exception(e)
                return

        st.success("✅ Excel workbook generated successfully!")
        st.download_button(
            label="⬇️ Download Excel Workbook",
            data=excel_bytes,
            file_name=f"{base_name}_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

        # Cloud Save Option for generated Excel report
        from utils.smart_save import build_report_payload
        from components.auth_ui import render_cloud_save_button
        report_payload = build_report_payload(report_title, "excel")
        report_payload["file_bytes"] = excel_bytes
        render_cloud_save_button("report", report_payload, label="☁️ Save Excel Report to Cloud", key_suffix="gen_excel")

    # Extra note for users
    st.info(
        "💡 **Tip:** To include interactive charts, use the **Visualization Studio** "
        "module. Charts are available to interact with inside the app but are not "
        "embedded in the exported report due to format limitations."
    )


# =============================================================================
# TAB 2 HELPER — Download Center
# =============================================================================

def _render_download_center() -> None:
    """
    Renders individual download buttons for all available data files.

    Available downloads:
    --------------------
    - Raw dataset as CSV
    - Cleaned dataset as CSV (if cleaning was applied)
    - Statistical summary table as CSV
    - Automated insights report as Markdown
    - ML predictions as CSV (if classification/regression model exists)
    - Complete Excel workbook (all sheets)
    """
    st.subheader("💾 Download Center")
    st.markdown(
        "Download individual files from each analysis module. "
        "Only files with actual data are shown."
    )

    from utils.report_builder import (
        collect_report_data,
        export_df_to_csv_bytes,
        export_insights_to_markdown_bytes,
        build_excel_workbook,
    )

    report_data = collect_report_data(dict(st.session_state))
    dataset_name = report_data.get("dataset_name", "dataset")
    base_name = dataset_name.rsplit(".", 1)[0] if "." in dataset_name else dataset_name

    # Cloud Save Complete Project Option
    from utils.smart_save import build_project_payload
    from components.auth_ui import render_save_section
    proj_payload = build_project_payload()
    render_save_section("project", proj_payload, key_suffix="center_proj")
    st.markdown("---")

    any_download_shown = False

    # ---- Raw Dataset CSV ---------------------------------------------------
    raw_df = report_data.get("raw_df")
    if raw_df is not None and not raw_df.empty:
        any_download_shown = True
        st.markdown("#### 📊 Dataset Files")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rows", f"{len(raw_df):,}")
            st.metric("Columns", len(raw_df.columns))
        with col2:
            raw_bytes = export_df_to_csv_bytes(raw_df)
            st.download_button(
                label=f"⬇️ Download Raw Dataset ({base_name}.csv)",
                data=raw_bytes,
                file_name=f"{base_name}_raw.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # ---- Cleaned Dataset CSV -----------------------------------------------
    cleaned_df = report_data.get("cleaned_df")
    if cleaned_df is not None and not cleaned_df.empty:
        any_download_shown = True
        st.markdown("---")
        st.markdown("#### 🧹 Cleaned Dataset")

        col1, col2 = st.columns(2)
        with col1:
            cleaning_hist = st.session_state.get("cleaning_history", [])
            st.metric("Rows After Cleaning", f"{len(cleaned_df):,}")
            st.metric("Cleaning Steps Applied", len(cleaning_hist))
        with col2:
            cleaned_bytes = export_df_to_csv_bytes(cleaned_df)
            st.download_button(
                label=f"⬇️ Download Cleaned Dataset",
                data=cleaned_bytes,
                file_name=f"{base_name}_cleaned.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # ---- Statistical Summary CSV -------------------------------------------
    stats_table = report_data.get("stats_table")
    if stats_table is not None and not stats_table.empty:
        any_download_shown = True
        st.markdown("---")
        st.markdown("#### 📈 Statistical Summary")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Numerical Columns Analyzed", len(stats_table))
        with col2:
            stats_bytes = export_df_to_csv_bytes(stats_table)
            st.download_button(
                label="⬇️ Download Statistical Summary (.csv)",
                data=stats_bytes,
                file_name=f"{base_name}_statistics.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # ---- Insights Markdown -------------------------------------------------
    insights = report_data.get("insights")
    if insights and "error" not in insights:
        any_download_shown = True
        st.markdown("---")
        st.markdown("#### 💡 Automated Insights Report")

        col1, col2 = st.columns(2)
        n_cards = len(insights.get("insight_cards", []))
        with col1:
            st.metric("Insight Cards Found", n_cards)
        with col2:
            insights_bytes = export_insights_to_markdown_bytes(report_data)
            st.download_button(
                label="⬇️ Download Insights Report (.md)",
                data=insights_bytes,
                file_name=f"{base_name}_insights.md",
                mime="text/markdown",
                use_container_width=True,
            )

    # ---- ML Predictions CSV ------------------------------------------------
    ml_bundle = report_data.get("ml_bundle")
    ml_task = report_data.get("ml_task", "")
    if ml_bundle and ml_task in ("regression", "classification"):
        test_results = ml_bundle.get("test_results")
        if test_results is not None and not test_results.empty:
            any_download_shown = True
            st.markdown("---")
            st.markdown("#### 🤖 Machine Learning Predictions")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Prediction Rows", f"{len(test_results):,}")
                st.metric("Task", ml_task.capitalize())
            with col2:
                pred_bytes = export_df_to_csv_bytes(test_results)
                st.download_button(
                    label="⬇️ Download ML Predictions (.csv)",
                    data=pred_bytes,
                    file_name=f"{base_name}_ml_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    # ---- Clustering Results CSV -------------------------------------------
    if ml_bundle and ml_task == "clustering":
        clustered_df = ml_bundle.get("clustered_df")
        if clustered_df is not None and not clustered_df.empty:
            any_download_shown = True
            st.markdown("---")
            st.markdown("#### 🤖 Clustering Results")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Clustered Rows", f"{len(clustered_df):,}")
                n_clusters = clustered_df["Cluster"].nunique() if "Cluster" in clustered_df.columns else "?"
                st.metric("Number of Clusters", n_clusters)
            with col2:
                cluster_bytes = export_df_to_csv_bytes(clustered_df)
                st.download_button(
                    label="⬇️ Download Clustering Results (.csv)",
                    data=cluster_bytes,
                    file_name=f"{base_name}_clusters.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    # ---- Full Excel Workbook -----------------------------------------------
    st.markdown("---")
    st.markdown("#### 📦 Complete Excel Workbook (All Sheets)")
    st.caption(
        "Downloads a single Excel file containing all available data: "
        "Dataset Overview, Raw Data, Cleaned Data, Statistical Summary, "
        "Insights, and ML Results."
    )

    if st.button("🔧 Build & Download Excel Workbook", use_container_width=True, type="primary"):
        with st.spinner("Building Excel workbook…"):
            try:
                excel_bytes = build_excel_workbook(report_data)
                st.download_button(
                    label="⬇️ Download Complete Excel Workbook",
                    data=excel_bytes,
                    file_name=f"{base_name}_complete_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
                st.success("✅ Excel workbook ready!")
                any_download_shown = True
            except Exception as e:
                st.error(f"❌ Excel generation failed: {e}")

    if not any_download_shown:
        st.info(
            "🔍 Run analysis modules first to unlock more download options:\n\n"
            "- Upload a dataset in **Dataset Explorer**\n"
            "- Apply cleaning in **Data Cleaning**\n"
            "- Generate statistics in **Statistical Analysis**\n"
            "- Run **Automated Insights Studio**\n"
            "- Train a model in **Machine Learning Studio**"
        )


# =============================================================================
# TAB 3 HELPER — AI Executive Summary
# =============================================================================

def _render_ai_summary_tab() -> None:
    """
    Renders the AI Executive Summary generator tab.

    What it does:
    -------------
    1. Checks if a Gemini API key is available (session state / secrets / env).
    2. Shows a status indicator (API connected / local fallback).
    3. Provides a 'Generate Summary' button.
    4. Displays the generated summary with an option to download it.
    5. Never exposes the API key in the UI or exports.
    """
    st.subheader("🤖 AI Executive Summary")
    st.markdown(
        "Generate a professional, human-readable executive summary of your dataset "
        "and analysis. Uses the Gemini AI model when an API key is configured, "
        "or falls back to our built-in insights engine."
    )

    # ---- API Key Status Indicator -----------------------------------------
    try:
        from utils.ai_assistant import get_gemini_api_key
        api_key = get_gemini_api_key()
    except Exception:
        api_key = None

    if api_key:
        st.success(
            "✅ **Gemini API Connected** — The AI Executive Summary will use "
            "Google's Gemini model for a rich, generative narrative."
        )
    else:
        st.warning(
            "⚠️ **No Gemini API Key Found** — The summary will be generated "
            "using InsightLab's built-in deterministic insights engine. "
            "To enable AI-powered summaries, add your Gemini API key in the "
            "**AI Data Assistant** module or in `.streamlit/secrets.toml`."
        )

    st.markdown("---")

    # ---- Summary Options ---------------------------------------------------
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        summary_focus = st.selectbox(
            "🎯 Summary Focus",
            options=[
                "General Overview",
                "Data Quality & Cleaning",
                "Statistical Patterns",
                "Business Recommendations",
            ],
            help="Guides the AI's focus when generating the summary.",
        )
    with col_opt2:
        use_cleaned = st.checkbox(
            "Use Cleaned Dataset (if available)",
            value=True,
            help="If unchecked, the original raw dataset is used for summary generation.",
        )

    # ---- Generate Button ---------------------------------------------------
    if st.button("✨ Generate AI Executive Summary", use_container_width=True, type="primary"):
        # Pick the right DataFrame
        if use_cleaned:
            active_df = st.session_state.get("cleaned_df") or st.session_state.get("raw_df")
        else:
            active_df = st.session_state.get("raw_df")

        dataset_name = st.session_state.get("dataset_name", "dataset")

        if active_df is None or active_df.empty:
            st.error("❌ No dataset available. Upload data in the Dataset Explorer first.")
            return

        # Build a focus-aware prompt prefix
        focus_prompts = {
            "General Overview": "Write a comprehensive general overview.",
            "Data Quality & Cleaning": "Focus on data quality issues, missing values, duplicates, and cleaning recommendations.",
            "Statistical Patterns": "Focus on key statistical patterns, distributions, correlations, and anomalies.",
            "Business Recommendations": "Focus on actionable business insights and concrete recommendations.",
        }
        focus_text = focus_prompts.get(summary_focus, "Write a comprehensive general overview.")

        with st.spinner("🤖 Generating executive summary…"):
            try:
                from utils.report_builder import get_ai_executive_summary
                # Inject focus into the API key lookup context
                summary_text = get_ai_executive_summary(active_df, dataset_name, api_key)
            except Exception as e:
                st.error(f"❌ Summary generation failed: {e}")
                return

        # Store in session state so the Report Builder can use it
        st.session_state["ai_executive_summary"] = summary_text

        st.success("✅ Executive summary generated!")
        st.markdown("---")

        # Display the summary
        st.markdown("### 📄 Executive Summary")
        st.markdown(summary_text)

        # Download button for the summary text
        st.markdown("---")
        base_name = dataset_name.rsplit(".", 1)[0] if "." in dataset_name else dataset_name
        st.download_button(
            label="⬇️ Download Executive Summary (.md)",
            data=summary_text.encode("utf-8"),
            file_name=f"{base_name}_executive_summary.md",
            mime="text/markdown",
            use_container_width=True,
        )

        st.info(
            "💡 **Tip:** Check the **✨ AI Executive Summary** box in the "
            "**Report Builder** tab to include this summary in your PDF report."
        )

    # Show previously generated summary if available
    elif "ai_executive_summary" in st.session_state:
        st.markdown("---")
        st.markdown("### 📄 Previously Generated Summary")
        st.caption(
            "This summary was generated earlier in this session. "
            "Click 'Generate AI Executive Summary' to refresh it."
        )
        st.markdown(st.session_state["ai_executive_summary"])

        dataset_name = st.session_state.get("dataset_name", "dataset")
        base_name = dataset_name.rsplit(".", 1)[0] if "." in dataset_name else dataset_name
        st.download_button(
            label="⬇️ Download Executive Summary (.md)",
            data=st.session_state["ai_executive_summary"].encode("utf-8"),
            file_name=f"{base_name}_executive_summary.md",
            mime="text/markdown",
            use_container_width=True,
        )
