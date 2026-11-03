"""
report_builder.py
=================
Report Builder Engine for InsightLab Phase 8 — Professional Reporting & Export Studio.

What This Module Does:
----------------------
This module is the "engine" behind all report generation. It:
  1. Collects all available analysis results from Streamlit session state.
  2. Builds a beautifully styled PDF report using ReportLab.
  3. Builds a multi-sheet Excel workbook using openpyxl + Pandas.
  4. Generates an AI Executive Summary using the Gemini API (with fallback).

Key Libraries Used:
-------------------
- io.BytesIO  : A memory buffer that behaves like a file. We write PDF/Excel bytes
                INTO this buffer instead of saving to disk, then hand those bytes
                directly to Streamlit's download button. No temporary files needed.

- ReportLab   : A Python PDF generation library. We use "platypus" (its high-level
                API) which works like laying out a document with "Flowable" building
                blocks — paragraphs, tables, spacers, horizontal lines.

- openpyxl    : A library for reading and writing Excel (.xlsx) files. We use it via
                Pandas (pd.ExcelWriter) to write multiple sheets easily.

Design Rules:
-------------
- NEVER fabricate data, statistics, or AI summaries.
- NEVER expose API keys in any output.
- Always use session_state values as-is; only format, never invent.
- Gracefully handle missing data (no dataset, no ML, no API key).
"""

import io
import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# =============================================================================
# SECTION 1: DATA COLLECTION
# =============================================================================

def collect_report_data(session_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pulls all available analysis results from Streamlit session state and
    assembles them into a single structured dictionary for report generation.

    Parameters:
    -----------
    session_state : dict
        A snapshot of st.session_state (passed in as a plain dict so this
        function can be tested independently of Streamlit).

    Returns:
    --------
    dict with keys:
        - 'has_dataset'      : bool — is there any data at all?
        - 'dataset_name'     : str  — filename
        - 'raw_df'           : pd.DataFrame or None
        - 'cleaned_df'       : pd.DataFrame or None
        - 'cleaning_history' : list of str — operations applied
        - 'ml_bundle'        : dict or None — trained model information
        - 'ml_task'          : str or None — "regression"/"classification"/"clustering"
        - 'chat_history'     : list of dicts or None
        - 'quality_score'    : dict or None — from compute_dataset_quality_score
        - 'stats_table'      : pd.DataFrame or None — statistical summary table
        - 'insights'         : dict or None — from generate_automated_insights
        - 'generated_at'     : str — ISO timestamp
    """
    # Extract the active DataFrame (prefer cleaned, fall back to raw)
    raw_df: Optional[pd.DataFrame] = session_state.get("raw_df")
    cleaned_df: Optional[pd.DataFrame] = session_state.get("cleaned_df")
    active_df = cleaned_df if (cleaned_df is not None and not cleaned_df.empty) else raw_df

    has_dataset = active_df is not None and not active_df.empty

    # Quality score & statistics — only compute when a dataset is available
    quality_score = None
    stats_table = None
    insights = None

    if has_dataset:
        try:
            from utils.insights_engine import compute_dataset_quality_score, generate_automated_insights
            quality_score = compute_dataset_quality_score(active_df)
        except Exception:
            quality_score = None

        try:
            from utils.statistical_analyzer import compute_dataset_statistics_table
            stats_table = compute_dataset_statistics_table(active_df)
        except Exception:
            stats_table = None

        try:
            from utils.insights_engine import generate_automated_insights
            insights = generate_automated_insights(active_df)
        except Exception:
            insights = None

    return {
        "has_dataset": has_dataset,
        "dataset_name": session_state.get("dataset_name", "dataset"),
        "raw_df": raw_df,
        "cleaned_df": cleaned_df,
        "active_df": active_df,
        "cleaning_history": session_state.get("cleaning_history", []),
        "ml_bundle": session_state.get("ml_model_bundle"),
        "ml_task": session_state.get("ml_task"),
        "chat_history": session_state.get("chat_history", []),
        "quality_score": quality_score,
        "stats_table": stats_table,
        "insights": insights,
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# =============================================================================
# SECTION 2: AI EXECUTIVE SUMMARY
# =============================================================================

def get_ai_executive_summary(
    df: pd.DataFrame,
    dataset_name: str,
    api_key: Optional[str]
) -> str:
    """
    Generates an AI-powered executive summary of the dataset.

    If an API key is available, calls the Gemini API through the existing
    ai_assistant utility. Otherwise, falls back to the deterministic
    insights markdown report generated by insights_engine.

    Parameters:
    -----------
    df          : pd.DataFrame — the active dataset.
    dataset_name: str — human-readable filename.
    api_key     : str or None — Gemini API key.

    Returns:
    --------
    str — executive summary text (markdown format).
    """
    if df is None or df.empty:
        return "No dataset available for executive summary generation."

    # Try Gemini API first
    if api_key:
        try:
            from utils.ai_assistant import build_dataset_context_summary, query_gemini_assistant
            context = build_dataset_context_summary(df, dataset_name)
            prompt = (
                "You are a professional data analyst. Write a concise executive summary "
                "(3-5 paragraphs, plain business language) of this dataset for a report. "
                "Include: what the dataset contains, key patterns, any data quality issues, "
                "and 2-3 actionable recommendations. Do NOT use bullet-point lists — use "
                "flowing paragraphs. Do NOT invent facts not in the context provided."
            )
            summary = query_gemini_assistant(prompt, context, api_key)
            if summary and not summary.startswith("❌"):
                return summary
        except Exception:
            pass  # Fall through to local fallback

    # Local fallback — deterministic markdown report from insights engine
    try:
        from utils.insights_engine import generate_automated_insights, generate_insights_markdown_report
        insights = generate_automated_insights(df)
        return generate_insights_markdown_report(insights, dataset_name)
    except Exception as e:
        return f"Executive summary could not be generated: {e}"


# =============================================================================
# SECTION 3: PDF REPORT BUILDER
# =============================================================================

def build_pdf_report(report_data: Dict[str, Any], config: Dict[str, Any]) -> bytes:
    """
    Builds a professional PDF report and returns its raw bytes.

    Why return bytes?
    -----------------
    Streamlit's `st.download_button` accepts bytes directly via the `data=`
    argument. We write everything into an io.BytesIO buffer (an in-memory
    file) and return its content as bytes — no temp files needed.

    Parameters:
    -----------
    report_data : dict — output of collect_report_data()
    config      : dict — user selections:
        - 'title'              : str
        - 'include_overview'   : bool
        - 'include_statistics' : bool
        - 'include_cleaning'   : bool
        - 'include_insights'   : bool
        - 'include_ml'         : bool
        - 'include_ai_summary' : bool
        - 'ai_summary_text'    : str (pre-generated summary, if available)

    Returns:
    --------
    bytes — raw PDF content.
    """
    # ReportLab imports — lazy import so the module doesn't crash at startup
    # if ReportLab is not yet installed.
    try:
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            HRFlowable, PageBreak
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
    except ImportError as e:
        raise ImportError(
            f"ReportLab is not installed. Run: pip install reportlab\n{e}"
        )

    # -------------------------------------------------------------------------
    # 3a. Document Setup
    # -------------------------------------------------------------------------
    # BytesIO is an in-memory byte stream — exactly like a file but stored in RAM.
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2 * cm,
        title=config.get("title", "InsightLab Report"),
        author="InsightLab",
    )

    # -------------------------------------------------------------------------
    # 3b. Styles
    # -------------------------------------------------------------------------
    # getSampleStyleSheet() gives us a set of ready-made styles: Normal, Heading1, etc.
    styles = getSampleStyleSheet()

    # Define custom styles specific to InsightLab
    BRAND_BLUE = colors.HexColor("#1E3A5F")    # Dark navy — InsightLab primary
    ACCENT_TEAL = colors.HexColor("#17A2B8")   # Teal accent
    LIGHT_GRAY = colors.HexColor("#F5F5F5")    # Table row background
    MID_GRAY = colors.HexColor("#CCCCCC")      # Borders
    WHITE = colors.white
    BLACK = colors.black

    title_style = ParagraphStyle(
        "InsightTitle",
        parent=styles["Title"],
        fontSize=26,
        textColor=BRAND_BLUE,
        spaceAfter=6,
        spaceBefore=12,
        alignment=TA_CENTER,
    )
    subtitle_style = ParagraphStyle(
        "InsightSubtitle",
        parent=styles["Normal"],
        fontSize=11,
        textColor=colors.HexColor("#555555"),
        spaceAfter=4,
        alignment=TA_CENTER,
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading1"],
        fontSize=14,
        textColor=BRAND_BLUE,
        spaceBefore=18,
        spaceAfter=6,
        borderPadding=(0, 0, 4, 0),
    )
    sub_heading = ParagraphStyle(
        "SubHeading",
        parent=styles["Heading2"],
        fontSize=11,
        textColor=ACCENT_TEAL,
        spaceBefore=10,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "InsightBody",
        parent=styles["Normal"],
        fontSize=9,
        leading=14,
        spaceAfter=4,
        textColor=BLACK,
    )
    small_style = ParagraphStyle(
        "InsightSmall",
        parent=styles["Normal"],
        fontSize=8,
        leading=12,
        textColor=colors.HexColor("#777777"),
    )
    caption_style = ParagraphStyle(
        "InsightCaption",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.HexColor("#999999"),
        alignment=TA_CENTER,
        spaceAfter=6,
    )

    # -------------------------------------------------------------------------
    # 3c. Helper: build a styled ReportLab Table from a Pandas DataFrame
    # -------------------------------------------------------------------------
    def df_to_rl_table(df: pd.DataFrame, max_rows: int = 50) -> Table:
        """
        Converts a Pandas DataFrame to a ReportLab Table with alternating
        row shading and a coloured header row.

        What is a ReportLab Table?
        --------------------------
        A Table is a grid of cells. Each cell can contain text or a Paragraph.
        TableStyle rules are applied as (command, start_cell, end_cell, value).
        Coordinates are (column, row) where (0, 0) = top-left.
        """
        display_df = df.head(max_rows)

        # Build data grid: first row = headers, subsequent rows = data values
        header_row = [
            Paragraph(f"<b>{str(col)}</b>", small_style) for col in display_df.columns
        ]
        data = [header_row]

        for _, row in display_df.iterrows():
            data.append([
                Paragraph(str(val) if val is not None and not (
                    isinstance(val, float) and np.isnan(val)
                ) else "—", small_style)
                for val in row
            ])

        # Calculate column widths that fill the page (A4 width minus margins = ~17cm)
        page_width = 17 * cm
        n_cols = len(display_df.columns)
        col_width = page_width / n_cols

        tbl = Table(data, colWidths=[col_width] * n_cols, repeatRows=1)

        # Apply table styling
        tbl.setStyle(TableStyle([
            # Header row
            ("BACKGROUND", (0, 0), (-1, 0), BRAND_BLUE),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
            ("TOPPADDING", (0, 0), (-1, 0), 6),
            # Alternating row colors
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_GRAY]),
            # Grid lines
            ("GRID", (0, 0), (-1, -1), 0.5, MID_GRAY),
            ("FONTSIZE", (0, 1), (-1, -1), 8),
            ("TOPPADDING", (0, 1), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        return tbl

    # -------------------------------------------------------------------------
    # 3d. Helper: build a two-column key-value info table
    # -------------------------------------------------------------------------
    def kv_table(pairs: List[tuple]) -> Table:
        """Builds a compact key-value info table. pairs = [(label, value), ...]"""
        data = []
        for label, value in pairs:
            data.append([
                Paragraph(f"<b>{label}</b>", small_style),
                Paragraph(str(value), small_style),
            ])
        tbl = Table(data, colWidths=[5 * cm, 12 * cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), LIGHT_GRAY),
            ("GRID", (0, 0), (-1, -1), 0.5, MID_GRAY),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ]))
        return tbl

    # -------------------------------------------------------------------------
    # 3e. Helper: add a section divider
    # -------------------------------------------------------------------------
    def hr():
        return HRFlowable(
            width="100%", thickness=1, color=ACCENT_TEAL,
            spaceAfter=6, spaceBefore=2
        )

    # =========================================================================
    # 3f. Build the document "story" (list of Flowables)
    # =========================================================================
    story = []

    # ---- TITLE PAGE ---------------------------------------------------------
    title_text = config.get("title", "InsightLab Analysis Report")
    dataset_name = report_data.get("dataset_name", "Unknown Dataset")
    generated_at = report_data.get("generated_at", "")

    story.append(Spacer(1, 2 * cm))
    story.append(Paragraph("🔬 InsightLab", subtitle_style))
    story.append(Paragraph(title_text, title_style))
    story.append(hr())
    story.append(Paragraph(f"Dataset: <b>{dataset_name}</b>", subtitle_style))
    story.append(Paragraph(f"Generated: {generated_at}", caption_style))

    # Optional intro paragraph
    intro = config.get("intro_text", "").strip()
    if intro:
        story.append(Spacer(1, 0.5 * cm))
        story.append(Paragraph(intro, body_style))

    story.append(Spacer(1, 0.5 * cm))

    # ---- SECTION 1: DATASET OVERVIEW ----------------------------------------
    if config.get("include_overview", True) and report_data.get("has_dataset"):
        story.append(PageBreak())
        story.append(Paragraph("1. Dataset Overview", section_heading))
        story.append(hr())

        active_df = report_data.get("active_df")
        raw_df = report_data.get("raw_df")
        cleaned_df = report_data.get("cleaned_df")

        # Dimensions
        pairs = [
            ("Dataset Name", dataset_name),
            ("Total Rows", f"{len(active_df):,}"),
            ("Total Columns", f"{len(active_df.columns)}"),
            ("Numerical Columns", ", ".join(
                active_df.select_dtypes(include=[np.number]).columns.tolist()
            ) or "None"),
            ("Categorical Columns", ", ".join(
                active_df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
            ) or "None"),
            ("Missing Values", f"{int(active_df.isnull().sum().sum()):,} cells"),
            ("Duplicate Rows", f"{int(active_df.duplicated().sum()):,}"),
        ]
        if raw_df is not None and cleaned_df is not None and not cleaned_df.empty:
            pairs.append(("Original Rows", f"{len(raw_df):,}"))
            pairs.append(("Cleaned Rows", f"{len(cleaned_df):,}"))

        story.append(kv_table(pairs))
        story.append(Spacer(1, 0.4 * cm))

        # Data Quality Score
        quality = report_data.get("quality_score")
        if quality:
            story.append(Paragraph("Data Quality Score", sub_heading))
            story.append(kv_table([
                ("Score", f"{quality['score']} / 100"),
                ("Grade", quality['grade']),
                ("Missing Rate", f"{quality['missing_pct']}%"),
                ("Duplicate Rate", f"{quality['duplicate_pct']}%"),
            ]))
            if quality.get("issues"):
                story.append(Spacer(1, 0.2 * cm))
                story.append(Paragraph("<b>Quality Issues Detected:</b>", body_style))
                for issue in quality["issues"]:
                    story.append(Paragraph(f"• {issue}", body_style))

        # Column schema
        story.append(Spacer(1, 0.4 * cm))
        story.append(Paragraph("Column Schema", sub_heading))
        schema_data = [
            [Paragraph("<b>Column</b>", small_style),
             Paragraph("<b>Data Type</b>", small_style),
             Paragraph("<b>Non-Null Count</b>", small_style),
             Paragraph("<b>Unique Values</b>", small_style)]
        ]
        for col in active_df.columns:
            schema_data.append([
                Paragraph(str(col), small_style),
                Paragraph(str(active_df[col].dtype), small_style),
                Paragraph(str(active_df[col].count()), small_style),
                Paragraph(str(active_df[col].nunique()), small_style),
            ])
        schema_tbl = Table(
            schema_data,
            colWidths=[5 * cm, 3.5 * cm, 4.5 * cm, 4 * cm],
            repeatRows=1
        )
        schema_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), BRAND_BLUE),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_GRAY]),
            ("GRID", (0, 0), (-1, -1), 0.5, MID_GRAY),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(schema_tbl)

    # ---- SECTION 2: STATISTICAL ANALYSIS ------------------------------------
    if config.get("include_statistics", True) and report_data.get("stats_table") is not None:
        stats_df = report_data["stats_table"]
        if not stats_df.empty:
            story.append(PageBreak())
            story.append(Paragraph("2. Statistical Analysis Summary", section_heading))
            story.append(hr())
            story.append(Paragraph(
                "The table below shows key descriptive statistics for all numerical columns.",
                body_style
            ))
            story.append(Spacer(1, 0.3 * cm))
            story.append(df_to_rl_table(stats_df, max_rows=30))
            story.append(Spacer(1, 0.3 * cm))
            story.append(Paragraph(
                "Glossary: Mean = average; Median = middle value; Std Dev = standard deviation "
                "(spread of data); Skewness = symmetry of distribution (0 = symmetric).",
                small_style
            ))

    # ---- SECTION 3: DATA CLEANING HISTORY -----------------------------------
    if config.get("include_cleaning", True):
        cleaning_history = report_data.get("cleaning_history", [])
        if cleaning_history:
            story.append(PageBreak())
            story.append(Paragraph("3. Data Cleaning History", section_heading))
            story.append(hr())
            story.append(Paragraph(
                "The following data cleaning operations were applied during this session:",
                body_style
            ))
            story.append(Spacer(1, 0.2 * cm))
            for i, step in enumerate(cleaning_history, 1):
                story.append(Paragraph(f"  {i}. {step}", body_style))

            # Before / After summary
            raw_df = report_data.get("raw_df")
            cleaned_df = report_data.get("cleaned_df")
            if raw_df is not None and cleaned_df is not None and not cleaned_df.empty:
                story.append(Spacer(1, 0.4 * cm))
                story.append(Paragraph("Before / After Comparison", sub_heading))
                story.append(kv_table([
                    ("Original Rows", f"{len(raw_df):,}"),
                    ("Cleaned Rows", f"{len(cleaned_df):,}"),
                    ("Rows Removed", f"{len(raw_df) - len(cleaned_df):,}"),
                    ("Original Missing Cells", f"{int(raw_df.isnull().sum().sum()):,}"),
                    ("Cleaned Missing Cells", f"{int(cleaned_df.isnull().sum().sum()):,}"),
                ]))

    # ---- SECTION 4: AUTOMATED INSIGHTS -------------------------------------
    if config.get("include_insights", True):
        insights = report_data.get("insights")
        if insights and "error" not in insights:
            story.append(PageBreak())
            story.append(Paragraph("4. Automated Insights", section_heading))
            story.append(hr())

            # Summary cards
            cards = insights.get("insight_cards", [])
            if cards:
                story.append(Paragraph("Key Findings", sub_heading))
                for card in cards[:8]:  # Limit to 8 cards in PDF
                    story.append(Spacer(1, 0.15 * cm))
                    story.append(Paragraph(f"<b>{card.get('title', '')}</b>", body_style))
                    story.append(Paragraph(card.get("finding", ""), body_style))
                    story.append(Paragraph(
                        f"<i>{card.get('explanation', '')}</i>", small_style
                    ))

            # Strong correlations
            strong_corrs = insights.get("strong_correlations", [])
            if strong_corrs:
                story.append(Spacer(1, 0.3 * cm))
                story.append(Paragraph("Strong Correlations Discovered", sub_heading))
                for c in strong_corrs[:6]:
                    story.append(Paragraph(
                        f"• <b>{c['feature_1']}</b> & <b>{c['feature_2']}</b>: "
                        f"r = {c['correlation']} — {c['interpretation']}",
                        body_style
                    ))
                story.append(Spacer(1, 0.2 * cm))
                story.append(Paragraph(
                    "⚠️ Reminder: Correlation describes association between variables, "
                    "NOT causation. A high correlation does NOT mean one variable causes the other.",
                    small_style
                ))

            # Recommendations
            recommendations = insights.get("recommendations", [])
            if recommendations:
                story.append(Spacer(1, 0.3 * cm))
                story.append(Paragraph("Smart Recommendations", sub_heading))
                for rec in recommendations[:5]:
                    story.append(Paragraph(
                        f"• <b>{rec.get('action', '')}</b>: {rec.get('text', '')}",
                        body_style
                    ))
                    story.append(Paragraph(
                        f"  <i>Why: {rec.get('reason', '')}</i>", small_style
                    ))

    # ---- SECTION 5: MACHINE LEARNING RESULTS --------------------------------
    if config.get("include_ml", True) and report_data.get("ml_bundle"):
        ml = report_data["ml_bundle"]
        ml_task = report_data.get("ml_task", "").capitalize()

        story.append(PageBreak())
        story.append(Paragraph("5. Machine Learning Model Results", section_heading))
        story.append(hr())

        # Model summary
        story.append(Paragraph("Model Configuration", sub_heading))
        pairs = [
            ("Task Type", ml_task),
            ("Algorithm", ml.get("algorithm", "—")),
            ("Target Column", ml.get("target_col", "—")),
            ("Feature Columns", ", ".join(ml.get("feature_cols", []))),
        ]
        story.append(kv_table(pairs))
        story.append(Spacer(1, 0.3 * cm))

        # Evaluation metrics
        metrics = ml.get("metrics", {})
        if metrics:
            story.append(Paragraph("Evaluation Metrics", sub_heading))
            metric_pairs = [(k, str(round(v, 4) if isinstance(v, float) else v))
                            for k, v in metrics.items()]
            story.append(kv_table(metric_pairs))

            story.append(Spacer(1, 0.2 * cm))
            if "r2" in str(metrics).lower() or "r2_score" in metrics:
                story.append(Paragraph(
                    "R² Score: Proportion of target variance explained by the model. "
                    "1.0 = perfect predictions; 0.0 = no better than predicting the mean.",
                    small_style
                ))
            if "accuracy" in metrics:
                story.append(Paragraph(
                    "Accuracy: The percentage of correctly classified records "
                    "(correct predictions ÷ total predictions × 100).",
                    small_style
                ))

        # Feature importance
        fi = ml.get("feature_importance")
        if fi:
            story.append(Spacer(1, 0.3 * cm))
            story.append(Paragraph("Feature Importance", sub_heading))
            story.append(Paragraph(
                "Feature importance shows how much each input column influenced the model's predictions. "
                "Higher scores = more influential.", body_style
            ))
            fi_pairs = [
                (feature, f"{importance:.4f}")
                for feature, importance in sorted(
                    fi.items(), key=lambda x: x[1], reverse=True
                )
            ]
            story.append(kv_table(fi_pairs))

    # ---- SECTION 6: AI EXECUTIVE SUMMARY -----------------------------------
    if config.get("include_ai_summary", False):
        ai_text = config.get("ai_summary_text", "").strip()
        if ai_text:
            story.append(PageBreak())
            story.append(Paragraph("6. AI Executive Summary", section_heading))
            story.append(hr())

            # Split the AI text by lines and render each as a paragraph
            for line in ai_text.split("\n"):
                line = line.strip()
                if not line:
                    story.append(Spacer(1, 0.15 * cm))
                elif line.startswith("##"):
                    story.append(Paragraph(line.lstrip("#").strip(), sub_heading))
                elif line.startswith("#"):
                    story.append(Paragraph(line.lstrip("#").strip(), sub_heading))
                else:
                    story.append(Paragraph(line, body_style))

    # ---- FOOTER NOTE -------------------------------------------------------
    story.append(Spacer(1, 1 * cm))
    story.append(hr())
    story.append(Paragraph(
        f"Generated by InsightLab | {generated_at} | "
        "For interactive charts, open the Visualization Studio inside the application.",
        caption_style
    ))

    # =========================================================================
    # 3g. Build the PDF and return bytes
    # =========================================================================
    doc.build(story)

    # Move the buffer's read position to the start, then read all bytes
    buffer.seek(0)
    return buffer.read()


# =============================================================================
# SECTION 4: EXCEL WORKBOOK BUILDER
# =============================================================================

def build_excel_workbook(report_data: Dict[str, Any]) -> bytes:
    """
    Builds a multi-sheet Excel workbook and returns its raw bytes.

    Sheet Structure:
    ----------------
    1. "Dataset Overview"  — key metrics (dimensions, quality score, schema).
    2. "Raw Data"          — original uploaded data (max 500 rows).
    3. "Cleaned Data"      — cleaned working copy (max 500 rows), if available.
    4. "Statistical Summary" — full statistics table for all numerical columns.
    5. "Insights Summary"  — top findings and recommendations in text form.
    6. "ML Results"        — machine learning metrics and feature importance.

    What is io.BytesIO?
    -------------------
    openpyxl normally saves to a file path. By pointing ExcelWriter to a
    BytesIO buffer, all Excel bytes are written to RAM and we can hand them
    directly to Streamlit's download button.

    Parameters:
    -----------
    report_data : dict — output of collect_report_data()

    Returns:
    --------
    bytes — raw .xlsx content.
    """
    buffer = io.BytesIO()

    # pd.ExcelWriter is Pandas' Excel-writing context manager.
    # 'xlsxwriter' or 'openpyxl' are the engines; openpyxl supports formatting.
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:

        dataset_name = report_data.get("dataset_name", "dataset")
        active_df = report_data.get("active_df")
        raw_df = report_data.get("raw_df")
        cleaned_df = report_data.get("cleaned_df")
        stats_table = report_data.get("stats_table")
        ml_bundle = report_data.get("ml_bundle")
        insights = report_data.get("insights")
        quality = report_data.get("quality_score")
        cleaning_history = report_data.get("cleaning_history", [])

        # ------------------------------------------------------------------
        # Sheet 1: Dataset Overview
        # ------------------------------------------------------------------
        overview_rows = []
        if active_df is not None:
            overview_rows.extend([
                {"Metric": "Dataset Name", "Value": dataset_name},
                {"Metric": "Total Rows", "Value": len(active_df)},
                {"Metric": "Total Columns", "Value": len(active_df.columns)},
                {"Metric": "Missing Cells", "Value": int(active_df.isnull().sum().sum())},
                {"Metric": "Duplicate Rows", "Value": int(active_df.duplicated().sum())},
                {"Metric": "Numerical Columns",
                 "Value": ", ".join(active_df.select_dtypes(include=[np.number]).columns.tolist())},
                {"Metric": "Categorical Columns",
                 "Value": ", ".join(active_df.select_dtypes(include=["object", "category", "string"]).columns.tolist())},
            ])
        if quality:
            overview_rows.extend([
                {"Metric": "Quality Score", "Value": f"{quality['score']} / 100"},
                {"Metric": "Quality Grade", "Value": quality['grade']},
                {"Metric": "Missing Rate (%)", "Value": quality['missing_pct']},
                {"Metric": "Duplicate Rate (%)", "Value": quality['duplicate_pct']},
            ])
        if cleaning_history:
            for i, step in enumerate(cleaning_history, 1):
                overview_rows.append({"Metric": f"Cleaning Step {i}", "Value": step})

        if overview_rows:
            pd.DataFrame(overview_rows).to_excel(
                writer, sheet_name="Dataset Overview", index=False
            )
        else:
            pd.DataFrame([{"Metric": "No data uploaded", "Value": ""}]).to_excel(
                writer, sheet_name="Dataset Overview", index=False
            )

        # ------------------------------------------------------------------
        # Sheet 2: Raw Data (max 500 rows)
        # ------------------------------------------------------------------
        if raw_df is not None and not raw_df.empty:
            raw_df.head(500).to_excel(writer, sheet_name="Raw Data", index=False)
        else:
            pd.DataFrame([{"Note": "No raw data available"}]).to_excel(
                writer, sheet_name="Raw Data", index=False
            )

        # ------------------------------------------------------------------
        # Sheet 3: Cleaned Data (max 500 rows)
        # ------------------------------------------------------------------
        if cleaned_df is not None and not cleaned_df.empty:
            cleaned_df.head(500).to_excel(writer, sheet_name="Cleaned Data", index=False)
        else:
            pd.DataFrame([{"Note": "No cleaning operations applied yet"}]).to_excel(
                writer, sheet_name="Cleaned Data", index=False
            )

        # ------------------------------------------------------------------
        # Sheet 4: Statistical Summary
        # ------------------------------------------------------------------
        if stats_table is not None and not stats_table.empty:
            stats_table.to_excel(writer, sheet_name="Statistical Summary", index=False)
        else:
            pd.DataFrame([{"Note": "No numerical columns available for statistics"}]).to_excel(
                writer, sheet_name="Statistical Summary", index=False
            )

        # ------------------------------------------------------------------
        # Sheet 5: Insights Summary
        # ------------------------------------------------------------------
        insights_rows = []
        if insights and "error" not in insights:
            cards = insights.get("insight_cards", [])
            for card in cards:
                insights_rows.append({
                    "Type": card.get("type", ""),
                    "Title": card.get("title", ""),
                    "Finding": card.get("finding", ""),
                    "Explanation": card.get("explanation", ""),
                    "Columns": ", ".join(card.get("columns", [])),
                })

            # Add correlations
            strong_corrs = insights.get("strong_correlations", [])
            for c in strong_corrs:
                insights_rows.append({
                    "Type": "correlation",
                    "Title": f"{c['feature_1']} & {c['feature_2']}",
                    "Finding": f"r = {c['correlation']}",
                    "Explanation": c.get("interpretation", ""),
                    "Columns": f"{c['feature_1']}, {c['feature_2']}",
                })

        if insights_rows:
            pd.DataFrame(insights_rows).to_excel(
                writer, sheet_name="Insights Summary", index=False
            )
        else:
            pd.DataFrame([{"Note": "No insights available. Run Automated Insights Studio first."}]).to_excel(
                writer, sheet_name="Insights Summary", index=False
            )

        # ------------------------------------------------------------------
        # Sheet 6: ML Results
        # ------------------------------------------------------------------
        ml_rows = []
        if ml_bundle:
            metrics = ml_bundle.get("metrics", {})
            for k, v in metrics.items():
                ml_rows.append({"Section": "Evaluation Metric", "Key": k, "Value": v})

            fi = ml_bundle.get("feature_importance")
            if fi:
                for feature, importance in sorted(fi.items(), key=lambda x: x[1], reverse=True):
                    ml_rows.append({
                        "Section": "Feature Importance",
                        "Key": feature,
                        "Value": round(importance, 6)
                    })

            ml_rows.insert(0, {"Section": "Configuration", "Key": "Task", "Value": report_data.get("ml_task", "")})
            ml_rows.insert(1, {"Section": "Configuration", "Key": "Algorithm", "Value": ml_bundle.get("algorithm", "")})
            ml_rows.insert(2, {"Section": "Configuration", "Key": "Target Column", "Value": ml_bundle.get("target_col", "")})
            ml_rows.insert(3, {"Section": "Configuration", "Key": "Feature Columns", "Value": ", ".join(ml_bundle.get("feature_cols", []))})

        if ml_rows:
            pd.DataFrame(ml_rows).to_excel(writer, sheet_name="ML Results", index=False)
        else:
            pd.DataFrame([{"Note": "No ML model trained. Run Machine Learning Studio first."}]).to_excel(
                writer, sheet_name="ML Results", index=False
            )

    # Return the buffer content as bytes
    buffer.seek(0)
    return buffer.read()


# =============================================================================
# SECTION 5: INDIVIDUAL CSV EXPORTS
# =============================================================================

def export_df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Exports a DataFrame to CSV bytes (UTF-8 with BOM for Excel compatibility).

    The BOM (Byte Order Mark) ensures Excel opens the CSV with correct
    character encoding, especially for special characters.
    """
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def export_insights_to_markdown_bytes(report_data: Dict[str, Any]) -> bytes:
    """
    Exports the automated insights report as UTF-8 markdown bytes.
    """
    try:
        from utils.insights_engine import generate_insights_markdown_report
        insights = report_data.get("insights")
        dataset_name = report_data.get("dataset_name", "dataset")
        if insights and "error" not in insights:
            md_text = generate_insights_markdown_report(insights, dataset_name)
        else:
            md_text = "# InsightLab Automated Insights\n\nNo insights available."
        return md_text.encode("utf-8")
    except Exception as e:
        return f"# Error\n\n{e}".encode("utf-8")
