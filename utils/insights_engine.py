"""
insights_engine.py
==================
Automated Insights Engine for InsightLab (Phase 5).

Core Principles:
----------------
1. Deterministic & Rigorous:
   All findings are computed strictly using statistical formulas with Pandas & NumPy.
   No hallucinated, fabricated, or ungrounded statements.

2. Statistical Soundness:
   - Outliers are detected using the standard Tukey's Interquartile Range (IQR) method (1.5 * IQR).
   - Health score is computed as a weighted index of completeness and uniqueness.
   - Correlations are clearly labeled with the reminder that correlation != causation.

3. Beginner-Friendly Explanations:
   Translates complex mathematical metrics (skewness, variance, IQR, Pearson r) into
   intuitive, actionable business narratives.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from utils.statistical_analyzer import (
    compute_column_statistics,
    compute_correlation_matrix,
    get_top_correlations,
    interpret_skewness
)


def detect_outliers_iqr(series: pd.Series) -> Dict[str, Any]:
    """
    Detects numerical anomalies/outliers using Tukey's Interquartile Range (IQR) method.

    What is Tukey's IQR Method?
    ---------------------------
    - Q1 = 25th percentile (bottom quarter of data)
    - Q3 = 75th percentile (top quarter of data)
    - IQR = Q3 - Q1 (the middle 50% range)
    - Lower Fence = Q1 - 1.5 * IQR
    - Upper Fence = Q3 + 1.5 * IQR

    Any value beyond these fences is mathematically flagged as an outlier.
    """
    clean = series.dropna()
    if len(clean) < 4:
        return {
            "has_outliers": False,
            "outlier_count": 0,
            "outlier_pct": 0.0,
            "lower_bound": None,
            "upper_bound": None,
            "outlier_values": []
        }

    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    iqr = q3 - q1

    lower_bound = round(q1 - 1.5 * iqr, 3)
    upper_bound = round(q3 + 1.5 * iqr, 3)

    outliers = clean[(clean < lower_bound) | (clean > upper_bound)].tolist()
    count = len(outliers)
    pct = round((count / len(clean)) * 100, 2)

    return {
        "has_outliers": count > 0,
        "outlier_count": count,
        "outlier_pct": pct,
        "q1": round(q1, 3),
        "q3": round(q3, 3),
        "iqr": round(iqr, 3),
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_values": outliers[:10]  # First 10 samples
    }


def compute_dataset_quality_score(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes an objective Data Quality & Cleanliness Score (0 to 100).

    Scoring Criteria:
    -----------------
    - Completeness (60% weight): Ratio of populated cells vs total cells.
    - Uniqueness (40% weight): Ratio of unique rows vs total rows.
    """
    if df is None or df.empty:
        return {
            "score": 0,
            "grade": "F",
            "missing_pct": 100.0,
            "duplicate_pct": 0.0,
            "issues": ["Dataset is empty."]
        }

    total_rows = len(df)
    total_cells = total_rows * len(df.columns)
    missing_cells = int(df.isnull().sum().sum())
    missing_pct = round((missing_cells / total_cells) * 100, 2) if total_cells > 0 else 0.0

    duplicate_rows = int(df.duplicated().sum())
    dup_pct = round((duplicate_rows / total_rows) * 100, 2) if total_rows > 0 else 0.0

    # Calculate weighted health score
    completeness_score = max(0.0, 100.0 - (missing_pct * 2.0))
    uniqueness_score = max(0.0, 100.0 - (dup_pct * 2.5))
    final_score = int(round(completeness_score * 0.6 + uniqueness_score * 0.4))
    final_score = max(0, min(100, final_score))

    if final_score >= 90:
        grade = "A (Excellent)"
    elif final_score >= 80:
        grade = "B (Good)"
    elif final_score >= 70:
        grade = "C (Fair)"
    elif final_score >= 60:
        grade = "D (Poor)"
    else:
        grade = "F (Critical Issues)"

    # Identify specific actionable quality issues
    issues = []
    if duplicate_rows > 0:
        issues.append(f"{duplicate_rows} duplicate row(s) detected ({dup_pct}% of total records).")
    if missing_cells > 0:
        issues.append(f"{missing_cells} missing cell(s) across dataset ({missing_pct}% total).")

    # Detect constant columns (only 1 unique value)
    for col in df.columns:
        if df[col].nunique(dropna=True) == 1:
            issues.append(f"Column '{col}' is constant (contains only 1 unique value).")

    # Detect high missing rate columns (> 20%)
    for col in df.columns:
        col_missing = df[col].isnull().sum()
        col_pct = (col_missing / total_rows) * 100
        if col_pct > 20:
            issues.append(f"Column '{col}' has a severe missing rate of {col_pct:.1f}%.")

    return {
        "score": final_score,
        "grade": grade,
        "missing_pct": missing_pct,
        "duplicate_pct": dup_pct,
        "duplicate_count": duplicate_rows,
        "missing_count": missing_cells,
        "issues": issues
    }


def analyze_category_performers(df: pd.DataFrame, cat_col: str, num_col: str) -> Optional[Dict[str, Any]]:
    """
    Identifies top and bottom performing categories for a numerical metric.
    """
    clean_df = df[[cat_col, num_col]].dropna()
    if clean_df.empty or clean_df[cat_col].nunique() < 2:
        return None

    grouped = clean_df.groupby(cat_col)[num_col].agg(["sum", "mean", "count"]).reset_index()
    grouped_sum = grouped.sort_values(by="sum", ascending=False)

    total_sum = grouped_sum["sum"].sum()
    top_cat = grouped_sum.iloc[0]
    bottom_cat = grouped_sum.iloc[-1]

    top_share = round((top_cat["sum"] / total_sum) * 100, 1) if total_sum != 0 else 0.0
    bottom_share = round((bottom_cat["sum"] / total_sum) * 100, 1) if total_sum != 0 else 0.0

    return {
        "cat_col": cat_col,
        "num_col": num_col,
        "top_category": str(top_cat[cat_col]),
        "top_sum": round(float(top_cat["sum"]), 2),
        "top_mean": round(float(top_cat["mean"]), 2),
        "top_share_pct": top_share,
        "bottom_category": str(bottom_cat[cat_col]),
        "bottom_sum": round(float(bottom_cat["sum"]), 2),
        "bottom_mean": round(float(bottom_cat["mean"]), 2),
        "bottom_share_pct": bottom_share,
        "total_categories": len(grouped)
    }


def generate_automated_insights(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Main analytical engine that scans the dataset and produces structured findings,
    insight cards, relationship discoveries, anomaly reports, and smart recommendations.
    """
    if df is None or df.empty:
        return {"error": "Dataset is empty."}

    num_rows, num_cols = df.shape
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    # 1. Dataset Quality
    quality = compute_dataset_quality_score(df)

    # 2. Key Statistical Insights & Cards
    cards: List[Dict[str, Any]] = []

    # A. Anomaly / Outlier Scan across numerical columns
    outlier_reports = {}
    for col in numeric_cols:
        outlier_data = detect_outliers_iqr(df[col])
        if outlier_data["has_outliers"]:
            outlier_reports[col] = outlier_data
            cards.append({
                "type": "anomaly",
                "badge": "⚠️ Outliers Detected",
                "title": f"Unusual Values Detected in `{col}`",
                "finding": f"Found **{outlier_data['outlier_count']}** unusual records ({outlier_data['outlier_pct']}% of column data).",
                "explanation": (
                    f"Values falling below {outlier_data['lower_bound']} or above {outlier_data['upper_bound']} "
                    f"deviate significantly from the expected range (based on 1.5 × IQR). "
                    f"Examples: {outlier_data['outlier_values'][:3]}."
                ),
                "columns": [col],
                "chart_recommendation": {
                    "chart_type": "box",
                    "col": col
                }
            })

    # B. Category Leader Insights
    category_insights = []
    if categorical_cols and numeric_cols:
        # Check first 2 meaningful categorical columns against the first 2 numerical columns
        for c_col in categorical_cols[:2]:
            for n_col in numeric_cols[:2]:
                performer = analyze_category_performers(df, c_col, n_col)
                if performer and performer["top_share_pct"] > 30.0:
                    category_insights.append(performer)
                    cards.append({
                        "type": "performance",
                        "badge": "🏆 Category Leader",
                        "title": f"`{performer['top_category']}` Dominates `{performer['num_col']}`",
                        "finding": (
                            f"**{performer['top_category']}** accounts for **{performer['top_share_pct']}%** "
                            f"of total {performer['num_col']} ({performer['top_sum']:,}), averaging {performer['top_mean']:,} per record."
                        ),
                        "explanation": (
                            f"In contrast, the lowest contributor is **{performer['bottom_category']}** "
                            f"accounting for only {performer['bottom_share_pct']}% of {performer['num_col']}."
                        ),
                        "columns": [c_col, n_col],
                        "chart_recommendation": {
                            "chart_type": "bar",
                            "x_col": c_col,
                            "y_col": n_col
                        }
                    })

    # C. Strong Correlation Insights
    corr_matrix = compute_correlation_matrix(df)
    top_corrs = get_top_correlations(corr_matrix) if not corr_matrix.empty else []
    strong_corrs = [p for p in top_corrs if p["abs_corr"] >= 0.5]

    for corr_item in strong_corrs[:3]:
        r_val = corr_item["correlation"]
        direction = "Positive" if r_val > 0 else "Negative"
        cards.append({
            "type": "relationship",
            "badge": f"🔗 Strong {direction} Correlation (r = {r_val})",
            "title": f"Relationship: `{corr_item['feature_1']}` & `{corr_item['feature_2']}`",
            "finding": f"Strong linear association of **r = {r_val:.2f}** discovered.",
            "explanation": (
                f"{corr_item['interpretation']} "
                f"*(Note: Correlation describes an observed association, never proof of causation.)*"
            ),
            "columns": [corr_item["feature_1"], corr_item["feature_2"]],
            "chart_recommendation": {
                "chart_type": "scatter",
                "x_col": corr_item["feature_1"],
                "y_col": corr_item["feature_2"]
            }
        })

    # D. Distribution Skewness Insights
    for col in numeric_cols[:3]:
        stats = compute_column_statistics(df[col])
        if stats["skewness"] is not None and abs(stats["skewness"]) >= 1.0:
            skew_desc = "Right-Skewed (tail stretches high)" if stats["skewness"] > 0 else "Left-Skewed (tail stretches low)"
            cards.append({
                "type": "distribution",
                "badge": "📊 Heavy Distribution Skew",
                "title": f"Skewed Distribution in `{col}`",
                "finding": f"Skewness score is **{stats['skewness']}** ({skew_desc}).",
                "explanation": (
                    f"Because of this skewness, the **Median ({stats['median']:,})** represents a more reliable "
                    f"central typical value than the **Mean ({stats['mean']:,})**."
                ),
                "columns": [col],
                "chart_recommendation": {
                    "chart_type": "histogram",
                    "col": col
                }
            })

    # 3. Smart Recommendations
    recommendations = []
    if quality["duplicate_count"] > 0:
        recommendations.append({
            "action": "Deduplication",
            "text": f"Remove the {quality['duplicate_count']} duplicate records using the Data Cleaning Studio to prevent double-counting.",
            "reason": "Duplicate records inflate totals and artificially distort means and statistical variances."
        })

    if quality["missing_count"] > 0:
        recommendations.append({
            "action": "Missing Value Strategy",
            "text": f"Address {quality['missing_count']} missing cells using median imputation for skewed numerical columns.",
            "reason": "Preserves sample size without introducing artificial distortion."
        })

    if outlier_reports:
        outlier_cols_str = ", ".join([f"`{c}`" for c in list(outlier_reports.keys())[:2]])
        recommendations.append({
            "action": "Outlier Investigation",
            "text": f"Investigate extreme values in {outlier_cols_str}.",
            "reason": "Confirm whether outliers are data entry typos or represent legitimate high-value transactions."
        })

    if strong_corrs:
        top_p = strong_corrs[0]
        recommendations.append({
            "action": "Bivariate Deep Dive",
            "text": f"Explore the scatter trend between `{top_p['feature_1']}` and `{top_p['feature_2']}` in the Visualization Studio.",
            "reason": f"With r = {top_p['correlation']}, tracking these two metrics together can help forecast performance."
        })

    # Default general recommendation if clean
    if not recommendations:
        recommendations.append({
            "action": "Exploratory Visualizations",
            "text": "Your dataset is remarkably clean! Proceed to the Visualization Studio to build executive dashboards.",
            "reason": "No structural data defects or heavy missing values were found."
        })

    return {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "quality": quality,
        "cards": cards,
        "outlier_reports": outlier_reports,
        "top_correlations": top_corrs,
        "strong_correlations": strong_corrs,
        "category_insights": category_insights,
        "recommendations": recommendations
    }


def generate_insights_markdown_report(insights: Dict[str, Any], dataset_name: str) -> str:
    """
    Builds a downloadable, comprehensive text/markdown report summarizing all insights.
    """
    quality = insights.get("quality", {})
    cards = insights.get("cards", [])
    recommendations = insights.get("recommendations", [])

    lines = [
        f"# 🔬 InsightLab — Automated Insights Report",
        f"**Dataset:** `{dataset_name}`",
        f"**Records:** {insights.get('num_rows', 0):,} rows | {insights.get('num_cols', 0)} columns",
        f"**Data Cleanliness Score:** {quality.get('score', 0)}/100 ({quality.get('grade', 'N/A')})",
        "",
        "---",
        "",
        "## 1. Executive Summary & Data Health",
        f"- **Missing Cells:** {quality.get('missing_count', 0)} ({quality.get('missing_pct', 0.0)}%)",
        f"- **Duplicate Records:** {quality.get('duplicate_count', 0)} ({quality.get('duplicate_pct', 0.0)}%)",
        f"- **Numerical Columns:** {len(insights.get('numeric_cols', []))}",
        f"- **Categorical Columns:** {len(insights.get('categorical_cols', []))}",
        ""
    ]

    if quality.get("issues"):
        lines.append("### Flagged Quality Issues:")
        for issue in quality["issues"]:
            lines.append(f"- {issue}")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## 2. Key Statistical Insights & Findings",
        ""
    ])

    if not cards:
        lines.append("No critical patterns or anomalies detected.")
    else:
        for idx, card in enumerate(cards, 1):
            lines.append(f"### {idx}. {card['title']}")
            lines.append(f"**Finding:** {card['finding']}")
            lines.append(f"**Explanation:** {card['explanation']}")
            lines.append(f"**Columns Involved:** {', '.join(card['columns'])}")
            lines.append("")

    lines.extend([
        "---",
        "",
        "## 3. Relationships & Correlations",
        ""
    ])
    strong_corrs = insights.get("strong_correlations", [])
    if strong_corrs:
        for c in strong_corrs:
            lines.append(f"- **`{c['feature_1']}` & `{c['feature_2']}`:** r = {c['correlation']} ({c['interpretation']})")
    else:
        lines.append("No strong linear relationships (|r| >= 0.5) discovered.")

    lines.append("")
    lines.append("> *Reminder: Correlation describes association, not causality.*")
    lines.append("")

    lines.extend([
        "---",
        "",
        "## 4. Smart Recommendations & Next Steps",
        ""
    ])
    for idx, rec in enumerate(recommendations, 1):
        lines.append(f"**{idx}. {rec['action']}:** {rec['text']}")
        lines.append(f"   *Why:* {rec['reason']}")
        lines.append("")

    lines.append("---")
    lines.append("*Generated by InsightLab Automated Insights Studio.*")

    return "\n".join(lines)
