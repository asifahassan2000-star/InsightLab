"""
visualizer.py
=============
This module handles chart construction, data type validation, and export logic
for the Interactive Visualization Studio (Phase 4).

Why separate chart building from the Streamlit UI?
--------------------------------------------------
1. Separation of Concerns: Keeps plotting logic modular, testable, and reusable.
2. Defensive Charting: Validates that column data types are statistically
   compatible with the requested chart type before plotting.
3. Extensibility: Makes it easy to add new chart types without rewriting the UI.
"""

from typing import Tuple, Optional, List, Dict, Any
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# Supported themes in Plotly
PLOTLY_THEMES = [
    "plotly_white",
    "plotly",
    "plotly_dark",
    "seaborn",
    "simple_white",
    "ggplot2"
]

# Supported Color Palettes
COLOR_PALETTES = {
    "Default (Plotly)": px.colors.qualitative.Plotly,
    "Bold": px.colors.qualitative.Bold,
    "Vivid": px.colors.qualitative.Vivid,
    "Pastel": px.colors.qualitative.Pastel,
    "Safe (Colorblind-Friendly)": px.colors.qualitative.Safe,
    "Prism": px.colors.qualitative.Prism
}


def validate_chart_inputs(
    chart_type: str,
    df: pd.DataFrame,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    numeric_cols: Optional[List[str]] = None
) -> Tuple[bool, Optional[str]]:
    """
    Validates that user-selected columns are statistically and logically
    compatible with the chosen chart type. Prevents misleading visualizations.

    Returns:
    --------
    Tuple[bool, Optional[str]]:
        (is_valid, error_or_warning_message)
    """
    if df is None or df.empty:
        return False, "The dataset is empty. Please upload or select a valid dataset."

    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if chart_type == "Bar Chart":
        if not x_col:
            return False, "Please select an X-axis category column for the bar chart."
        if y_col and y_col not in numeric_cols:
            return False, f"Bar chart values (Y-axis) must be numerical. '{y_col}' is categorical/text."

    elif chart_type == "Line Chart":
        if not x_col or not y_col:
            return False, "Line charts require both an X-axis (e.g., date or sequence) and a Y-axis metric."
        if y_col not in numeric_cols:
            return False, f"Line chart trends (Y-axis) must be numerical. '{y_col}' is non-numeric."

    elif chart_type == "Pie Chart":
        if not x_col:
            return False, "Pie charts require a category column for slice labels."
        if y_col and y_col not in numeric_cols:
            return False, f"Pie chart slice sizes (values) must be numerical. '{y_col}' is non-numeric."
        if df[x_col].nunique() > 25:
            return False, (
                f"Column '{x_col}' has {df[x_col].nunique()} unique values! "
                "Pie charts with more than 20 slices become cluttered and misleading. "
                "Consider a horizontal bar chart instead."
            )

    elif chart_type == "Histogram":
        if not x_col:
            return False, "Please select a numerical column to plot its frequency distribution."
        if x_col not in numeric_cols:
            return False, (
                f"Histograms are designed for continuous numerical data to show distribution frequencies. "
                f"'{x_col}' is categorical/text. Use a Bar Chart instead."
            )

    elif chart_type == "Scatter Plot":
        if not x_col or not y_col:
            return False, "Scatter plots require both an X-axis and Y-axis variable to measure relationships."
        if x_col not in numeric_cols:
            return False, f"Scatter plot X-axis must be numerical. '{x_col}' is non-numeric."
        if y_col not in numeric_cols:
            return False, f"Scatter plot Y-axis must be numerical. '{y_col}' is non-numeric."

    elif chart_type == "Correlation Heatmap":
        if len(numeric_cols) < 2:
            return False, "Correlation heatmaps require at least two numerical columns in the dataset."

    return True, None


def build_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: Optional[str] = None,
    color_col: Optional[str] = None,
    aggregation: str = "Sum",
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    theme: str = "plotly_white",
    color_sequence: Optional[List[str]] = None,
    barmode: str = "group"
) -> go.Figure:
    """Creates an interactive Plotly Bar Chart with customizable aggregation."""
    # Handle missing values in chosen columns
    subset_cols = [c for c in [x_col, y_col, color_col] if c]
    plot_df = df.dropna(subset=subset_cols).copy()

    if plot_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data remaining after removing missing values.", showarrow=False)
        return fig

    # If no Y column is provided or user selected 'Count', aggregate frequency
    if y_col is None or aggregation == "Count":
        if color_col and color_col != x_col:
            grouped = plot_df.groupby([x_col, color_col]).size().reset_index(name="Count")
            fig = px.bar(
                grouped,
                x=x_col,
                y="Count",
                color=color_col,
                barmode=barmode,
                color_discrete_sequence=color_sequence
            )
        else:
            grouped = plot_df[x_col].value_counts().reset_index()
            grouped.columns = [x_col, "Count"]
            fig = px.bar(
                grouped,
                x=x_col,
                y="Count",
                color=x_col,
                color_discrete_sequence=color_sequence
            )
        y_effective = "Count"
    else:
        # Group and aggregate
        group_cols = [x_col]
        if color_col and color_col != x_col:
            group_cols.append(color_col)

        if aggregation == "Sum":
            grouped = plot_df.groupby(group_cols, as_index=False)[y_col].sum()
        elif aggregation == "Mean":
            grouped = plot_df.groupby(group_cols, as_index=False)[y_col].mean()
            grouped[y_col] = grouped[y_col].round(2)
        elif aggregation == "Median":
            grouped = plot_df.groupby(group_cols, as_index=False)[y_col].median()
            grouped[y_col] = grouped[y_col].round(2)
        else:
            grouped = plot_df.groupby(group_cols, as_index=False)[y_col].sum()

        fig = px.bar(
            grouped,
            x=x_col,
            y=y_col,
            color=color_col if color_col else x_col,
            barmode=barmode,
            color_discrete_sequence=color_sequence
        )
        y_effective = f"{aggregation} of {y_col}"

    fig.update_layout(
        title=title or f"{y_effective} by {x_col}",
        xaxis_title=x_label or x_col,
        yaxis_title=y_label or y_effective,
        template=theme,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def build_line_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    theme: str = "plotly_white",
    color_sequence: Optional[List[str]] = None,
    show_markers: bool = True
) -> go.Figure:
    """Creates an interactive Plotly Line Chart for trends over time or sequence."""
    subset_cols = [c for c in [x_col, y_col, color_col] if c]
    plot_df = df.dropna(subset=subset_cols).copy()

    # Sort by X axis for smooth, meaningful line connections
    plot_df = plot_df.sort_values(by=x_col)

    fig = px.line(
        plot_df,
        x=x_col,
        y=y_col,
        color=color_col,
        markers=show_markers,
        color_discrete_sequence=color_sequence
    )
    fig.update_layout(
        title=title or f"Trend of {y_col} across {x_col}",
        xaxis_title=x_label or x_col,
        yaxis_title=y_label or y_col,
        template=theme,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def build_pie_chart(
    df: pd.DataFrame,
    names_col: str,
    values_col: Optional[str] = None,
    is_donut: bool = False,
    title: str = "",
    theme: str = "plotly_white",
    color_sequence: Optional[List[str]] = None
) -> go.Figure:
    """Creates an interactive Plotly Pie or Donut chart for proportions."""
    subset_cols = [c for c in [names_col, values_col] if c]
    plot_df = df.dropna(subset=subset_cols).copy()

    if values_col:
        # Group sum
        grouped = plot_df.groupby(names_col, as_index=False)[values_col].sum()
        fig = px.pie(
            grouped,
            names=names_col,
            values=values_col,
            hole=0.4 if is_donut else 0.0,
            color_discrete_sequence=color_sequence
        )
        val_name = f"Sum of {values_col}"
    else:
        # Frequency count
        counts = plot_df[names_col].value_counts().reset_index()
        counts.columns = [names_col, "Count"]
        fig = px.pie(
            counts,
            names=names_col,
            values="Count",
            hole=0.4 if is_donut else 0.0,
            color_discrete_sequence=color_sequence
        )
        val_name = "Count"

    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(
        title=title or f"Proportion of {val_name} by {names_col}",
        template=theme,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def build_histogram(
    df: pd.DataFrame,
    x_col: str,
    color_col: Optional[str] = None,
    nbins: int = 20,
    marginal: str = "box",
    title: str = "",
    x_label: str = "",
    y_label: str = "Frequency",
    theme: str = "plotly_white",
    color_sequence: Optional[List[str]] = None
) -> go.Figure:
    """Creates an interactive Plotly Histogram with optional marginal box/rug."""
    subset_cols = [c for c in [x_col, color_col] if c]
    plot_df = df.dropna(subset=subset_cols).copy()

    marginal_opt = marginal if marginal in ["box", "violin", "rug"] else None

    fig = px.histogram(
        plot_df,
        x=x_col,
        color=color_col,
        nbins=nbins,
        marginal=marginal_opt,
        color_discrete_sequence=color_sequence
    )
    fig.update_layout(
        title=title or f"Distribution Frequency: {x_col}",
        xaxis_title=x_label or x_col,
        yaxis_title=y_label,
        template=theme,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def build_scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    size_col: Optional[str] = None,
    add_trendline: bool = False,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    theme: str = "plotly_white",
    color_sequence: Optional[List[str]] = None
) -> go.Figure:
    """Creates an interactive Plotly Scatter Plot with optional color, size, and OLS trendline."""
    subset_cols = [c for c in [x_col, y_col, color_col, size_col] if c]
    plot_df = df.dropna(subset=subset_cols).copy()

    # Ensure size column only contains positive non-zero values if specified
    valid_size = size_col
    if size_col:
        # If size column has negative or zero numbers, shift/scale to avoid Plotly errors
        if (plot_df[size_col] <= 0).any():
            min_val = plot_df[size_col].min()
            plot_df[f"{size_col}_scaled"] = plot_df[size_col] - min_val + 1
            valid_size = f"{size_col}_scaled"

    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        color=color_col,
        size=valid_size,
        color_discrete_sequence=color_sequence
    )

    # Add NumPy-based OLS linear trendline if requested
    if add_trendline and len(plot_df) > 1 and plot_df[x_col].nunique() > 1:
        x_vals = plot_df[x_col].values
        y_vals = plot_df[y_col].values
        slope, intercept = np.polyfit(x_vals, y_vals, 1)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 50)
        y_line = slope * x_line + intercept

        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name=f"Trendline: y = {slope:.2f}x + {intercept:.2f}",
                line=dict(color="#d62728", dash="dash", width=2)
            )
        )

    fig.update_layout(
        title=title or f"{y_col} vs. {x_col}",
        xaxis_title=x_label or x_col,
        yaxis_title=y_label or y_col,
        template=theme,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def build_correlation_heatmap(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "pearson",
    colorscale: str = "RdBu_r",
    title: str = "",
    theme: str = "plotly_white"
) -> go.Figure:
    """Creates an interactive Plotly Heatmap for the correlation matrix."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target_cols = [c for c in columns if c in numeric_cols] if columns else numeric_cols

    if len(target_cols) < 2:
        fig = go.Figure()
        fig.add_annotation(text="At least 2 numerical columns are required for a correlation heatmap.", showarrow=False)
        return fig

    corr_matrix = df[target_cols].corr(method=method).round(3)

    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale=colorscale,
        zmin=-1,
        zmax=1
    )
    fig.update_layout(
        title=title or f"{method.capitalize()} Correlation Heatmap",
        template=theme,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def export_figure_to_html(fig: go.Figure) -> str:
    """
    Exports a Plotly figure to a self-contained, interactive HTML string.
    Can be downloaded and opened in any browser without needing Python.
    """
    return fig.to_html(include_plotlyjs="cdn", full_html=True)
