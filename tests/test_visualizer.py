"""
Unit Tests for Visualizer (`utils/visualizer.py`)
=================================================
Tests all Phase 4 charting and validation functions:
- Input validation & defense against incompatible data types
- Bar chart generation (Count, Sum, Mean)
- Line chart generation
- Pie & Donut chart generation
- Histogram generation with marginals
- Scatter plot generation with trendline
- Correlation heatmap generation
- Standalone HTML figure export
"""

import unittest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.visualizer import (
    validate_chart_inputs,
    build_bar_chart,
    build_line_chart,
    build_pie_chart,
    build_histogram,
    build_scatter_plot,
    build_correlation_heatmap,
    export_figure_to_html
)


class TestVisualizer(unittest.TestCase):

    def setUp(self):
        """Create a synthetic dataset for testing all visualization types."""
        self.df = pd.DataFrame({
            "category": ["Electronics", "Furniture", "Office", "Electronics", "Furniture"],
            "date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"],
            "sales": [500.0, 150.0, 45.0, 900.0, 220.0],
            "profit": [100.0, 30.0, 10.0, 250.0, 50.0],
            "quantity": [2, 1, 5, 3, 2]
        })

    def test_validation_bar_chart(self):
        """Test bar chart input validation."""
        # Valid: Category on X, numerical on Y
        valid, msg = validate_chart_inputs("Bar Chart", self.df, x_col="category", y_col="sales")
        self.assertTrue(valid)
        self.assertIsNone(msg)

        # Incompatible: Text on Y-axis
        valid, msg = validate_chart_inputs("Bar Chart", self.df, x_col="category", y_col="date")
        self.assertFalse(valid)
        self.assertIn("must be numerical", msg)

    def test_validation_scatter_plot(self):
        """Test scatter plot requires two numerical columns."""
        # Valid: numerical X and numerical Y
        valid, msg = validate_chart_inputs("Scatter Plot", self.df, x_col="sales", y_col="profit")
        self.assertTrue(valid)

        # Incompatible: category on X
        valid, msg = validate_chart_inputs("Scatter Plot", self.df, x_col="category", y_col="profit")
        self.assertFalse(valid)
        self.assertIn("must be numerical", msg)

    def test_validation_histogram(self):
        """Test histogram rejects non-numeric continuous distributions."""
        valid, msg = validate_chart_inputs("Histogram", self.df, x_col="category")
        self.assertFalse(valid)
        self.assertIn("Histograms are designed for continuous numerical data", msg)

        valid, msg = validate_chart_inputs("Histogram", self.df, x_col="sales")
        self.assertTrue(valid)

    def test_build_bar_chart(self):
        """Test creating a bar chart with Sum and Count aggregations."""
        fig_sum = build_bar_chart(self.df, x_col="category", y_col="sales", aggregation="Sum")
        self.assertIsInstance(fig_sum, go.Figure)

        fig_count = build_bar_chart(self.df, x_col="category", aggregation="Count")
        self.assertIsInstance(fig_count, go.Figure)

    def test_build_line_chart(self):
        """Test creating a line chart."""
        fig_line = build_line_chart(self.df, x_col="date", y_col="sales")
        self.assertIsInstance(fig_line, go.Figure)

    def test_build_pie_chart(self):
        """Test creating standard pie and donut charts."""
        fig_pie = build_pie_chart(self.df, names_col="category", values_col="sales", is_donut=False)
        self.assertIsInstance(fig_pie, go.Figure)

        fig_donut = build_pie_chart(self.df, names_col="category", values_col="sales", is_donut=True)
        self.assertIsInstance(fig_donut, go.Figure)

    def test_build_histogram(self):
        """Test creating a histogram."""
        fig_hist = build_histogram(self.df, x_col="sales", nbins=5, marginal="box")
        self.assertIsInstance(fig_hist, go.Figure)

    def test_build_scatter_plot(self):
        """Test creating a scatter plot with OLS trendline."""
        fig_scatter = build_scatter_plot(
            self.df,
            x_col="sales",
            y_col="profit",
            color_col="category",
            add_trendline=True
        )
        self.assertIsInstance(fig_scatter, go.Figure)
        # Verify trendline trace exists in figure data
        trace_names = [trace.name for trace in fig_scatter.data if trace.name]
        self.assertTrue(any("Trendline" in name for name in trace_names))

    def test_build_correlation_heatmap(self):
        """Test creating a correlation heatmap."""
        fig_heat = build_correlation_heatmap(self.df, columns=["sales", "profit", "quantity"])
        self.assertIsInstance(fig_heat, go.Figure)

    def test_export_figure_to_html(self):
        """Test exporting figure to self-contained HTML."""
        fig = build_bar_chart(self.df, x_col="category", y_col="sales")
        html_str = export_figure_to_html(fig)
        self.assertIsInstance(html_str, str)
        self.assertIn("html", html_str.lower())
        self.assertIn("plotly", html_str.lower())


if __name__ == "__main__":
    unittest.main()
