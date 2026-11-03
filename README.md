# 🔬 InsightLab — AI-Powered Data Analysis Platform

InsightLab is an interactive, modern data analysis platform built with Python and Streamlit.

---

## 🎯 Features

### Phase 1: Dataset Explorer
1. **File Ingestion**: Seamless upload for `.csv`, `.xlsx`, and `.xls` files.
2. **Instant Preview**: Displays the first 10 rows with an interactive row-count slider.
3. **Dataset Dimensions**: Total rows, columns, duplicate rows, missing ratio, and memory usage.
4. **Column & Data Type Breakdown**: Column names, inferred data types, non-null counts, and missing percentages.
5. **Statistical Insights**: Numerical summary (`df.describe()`) and categorical breakdown.
6. **Graceful Error Handling**: Safe handling of corrupted, empty, or unsupported files.

### Phase 2: Data Cleaning Studio
1. **Non-Destructive Architecture**: Original raw datasets are never overwritten (`df.copy()`).
2. **Quality & Health Inspection**: Identifies duplicate rows and columns with missing values.
3. **Duplicate Row Removal**: Eliminates exact duplicates with one click.
4. **Missing Value Treatment**:
   - **Row Removal (`dropna`)**: Drop rows missing in any column or targeted columns.
   - **Numerical Imputation (`fillna`)**: Impute missing numbers with **Mean** or **Median**.
   - **Categorical Imputation**: Impute missing text/categories with **Mode** (most frequent).
5. **Before-vs-After Statistics**: Real-time delta comparison cards for rows, columns, missing cells, and duplicates.
6. **Cleaning Audit Log**: Detailed step-by-step history of transformations performed.
7. **Cleaned Data Preview & Export**: Interactive table preview and one-click download in **CSV** or **Excel (.xlsx)**.
8. **One-Click Revert**: Reset all cleaning transformations to return to the original data anytime.

### Phase 3: Statistical Analysis Studio
1. **Measures of Central Tendency**: Calculates Mean, Median, and Mode for all numerical variables.
2. **Measures of Dispersion & Spread**: Computes Minimum, Maximum, Range, Variance ($s^2$), Standard Deviation ($s$), and Interquartile Range (IQR).
3. **Distribution & Shape Analysis**:
   - Computes Skewness and Kurtosis with automated shape interpretations (Symmetric, Right-Skewed, Left-Skewed).
   - Interactive Plotly Histograms with Box marginals and Box & Whisker plots with outlier points.
4. **Bivariate Correlation Analysis**:
   - Pearson (linear) and Spearman (rank-order) correlation matrices.
   - Interactive Plotly Heatmap with cell values.
   - Ranked correlation pairs table with plain-English interpretations (Strong Positive, Moderate, Negative, etc.).
   - Interactive Bivariate Scatter Plot with NumPy OLS trendline fitting.
5. **Category-Wise & Department-Wise Comparisons**:
   - GroupBy aggregations of numerical metrics by categorical segments (e.g. `salary` by `department`, `profit` by `category`).
   - Grouped interactive Plotly Bar charts (Mean comparison) and Grouped Box plots (distribution comparison across categories).
6. **Centralized Summary Table**: Filterable statistical table summarizing all numerical columns.

### Phase 4: Interactive Visualization Studio
1. **6 Dynamic Chart Types**:
   - **Bar Charts**: Category comparisons with Sum, Mean, Median, and Count aggregations.
   - **Line Charts**: Time-series and sequential trend tracking with markers.
   - **Pie & Donut Charts**: Proportional composition with hollow center option.
   - **Histograms**: Continuous distribution frequencies with box, violin, and rug marginals.
   - **Scatter Plots**: Bivariate relationships with bubble sizing and NumPy OLS linear trendlines.
   - **Correlation Heatmaps**: Interactive multivariate matrix with selectable color scales.
2. **Defensive Data Type Validation**: Prevents misleading visual outputs (e.g. text columns on scatter plots).
3. **Interactive Pre-Filters**: Filter by category or numeric range before generating charts.
4. **Full Styling Control**: Customizable titles, X/Y axis labels, Plotly themes, and color palettes.
5. **Standalone Export**: Download interactive charts as self-contained `.html` or export plotted data as `.csv`.

### Phase 5: Automated Insights Studio
1. **Executive Quality Scorecard**: Computes a deterministic Data Cleanliness Score (0 to 100) and letter grade (A-F) based on completeness and uniqueness.
2. **Automated Discovery Cards**:
   - Short descriptive title and badge.
   - Exact mathematical finding (outliers, category leaders, strong correlations, skewness).
   - Beginner-friendly narrative explanation.
   - Embedded interactive Plotly chart corresponding directly to the finding.
3. **Anomaly & Outlier Radar**: Scans numerical columns using Tukey's Interquartile Range (IQR) method (1.5 × IQR fences) without mutating the dataset.
4. **Relationship Discovery**: Identifies strong positive and negative associations ($|r| \ge 0.5$) with mandatory reminder that *correlation does not imply causation*.
5. **Smart Recommendations**: Suggests actionable investigative questions and workflow next steps.
6. **Downloadable Insights Report**: Export comprehensive findings in standard Markdown format (`.md`).

### Machine Learning Studio
1. **Machine Learning Primer**: Beginner-friendly breakdown of Regression, Classification, and Clustering.
2. **Guided 4-Step Training Wizard**:
   - **Step 1 (Dataset Selection)**: Train on raw or cleaned datasets with quick preview.
   - **Step 2 (Task Selection)**: Choose between Regression, Classification, and Clustering.
   - **Step 3 (Configuration)**: Dynamic target selection, recommended input features, and algorithm choices (Random Forest, Linear Regression, Logistic Regression, K-Means).
   - **Step 4 (Safe Training Pipeline)**: Encapsulated Scikit-learn Pipelines (`ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, `SimpleImputer`) to eliminate data leakage.
3. **Comprehensive Diagnostics & Visuals**:
   - **Regression**: MAE, RMSE, $R^2$ Score, and interactive Actual vs. Predicted scatter plot with $y=x$ ideal reference line.
   - **Classification**: Accuracy, Precision, Recall, F1-Score, and interactive Confusion Matrix heatmap.
   - **Clustering**: K-Means segment distribution bar chart, 2D scatter visualization, and Silhouette Score.
   - **Feature Importance**: Ranked bar chart of top predictive features.
4. **Prediction Studio**: Interactive form to input new records, generate real-time model predictions with confidence probabilities, and download results as CSV.

### Phase 7: AI Data Assistant
1. **Chat With Your Data**: Natural language conversational interface allowing users to ask questions about columns, records, anomalies, and categories.
2. **Pedagogical AI Tutor**: Explains statistical findings, data cleaning rationales, visualizations, and machine learning metrics with beginner analogies.
3. **Grounded Statistical Context (Zero Hallucination)**: Compiles verified mathematical facts, value counts, and correlation tables to strictly anchor AI answers.
4. **Privacy-Preserving**: Transmits structural metadata and statistical aggregations—never raw personal records or sensitive customer details.
5. **Multi-Mode Support (Gemini SDK & Local Engine)**: Uses Google GenAI SDK (`google-genai` with Gemini 3.8 Flash) when configured, and falls back to a deterministic local Pandas engine when running offline.

---

## 🏗️ Project Architecture

```
insightlab/
├── app.py                         # Main entry point and sidebar router
├── components/
│   ├── __init__.py
│   ├── dataset_explorer.py        # UI: dataset exploration (Phase 1)
│   ├── data_cleaning.py           # UI: data cleaning workbench (Phase 2)
│   ├── statistical_analysis.py    # UI: statistical analysis studio (Phase 3)
│   ├── visualization_studio.py    # UI: interactive visualization studio (Phase 4)
│   ├── insights_studio.py         # UI: automated insights studio (Phase 5)
│   ├── machine_learning_studio.py # UI: machine learning studio (Phase 6)
│   └── ai_assistant.py            # UI: AI Data Assistant (Phase 7)
├── utils/
│   ├── __init__.py
│   ├── data_loader.py             # Ingestion & schema inspection
│   ├── data_cleaner.py            # Non-destructive data cleaning & exports
│   ├── statistical_analyzer.py    # Vectorized statistical computations
│   ├── visualizer.py              # Dynamic Plotly builders & type validation
│   ├── insights_engine.py         # Deterministic outlier & pattern detection
│   ├── ml_engine.py               # Scikit-learn pipelines & training engine
│   └── ai_assistant.py            # Gemini integration & grounded context engine
├── sample_data/                   # Ready-to-use sample files for testing
│   ├── sales_data.csv             # E-commerce dataset with categories & metrics
│   ├── employee_records.xlsx      # HR dataset with departments & salaries
│   └── empty_file.csv             # Empty test file for validating error handling
├── tests/
│   ├── test_data_loader.py        # Automated unit tests for data loader (6 tests)
│   ├── test_data_cleaner.py       # Automated unit tests for data cleaner (11 tests)
│   ├── test_statistical_analyzer.py # Automated unit tests for stats (7 tests)
│   ├── test_visualizer.py         # Automated unit tests for visualizer (10 tests)
│   ├── test_insights_engine.py    # Automated unit tests for insights (7 tests)
│   ├── test_ml_engine.py          # Automated unit tests for ML (8 tests)
│   └── test_ai_assistant.py       # Automated unit tests for AI assistant (5 tests)
└── requirements.txt               # Python dependencies
```

### Why Modular Code?
- **Separation of Concerns**: `utils/data_loader.py` handles the data logic, while `components/dataset_explorer.py` handles how it looks on screen.
- **Maintainability**: If we need to change how Excel files are parsed or add support for `.parquet` files later, we only edit `data_loader.py` without touching the UI.
- **Testability**: We can write automated unit tests for `data_loader.py` directly without having to launch a web browser.

---

## 🚀 How to Run InsightLab

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Streamlit Application
```bash
streamlit run app.py
```
*(If `streamlit` is not directly on your system PATH, use: `python -m streamlit run app.py`)*

### 3. Open in Browser
Streamlit will automatically open your default browser to `http://localhost:8501`.

---

## 🧪 Running Automated Tests

To run the automated test suite and ensure all validation logic works:
```bash
python -m unittest discover tests
```

---

## 📚 Key Concepts for Beginners

### 1. What is Pandas?
Pandas is Python's leading data manipulation library. Its primary data structure is the **DataFrame** (`pd.DataFrame`), which represents tabular data with rows and columns (just like an Excel sheet or SQL database table).

### 2. What is Streamlit?
Streamlit turns Python scripts into interactive web apps. Every time a user interacts with a widget (clicks a button, selects a file, moves a slider), Streamlit re-runs the Python script from top to bottom, updating the interface dynamically.

### 3. Key Pandas Operations in this Phase:
- `pd.read_csv()` / `pd.read_excel()`: Parses raw files into in-memory DataFrames.
- `df.shape`: Returns a tuple `(rows, columns)` indicating dimensions.
- `df.head(10)`: Selects the first 10 rows.
- `df.dtypes`: Shows the data type of each column (`int64`, `float64`, `object`, `datetime64`).
- `df.isnull().sum()`: Counts missing values per column.
- `df.describe()`: Computes essential summary statistics.
