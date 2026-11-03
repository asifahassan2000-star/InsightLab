"""
ml_engine.py
============
Machine Learning Engine for InsightLab (Phase 6).

Key Principles:
---------------
1. No Data Leakage:
   Preprocessing (imputation, scaling, one-hot encoding) is strictly encapsulated
   inside Scikit-learn Pipelines (`sklearn.pipeline.Pipeline`).
   Transformers are fit ONLY on the training split (`X_train`) and evaluated on (`X_test`).

2. Beginner-Friendly & Resilient:
   Automatically handles missing values and text/categorical features.
   Validates dataset sizes and targets to prevent cryptic Scikit-learn stack traces.

3. Tasks Supported:
   - Regression: Predict continuous numerical targets (e.g., profit, salary).
   - Classification: Predict discrete category labels (e.g., department, churn).
   - Clustering: Unsupervised grouping of similar observations (K-Means).
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    silhouette_score
)


def get_feature_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str]
) -> ColumnTransformer:
    """
    Builds a robust Scikit-learn ColumnTransformer.

    Why ColumnTransformer?
    ----------------------
    Real-world data contains both numbers (e.g. salary) and text (e.g. department).
    - Numbers need median imputation and scaling (StandardScaler).
    - Text needs most-frequent imputation and One-Hot Encoding (OHE).
    ColumnTransformer applies the appropriate recipe to each column type automatically.
    """
    transformers = []

    if numeric_features:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", num_pipeline, numeric_features))

    if categorical_features:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", cat_pipeline, categorical_features))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def train_regression_model(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    algorithm: str = "Random Forest",
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Trains a regression model to predict continuous numerical values.

    Returns:
    --------
    Dict containing:
    - 'pipeline': Trained Pipeline object ready for inference
    - 'metrics': MAE, RMSE, R2
    - 'test_results': DataFrame with Actual, Predicted, and Residuals
    - 'feature_importance': DataFrame of features and their relative weights
    """
    if len(df) < 5:
        raise ValueError(f"Dataset has only {len(df)} rows. Regression requires at least 5 rows.")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    if not feature_cols:
        raise ValueError("Please select at least one feature column for training.")

    # Drop rows where target is NaN
    clean_df = df.dropna(subset=[target_col]).copy()
    if len(clean_df) < 5:
        raise ValueError("Fewer than 5 rows remain after removing missing target values.")

    X = clean_df[feature_cols]
    y = clean_df[target_col].astype(float)

    # Separate numeric and categorical features
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist()

    # Split into train and test sets to evaluate generalization
    # If dataset is small (e.g. 5-8 rows), use 1 or 2 test samples
    actual_test_size = max(1, int(len(clean_df) * test_size))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=actual_test_size, random_state=random_state
    )

    # Choose model
    if algorithm == "Linear Regression":
        regressor = LinearRegression()
    else:
        regressor = RandomForestRegressor(n_estimators=50, random_state=random_state)

    preprocessor = get_feature_preprocessor(num_cols, cat_cols)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])

    # Fit pipeline strictly on training data (prevents data leakage)
    pipeline.fit(X_train, y_train)

    # Predict on test data
    y_pred = pipeline.predict(X_test)

    # Calculate evaluation metrics
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred)) if len(y_test) > 1 else 1.0

    test_results = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": np.round(y_pred, 2),
        "Absolute Error": np.round(np.abs(y_test.values - y_pred), 2)
    })

    # Extract feature names & importance if available
    importance_df = pd.DataFrame()
    try:
        fitted_prep = pipeline.named_steps["preprocessor"]
        ohe_cols = fitted_prep.get_feature_names_out()
        cleaned_feature_names = [c.replace("num__", "").replace("cat__", "") for c in ohe_cols]

        if hasattr(regressor, "feature_importances_"):
            importances = regressor.feature_importances_
            importance_df = pd.DataFrame({
                "Feature": cleaned_feature_names,
                "Importance": np.round(importances, 4)
            }).sort_values(by="Importance", ascending=False).head(10)
        elif hasattr(regressor, "coef_"):
            importances = np.abs(regressor.coef_)
            importance_df = pd.DataFrame({
                "Feature": cleaned_feature_names,
                "Importance": np.round(importances, 4)
            }).sort_values(by="Importance", ascending=False).head(10)
    except Exception:
        importance_df = pd.DataFrame()

    return {
        "pipeline": pipeline,
        "algorithm": algorithm,
        "target_col": target_col,
        "feature_cols": feature_cols,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "metrics": {
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "r2": round(r2, 3)
        },
        "test_results": test_results,
        "feature_importance": importance_df
    }


def train_classification_model(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    algorithm: str = "Random Forest",
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Trains a classification model to predict categorical labels.
    """
    if len(df) < 5:
        raise ValueError(f"Dataset has only {len(df)} rows. Classification requires at least 5 rows.")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    if not feature_cols:
        raise ValueError("Please select at least one feature column for training.")

    clean_df = df.dropna(subset=[target_col]).copy()
    if clean_df[target_col].nunique() < 2:
        raise ValueError(f"Target column '{target_col}' has only 1 unique class. Classification requires at least 2 distinct classes.")

    X = clean_df[feature_cols]
    y = clean_df[target_col].astype(str)

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist()

    # Split train/test
    actual_test_size = max(1, int(len(clean_df) * test_size))
    # Try stratified split if class counts allow, otherwise standard split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=actual_test_size, random_state=random_state, stratify=y
        )
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=actual_test_size, random_state=random_state
        )

    if algorithm == "Logistic Regression":
        classifier = LogisticRegression(max_iter=500, random_state=random_state)
    else:
        classifier = RandomForestClassifier(n_estimators=50, random_state=random_state)

    preprocessor = get_feature_preprocessor(num_cols, cat_cols)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    classes = sorted(list(set(y.unique())))
    cm = confusion_matrix(y_test, y_pred, labels=classes)

    # Feature Importance
    importance_df = pd.DataFrame()
    try:
        fitted_prep = pipeline.named_steps["preprocessor"]
        ohe_cols = fitted_prep.get_feature_names_out()
        cleaned_feature_names = [c.replace("num__", "").replace("cat__", "") for c in ohe_cols]

        if hasattr(classifier, "feature_importances_"):
            importances = classifier.feature_importances_
            importance_df = pd.DataFrame({
                "Feature": cleaned_feature_names,
                "Importance": np.round(importances, 4)
            }).sort_values(by="Importance", ascending=False).head(10)
    except Exception:
        importance_df = pd.DataFrame()

    return {
        "pipeline": pipeline,
        "algorithm": algorithm,
        "target_col": target_col,
        "feature_cols": feature_cols,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "classes": classes,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "metrics": {
            "accuracy": round(acc, 3),
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "f1_score": round(f1, 3)
        },
        "confusion_matrix": cm,
        "feature_importance": importance_df
    }


def train_clustering_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    n_clusters: int = 3,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Trains a K-Means clustering model for unsupervised grouping.
    """
    if len(df) < n_clusters:
        raise ValueError(f"Cannot form {n_clusters} clusters with only {len(df)} rows.")

    if not feature_cols:
        raise ValueError("Please select at least one feature for clustering.")

    X = df[feature_cols].copy()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist()

    preprocessor = get_feature_preprocessor(num_cols, cat_cols)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clusterer", kmeans)
    ])

    pipeline.fit(X)
    cluster_labels = pipeline.named_steps["clusterer"].labels_

    # Create clustered copy of the dataset
    clustered_df = df.copy()
    clustered_df["Cluster"] = [f"Cluster {c + 1}" for c in cluster_labels]

    # Cluster distribution
    cluster_distribution = clustered_df["Cluster"].value_counts().reset_index()
    cluster_distribution.columns = ["Cluster", "Count"]

    # Compute silhouette score if dataset is sufficient
    transformed_X = pipeline.named_steps["preprocessor"].transform(X)
    silhouette = 0.0
    if len(df) > n_clusters and n_clusters > 1:
        try:
            silhouette = float(silhouette_score(transformed_X, cluster_labels))
        except Exception:
            silhouette = 0.0

    return {
        "pipeline": pipeline,
        "algorithm": "K-Means",
        "n_clusters": n_clusters,
        "feature_cols": feature_cols,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "clustered_df": clustered_df,
        "cluster_distribution": cluster_distribution,
        "silhouette_score": round(silhouette, 3)
    }


def predict_single_instance(
    model_bundle: Dict[str, Any],
    input_values: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Takes user-entered values from Prediction Studio and generates a prediction.
    """
    pipeline = model_bundle["pipeline"]
    feature_cols = model_bundle["feature_cols"]

    # Convert single dictionary to 1-row DataFrame
    input_df = pd.DataFrame([input_values])[feature_cols]

    prediction = pipeline.predict(input_df)[0]
    result = {"prediction": prediction}

    # If classification, get prediction probabilities
    if "classes" in model_bundle and hasattr(pipeline.named_steps.get("classifier"), "predict_proba"):
        try:
            probs = pipeline.predict_proba(input_df)[0]
            classes = model_bundle["classes"]
            prob_dict = {str(c): round(float(p) * 100, 1) for c, p in zip(classes, probs)}
            result["probabilities"] = prob_dict
            result["confidence"] = max(prob_dict.values())
        except Exception:
            pass

    return result
