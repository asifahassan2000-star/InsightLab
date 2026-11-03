"""
machine_learning_studio.py
==========================
This module renders the 'Machine Learning Studio' in InsightLab (Phase 6).

Key Principles:
---------------
1. Beginner-Friendly Workflow:
   Guided 4-step wizard:
   Step 1: Select Dataset
   Step 2: Choose Task (Regression, Classification, or Clustering)
   Step 3: Configure Target, Features & Algorithm
   Step 4: Train & Evaluate Model

2. Interactive Prediction Studio:
   Allows users to test their trained model with custom inputs and download results.

3. Complete Transparency & Leakage Prevention:
   Uses Scikit-learn Pipelines to guarantee no data leakage between train and test splits.
"""

from typing import Optional, List, Dict, Any
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import load_dataset
from utils.ml_engine import (
    train_regression_model,
    train_classification_model,
    train_clustering_model,
    predict_single_instance
)


def render_machine_learning_studio():
    """Renders the Machine Learning Studio UI."""
    st.markdown("## 🤖 Machine Learning Studio")
    st.markdown(
        "Build, evaluate, and test machine learning models through a guided, beginner-friendly workflow. "
        "No complex math or boilerplate code required."
    )

    # --------------------------------------------------------------------------
    # 1. Machine Learning Dashboard / Task Primer
    # --------------------------------------------------------------------------
    with st.expander("🎓 Machine Learning 101: Understanding the 3 Core Tasks", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### 📈 Regression")
            st.markdown(
                "**Goal:** Predict a continuous numerical value.\n\n"
                "- *Real-world example:* Predicting house prices, annual salary, or quarterly profit.\n"
                "- *Key metrics:* MAE, RMSE, $R^2$ Score."
            )
        with c2:
            st.markdown("#### 🎯 Classification")
            st.markdown(
                "**Goal:** Predict a discrete category or label.\n\n"
                "- *Real-world example:* Predicting customer churn (Yes/No), product category, or loan risk.\n"
                "- *Key metrics:* Accuracy, Precision, Recall, F1-Score."
            )
        with c3:
            st.markdown("#### 🧩 Clustering")
            st.markdown(
                "**Goal:** Discover hidden natural groupings automatically without labels.\n\n"
                "- *Real-world example:* Customer market segmentation, store grouping.\n"
                "- *Key algorithm:* K-Means Clustering."
            )

    # --------------------------------------------------------------------------
    # Step 1: Select Dataset
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Step 1: Select Dataset")

    has_raw = "raw_df" in st.session_state and st.session_state["raw_df"] is not None
    has_cleaned = "cleaned_df" in st.session_state and st.session_state["cleaned_df"] is not None

    if not has_raw and not has_cleaned:
        st.info("ℹ️ No dataset loaded. Upload a file below or use the built-in sample data to train a model.")
        uploaded_file = st.file_uploader("Upload dataset for Machine Learning", type=["csv", "xlsx", "xls"], key="ml_uploader")
        if uploaded_file:
            loaded_df, err = load_dataset(uploaded_file)
            if not err:
                st.session_state["raw_df"] = loaded_df.copy()
                st.session_state["cleaned_df"] = loaded_df.copy()
                st.session_state["dataset_name"] = uploaded_file.name
                st.rerun()
            else:
                st.error(f"Failed to load file: {err}")
                return
        return

    dataset_name = st.session_state.get("dataset_name", "dataset")
    col_ds1, col_ds2 = st.columns([3, 2])
    with col_ds1:
        st.write(f"**Active Dataset:** `{dataset_name}`")
    with col_ds2:
        if has_cleaned and has_raw:
            data_choice = st.radio(
                "Select Data Version to Train On:",
                options=["Cleaned Working Dataset (Phase 2)", "Original Raw Dataset (Phase 1)"],
                horizontal=True
            )
            df = st.session_state["cleaned_df"] if "Cleaned" in data_choice else st.session_state["raw_df"]
        else:
            df = st.session_state.get("cleaned_df", st.session_state.get("raw_df"))

    st.caption(f"Dataset Dimensions: **{len(df):,} rows** × **{len(df.columns)} columns**")
    with st.expander("👀 Quick Data Preview", expanded=False):
        st.dataframe(df.head(5), use_container_width=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist()
    all_cols = df.columns.tolist()

    # --------------------------------------------------------------------------
    # Step 2: Choose ML Task
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Step 2: Choose Machine Learning Task")

    task = st.radio(
        "What type of problem are you solving?",
        options=["📈 Regression (Predict a Number)", "🎯 Classification (Predict a Category)", "🧩 Clustering (Discover Groups)"],
        horizontal=True
    )

    # --------------------------------------------------------------------------
    # Step 3: Configure Model
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Step 3: Configure Model")

    target_col = None
    feature_cols = []
    algorithm = ""
    n_clusters = 3

    if task.startswith("📈 Regression"):
        if not numeric_cols:
            st.error("⚠️ No numerical columns available to use as a regression target.")
            return

        col_t, col_algo = st.columns(2)
        with col_t:
            target_col = st.selectbox(
                "Target Column to Predict (Numerical):",
                options=numeric_cols,
                help="The specific numerical outcome your model will learn to predict."
            )
        with col_algo:
            algorithm = st.selectbox(
                "Choose Algorithm:",
                options=["Random Forest (Recommended)", "Linear Regression"],
                help="Random Forest captures complex non-linear patterns. Linear Regression is simpler and transparent."
            )

        default_features = [c for c in all_cols if c != target_col]
        feature_cols = st.multiselect(
            "Select Input Features (Predictors):",
            options=default_features,
            default=default_features,
            help="The information the model uses to make its prediction."
        )

    elif task.startswith("🎯 Classification"):
        potential_targets = categorical_cols + [c for c in numeric_cols if df[c].nunique() <= 10]
        if not potential_targets:
            potential_targets = all_cols

        col_t, col_algo = st.columns(2)
        with col_t:
            target_col = st.selectbox(
                "Target Column to Predict (Category):",
                options=potential_targets,
                help="The categorical label or class your model will learn to identify."
            )
        with col_algo:
            algorithm = st.selectbox(
                "Choose Algorithm:",
                options=["Random Forest (Recommended)", "Logistic Regression"],
                help="Random Forest handles multiple categories robustly. Logistic Regression is ideal for binary (Yes/No) tasks."
            )

        default_features = [c for c in all_cols if c != target_col]
        feature_cols = st.multiselect(
            "Select Input Features (Predictors):",
            options=default_features,
            default=default_features,
            help="The features the model uses to classify observations."
        )

    elif task.startswith("🧩 Clustering"):
        col_k, col_spacer = st.columns(2)
        with col_k:
            n_clusters = st.slider("Number of Clusters (k):", min_value=2, max_value=min(6, max(2, len(df) - 1)), value=3)

        feature_cols = st.multiselect(
            "Select Features for Clustering:",
            options=numeric_cols if numeric_cols else all_cols,
            default=numeric_cols if numeric_cols else all_cols[:3],
            help="Columns used to calculate similarity between records."
        )

    # --------------------------------------------------------------------------
    # Step 4: Train Model
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Step 4: Train Model")

    if not feature_cols:
        st.warning("Please select at least one feature before training.")
        return

    clean_algo_name = algorithm.replace(" (Recommended)", "")

    if st.button("🚀 Train Model", type="primary", use_container_width=True):
        with st.spinner(f"Training {clean_algo_name} model via Scikit-learn Pipeline..."):
            try:
                if task.startswith("📈 Regression"):
                    result = train_regression_model(
                        df=df,
                        target_col=target_col,
                        feature_cols=feature_cols,
                        algorithm=clean_algo_name,
                        test_size=0.2
                    )
                    st.session_state["ml_model_bundle"] = result
                    st.session_state["ml_task"] = "regression"
                    st.success("✅ Regression model trained successfully!")

                elif task.startswith("🎯 Classification"):
                    result = train_classification_model(
                        df=df,
                        target_col=target_col,
                        feature_cols=feature_cols,
                        algorithm=clean_algo_name,
                        test_size=0.2
                    )
                    st.session_state["ml_model_bundle"] = result
                    st.session_state["ml_task"] = "classification"
                    st.success("✅ Classification model trained successfully!")

                elif task.startswith("🧩 Clustering"):
                    result = train_clustering_model(
                        df=df,
                        feature_cols=feature_cols,
                        n_clusters=n_clusters
                    )
                    st.session_state["ml_model_bundle"] = result
                    st.session_state["ml_task"] = "clustering"
                    st.success("✅ Clustering model converged successfully!")

            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")
                st.info("💡 Ensure your target column has sufficient variety and selected features are valid.")
                return

    # --------------------------------------------------------------------------
    # Step 5: Model Results & Prediction Studio
    # --------------------------------------------------------------------------
    if "ml_model_bundle" not in st.session_state or not st.session_state["ml_model_bundle"]:
        return

    model_bundle = st.session_state["ml_model_bundle"]
    ml_task = st.session_state.get("ml_task", "")

    st.markdown("---")
    st.markdown("### 🏆 Model Results & Insights")

    tab_eval, tab_predict, tab_guide = st.tabs([
        "📊 Model Performance & Diagnostics",
        "🔮 Prediction Studio (Interactive Testing)",
        "🎓 Machine Learning Guide"
    ])

    # --------------------------------------------------------------------------
    # TAB 1: MODEL EVALUATION
    # --------------------------------------------------------------------------
    with tab_eval:
        if ml_task == "regression":
            metrics = model_bundle["metrics"]
            st.markdown(f"#### Regression Metrics: Predicting `{model_bundle['target_col']}`")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("R² Score (Goodness of Fit)", f"{metrics['r2']:.3f}", help="1.0 is perfect prediction. Above 0.7 is typically strong.")
            with m2:
                st.metric("MAE (Mean Absolute Error)", f"{metrics['mae']:.2f}", help="Average magnitude of errors in target units.")
            with m3:
                st.metric("RMSE (Root Mean Squared Error)", f"{metrics['rmse']:.2f}", help="Penalizes larger prediction errors more heavily.")

            # Actual vs Predicted Scatter Plot
            test_df = model_bundle["test_results"]
            if not test_df.empty:
                st.markdown("##### Actual vs. Predicted Values (Test Set)")
                fig_reg = px.scatter(
                    test_df,
                    x="Actual",
                    y="Predicted",
                    title="Actual vs. Predicted (Points close to the red line indicate accurate predictions)",
                    color_discrete_sequence=["#1f77b4"]
                )
                min_val = min(test_df["Actual"].min(), test_df["Predicted"].min())
                max_val = max(test_df["Actual"].max(), test_df["Predicted"].max())
                fig_reg.add_trace(
                    go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode="lines", name="Perfect Prediction (y=x)", line=dict(color="red", dash="dash"))
                )
                fig_reg.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_reg, use_container_width=True)

        elif ml_task == "classification":
            metrics = model_bundle["metrics"]
            st.markdown(f"#### Classification Metrics: Predicting `{model_bundle['target_col']}`")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Accuracy", f"{metrics['accuracy'] * 100:.1f}%")
            with m2:
                st.metric("Precision", f"{metrics['precision'] * 100:.1f}%")
            with m3:
                st.metric("Recall", f"{metrics['recall'] * 100:.1f}%")
            with m4:
                st.metric("F1-Score", f"{metrics['f1_score']:.3f}")

            # Confusion Matrix Heatmap
            st.markdown("##### Confusion Matrix (Test Set)")
            st.caption("Diagonal cells show correct predictions. Off-diagonal cells reveal classification mistakes.")
            cm = model_bundle["confusion_matrix"]
            classes = model_bundle["classes"]
            fig_cm = px.imshow(
                cm,
                x=[f"Pred: {c}" for c in classes],
                y=[f"Actual: {c}" for c in classes],
                text_auto=True,
                color_continuous_scale="Blues",
                title="Confusion Matrix"
            )
            fig_cm.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_cm, use_container_width=True)

        elif ml_task == "clustering":
            st.markdown("#### K-Means Cluster Segmentation")
            c_dist = model_bundle["cluster_distribution"]
            st.metric("Clusters Formed", f"{model_bundle['n_clusters']}", f"Silhouette Score: {model_bundle['silhouette_score']}")

            col_cp1, col_cp2 = st.columns(2)
            with col_cp1:
                st.markdown("##### Cluster Distribution")
                fig_dist = px.bar(c_dist, x="Cluster", y="Count", text_auto=True, color="Cluster")
                fig_dist.update_layout(template="plotly_white", showlegend=False, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_dist, use_container_width=True)

            with col_cp2:
                clustered_data = model_bundle["clustered_df"]
                f_cols = model_bundle["feature_cols"]
                if len(f_cols) >= 2:
                    st.markdown("##### Cluster Scatter Representation")
                    fig_sc = px.scatter(
                        clustered_data,
                        x=f_cols[0],
                        y=f_cols[1],
                        color="Cluster",
                        title=f"{f_cols[1]} vs {f_cols[0]} by Cluster"
                    )
                    fig_sc.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart(fig_sc, use_container_width=True)

        # Feature Importance (Supported in Regression & Classification)
        if "feature_importance" in model_bundle and not model_bundle["feature_importance"].empty:
            st.markdown("---")
            st.markdown("#### 🌟 Top Feature Importances")
            st.caption("Indicates which input variables had the greatest statistical influence on the model's decisions.")
            fi_df = model_bundle["feature_importance"]
            fig_fi = px.bar(
                fi_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top Predictive Features",
                color="Importance",
                color_continuous_scale="Blues"
            )
            fig_fi.update_layout(template="plotly_white", yaxis=dict(autorange="reversed"), margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_fi, use_container_width=True)

    # --------------------------------------------------------------------------
    # TAB 2: PREDICTION STUDIO (INTERACTIVE TESTING)
    # --------------------------------------------------------------------------
    with tab_predict:
        st.subheader("Interactive Prediction Form")
        st.caption("Input custom feature values below to generate a real-time prediction using your trained model pipeline.")

        if ml_task == "clustering":
            st.info("Clustering is an unsupervised grouping task. View the segmented records in Tab 1.")
        else:
            with st.form("prediction_input_form"):
                st.markdown("##### Enter Values for Prediction:")
                input_values = {}
                feature_cols = model_bundle["feature_cols"]
                num_cols = model_bundle["num_cols"]
                cat_cols = model_bundle["cat_cols"]

                # Display input widgets in a balanced 2-column layout
                p_cols = st.columns(2)
                for idx, feat in enumerate(feature_cols):
                    col_target = p_cols[idx % 2]
                    with col_target:
                        if feat in num_cols:
                            default_val = float(df[feat].median()) if pd.notna(df[feat].median()) else 0.0
                            input_values[feat] = st.number_input(
                                f"{feat}:",
                                value=round(default_val, 2),
                                key=f"pred_input_{feat}"
                            )
                        else:
                            options = sorted([str(x) for x in df[feat].dropna().unique().tolist()])
                            input_values[feat] = st.selectbox(
                                f"{feat}:",
                                options=options if options else ["Unknown"],
                                key=f"pred_input_{feat}"
                            )

                submit_pred = st.form_submit_button("🔮 Generate Prediction", use_container_width=True)

            if submit_pred:
                pred_result = predict_single_instance(model_bundle, input_values)
                pred_val = pred_result["prediction"]

                with st.container(border=True):
                    if ml_task == "regression":
                        st.markdown(f"### 🎯 Predicted `{model_bundle['target_col']}`: **{float(pred_val):,.2f}**")
                    else:
                        conf_str = f" (Confidence: {pred_result.get('confidence', 0)}%)" if "confidence" in pred_result else ""
                        st.markdown(f"### 🎯 Predicted Class: **{pred_val}**{conf_str}")
                        if "probabilities" in pred_result:
                            st.write("**Class Probability Distribution:**")
                            st.dataframe(pd.DataFrame([pred_result["probabilities"]]), use_container_width=True)

                # Export single prediction to CSV
                res_export_df = pd.DataFrame([dict(**input_values, Predicted_Output=pred_val)])
                csv_bytes = res_export_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Prediction as CSV",
                    data=csv_bytes,
                    file_name="insightlab_prediction.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    # --------------------------------------------------------------------------
    # TAB 3: MACHINE LEARNING GUIDE
    # --------------------------------------------------------------------------
    with tab_guide:
        st.markdown("#### 🎓 Beginner's Guide: How This Machine Learning Works")
        st.markdown("""
        ### 1. What is an 80/20 Train/Test Split?
        When training a machine learning model, you must never test the model on the exact same data it learned from.
        - **Training Set (80%):** The textbook the model studies to find patterns.
        - **Testing Set (20%):** The unseen final exam used to verify if the model truly learned or just memorized answers.

        ---

        ### 2. What is Data Leakage & Why Pipelines Matter?
        If you calculate the average of a column on your entire dataset before splitting it, the model secretly peeked at the test set's data!
        This is called **Data Leakage**.
        InsightLab prevents data leakage by using **Scikit-learn Pipelines**: imputers and scalers are calculated strictly from the training split.

        ---

        ### 3. Understanding the Evaluation Metrics
        - **R² Score (Regression):** Measures how much of the variance in the target is explained by your inputs. $1.0$ is a perfect prediction, $0.0$ is no better than guessing the average.
        - **MAE (Mean Absolute Error):** The average difference between predictions and real values in target units.
        - **Accuracy (Classification):** Percentage of total predictions that were correct.
        - **F1-Score:** The harmonic balance between Precision (avoiding false alarms) and Recall (avoiding missed targets).
        """)

    # --------------------------------------------------------------------------
    # Cloud Save Section (Guest-friendly optional permanent cloud save)
    # --------------------------------------------------------------------------
    if st.session_state.get("ml_model_bundle"):
        from utils.smart_save import build_ml_model_payload
        from components.auth_ui import render_save_section
        ml_payload = build_ml_model_payload()
        render_save_section("model", ml_payload, key_suffix="ml_studio")
