"""
InsightLab - AI-Powered Data Analysis Platform
==============================================
Main Entry Point (`app.py`)

This file configures the Streamlit application, establishes the sidebar
navigation, and renders the active module (Dataset Explorer).

How to run this application:
----------------------------
Run the following command in your terminal:
    streamlit run app.py
"""

import streamlit as st
from components.dataset_explorer import render_dataset_explorer
from components.data_cleaning import render_data_cleaning
from components.statistical_analysis import render_statistical_analysis
from components.visualization_studio import render_visualization_studio
from components.insights_studio import render_insights_studio
from components.machine_learning_studio import render_machine_learning_studio
from components.ai_assistant import render_ai_assistant
from components.export_studio import render_export_studio
from components.auth_ui import render_auth_header, render_auth_page
from components.user_workspace import render_user_workspace


def main():
    # --------------------------------------------------------------------------
    # 1. Page Configuration (Must be the first Streamlit command)
    # --------------------------------------------------------------------------
    st.set_page_config(
        page_title="InsightLab - Data Analysis Platform",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Hide the default Streamlit Deploy button
    st.markdown(
        """
        <style>
        .stAppDeployButton,
        .stDeployButton,
        [data-testid="stAppDeployButton"],
        [data-testid="stDeployButton"] {
            display: none !important;
            visibility: hidden !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------------------------
    # 2. Sidebar Setup & Navigation
    # --------------------------------------------------------------------------
    with st.sidebar:
        st.title("🔬 InsightLab")
        st.markdown("**AI-Powered Data Analysis Platform**")
        st.caption("A beginner-friendly tool to inspect, clean, visualize, and analyze datasets.")
        st.markdown("---")

        # Authentication Status / Guest Indicator
        render_auth_header()
        st.markdown("---")

        st.subheader("🧭 Navigation")
        # Navigation options
        nav_options = [
            "📊 Dataset Explorer",
            "🧹 Data Cleaning (Phase 2)",
            "📈 Statistical Analysis (Phase 3)",
            "🎨 Visualization Studio (Phase 4)",
            "💡 Automated Insights Studio (Phase 5)",
            "🤖 Machine Learning Studio",
            "💬 AI Data Assistant (Phase 7)",
            "📄 Export & Report Studio (Phase 8)",
            "🏠 Personal Workspace"
        ]

        default_index = 0
        if st.session_state.get("_nav_to_workspace"):
            st.session_state.pop("_nav_to_workspace", None)
            default_index = 8

        app_mode = st.radio(
            "Select Module:",
            options=nav_options,
            index=default_index,
            help="Choose the module you wish to use."
        )

        st.markdown("---")
        st.markdown("### 🛠️ Tech Stack")
        st.markdown(
            """
            - **Framework:** Streamlit
            - **Data Handling:** Pandas & NumPy
            - **Machine Learning:** Scikit-learn
            - **Generative AI:** Google Gemini SDK
            - **Cloud & Auth:** Google Firebase
            - **Visuals:** Plotly
            - **Language:** Python
            """
        )

        st.markdown("---")
        st.caption("InsightLab Cloud & Workspace active.")

    # --------------------------------------------------------------------------
    # 3. Router: Render the Selected Module
    # --------------------------------------------------------------------------
    # If the user opened the Auth page (Sign In / Sign Up)
    if st.session_state.get("_show_auth_dialog"):
        render_auth_page()
        return

    if app_mode == "📊 Dataset Explorer":
        render_dataset_explorer()
    elif app_mode == "🧹 Data Cleaning (Phase 2)":
        render_data_cleaning()
    elif app_mode == "📈 Statistical Analysis (Phase 3)":
        render_statistical_analysis()
    elif app_mode == "🎨 Visualization Studio (Phase 4)":
        render_visualization_studio()
    elif app_mode == "💡 Automated Insights Studio (Phase 5)":
        render_insights_studio()
    elif app_mode == "🤖 Machine Learning Studio":
        render_machine_learning_studio()
    elif app_mode == "💬 AI Data Assistant (Phase 7)":
        render_ai_assistant()
    elif app_mode == "📄 Export & Report Studio (Phase 8)":
        render_export_studio()
    elif app_mode == "🏠 Personal Workspace":
        render_user_workspace()
    else:
        st.header(app_mode)
        st.info("🚧 InsightLab Module")
        st.markdown("Use the navigation menu in the sidebar to explore all available modules.")


if __name__ == "__main__":
    main()
