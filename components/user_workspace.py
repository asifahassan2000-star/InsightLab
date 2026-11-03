"""
user_workspace.py
=================
Personal User Workspace for InsightLab.

What This Module Does:
----------------------
Renders the authenticated user's private dashboard containing 6 sections:
  1. 📂 My Projects — Saved project snapshots (dataset + cleaning + ML state)
  2. 📊 My Datasets — Saved datasets (metadata and cloud storage files)
  3. 📄 My Reports — Saved PDF and Excel report records
  4. 📈 My Analysis History — Statistical summaries & correlation discoveries
  5. 🤖 My Saved Models — Machine learning configurations and metrics
  6. ⚙️ Account Settings — User profile, display name update, password reset, sign out

Actions Supported on Saved Items:
---------------------------------
- **Open / Inspect** : View the complete metadata and configuration.
- **Rename**         : Update the title of any saved item in Cloud Firestore.
- **Delete**         : Remove the item from Cloud Firestore with confirmation.
- **Export / Download**: Export the saved metadata as JSON / CSV.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, List, Optional
import datetime

from utils.firebase_config import is_firebase_configured
from utils.firebase_auth import (
    get_current_user,
    get_valid_id_token,
    is_authenticated,
    update_profile,
    send_password_reset,
    sign_out,
)
from utils.firebase_db import (
    get_user_items,
    delete_user_item,
    rename_user_item,
)


def render_user_workspace() -> None:
    """
    Renders the Personal Workspace Dashboard.
    If the user is a guest, displays a welcoming guest card with a button to sign in.
    """
    st.title("🏠 Personal Workspace")
    st.markdown("Your central cloud hub for saved projects, datasets, analysis, and trained models.")
    st.markdown("---")

    if not is_firebase_configured():
        st.info(
            "☁️ **Firebase is not yet configured for InsightLab.**\n\n"
            "InsightLab is currently running in **100% Guest Mode**. All analysis features, "
            "visualizations, ML models, and PDF/Excel downloads are completely available.\n\n"
            "To connect your Firebase project for cloud saving, add your credentials in `.streamlit/secrets.toml`."
        )
        return

    user = get_current_user()

    # Guest user view
    if not user:
        _render_guest_workspace_prompt()
        return

    # Authenticated user view
    display_name = user.get("display_name") or user.get("email", "Analyst")
    st.success(f"👋 Welcome to your workspace, **{display_name}** (`{user.get('email')}`)!")

    # Tabs for the 6 sections
    tab_projects, tab_datasets, tab_reports, tab_analysis, tab_models, tab_settings = st.tabs([
        "📂 My Projects",
        "📊 My Datasets",
        "📄 My Reports",
        "📈 My Analysis History",
        "🤖 My Saved Models",
        "⚙️ Account Settings",
    ])

    uid = user["uid"]
    id_token = get_valid_id_token()

    with tab_projects:
        _render_items_section(
            uid=uid,
            id_token=id_token,
            subcollection="projects",
            title="My Saved Projects",
            empty_msg="No saved projects yet. Open any analysis module and click 'Save to Cloud'!",
            icon="📂"
        )

    with tab_datasets:
        _render_items_section(
            uid=uid,
            id_token=id_token,
            subcollection="datasets",
            title="My Datasets",
            empty_msg="No datasets saved yet. Upload a dataset in Dataset Explorer and save it to cloud.",
            icon="📊"
        )

    with tab_reports:
        _render_items_section(
            uid=uid,
            id_token=id_token,
            subcollection="reports",
            title="My Generated Reports",
            empty_msg="No reports saved yet. Generate a PDF or Excel report in Export Studio and save it.",
            icon="📄"
        )

    with tab_analysis:
        _render_items_section(
            uid=uid,
            id_token=id_token,
            subcollection="analyses",
            title="My Analysis History",
            empty_msg="No analysis records saved yet. Save findings from Statistical Analysis or Insights Studio.",
            icon="📈"
        )

    with tab_models:
        _render_items_section(
            uid=uid,
            id_token=id_token,
            subcollection="models",
            title="My Saved Models",
            empty_msg="No machine learning models saved yet. Train a model in Machine Learning Studio and save it.",
            icon="🤖"
        )

    with tab_settings:
        _render_account_settings(user, id_token)


def _render_guest_workspace_prompt() -> None:
    """Friendly welcome message when a guest visits the Personal Workspace tab."""
    st.info("👤 **You are currently browsing as a Guest.**")
    st.markdown(
        """
        InsightLab lets you explore, clean, visualize, analyze, and train models completely free without an account!
        
        ### Why create a free account?
        - ☁️ **Save your projects** and resume where you left off.
        - 📊 **Track multiple datasets** and their data cleaning steps.
        - 📄 **Store generated reports** in your personal cloud history.
        - 🤖 **Preserve machine learning models** and test results.
        """
    )
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🔑 Sign In / Sign Up", key="ws_signin_btn", type="primary", use_container_width=True):
            st.session_state["_show_auth_dialog"] = True
            st.rerun()


def _render_items_section(
    uid: str,
    id_token: str,
    subcollection: str,
    title: str,
    empty_msg: str,
    icon: str
) -> None:
    """Generic CRUD section renderer for any user subcollection."""
    st.subheader(f"{icon} {title}")

    # Fetch items from Firestore
    with st.spinner(f"Loading {title.lower()}..."):
        items = get_user_items(uid, subcollection, id_token)

    if not items:
        st.info(f"ℹ️ {empty_msg}")
        return

    st.caption(f"Showing **{len(items)}** saved item(s) in Cloud Firestore.")

    # Render each item as an expandable card
    for idx, item in enumerate(items):
        doc_id = item.get("_id", f"item_{idx}")
        name = item.get("name") or f"Untitled {subcollection[:-1].capitalize()}"
        saved_at = item.get("saved_at") or item.get("_created", "Unknown date")

        # Clean display date
        clean_date = saved_at[:16].replace("T", " ") if "T" in saved_at else saved_at

        with st.expander(f"{icon} **{name}** — *Saved on {clean_date}*", expanded=(idx == 0)):
            col_info, col_actions = st.columns([3, 1])

            with col_info:
                st.markdown(f"**Item ID:** `{doc_id}`")
                # Display specific fields cleanly
                for k, v in item.items():
                    if k in ("_id", "_path", "_created", "_updated", "name"):
                        continue
                    if isinstance(v, dict):
                        st.markdown(f"**{k.replace('_', ' ').capitalize()}:**")
                        st.json(v)
                    elif isinstance(v, list):
                        st.markdown(f"**{k.replace('_', ' ').capitalize()} ({len(v)}):** {', '.join(str(x) for x in v[:5])}{'...' if len(v) > 5 else ''}")
                    else:
                        st.markdown(f"**{k.replace('_', ' ').capitalize()}:** {v}")

            with col_actions:
                st.markdown("##### Actions")

                # Rename form
                rename_key = f"rename_input_{subcollection}_{doc_id}"
                new_name = st.text_input("New Name", value=name, key=rename_key)
                if st.button("✏️ Rename", key=f"btn_rename_{subcollection}_{doc_id}", use_container_width=True):
                    if new_name.strip() and new_name.strip() != name:
                        with st.spinner("Renaming..."):
                            success, _ = rename_user_item(uid, subcollection, doc_id, new_name.strip(), id_token)
                        if success:
                            st.success("Renamed!")
                            st.rerun()
                        else:
                            st.error("Failed to rename.")

                st.markdown("---")
                # Delete button
                if st.button("🗑️ Delete", key=f"btn_del_{subcollection}_{doc_id}", type="secondary", use_container_width=True):
                    with st.spinner("Deleting..."):
                        success, _ = delete_user_item(uid, subcollection, doc_id, id_token)
                    if success:
                        st.success("Deleted successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to delete.")


def _render_account_settings(user: Dict[str, Any], id_token: str) -> None:
    """Renders profile management and password reset options."""
    st.subheader("⚙️ Account Settings")

    col_profile, col_security = st.columns(2)

    with col_profile:
        st.markdown("#### 👤 Profile Details")
        st.markdown(f"- **Email:** `{user.get('email')}`")
        st.markdown(f"- **Firebase UID:** `{user.get('uid')}`")
        current_name = user.get("display_name", "")
        st.markdown(f"- **Display Name:** {current_name or '*(Not set)*'}")

        st.markdown("---")
        st.markdown("#### ✏️ Update Display Name")
        new_display_name = st.text_input("Display Name", value=current_name, key="setting_display_name")
        if st.button("Save Profile Name", key="btn_save_profile", type="primary"):
            if new_display_name.strip():
                with st.spinner("Updating profile..."):
                    success, msg = update_profile(id_token, new_display_name.strip())
                if success:
                    user["display_name"] = new_display_name.strip()
                    st.session_state["firebase_user"] = user
                    st.success("✅ Profile updated successfully!")
                    st.rerun()
                else:
                    st.error(f"❌ {msg}")

    with col_security:
        st.markdown("#### 🔒 Security & Password")
        st.markdown("Need to change your password? Request a secure password reset link sent to your registered email.")
        if st.button("📧 Send Password Reset Email", key="btn_reset_pwd_settings", use_container_width=True):
            with st.spinner("Sending email..."):
                success, msg = send_password_reset(user.get("email", ""))
            if success:
                st.success(f"✅ {msg}")
            else:
                st.error(f"❌ {msg}")

        st.markdown("---")
        st.markdown("#### 🚪 Session")
        st.markdown("Sign out of InsightLab on this browser session.")
        if st.button("Sign Out of Account", key="btn_signout_settings", type="secondary", use_container_width=True):
            sign_out()
            st.success("Signed out.")
            st.rerun()
