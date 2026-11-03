"""
auth_ui.py
==========
Authentication UI Components for InsightLab.

What This Module Does:
----------------------
Provides all the visible authentication pieces users interact with:

1. **Sidebar Auth Header** — Shows "Guest" badge or user profile with Sign Out.
2. **Sign In Dialog** — Modal popup with email + password login form.
3. **Sign Up Dialog** — Modal popup with registration form.
4. **Password Reset Dialog** — Modal popup for forgotten passwords.
5. **Cloud Save Button** — Reusable button for any module; prompts login if guest.
6. **Auth Required Dialog** — Friendly popup when guest tries to save.

Design Principles:
------------------
- Guest users are NEVER blocked or interrupted.
- Auth dialogs appear ONLY when the user clicks "Sign In" or a "Save to Cloud" button.
- The sidebar auth area is compact and subtle — it does not dominate the layout.
- After login, the user returns to exactly where they were, with all work preserved.

Beginner Concept — What is @st.dialog?
---------------------------------------
Streamlit's @st.dialog decorator creates a "modal popup" — a floating window
that appears on top of the current page. It's perfect for login forms because:
  1. It doesn't navigate away from the current page.
  2. When closed, the user's work is still there.
  3. It provides a focused, distraction-free form experience.
The dialog function is called like a regular function, and Streamlit handles
the popup rendering automatically.
"""

import streamlit as st
from typing import Optional, Dict, Any

from utils.firebase_config import is_firebase_configured
from utils.firebase_auth import (
    sign_up,
    sign_in,
    send_password_reset,
    update_profile,
    sign_out,
    get_current_user,
    is_authenticated,
    store_user_session,
)


# =============================================================================
# SECTION 1: SIDEBAR AUTH HEADER
# =============================================================================

def render_auth_header() -> None:
    """
    Renders the authentication status in the sidebar.

    When NOT logged in:
        👤 Guest Mode
        [🔑 Sign In]

    When logged in:
        👤 user@example.com
        [My Workspace] [Sign Out]

    This function should be called INSIDE `with st.sidebar:` in app.py.
    """
    user = get_current_user() if is_firebase_configured() else None

    if user:
        # --- LOGGED IN ---
        display = user.get("display_name") or user.get("email", "User")
        st.markdown(f"👤 **{display}**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🏠 Workspace", key="sidebar_workspace_btn", use_container_width=True):
                st.session_state["_nav_to_workspace"] = True
                st.rerun()
        with col2:
            if st.button("🚪 Sign Out", key="sidebar_signout_btn", use_container_width=True):
                sign_out()
                st.rerun()
    else:
        # --- GUEST MODE ---
        st.markdown("👤 **Guest Mode**")
        st.caption("Full access without login. Sign in to save permanently.")
        if st.button("🔑 Sign In / Sign Up", key="sidebar_signin_btn", use_container_width=True):
            st.session_state["_show_auth_dialog"] = True
            st.rerun()


# =============================================================================
# SECTION 2: AUTH DIALOG (inline — compatible with all Streamlit versions)
# =============================================================================

def render_auth_page() -> None:
    """
    Renders a full authentication page with Login, Sign Up, and Password Reset tabs.

    This is rendered as the MAIN CONTENT (not a dialog popup) when the user
    clicks Sign In. It replaces the current module view temporarily.
    After successful auth, the user is redirected back to their previous module.
    """
    st.title("🔐 InsightLab Account")
    st.markdown("Sign in to save your work permanently, or sign up for a free account.")
    st.markdown("---")

    if not is_firebase_configured():
        st.warning("⚠️ **Firebase is not yet configured in `.streamlit/secrets.toml`.**")
        st.markdown(
            """
            To enable permanent cloud accounts and saving for InsightLab, configure your separate Firebase project:
            
            1. Go to the [Firebase Console](https://console.firebase.google.com/) and create a new project called **`insightlab`** (free Spark plan).
            2. In **Project Settings** → **General**, copy your **Web API Key** and **Project ID**.
            3. In **Build** → **Authentication**, click **Get Started** and enable **Email/Password**.
            4. In **Build** → **Firestore Database**, click **Create Database** (Start in production mode or test mode).
            5. Create or open `.streamlit/secrets.toml` in your project folder and add:
            ```toml
            [firebase]
            api_key = "AIzaSy..."
            project_id = "insightlab-XXXXX"
            storage_bucket = "insightlab-XXXXX.appspot.com" # optional
            ```
            """
        )
        if st.button("⬅️ Continue as Guest (All Features Work Free!)", use_container_width=True, type="primary"):
            st.session_state.pop("_show_auth_dialog", None)
            st.rerun()
        return

    tab_login, tab_signup, tab_reset = st.tabs(["🔑 Sign In", "📝 Sign Up", "🔄 Reset Password"])

    # ---- TAB 1: SIGN IN ----------------------------------------------------
    with tab_login:
        st.subheader("Welcome Back")
        with st.form("login_form", clear_on_submit=False):
            login_email = st.text_input("Email Address", key="login_email_input",
                                        placeholder="you@example.com")
            login_password = st.text_input("Password", type="password", key="login_password_input",
                                           placeholder="Enter your password")
            login_submitted = st.form_submit_button("🔑 Sign In", use_container_width=True,
                                                     type="primary")

        if login_submitted:
            if not login_email or not login_password:
                st.error("Please enter both email and password.")
            else:
                with st.spinner("Signing in..."):
                    success, result = sign_in(login_email, login_password)
                if success:
                    store_user_session(result)
                    # Create/update user profile in Firestore
                    _save_user_profile_on_login(result)
                    st.success(f"✅ Welcome back, {result.get('display_name') or result.get('email')}!")
                    st.session_state.pop("_show_auth_dialog", None)
                    _handle_pending_save()
                    st.rerun()
                else:
                    st.error(f"❌ {result.get('error', 'Login failed.')}")

    # ---- TAB 2: SIGN UP ----------------------------------------------------
    with tab_signup:
        st.subheader("Create Your Free Account")
        st.caption("Your account lets you save projects, datasets, and reports to the cloud.")
        with st.form("signup_form", clear_on_submit=False):
            signup_name = st.text_input("Your Name", key="signup_name_input",
                                        placeholder="e.g. Sarah")
            signup_email = st.text_input("Email Address", key="signup_email_input",
                                         placeholder="you@example.com")
            signup_password = st.text_input("Password (min 6 characters)", type="password",
                                            key="signup_password_input")
            signup_confirm = st.text_input("Confirm Password", type="password",
                                           key="signup_confirm_input")
            signup_submitted = st.form_submit_button("📝 Create Account", use_container_width=True,
                                                      type="primary")

        if signup_submitted:
            if not signup_email or not signup_password:
                st.error("Please fill in all required fields.")
            elif signup_password != signup_confirm:
                st.error("Passwords do not match. Please re-enter.")
            elif len(signup_password) < 6:
                st.error("Password must be at least 6 characters long.")
            else:
                with st.spinner("Creating your account..."):
                    success, result = sign_up(signup_email, signup_password, signup_name)
                if success:
                    store_user_session(result)
                    _save_user_profile_on_login(result)
                    st.success(f"✅ Account created! Welcome, {signup_name or signup_email}!")
                    st.balloons()
                    st.session_state.pop("_show_auth_dialog", None)
                    _handle_pending_save()
                    st.rerun()
                else:
                    st.error(f"❌ {result.get('error', 'Registration failed.')}")

    # ---- TAB 3: PASSWORD RESET ---------------------------------------------
    with tab_reset:
        st.subheader("Forgot Your Password?")
        st.markdown("Enter your email and we'll send you a reset link.")
        with st.form("reset_form", clear_on_submit=False):
            reset_email = st.text_input("Email Address", key="reset_email_input",
                                        placeholder="you@example.com")
            reset_submitted = st.form_submit_button("📧 Send Reset Link", use_container_width=True)

        if reset_submitted:
            if not reset_email:
                st.error("Please enter your email address.")
            else:
                with st.spinner("Sending reset email..."):
                    success, message = send_password_reset(reset_email)
                if success:
                    st.success(f"✅ {message}")
                else:
                    st.error(f"❌ {message}")

    # ---- BACK BUTTON -------------------------------------------------------
    st.markdown("---")
    if st.button("⬅️ Back to InsightLab (continue as guest)", use_container_width=True):
        st.session_state.pop("_show_auth_dialog", None)
        st.rerun()

    st.caption(
        "💡 You don't need an account to use InsightLab. All analysis features "
        "work without signing in. An account is only needed to save your work "
        "permanently to the cloud."
    )


# =============================================================================
# SECTION 3: CLOUD SAVE BUTTON (reusable across modules)
# =============================================================================

def render_cloud_save_button(
    save_type: str,
    save_data: Dict[str, Any],
    label: str = "☁️ Save to Cloud",
    key_suffix: str = "",
) -> bool:
    """
    Renders a "Save to Cloud" button that any module can use.

    If the user is authenticated → saves immediately.
    If the user is a guest → shows a friendly prompt to sign in.

    Parameters:
    -----------
    save_type  : str — "project", "dataset", "analysis", "model", or "report"
    save_data  : dict — the data to save (module-specific)
    label      : str — button label text
    key_suffix : str — unique suffix for the Streamlit button key

    Returns:
    --------
    bool — True if save was completed (user is authenticated and save succeeded).

    How the "Smart Save" Flow Works:
    --------------------------------
    1. User clicks "Save to Cloud" button.
    2. If logged in → save immediately and show success.
    3. If guest → store the save request in session_state["pending_save"]
       and redirect to the sign-in page.
    4. After sign-in, _handle_pending_save() detects the pending request
       and completes the save automatically.
    5. User doesn't have to click save again.
    """
    if not is_firebase_configured():
        return False

    btn_key = f"cloud_save_{save_type}_{key_suffix}"

    if st.button(label, key=btn_key, use_container_width=True):
        if is_authenticated():
            return _execute_save(save_type, save_data)
        else:
            # Store the pending save and redirect to auth
            st.session_state["pending_save"] = {
                "type": save_type,
                "data": save_data,
            }
            st.session_state["_show_auth_dialog"] = True
            st.info(
                "☁️ **Sign in to save your work permanently.**\n\n"
                "Your analysis will be preserved. After signing in, "
                "your work will be saved automatically."
            )
            st.rerun()

    return False


def render_save_section(save_type: str, save_data: Dict[str, Any], key_suffix: str = "") -> None:
    """
    Renders a complete cloud save section with explanatory text.
    Designed to be appended at the bottom of any module.
    """
    if not is_firebase_configured():
        return

    st.markdown("---")
    st.markdown("#### ☁️ Save to Cloud")

    if is_authenticated():
        user = get_current_user()
        st.caption(f"Signed in as **{user.get('display_name') or user.get('email')}**")
    else:
        st.caption("Sign in to save your work permanently. Your current session data will be preserved.")

    render_cloud_save_button(save_type, save_data, key_suffix=key_suffix)


# =============================================================================
# SECTION 4: SAVE EXECUTION
# =============================================================================

def _execute_save(save_type: str, save_data: Dict[str, Any]) -> bool:
    """
    Executes the actual cloud save operation.

    Routes to the appropriate firebase_db function based on save_type.
    """
    from utils.firebase_auth import get_current_user, get_valid_id_token
    from utils.firebase_db import (
        save_project,
        save_dataset_metadata,
        save_report_metadata,
        save_analysis,
        save_model_info,
    )

    user = get_current_user()
    if not user:
        st.error("Not authenticated. Please sign in first.")
        return False

    uid = user["uid"]
    id_token = get_valid_id_token()
    if not id_token:
        st.error("Session expired. Please sign in again.")
        return False

    save_functions = {
        "project": save_project,
        "dataset": save_dataset_metadata,
        "report": save_report_metadata,
        "analysis": save_analysis,
        "model": save_model_info,
    }

    save_fn = save_functions.get(save_type)
    if not save_fn:
        st.error(f"Unknown save type: {save_type}")
        return False

    # Deduplication check to prevent duplicate saves during Streamlit reruns
    save_fingerprint = f"{save_type}_{save_data.get('name')}_{uid}"
    if st.session_state.get("_last_saved_fingerprint") == save_fingerprint:
        st.info(f"ℹ️ {save_type.capitalize()} was already saved in this session.")
        return True

    # Optional: If save_data contains file_bytes and Cloud Storage is available, upload file
    file_bytes = save_data.pop("file_bytes", None)
    file_name = save_data.get("name", "file")
    
    if file_bytes is not None:
        try:
            from utils.firebase_storage import is_storage_available, upload_file, user_storage_path, get_content_type
            if is_storage_available():
                storage_path = user_storage_path(uid, f"{save_type}s", file_name)
                mime = get_content_type(file_name)
                ok, storage_res = upload_file(file_bytes, storage_path, id_token, mime)
                if ok and "download_url" in storage_res:
                    save_data["storage_url"] = storage_res["download_url"]
                    save_data["storage_path"] = storage_path
        except Exception:
            pass  # If storage fails or requires Blaze plan, continue saving metadata to Firestore

    with st.spinner(f"☁️ Saving {save_type} to cloud..."):
        success, result = save_fn(uid, save_data, id_token)

    if success:
        st.session_state["_last_saved_fingerprint"] = save_fingerprint
        st.success(f"✅ {save_type.capitalize()} saved to cloud successfully!")
        return True
    else:
        st.error(f"❌ Save failed: {result.get('error', 'Unknown error')}")
        return False


def _handle_pending_save() -> None:
    """
    Checks for a pending save operation and completes it after authentication.

    This is called automatically after a successful sign-in or sign-up.
    If the user clicked "Save to Cloud" before authenticating, their save
    request was stored in session_state["pending_save"]. We now complete it.
    """
    pending = st.session_state.get("pending_save")
    if not pending:
        return

    save_type = pending.get("type", "")
    save_data = pending.get("data", {})

    # Remove the pending marker BEFORE executing (prevents infinite loops)
    st.session_state.pop("pending_save", None)

    if save_type and save_data:
        _execute_save(save_type, save_data)


def _save_user_profile_on_login(user_data: Dict[str, Any]) -> None:
    """
    Creates or updates the user's profile in Firestore after login/signup.
    Runs silently — errors are logged but don't interrupt the user.
    """
    try:
        from utils.firebase_db import save_user_profile, update_last_login
        uid = user_data.get("uid", "")
        email = user_data.get("email", "")
        display_name = user_data.get("display_name", "")
        id_token = user_data.get("id_token", "")

        if uid and id_token:
            save_user_profile(uid, email, display_name, id_token)
    except Exception:
        pass  # Profile save is non-critical
