"""
firebase_auth.py
================
Firebase Authentication Engine for InsightLab.

What This Module Does:
----------------------
Handles all user authentication operations by calling the Firebase Auth REST
API directly over HTTPS. No Firebase SDK installation required — we use
plain HTTP requests with the `requests` library.

Why REST API Instead of Firebase SDK?
-------------------------------------
1. **Zero compatibility issues** — works with any Python version (3.7+).
2. **Lightweight** — only needs the `requests` library (already installed).
3. **Educational** — you can see exactly what data is sent and received.
4. **No admin credentials** — uses the public Web API Key, not a service account.

Firebase Auth REST API Flow:
----------------------------
  Sign Up:   POST .../accounts:signUp            → creates a new user
  Sign In:   POST .../accounts:signInWithPassword → returns an idToken
  Refresh:   POST .../token                       → refreshes an expired idToken
  Reset:     POST .../accounts:sendOobCode        → sends password reset email
  Profile:   POST .../accounts:update             → updates display name
  Lookup:    POST .../accounts:lookup              → gets user profile data

The idToken:
------------
When a user signs in, Firebase returns an **idToken** (a JSON Web Token / JWT).
This token is proof that the user is authenticated. It expires after 1 hour.
We also get a **refreshToken** which never expires and can be used to get
a new idToken without re-entering the password.

We store both tokens in `st.session_state` so they persist during the session.

Security:
---------
- Passwords are NEVER stored locally — they go directly to Firebase over HTTPS.
- Tokens are kept only in server-side session state (not in cookies or URLs).
- The Web API Key is NOT a secret — Firebase Security Rules protect data, not the key.
"""

import datetime
from typing import Any, Dict, Optional, Tuple

import requests

from utils.firebase_config import (
    AUTH_BASE_URL,
    TOKEN_REFRESH_URL,
    get_firebase_config,
    is_firebase_configured,
)


# =============================================================================
# DATA TYPES
# =============================================================================

# A "user dict" is the authenticated user's data stored in session state.
# Structure:
# {
#     "uid": "abc123...",
#     "email": "user@example.com",
#     "display_name": "John Doe",
#     "id_token": "eyJhbGci...",       ← Firebase JWT, expires in 1 hour
#     "refresh_token": "AGEhc0B...",   ← used to get a new id_token
#     "token_expiry": datetime,        ← when the id_token expires
# }


# =============================================================================
# SECTION 1: SIGN UP (CREATE ACCOUNT)
# =============================================================================

def sign_up(email: str, password: str, display_name: str = "") -> Tuple[bool, Dict[str, Any]]:
    """
    Creates a new user account with Firebase Authentication.

    Parameters:
    -----------
    email        : str — the user's email address.
    password     : str — must be at least 6 characters (Firebase requirement).
    display_name : str — optional friendly name (e.g. "Sarah").

    Returns:
    --------
    (success: bool, result: dict)
        On success: result = user data dict (uid, email, tokens, etc.)
        On failure: result = {"error": "Human-readable error message"}

    What happens behind the scenes:
    --------------------------------
    1. We POST to Firebase's signUp endpoint with email + password.
    2. Firebase creates the user and returns an idToken + refreshToken.
    3. If a display_name was provided, we call the update endpoint to set it.
    4. We package everything into a user dict and return it.
    """
    config = get_firebase_config()
    if not config["configured"]:
        return False, {"error": "Firebase is not configured. Add credentials to .streamlit/secrets.toml"}

    api_key = config["api_key"]
    url = f"{AUTH_BASE_URL}/accounts:signUp?key={api_key}"

    try:
        response = requests.post(url, json={
            "email": email,
            "password": password,
            "returnSecureToken": True,
        }, timeout=15)

        data = response.json()

        if "error" in data:
            error_msg = _translate_firebase_error(data["error"].get("message", ""))
            return False, {"error": error_msg}

        # Build user dict from the response
        user_data = _build_user_dict(data)

        # Set display name if provided
        if display_name.strip():
            _update_display_name(user_data["id_token"], display_name.strip(), api_key)
            user_data["display_name"] = display_name.strip()

        return True, user_data

    except requests.exceptions.ConnectionError:
        return False, {"error": "Cannot connect to Firebase. Please check your internet connection."}
    except requests.exceptions.Timeout:
        return False, {"error": "Request timed out. Please try again."}
    except Exception as e:
        return False, {"error": f"Unexpected error: {str(e)}"}


# =============================================================================
# SECTION 2: SIGN IN (LOG IN)
# =============================================================================

def sign_in(email: str, password: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Authenticates an existing user with email and password.

    Parameters:
    -----------
    email    : str — registered email address.
    password : str — the user's password.

    Returns:
    --------
    (success: bool, result: dict)
        On success: result = user data dict
        On failure: result = {"error": "Human-readable error message"}

    How Sign-In Works:
    ------------------
    1. We POST the email + password to Firebase's signInWithPassword endpoint.
    2. Firebase verifies the credentials and returns an idToken + refreshToken.
    3. We then call the lookup endpoint to fetch the display name.
    4. We package everything into a user dict.
    """
    config = get_firebase_config()
    if not config["configured"]:
        return False, {"error": "Firebase is not configured."}

    api_key = config["api_key"]
    url = f"{AUTH_BASE_URL}/accounts:signInWithPassword?key={api_key}"

    try:
        response = requests.post(url, json={
            "email": email,
            "password": password,
            "returnSecureToken": True,
        }, timeout=15)

        data = response.json()

        if "error" in data:
            error_msg = _translate_firebase_error(data["error"].get("message", ""))
            return False, {"error": error_msg}

        user_data = _build_user_dict(data)

        # Fetch display name via lookup
        lookup_result = _lookup_user(user_data["id_token"], api_key)
        if lookup_result:
            user_data["display_name"] = lookup_result.get("displayName", "")

        return True, user_data

    except requests.exceptions.ConnectionError:
        return False, {"error": "Cannot connect to Firebase. Please check your internet connection."}
    except requests.exceptions.Timeout:
        return False, {"error": "Request timed out. Please try again."}
    except Exception as e:
        return False, {"error": f"Unexpected error: {str(e)}"}


# =============================================================================
# SECTION 3: TOKEN REFRESH
# =============================================================================

def refresh_token(refresh_tok: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Exchanges a refresh token for a new id token.

    Why is this needed?
    -------------------
    Firebase id tokens expire after 1 hour. If the user keeps the app open
    for longer, their token becomes invalid. The refresh token (which doesn't
    expire) can be used to silently get a new id token without asking the
    user to log in again.

    Parameters:
    -----------
    refresh_tok : str — the refresh token from the original sign-in.

    Returns:
    --------
    (success, result) — same pattern as sign_in.
    """
    config = get_firebase_config()
    if not config["configured"]:
        return False, {"error": "Firebase is not configured."}

    api_key = config["api_key"]
    url = f"{TOKEN_REFRESH_URL}?key={api_key}"

    try:
        response = requests.post(url, data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_tok,
        }, timeout=15)

        data = response.json()

        if "error" in data:
            return False, {"error": "Session expired. Please log in again."}

        return True, {
            "id_token": data["id_token"],
            "refresh_token": data["refresh_token"],
            "token_expiry": datetime.datetime.now() + datetime.timedelta(
                seconds=int(data.get("expires_in", 3600))
            ),
        }

    except Exception:
        return False, {"error": "Could not refresh session. Please log in again."}


# =============================================================================
# SECTION 4: PASSWORD RESET
# =============================================================================

def send_password_reset(email: str) -> Tuple[bool, str]:
    """
    Sends a password reset email to the user.

    Parameters:
    -----------
    email : str — the registered email address.

    Returns:
    --------
    (success: bool, message: str)
        On success: message = confirmation string.
        On failure: message = error description.

    How It Works:
    -------------
    Firebase sends a pre-built password reset email with a link.
    The user clicks the link, enters a new password on a Firebase-hosted page,
    and their password is updated. We don't handle the new password ourselves.
    """
    config = get_firebase_config()
    if not config["configured"]:
        return False, "Firebase is not configured."

    api_key = config["api_key"]
    url = f"{AUTH_BASE_URL}/accounts:sendOobCode?key={api_key}"

    try:
        response = requests.post(url, json={
            "requestType": "PASSWORD_RESET",
            "email": email,
        }, timeout=15)

        data = response.json()

        if "error" in data:
            error_msg = _translate_firebase_error(data["error"].get("message", ""))
            return False, error_msg

        return True, f"Password reset email sent to {email}. Check your inbox."

    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to Firebase. Check your internet connection."
    except Exception as e:
        return False, f"Error: {str(e)}"


# =============================================================================
# SECTION 5: UPDATE PROFILE
# =============================================================================

def update_profile(id_token: str, display_name: str) -> Tuple[bool, str]:
    """
    Updates the authenticated user's display name.

    Parameters:
    -----------
    id_token     : str — the current valid Firebase id token.
    display_name : str — the new display name.

    Returns:
    --------
    (success: bool, message: str)
    """
    config = get_firebase_config()
    if not config["configured"]:
        return False, "Firebase is not configured."

    api_key = config["api_key"]

    try:
        success = _update_display_name(id_token, display_name, api_key)
        if success:
            return True, "Profile updated successfully."
        return False, "Failed to update profile."
    except Exception as e:
        return False, f"Error: {str(e)}"


# =============================================================================
# SECTION 6: SESSION MANAGEMENT HELPERS
# =============================================================================

def get_current_user() -> Optional[Dict[str, Any]]:
    """
    Returns the current authenticated user from Streamlit session state,
    or None if not logged in (guest mode).

    Also handles automatic token refresh when the token is about to expire.
    """
    try:
        import streamlit as st
        user_data = st.session_state.get("firebase_user")
        if not user_data:
            return None

        # Check if token needs refresh (within 5 minutes of expiry)
        expiry = user_data.get("token_expiry")
        if expiry and isinstance(expiry, datetime.datetime):
            buffer_time = datetime.timedelta(minutes=5)
            if datetime.datetime.now() > (expiry - buffer_time):
                # Silently refresh the token
                success, result = refresh_token(user_data["refresh_token"])
                if success:
                    user_data["id_token"] = result["id_token"]
                    user_data["refresh_token"] = result["refresh_token"]
                    user_data["token_expiry"] = result["token_expiry"]
                    st.session_state["firebase_user"] = user_data
                else:
                    # Token refresh failed — user must log in again
                    st.session_state.pop("firebase_user", None)
                    return None

        return user_data
    except Exception:
        return None


def is_authenticated() -> bool:
    """Returns True if a user is currently logged in."""
    return get_current_user() is not None


def get_valid_id_token() -> Optional[str]:
    """
    Returns a valid (non-expired) id token for the current user,
    refreshing if necessary. Returns None if not authenticated.
    """
    user = get_current_user()  # This auto-refreshes if needed
    if user:
        return user.get("id_token")
    return None


def sign_out() -> None:
    """
    Signs the user out by clearing their data from session state.

    Note: We don't need to call Firebase to "invalidate" the token —
    we simply forget it. The token will expire on its own after 1 hour.
    Saved work remains safe in Firestore.
    """
    try:
        import streamlit as st
        st.session_state.pop("firebase_user", None)
        st.session_state.pop("pending_save", None)
    except Exception:
        pass


def store_user_session(user_data: Dict[str, Any]) -> None:
    """
    Saves the authenticated user's data to Streamlit session state.
    Called after successful sign-in or sign-up.
    """
    try:
        import streamlit as st
        st.session_state["firebase_user"] = user_data
    except Exception:
        pass


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _build_user_dict(firebase_response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts a raw Firebase Auth API response into our standard user dict.

    Firebase returns fields like 'localId', 'idToken', etc.
    We normalize them into cleaner names.
    """
    expires_in = int(firebase_response.get("expiresIn", 3600))
    return {
        "uid": firebase_response.get("localId", ""),
        "email": firebase_response.get("email", ""),
        "display_name": firebase_response.get("displayName", ""),
        "id_token": firebase_response.get("idToken", ""),
        "refresh_token": firebase_response.get("refreshToken", ""),
        "token_expiry": datetime.datetime.now() + datetime.timedelta(seconds=expires_in),
    }


def _update_display_name(id_token: str, display_name: str, api_key: str) -> bool:
    """Updates the user's display name via the Firebase Auth update endpoint."""
    url = f"{AUTH_BASE_URL}/accounts:update?key={api_key}"
    try:
        response = requests.post(url, json={
            "idToken": id_token,
            "displayName": display_name,
            "returnSecureToken": False,
        }, timeout=10)
        return response.status_code == 200
    except Exception:
        return False


def _lookup_user(id_token: str, api_key: str) -> Optional[Dict[str, Any]]:
    """Fetches the user's profile data from Firebase."""
    url = f"{AUTH_BASE_URL}/accounts:lookup?key={api_key}"
    try:
        response = requests.post(url, json={
            "idToken": id_token,
        }, timeout=10)
        data = response.json()
        users = data.get("users", [])
        if users:
            return users[0]
    except Exception:
        pass
    return None


def _translate_firebase_error(error_code: str) -> str:
    """
    Converts Firebase's internal error codes into friendly human-readable messages.

    Firebase returns error codes like "EMAIL_EXISTS" or "WEAK_PASSWORD".
    These are not user-friendly, so we translate them.
    """
    translations = {
        "EMAIL_EXISTS": "An account with this email already exists. Try signing in instead.",
        "EMAIL_NOT_FOUND": "No account found with this email address.",
        "INVALID_PASSWORD": "Incorrect password. Please try again.",
        "INVALID_EMAIL": "Please enter a valid email address.",
        "WEAK_PASSWORD": "Password must be at least 6 characters long.",
        "USER_DISABLED": "This account has been disabled. Contact support.",
        "TOO_MANY_ATTEMPTS_TRY_LATER": "Too many failed attempts. Please try again later.",
        "INVALID_LOGIN_CREDENTIALS": "Invalid email or password. Please try again.",
        "MISSING_PASSWORD": "Please enter your password.",
        "MISSING_EMAIL": "Please enter your email address.",
    }
    # Check if the error code matches any known code (partial match)
    for code, message in translations.items():
        if code in error_code.upper():
            return message

    return f"Authentication error: {error_code}"
