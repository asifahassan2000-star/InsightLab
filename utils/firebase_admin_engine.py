"""
firebase_admin_engine.py
========================
Firebase Admin SDK Engine for InsightLab.

What This Module Does:
----------------------
Handles trusted server-side Firebase operations using the official
`firebase-admin` Python SDK.

Core Responsibilities:
----------------------
1. **Safe Initialization**:
   Initializes the Admin SDK singleton using service account credentials
   stored securely in `st.secrets["firebase_admin"]`.
   Does NOT crash if credentials are not yet configured.

2. **Server-Side Token Verification**:
   Cryptographically validates Firebase ID tokens via `auth.verify_id_token(id_token)`.
   Extracts the verified `uid` directly from the decoded JWT.
   NEVER trusts an unverified client UID!

3. **Ownership Validation**:
   Enforces that every database request to `/users/{userId}/...` is made
   by a verified token where `verified_uid == userId`.

4. **Trusted Admin Firestore Access**:
   Provides server-side access to Cloud Firestore when Admin credentials are active.
"""

from typing import Any, Dict, Optional, Tuple
import firebase_admin
from firebase_admin import auth, credentials, firestore


_admin_app: Optional[firebase_admin.App] = None


def get_admin_app() -> Optional[firebase_admin.App]:
    """
    Initializes or retrieves the Firebase Admin SDK App singleton.

    Reads credentials from Streamlit Secrets:
      - st.secrets["firebase_admin"]
    
    Returns:
    --------
    firebase_admin.App or None if credentials are not configured.
    """
    global _admin_app

    # If already initialized in this Python process, return existing instance
    if _admin_app is not None:
        return _admin_app

    try:
        # Check if default app was already initialized elsewhere
        _admin_app = firebase_admin.get_app()
        return _admin_app
    except ValueError:
        pass  # Not yet initialized

    # Attempt to load credentials from Streamlit secrets
    try:
        import streamlit as st
        admin_secrets = st.secrets.get("firebase_admin")
        if not admin_secrets:
            return None

        # Build credential dictionary
        # Handle both raw private_key strings and multiline strings
        cred_dict = dict(admin_secrets)
        if "private_key" in cred_dict:
            # Normalize escaped newlines if needed
            pk = str(cred_dict["private_key"]).replace("\\n", "\n").strip()
            cred_dict["private_key"] = pk

        # Only initialize if required service account keys exist
        required_keys = ["project_id", "private_key", "client_email"]
        if not all(k in cred_dict and cred_dict[k] for k in required_keys):
            return None

        cred = credentials.Certificate(cred_dict)
        _admin_app = firebase_admin.initialize_app(cred)
        return _admin_app

    except Exception:
        return None


def is_admin_sdk_initialized() -> bool:
    """Returns True if the Firebase Admin SDK is initialized and ready."""
    return get_admin_app() is not None


def verify_client_token(id_token: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Cryptographically verifies a Firebase ID token server-side using the Admin SDK.

    Parameters:
    -----------
    id_token : str — the raw JWT ID token sent from the client.

    Returns:
    --------
    (is_valid: bool, result: dict)
        On success: (True, decoded_token) where decoded_token["uid"] is verified.
        On failure: (False, {"error": "Reason"})
    """
    if not id_token:
        return False, {"error": "Missing authentication token."}

    app = get_admin_app()
    if app is None:
        # Admin SDK not initialized with service account;
        # Fall back to token inspection / REST verification
        return _fallback_verify_token(id_token)

    try:
        decoded_token = auth.verify_id_token(id_token, app=app)
        return True, decoded_token
    except auth.ExpiredIdTokenError:
        return False, {"error": "Authentication token has expired. Please sign in again."}
    except auth.InvalidIdTokenError:
        return False, {"error": "Invalid authentication token."}
    except Exception as e:
        return False, {"error": f"Token verification failed: {str(e)}"}


def verify_resource_ownership(id_token: str, expected_uid: str) -> Tuple[bool, str]:
    """
    Verifies that the caller's ID token belongs to the expected user UID.

    Never trusts a user ID supplied directly by the client!

    Returns:
    --------
    (is_owner: bool, verified_uid_or_error: str)
    """
    valid, result = verify_client_token(id_token)
    if not valid:
        return False, result.get("error", "Unauthorized")

    verified_uid = result.get("uid", "")
    if verified_uid != expected_uid:
        return False, f"Permission denied: token UID ({verified_uid}) does not match resource owner ({expected_uid})."

    return True, verified_uid


def get_admin_firestore_client():
    """Returns the Firestore client from the initialized Admin App, or None."""
    app = get_admin_app()
    if app is None:
        return None
    try:
        return firestore.client(app=app)
    except Exception:
        return None


def _fallback_verify_token(id_token: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Lightweight fallback token verification when Admin service account is not yet provided.
    Calls Firebase Auth Identity Toolkit accounts:lookup REST endpoint to ensure token is valid.
    """
    import requests
    from utils.firebase_config import AUTH_BASE_URL, get_firebase_config

    config = get_firebase_config()
    api_key = config.get("api_key")
    if not api_key:
        return False, {"error": "Firebase is not configured."}

    url = f"{AUTH_BASE_URL}/accounts:lookup?key={api_key}"
    try:
        resp = requests.post(url, json={"idToken": id_token}, timeout=10)
        data = resp.json()
        if "error" in data:
            return False, {"error": data["error"].get("message", "Token invalid")}

        users = data.get("users", [])
        if users:
            return True, {
                "uid": users[0].get("localId"),
                "email": users[0].get("email"),
                "email_verified": users[0].get("emailVerified", False),
            }
        return False, {"error": "User not found"}
    except Exception as e:
        return False, {"error": f"Token verification error: {str(e)}"}
