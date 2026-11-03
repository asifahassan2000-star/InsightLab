"""
firebase_config.py
==================
Firebase Configuration Manager for InsightLab.

What This Module Does:
----------------------
This is the single source of truth for all Firebase settings. Every other
Firebase module (auth, database, storage) imports its config from here.

How Firebase Configuration Works:
---------------------------------
Firebase needs a few pieces of information to connect to your project:

1. **API Key** (Web API Key):
   Found in Firebase Console → Project Settings → General.
   Used to call the Firebase Auth REST API (sign up, sign in, etc.).
   Despite the name, this key is safe to use in frontend/API calls —
   Firebase Security Rules protect your actual data, not this key.

2. **Project ID**:
   Your unique Firebase project identifier (e.g. "insightlab-abc123").
   Used to call the Firestore REST API and identify your database.

3. **Storage Bucket** (Optional):
   The Cloud Storage bucket name (e.g. "insightlab-abc123.appspot.com").
   Used for file uploads. May require a Blaze (pay-as-you-go) plan.

Where To Store These Values:
----------------------------
Create a file at:  <your-project>/.streamlit/secrets.toml

    [firebase]
    api_key = "AIzaSy..."
    project_id = "insightlab-abc123"
    storage_bucket = "insightlab-abc123.appspot.com"

Streamlit automatically reads this file and makes it available via
`st.secrets`. The file should NEVER be committed to Git / GitHub.

Beginner Note — What is .streamlit/secrets.toml?
-------------------------------------------------
Streamlit has a built-in secrets manager. You create a folder called
`.streamlit` in your project root, and inside it a file called `secrets.toml`.
TOML is a simple config format (like INI files). Streamlit reads this at
startup and exposes the values through `st.secrets["firebase"]["api_key"]`.
This keeps your private keys out of your code.
"""

from typing import Any, Dict, Optional


# =============================================================================
# CONFIGURATION LOADER
# =============================================================================

# Default project settings for InsightLab Firebase Project
DEFAULT_PROJECT_ID = "insightlab-58d6c"
DEFAULT_API_KEY = "AIzaSyDg8IglpO26e1zGVaAWt1gpgyBTxug9BTc"
DEFAULT_AUTH_DOMAIN = "insightlab-58d6c.firebaseapp.com"
DEFAULT_STORAGE_BUCKET = "insightlab-58d6c.firebasestorage.app"
DEFAULT_APP_ID = "1:1023407849681:web:5229e0dbf32a72158945f0"


def get_firebase_config() -> Dict[str, Optional[str]]:
    """
    Loads Firebase configuration from Streamlit secrets, with fallback to project defaults.
    """
    config = {
        "api_key": DEFAULT_API_KEY,
        "project_id": DEFAULT_PROJECT_ID,
        "auth_domain": DEFAULT_AUTH_DOMAIN,
        "storage_bucket": DEFAULT_STORAGE_BUCKET,
        "app_id": DEFAULT_APP_ID,
        "configured": True,
    }

    try:
        import streamlit as st
        firebase_section = st.secrets.get("firebase", {})
        if firebase_section.get("api_key"):
            config["api_key"] = firebase_section.get("api_key", "").strip()
        if firebase_section.get("project_id"):
            config["project_id"] = firebase_section.get("project_id", "").strip()
        if firebase_section.get("storage_bucket"):
            config["storage_bucket"] = firebase_section.get("storage_bucket", "").strip()
        if firebase_section.get("auth_domain"):
            config["auth_domain"] = firebase_section.get("auth_domain", "").strip()
    except Exception:
        # secrets.toml doesn't exist or isn't readable — defaults are active
        pass

    config["configured"] = bool(config["api_key"] and config["project_id"])
    return config


def is_firebase_configured() -> bool:
    """
    Quick check: is Firebase ready to use?

    Returns True only if api_key and project_id are both present
    in .streamlit/secrets.toml.
    """
    return get_firebase_config()["configured"]


# =============================================================================
# REST API BASE URLS
# =============================================================================

# Firebase Authentication REST API
# Documentation: https://firebase.google.com/docs/reference/rest/auth
AUTH_BASE_URL = "https://identitytoolkit.googleapis.com/v1"

# Token refresh endpoint (for refreshing expired tokens)
TOKEN_REFRESH_URL = "https://securetoken.googleapis.com/v1/token"

# Firestore REST API
# Documentation: https://firebase.google.com/docs/firestore/reference/rest
def get_firestore_base_url(project_id: str) -> str:
    """Returns the Firestore REST API base URL for a given project."""
    return (
        f"https://firestore.googleapis.com/v1/"
        f"projects/{project_id}/databases/(default)/documents"
    )

# Firebase Cloud Storage REST API
def get_storage_base_url(bucket: str) -> str:
    """Returns the Cloud Storage REST API base URL for a given bucket."""
    return f"https://firebasestorage.googleapis.com/v0/b/{bucket}/o"
