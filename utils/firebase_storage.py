"""
firebase_storage.py
===================
Firebase Cloud Storage Engine for InsightLab.

What This Module Does:
----------------------
Uploads and downloads files (CSV, Excel, PDF) to/from Firebase Cloud Storage
using the REST API.

IMPORTANT — Firebase Storage Plan Requirements:
-------------------------------------------------
Firebase Cloud Storage may require a Blaze (pay-as-you-go) plan for new
projects created after September 2024. The Spark (free) plan may not include
Cloud Storage on newer projects.

This module checks availability before uploading. If Storage is not available,
it returns a clear error message suggesting alternatives (like downloading
files directly from InsightLab instead).

Free Spark Plan Limits (when available):
  - 5 GB total storage
  - 1 GB/day download
  - 50,000 reads/day

Storage Path Structure:
-----------------------
All files are stored under the user's UID:

    users/{uid}/datasets/{filename}
    users/{uid}/reports/{filename}
    users/{uid}/exports/{filename}
    users/{uid}/models/{filename}

This ensures one user can never access another user's files.
"""

from typing import Any, Dict, Optional, Tuple

import requests

from utils.firebase_config import get_firebase_config, get_storage_base_url


# =============================================================================
# STORAGE AVAILABILITY CHECK
# =============================================================================

def is_storage_available() -> bool:
    """
    Checks if Firebase Cloud Storage is configured (bucket name is set).

    Note: This checks config only, not whether the bucket is actually
    accessible. The first upload/download will reveal connectivity issues.
    """
    config = get_firebase_config()
    return bool(config.get("storage_bucket"))


# =============================================================================
# FILE UPLOAD
# =============================================================================

def upload_file(
    file_bytes: bytes,
    storage_path: str,
    id_token: str,
    content_type: str = "application/octet-stream",
) -> Tuple[bool, Dict[str, Any]]:
    """
    Uploads a file to Firebase Cloud Storage.

    Parameters:
    -----------
    file_bytes    : bytes — the raw file content
    storage_path  : str — path inside the bucket (e.g. "users/uid/datasets/data.csv")
    id_token      : str — Firebase auth token
    content_type  : str — MIME type (e.g. "text/csv", "application/pdf")

    Returns:
    --------
    (success: bool, result: dict)
        On success: result = {"name": path, "size": bytes, "download_url": url}
        On failure: result = {"error": "message"}

    How Firebase Storage REST API Works:
    ------------------------------------
    We POST the raw file bytes to the storage endpoint with the path encoded
    in the URL. Firebase stores the file and returns metadata including a
    download token that can be used to retrieve the file later.
    """
    config = get_firebase_config()
    bucket = config.get("storage_bucket")
    if not bucket:
        return False, {
            "error": (
                "Cloud Storage is not configured. Add 'storage_bucket' to your "
                ".streamlit/secrets.toml file. Note: Firebase Cloud Storage may "
                "require the Blaze (pay-as-you-go) plan for new projects."
            )
        }

    base_url = get_storage_base_url(bucket)
    # URL-encode the storage path (forward slashes become %2F)
    encoded_path = requests.utils.quote(storage_path, safe="")
    url = f"{base_url}?uploadType=media&name={encoded_path}"

    try:
        response = requests.post(
            url,
            data=file_bytes,
            headers={
                "Authorization": f"Bearer {id_token}",
                "Content-Type": content_type,
            },
            timeout=60,  # Allow more time for large files
        )

        if response.status_code == 200:
            data = response.json()
            download_token = data.get("downloadTokens", "")
            download_url = (
                f"https://firebasestorage.googleapis.com/v0/b/{bucket}"
                f"/o/{encoded_path}?alt=media&token={download_token}"
            )
            return True, {
                "name": data.get("name", storage_path),
                "size": int(data.get("size", 0)),
                "content_type": data.get("contentType", content_type),
                "download_url": download_url,
                "download_token": download_token,
            }
        else:
            error_data = {}
            try:
                error_data = response.json()
            except Exception:
                pass
            error_msg = error_data.get("error", {}).get("message", f"HTTP {response.status_code}")

            # Provide helpful guidance for common errors
            if response.status_code == 402 or "billing" in str(error_msg).lower():
                error_msg = (
                    "Cloud Storage requires the Firebase Blaze plan. "
                    "Your files can still be downloaded directly from InsightLab. "
                    "To enable cloud storage, upgrade to Blaze in the Firebase Console."
                )
            elif response.status_code == 403:
                error_msg = (
                    "Permission denied. Check your Firebase Storage Security Rules "
                    "or ensure your account has access to the storage bucket."
                )

            return False, {"error": error_msg}

    except requests.exceptions.ConnectionError:
        return False, {"error": "Cannot connect to Firebase Storage. Check your internet."}
    except requests.exceptions.Timeout:
        return False, {"error": "Upload timed out. The file may be too large. Try again."}
    except Exception as e:
        return False, {"error": str(e)}


# =============================================================================
# FILE DOWNLOAD
# =============================================================================

def download_file(
    storage_path: str,
    id_token: str,
) -> Tuple[bool, bytes]:
    """
    Downloads a file from Firebase Cloud Storage.

    Parameters:
    -----------
    storage_path : str — path inside the bucket (e.g. "users/uid/datasets/data.csv")
    id_token     : str — Firebase auth token

    Returns:
    --------
    (success: bool, result: bytes or dict)
        On success: result = raw file bytes
        On failure: result = b"" (empty bytes)
    """
    config = get_firebase_config()
    bucket = config.get("storage_bucket")
    if not bucket:
        return False, b""

    base_url = get_storage_base_url(bucket)
    encoded_path = requests.utils.quote(storage_path, safe="")
    url = f"{base_url}/{encoded_path}?alt=media"

    try:
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {id_token}"},
            timeout=60,
        )

        if response.status_code == 200:
            return True, response.content
        else:
            return False, b""

    except Exception:
        return False, b""


# =============================================================================
# FILE DELETE
# =============================================================================

def delete_file(
    storage_path: str,
    id_token: str,
) -> Tuple[bool, str]:
    """
    Deletes a file from Firebase Cloud Storage.

    Parameters:
    -----------
    storage_path : str — path inside the bucket
    id_token     : str — Firebase auth token

    Returns:
    --------
    (success: bool, message: str)
    """
    config = get_firebase_config()
    bucket = config.get("storage_bucket")
    if not bucket:
        return False, "Storage not configured."

    base_url = get_storage_base_url(bucket)
    encoded_path = requests.utils.quote(storage_path, safe="")
    url = f"{base_url}/{encoded_path}"

    try:
        response = requests.delete(
            url,
            headers={"Authorization": f"Bearer {id_token}"},
            timeout=15,
        )

        if response.status_code in (200, 204):
            return True, "File deleted."
        elif response.status_code == 404:
            return True, "File already deleted or not found."
        else:
            return False, f"Failed to delete (HTTP {response.status_code})."

    except Exception as e:
        return False, str(e)


# =============================================================================
# HELPER: BUILD USER-SPECIFIC STORAGE PATHS
# =============================================================================

def user_storage_path(uid: str, category: str, filename: str) -> str:
    """
    Builds a storage path scoped to a specific user.

    Parameters:
    -----------
    uid      : str — Firebase UID
    category : str — "datasets", "reports", "exports", or "models"
    filename : str — the file name (e.g. "sales_data.csv")

    Returns:
    --------
    str — e.g. "users/abc123/datasets/sales_data.csv"
    """
    # Sanitize filename — remove dangerous characters
    safe_name = "".join(c for c in filename if c.isalnum() or c in "._- ").strip()
    if not safe_name:
        safe_name = "untitled"
    return f"users/{uid}/{category}/{safe_name}"


# =============================================================================
# CONTENT TYPE MAPPING
# =============================================================================

def get_content_type(filename: str) -> str:
    """Returns the MIME content type for a file based on its extension."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return {
        "csv": "text/csv",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "xls": "application/vnd.ms-excel",
        "pdf": "application/pdf",
        "md": "text/markdown",
        "json": "application/json",
        "pkl": "application/octet-stream",
        "joblib": "application/octet-stream",
    }.get(ext, "application/octet-stream")
