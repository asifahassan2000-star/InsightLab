"""
firebase_db.py
==============
Firestore Database Engine for InsightLab.

What This Module Does:
----------------------
Reads and writes data to Google Cloud Firestore using the REST API.
Every saved item (project, dataset, report, model) is stored as a Firestore
"document" inside a "collection," organized by the authenticated user's UID.

What is Firestore?
------------------
Cloud Firestore is a NoSQL document database. Think of it like a filing cabinet:

    Collection (folder)   → Document (file)         → Fields (contents)
    ──────────────────────────────────────────────────────────────────
    users/                → uid_abc123/              → email, display_name, ...
    users/uid_abc123/
      projects/           → proj_001/               → name, created_at, ...
      datasets/           → ds_001/                 → name, rows, columns, ...
      reports/            → rpt_001/                → name, type, sections, ...
      analyses/           → ana_001/                → type, results_summary, ...
      models/             → mdl_001/                → task, algorithm, metrics, ...

Each document is a JSON-like object (key-value pairs).
Documents are automatically identified by a unique ID.

REST API vs SDK:
----------------
We use the Firestore REST API (plain HTTP requests) instead of the
`firebase-admin` or `google-cloud-firestore` SDK. Benefits:
  - No heavy dependencies or Python version issues.
  - Requests are authenticated with the user's idToken, so Firestore
    Security Rules are enforced (users can only access their own data).
  - Educational — you can see exactly what's happening.

Firestore Value Types:
----------------------
Firestore REST API uses a typed value format:
  Python str   → {"stringValue": "hello"}
  Python int   → {"integerValue": "42"}        ← note: string representation
  Python float → {"doubleValue": 3.14}
  Python bool  → {"booleanValue": true}
  Python None  → {"nullValue": null}
  Python list  → {"arrayValue": {"values": [...]}}
  Python dict  → {"mapValue": {"fields": {...}}}

We handle this conversion automatically with helper functions.
"""

import datetime
import uuid
from typing import Any, Dict, List, Optional, Tuple

import requests

from utils.firebase_config import get_firebase_config, get_firestore_base_url
from utils.firebase_admin_engine import verify_resource_ownership, get_admin_firestore_client


# =============================================================================
# SECTION 1: VALUE CONVERTERS
# =============================================================================

def python_to_firestore(value: Any) -> Dict[str, Any]:
    """
    Converts a Python value to Firestore REST API typed-value format.

    Examples:
        python_to_firestore("hello")  → {"stringValue": "hello"}
        python_to_firestore(42)       → {"integerValue": "42"}
        python_to_firestore(3.14)     → {"doubleValue": 3.14}
        python_to_firestore(True)     → {"booleanValue": True}
        python_to_firestore(None)     → {"nullValue": None}
        python_to_firestore([1, 2])   → {"arrayValue": {"values": [...]}}
        python_to_firestore({"a": 1}) → {"mapValue": {"fields": {...}}}
    """
    if value is None:
        return {"nullValue": None}
    if isinstance(value, bool):
        # bool check MUST come before int because bool is a subclass of int
        return {"booleanValue": value}
    if isinstance(value, int):
        return {"integerValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, str):
        return {"stringValue": value}
    if isinstance(value, datetime.datetime):
        return {"timestampValue": value.isoformat() + "Z"}
    if isinstance(value, list):
        if not value:
            return {"arrayValue": {"values": []}}
        return {"arrayValue": {"values": [python_to_firestore(v) for v in value]}}
    if isinstance(value, dict):
        return {"mapValue": {"fields": {
            str(k): python_to_firestore(v) for k, v in value.items()
        }}}
    # Fallback — convert to string
    return {"stringValue": str(value)}


def firestore_to_python(value: Dict[str, Any]) -> Any:
    """
    Converts a Firestore REST API typed-value back to a Python value.

    This is the reverse of python_to_firestore().
    """
    if "stringValue" in value:
        return value["stringValue"]
    if "integerValue" in value:
        return int(value["integerValue"])
    if "doubleValue" in value:
        return float(value["doubleValue"])
    if "booleanValue" in value:
        return value["booleanValue"]
    if "nullValue" in value:
        return None
    if "timestampValue" in value:
        return value["timestampValue"]
    if "arrayValue" in value:
        values = value["arrayValue"].get("values", [])
        return [firestore_to_python(v) for v in values]
    if "mapValue" in value:
        fields = value["mapValue"].get("fields", {})
        return {k: firestore_to_python(v) for k, v in fields.items()}
    return None


def parse_document(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parses a full Firestore document response into a clean Python dict.

    Firestore returns documents in this shape:
    {
        "name": "projects/myproject/databases/(default)/documents/users/abc",
        "fields": { "email": {"stringValue": "a@b.com"}, ... },
        "createTime": "...",
        "updateTime": "..."
    }

    We convert it to:
    {
        "_id": "abc",
        "_path": "users/abc",
        "email": "a@b.com",
        ...
    }
    """
    result = {}

    # Extract document ID from the full resource name
    name = doc.get("name", "")
    if name:
        result["_id"] = name.split("/")[-1]
        # Extract the collection/document path after "documents/"
        if "/documents/" in name:
            result["_path"] = name.split("/documents/")[-1]

    # Convert all fields
    fields = doc.get("fields", {})
    for key, typed_value in fields.items():
        result[key] = firestore_to_python(typed_value)

    result["_created"] = doc.get("createTime", "")
    result["_updated"] = doc.get("updateTime", "")

    return result


# =============================================================================
# SECTION 2: CORE CRUD OPERATIONS
# =============================================================================

def generate_doc_id() -> str:
    """
    Generates a unique document ID.

    We use UUID4 (universally unique identifier, version 4) which produces
    a random 128-bit string. The chance of collision is astronomically low
    (1 in 2^122). Firestore can also auto-generate IDs, but using our own
    gives us more control.
    """
    return uuid.uuid4().hex[:20]


def create_document(
    collection_path: str,
    data: Dict[str, Any],
    id_token: str,
    document_id: Optional[str] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Creates a new document in Firestore.

    Parameters:
    -----------
    collection_path : str — e.g. "users/uid123/projects"
    data            : dict — Python dict of fields to store
    id_token        : str — Firebase auth token
    document_id     : str or None — custom ID; auto-generated if None

    Returns:
    --------
    (success: bool, result: dict)
        On success: result = parsed document with _id
        On failure: result = {"error": "message"}
    """
    config = get_firebase_config()
    if not config["configured"]:
        return False, {"error": "Firebase not configured."}

    base_url = get_firestore_base_url(config["project_id"])
    doc_id = document_id or generate_doc_id()

    # Build the full URL: base/collection?documentId=xxx
    url = f"{base_url}/{collection_path}?documentId={doc_id}"

    # Convert Python dict to Firestore format
    fields = {key: python_to_firestore(val) for key, val in data.items()}

    try:
        response = requests.post(
            url,
            json={"fields": fields},
            headers={"Authorization": f"Bearer {id_token}"},
            timeout=15,
        )

        if response.status_code in (200, 201):
            return True, parse_document(response.json())
        else:
            error = response.json().get("error", {})
            return False, {"error": error.get("message", f"HTTP {response.status_code}")}

    except requests.exceptions.ConnectionError:
        return False, {"error": "Cannot connect to Firestore. Check your internet."}
    except Exception as e:
        return False, {"error": str(e)}


def get_document(
    document_path: str,
    id_token: str,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Reads a single document from Firestore.

    Parameters:
    -----------
    document_path : str — e.g. "users/uid123/projects/proj001"
    id_token      : str — Firebase auth token

    Returns:
    --------
    (success: bool, result: dict)
    """
    config = get_firebase_config()
    if not config["configured"]:
        return False, {"error": "Firebase not configured."}

    base_url = get_firestore_base_url(config["project_id"])
    url = f"{base_url}/{document_path}"

    try:
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {id_token}"},
            timeout=15,
        )

        if response.status_code == 200:
            return True, parse_document(response.json())
        elif response.status_code == 404:
            return False, {"error": "Document not found."}
        else:
            error = response.json().get("error", {})
            return False, {"error": error.get("message", f"HTTP {response.status_code}")}

    except Exception as e:
        return False, {"error": str(e)}


def list_documents(
    collection_path: str,
    id_token: str,
    page_size: int = 100,
    order_by: Optional[str] = None,
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Lists all documents in a Firestore collection.

    Parameters:
    -----------
    collection_path : str — e.g. "users/uid123/projects"
    id_token        : str — Firebase auth token
    page_size       : int — maximum number of documents to return
    order_by        : str or None — field name to order by (optional)

    Returns:
    --------
    (success: bool, result: list[dict])
        On success: result = list of parsed document dicts
        On failure: result = [] (empty list)
    """
    config = get_firebase_config()
    if not config["configured"]:
        return False, []

    base_url = get_firestore_base_url(config["project_id"])
    url = f"{base_url}/{collection_path}?pageSize={page_size}"

    if order_by:
        url += f"&orderBy={order_by}"

    try:
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {id_token}"},
            timeout=15,
        )

        if response.status_code == 200:
            data = response.json()
            documents = data.get("documents", [])
            return True, [parse_document(doc) for doc in documents]
        else:
            return False, []

    except Exception:
        return False, []


def update_document(
    document_path: str,
    data: Dict[str, Any],
    id_token: str,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Updates (patches) an existing Firestore document.

    Parameters:
    -----------
    document_path : str — e.g. "users/uid123/projects/proj001"
    data          : dict — fields to update (only these fields change)
    id_token      : str — Firebase auth token

    Returns:
    --------
    (success: bool, result: dict)

    What is PATCH?
    --------------
    Unlike PUT (which replaces the entire document), PATCH only updates
    the fields you specify. Existing fields that aren't in `data` are
    left untouched. This is safer and more efficient.
    """
    config = get_firebase_config()
    if not config["configured"]:
        return False, {"error": "Firebase not configured."}

    base_url = get_firestore_base_url(config["project_id"])

    # Build update mask (tells Firestore which fields to update)
    field_paths = "&".join(f"updateMask.fieldPaths={k}" for k in data.keys())
    url = f"{base_url}/{document_path}?{field_paths}"

    fields = {key: python_to_firestore(val) for key, val in data.items()}

    try:
        response = requests.patch(
            url,
            json={"fields": fields},
            headers={"Authorization": f"Bearer {id_token}"},
            timeout=15,
        )

        if response.status_code == 200:
            return True, parse_document(response.json())
        else:
            error = response.json().get("error", {})
            return False, {"error": error.get("message", f"HTTP {response.status_code}")}

    except Exception as e:
        return False, {"error": str(e)}


def delete_document(
    document_path: str,
    id_token: str,
) -> Tuple[bool, str]:
    """
    Deletes a document from Firestore.

    Parameters:
    -----------
    document_path : str — e.g. "users/uid123/projects/proj001"
    id_token      : str — Firebase auth token

    Returns:
    --------
    (success: bool, message: str)

    Note: Firestore does NOT cascade deletes. Deleting a document does
    NOT delete its subcollections. We handle subcollection cleanup manually.
    """
    config = get_firebase_config()
    if not config["configured"]:
        return False, "Firebase not configured."

    base_url = get_firestore_base_url(config["project_id"])
    url = f"{base_url}/{document_path}"

    try:
        response = requests.delete(
            url,
            headers={"Authorization": f"Bearer {id_token}"},
            timeout=15,
        )

        if response.status_code == 200:
            return True, "Deleted successfully."
        elif response.status_code == 404:
            return False, "Document not found."
        else:
            error = response.json().get("error", {})
            return False, error.get("message", f"HTTP {response.status_code}")

    except Exception as e:
        return False, str(e)


# =============================================================================
# SECTION 3: INSIGHTLAB-SPECIFIC DATABASE OPERATIONS
# =============================================================================

def _user_collection(uid: str, subcollection: str) -> str:
    """Builds the Firestore path for a user's subcollection."""
    return f"users/{uid}/{subcollection}"


def save_user_profile(uid: str, email: str, display_name: str, id_token: str) -> Tuple[bool, Dict]:
    """
    Creates or updates the user's profile document.

    Enforces server-side token ownership check before writing.
    Stored at: users/{uid}
    """
    # Verify caller owns this UID
    is_owner, err = verify_resource_ownership(id_token, uid)
    if not is_owner:
        return False, {"error": err}

    data = {
        "email": email,
        "display_name": display_name,
        "created_at": datetime.datetime.utcnow().isoformat(),
        "last_login": datetime.datetime.utcnow().isoformat(),
    }
    return create_document("users", data, id_token, document_id=uid)


def update_last_login(uid: str, id_token: str) -> None:
    """Updates the user's last_login timestamp with ownership check."""
    is_owner, _ = verify_resource_ownership(id_token, uid)
    if not is_owner:
        return
    update_document(f"users/{uid}", {
        "last_login": datetime.datetime.utcnow().isoformat(),
    }, id_token)


def save_project(uid: str, project_data: Dict[str, Any], id_token: str) -> Tuple[bool, Dict]:
    """
    Saves a complete project snapshot.
    Enforces server-side token ownership check before writing.
    Stored at: users/{uid}/projects/{auto_id}
    """
    is_owner, err = verify_resource_ownership(id_token, uid)
    if not is_owner:
        return False, {"error": err}

    project_data["saved_at"] = datetime.datetime.utcnow().isoformat()
    project_data["updated_at"] = datetime.datetime.utcnow().isoformat()
    return create_document(_user_collection(uid, "projects"), project_data, id_token)


def save_dataset_metadata(uid: str, metadata: Dict[str, Any], id_token: str) -> Tuple[bool, Dict]:
    """
    Saves dataset metadata with ownership check.
    Stored at: users/{uid}/datasets/{auto_id}
    """
    is_owner, err = verify_resource_ownership(id_token, uid)
    if not is_owner:
        return False, {"error": err}

    metadata["saved_at"] = datetime.datetime.utcnow().isoformat()
    return create_document(_user_collection(uid, "datasets"), metadata, id_token)


def save_report_metadata(uid: str, report_data: Dict[str, Any], id_token: str) -> Tuple[bool, Dict]:
    """
    Saves report generation metadata with ownership check.
    Stored at: users/{uid}/reports/{auto_id}
    """
    is_owner, err = verify_resource_ownership(id_token, uid)
    if not is_owner:
        return False, {"error": err}

    report_data["generated_at"] = datetime.datetime.utcnow().isoformat()
    return create_document(_user_collection(uid, "reports"), report_data, id_token)


def save_analysis(uid: str, analysis_data: Dict[str, Any], id_token: str) -> Tuple[bool, Dict]:
    """
    Saves analysis results with ownership check.
    Stored at: users/{uid}/analyses/{auto_id}
    """
    is_owner, err = verify_resource_ownership(id_token, uid)
    if not is_owner:
        return False, {"error": err}

    analysis_data["saved_at"] = datetime.datetime.utcnow().isoformat()
    return create_document(_user_collection(uid, "analyses"), analysis_data, id_token)


def save_model_info(uid: str, model_data: Dict[str, Any], id_token: str) -> Tuple[bool, Dict]:
    """
    Saves machine learning model configuration and results with ownership check.
    Stored at: users/{uid}/models/{auto_id}
    """
    is_owner, err = verify_resource_ownership(id_token, uid)
    if not is_owner:
        return False, {"error": err}

    model_data["saved_at"] = datetime.datetime.utcnow().isoformat()
    return create_document(_user_collection(uid, "models"), model_data, id_token)


def get_user_items(uid: str, subcollection: str, id_token: str) -> List[Dict]:
    """
    Retrieves all items from a user's subcollection with ownership check.
    """
    is_owner, _ = verify_resource_ownership(id_token, uid)
    if not is_owner:
        return []

    success, docs = list_documents(_user_collection(uid, subcollection), id_token)
    return docs if success else []


def delete_user_item(uid: str, subcollection: str, doc_id: str, id_token: str) -> Tuple[bool, str]:
    """Deletes a specific item from a user's subcollection with ownership check."""
    is_owner, err = verify_resource_ownership(id_token, uid)
    if not is_owner:
        return False, err

    path = f"users/{uid}/{subcollection}/{doc_id}"
    return delete_document(path, id_token)


def rename_user_item(
    uid: str, subcollection: str, doc_id: str,
    new_name: str, id_token: str
) -> Tuple[bool, Dict]:
    """Renames a saved item by updating its 'name' field with ownership check."""
    is_owner, err = verify_resource_ownership(id_token, uid)
    if not is_owner:
        return False, {"error": err}

    path = f"users/{uid}/{subcollection}/{doc_id}"
    return update_document(path, {"name": new_name, "updated_at": datetime.datetime.utcnow().isoformat()}, id_token)
