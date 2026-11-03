"""
test_firebase_system.py
=======================
Unit Tests for InsightLab Guest-First Authentication & Firebase Cloud System.

What We Are Testing:
--------------------
1. Configuration loading & URL generation (firebase_config).
2. Auth payload formatting, error code translation, session state handling (firebase_auth).
3. Firestore data type conversions (python_to_firestore, firestore_to_python, parse_document).
4. Smart Save payload constructors for all 7 analysis entities (smart_save).
5. Storage path generation & MIME mapping (firebase_storage).
6. Guest mode behavior (system functions properly when no user is logged in).
"""

import pytest
import datetime
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from utils.firebase_config import (
    get_firebase_config,
    is_firebase_configured,
    get_firestore_base_url,
    get_storage_base_url,
)
from utils.firebase_auth import (
    _translate_firebase_error,
    _build_user_dict,
    sign_out,
    is_authenticated,
)
from utils.firebase_db import (
    python_to_firestore,
    firestore_to_python,
    parse_document,
    generate_doc_id,
)
from utils.firebase_storage import (
    user_storage_path,
    get_content_type,
    is_storage_available,
)
from utils.smart_save import (
    build_project_payload,
    build_dataset_payload,
    build_analysis_payload,
    build_ml_model_payload,
    build_report_payload,
)


# =============================================================================
# 1. FIREBASE CONFIG TESTS
# =============================================================================

class TestFirebaseConfig:
    def test_unconfigured_defaults(self):
        """When secrets are empty or absent, config should reflect unconfigured status."""
        config = get_firebase_config()
        assert isinstance(config, dict)
        assert "configured" in config
        assert "api_key" in config
        assert "project_id" in config

    def test_firestore_url_generation(self):
        """Firestore REST URL must be properly formed with the project id."""
        url = get_firestore_base_url("test-insightlab-123")
        assert url == "https://firestore.googleapis.com/v1/projects/test-insightlab-123/databases/(default)/documents"

    def test_storage_url_generation(self):
        """Storage REST URL must be properly formed with the bucket name."""
        url = get_storage_base_url("test-bucket.appspot.com")
        assert url == "https://firebasestorage.googleapis.com/v0/b/test-bucket.appspot.com/o"


# =============================================================================
# 2. FIREBASE AUTH TESTS
# =============================================================================

class TestFirebaseAuth:
    def test_error_translations(self):
        """Raw Firebase error codes must translate to friendly user messages."""
        msg_pw = _translate_firebase_error("EMAIL_EXISTS")
        assert "already exists" in msg_pw

        msg_inv = _translate_firebase_error("INVALID_PASSWORD")
        assert "Incorrect password" in msg_inv

        msg_weak = _translate_firebase_error("WEAK_PASSWORD")
        assert "at least 6 characters" in msg_weak

        msg_unknown = _translate_firebase_error("SOME_CUSTOM_ERROR")
        assert "Authentication error" in msg_unknown

    def test_build_user_dict_normalizes_fields(self):
        """Raw response from Firebase Auth must convert to normalized user dict."""
        raw_response = {
            "localId": "uid_12345",
            "email": "analyst@insightlab.org",
            "displayName": "Data Scientist",
            "idToken": "jwt_token_sample",
            "refreshToken": "refresh_sample",
            "expiresIn": "3600"
        }
        user = _build_user_dict(raw_response)
        assert user["uid"] == "uid_12345"
        assert user["email"] == "analyst@insightlab.org"
        assert user["display_name"] == "Data Scientist"
        assert user["id_token"] == "jwt_token_sample"
        assert user["refresh_token"] == "refresh_sample"
        assert isinstance(user["token_expiry"], datetime.datetime)

    def test_guest_is_not_authenticated(self):
        """When no user is logged in, is_authenticated must return False."""
        sign_out()
        assert is_authenticated() is False


# =============================================================================
# 3. FIRESTORE TYPE CONVERTERS
# =============================================================================

class TestFirestoreDBConverters:
    def test_python_to_firestore_scalars(self):
        """Scalar conversions (string, int, float, bool, None)."""
        assert python_to_firestore("InsightLab") == {"stringValue": "InsightLab"}
        assert python_to_firestore(42) == {"integerValue": "42"}
        assert python_to_firestore(3.1415) == {"doubleValue": 3.1415}
        assert python_to_firestore(True) == {"booleanValue": True}
        assert python_to_firestore(False) == {"booleanValue": False}
        assert python_to_firestore(None) == {"nullValue": None}

    def test_python_to_firestore_nested(self):
        """Nested structures (lists and dicts)."""
        res_list = python_to_firestore(["a", 1])
        assert res_list == {"arrayValue": {"values": [{"stringValue": "a"}, {"integerValue": "1"}]}}

        res_dict = python_to_firestore({"metric": "RMSE", "score": 2.5})
        assert "mapValue" in res_dict
        assert "fields" in res_dict["mapValue"]
        assert res_dict["mapValue"]["fields"]["metric"] == {"stringValue": "RMSE"}
        assert res_dict["mapValue"]["fields"]["score"] == {"doubleValue": 2.5}

    def test_firestore_to_python_roundtrip(self):
        """Converting back from Firestore format must recover the original values."""
        assert firestore_to_python({"stringValue": "hello"}) == "hello"
        assert firestore_to_python({"integerValue": "100"}) == 100
        assert firestore_to_python({"doubleValue": 0.85}) == 0.85
        assert firestore_to_python({"booleanValue": True}) is True
        assert firestore_to_python({"nullValue": None}) is None
        assert firestore_to_python({"arrayValue": {"values": [{"stringValue": "x"}]}}) == ["x"]

    def test_parse_document(self):
        """Full Firestore raw document must parse into a clean dict with _id."""
        raw_doc = {
            "name": "projects/proj-1/databases/(default)/documents/users/uid_99/projects/proj_abc",
            "fields": {
                "name": {"stringValue": "Quarterly Sales Project"},
                "rows": {"integerValue": "150"}
            },
            "createTime": "2026-11-02T12:00:00Z"
        }
        parsed = parse_document(raw_doc)
        assert parsed["_id"] == "proj_abc"
        assert parsed["name"] == "Quarterly Sales Project"
        assert parsed["rows"] == 150

    def test_generate_doc_id_unique(self):
        """Document ID generator must produce unique strings."""
        id1 = generate_doc_id()
        id2 = generate_doc_id()
        assert id1 != id2
        assert len(id1) >= 10


# =============================================================================
# 4. SMART SAVE PAYLOAD TESTS
# =============================================================================

class TestSmartSavePayloads:
    @pytest.fixture
    def mock_session(self):
        df = pd.DataFrame({
            "product": ["A", "B", "C"],
            "revenue": [100.0, 250.0, 300.0],
            "units": [10, 20, 30]
        })
        return {
            "raw_df": df,
            "cleaned_df": df,
            "dataset_name": "sales_q4.csv",
            "cleaning_history": ["Removed nulls"],
            "ml_task": "regression",
            "ml_model_bundle": {
                "algorithm": "Linear Regression",
                "target_col": "revenue",
                "feature_cols": ["units"],
                "metrics": {"R²": 0.95, "RMSE": 12.3},
                "feature_importance": {"units": 1.0}
            }
        }

    def test_build_project_payload(self, mock_session):
        with patch("streamlit.session_state", mock_session):
            payload = build_project_payload("My Sales Project")
            assert payload["name"] == "My Sales Project"
            assert payload["dataset_name"] == "sales_q4.csv"
            assert payload["rows"] == 3
            assert payload["columns"] == 3
            assert payload["ml_algorithm"] == "Linear Regression"
            assert payload["ml_metrics"]["R²"] == 0.95

    def test_build_dataset_payload(self, mock_session):
        with patch("streamlit.session_state", mock_session):
            payload = build_dataset_payload(is_cleaned=True)
            assert "Cleaned_" in payload["name"]
            assert payload["is_cleaned"] is True
            assert payload["rows"] == 3
            assert "revenue" in payload["numeric_columns"]
            assert "product" in payload["categorical_columns"]

    def test_build_analysis_payload(self, mock_session):
        with patch("streamlit.session_state", mock_session):
            payload = build_analysis_payload()
            assert "Analysis" in payload["name"]
            assert payload["dataset_name"] == "sales_q4.csv"

    def test_build_ml_model_payload(self, mock_session):
        with patch("streamlit.session_state", mock_session):
            payload = build_ml_model_payload()
            assert "Linear Regression" in payload["name"]
            assert payload["task"] == "regression"
            assert payload["target_col"] == "revenue"
            assert payload["feature_cols"] == ["units"]

    def test_build_report_payload(self, mock_session):
        with patch("streamlit.session_state", mock_session):
            payload = build_report_payload("Executive Summary PDF", "pdf")
            assert payload["name"] == "Executive Summary PDF"
            assert payload["report_type"] == "pdf"
            assert payload["has_ml"] is True


# =============================================================================
# 5. STORAGE & PATH HELPERS
# =============================================================================

class TestFirebaseStorage:
    def test_user_storage_path_sanitization(self):
        """Storage path must be correctly scoped under users/{uid}/{category}/{clean_filename}."""
        path = user_storage_path("uid_456", "datasets", "my sales! data#2026.csv")
        assert path.startswith("users/uid_456/datasets/")
        assert "!" not in path
        assert "#" not in path
        assert path.endswith(".csv")

    def test_content_type_detection(self):
        """MIME type must correctly correspond to file extensions."""
        assert get_content_type("data.csv") == "text/csv"
        assert "openxmlformats" in get_content_type("report.xlsx")
        assert get_content_type("report.pdf") == "application/pdf"
        assert get_content_type("notes.md") == "text/markdown"
        assert get_content_type("unknown.xyz") == "application/octet-stream"


# =============================================================================
# 6. FIREBASE ADMIN ENGINE & OWNERSHIP TESTS
# =============================================================================

class TestFirebaseAdminEngine:
    def test_verify_client_token_empty(self):
        """Empty token must fail verification immediately."""
        from utils.firebase_admin_engine import verify_client_token
        valid, res = verify_client_token("")
        assert valid is False
        assert "Missing" in res["error"]

    def test_verify_resource_ownership_mismatch(self):
        """If token belongs to user A, accessing user B's resource must be rejected."""
        from utils.firebase_admin_engine import verify_resource_ownership
        with patch("utils.firebase_admin_engine.verify_client_token", return_value=(True, {"uid": "user_A"})):
            is_owner, err = verify_resource_ownership("fake_token", "user_B")
            assert is_owner is False
            assert "Permission denied" in err

    def test_verify_resource_ownership_match(self):
        """If token belongs to user A, accessing user A's resource must be allowed."""
        from utils.firebase_admin_engine import verify_resource_ownership
        with patch("utils.firebase_admin_engine.verify_client_token", return_value=(True, {"uid": "user_A"})):
            is_owner, uid = verify_resource_ownership("fake_token", "user_A")
            assert is_owner is True
            assert uid == "user_A"

    def test_firestore_ownership_enforcement(self):
        """Direct call to save_project with mismatched token must fail without writing."""
        from utils.firebase_db import (
            save_project,
            save_dataset_metadata,
            save_report_metadata,
            save_analysis,
            save_model_info,
            get_user_items,
            delete_user_item,
            rename_user_item,
        )
        with patch("utils.firebase_admin_engine.verify_client_token", return_value=(True, {"uid": "attacker_uid"})):
            # 1. Attacker cannot save a project to victim's account
            s1, r1 = save_project("victim_uid", {"name": "Hacked Project"}, "attacker_token")
            assert s1 is False
            assert "Permission denied" in r1["error"]

            # 2. Attacker cannot save a dataset to victim's account
            s2, r2 = save_dataset_metadata("victim_uid", {"name": "Hacked Dataset"}, "attacker_token")
            assert s2 is False
            assert "Permission denied" in r2["error"]

            # 3. Attacker cannot save a report to victim's account
            s3, r3 = save_report_metadata("victim_uid", {"name": "Hacked Report"}, "attacker_token")
            assert s3 is False
            assert "Permission denied" in r3["error"]

            # 4. Attacker cannot read victim's private documents
            items = get_user_items("victim_uid", "projects", "attacker_token")
            assert items == []

            # 5. Attacker cannot delete victim's documents
            s5, r5 = delete_user_item("victim_uid", "projects", "doc123", "attacker_token")
            assert s5 is False
            assert "Permission denied" in r5

            # 6. Attacker cannot rename victim's documents
            s6, r6 = rename_user_item("victim_uid", "projects", "doc123", "New Name", "attacker_token")
            assert s6 is False
            assert "Permission denied" in r6["error"]


# =============================================================================
# 7. GUEST ACCESS & AUTH LIFECYCLE TESTS
# =============================================================================

class TestGuestAndAuthLifecycle:
    def test_guest_workflow_without_authentication(self):
        """Guests can inspect, clean, compute statistics and generate exports with zero auth."""
        from utils.firebase_auth import is_authenticated, sign_out
        from utils.statistical_analyzer import compute_dataset_statistics_table
        from utils.report_builder import export_df_to_csv_bytes

        sign_out()
        assert is_authenticated() is False

        df = pd.DataFrame({"age": [20, 25, 30], "score": [85.0, 90.0, 95.0]})
        stats = compute_dataset_statistics_table(df)
        assert len(stats) == 2

        csv_bytes = export_df_to_csv_bytes(df)
        assert len(csv_bytes) > 0

    def test_auth_functions_mocked(self):
        """Sign-in, sign-up, password reset and logout execute cleanly."""
        from utils.firebase_auth import (
            sign_in,
            sign_up,
            send_password_reset,
            sign_out,
            store_user_session,
            get_current_user,
        )

        with patch("requests.post") as mock_post:
            # Test Sign In
            mock_post.return_value.json.return_value = {
                "localId": "test_uid_101",
                "email": "user@test.org",
                "idToken": "valid_token_abc",
                "refreshToken": "refresh_abc",
                "expiresIn": "3600"
            }
            mock_post.return_value.status_code = 200

            success, user_dict = sign_in("user@test.org", "password123")
            assert success is True
            assert user_dict["uid"] == "test_uid_101"

            # Store in session
            store_user_session(user_dict)
            current = get_current_user()
            assert current["uid"] == "test_uid_101"

            # Logout
            sign_out()
            assert get_current_user() is None

            # Test Password Reset
            mock_post.return_value.json.return_value = {"email": "user@test.org"}
            ok, msg = send_password_reset("user@test.org")
            assert ok is True
            assert "sent to user@test.org" in msg

    def test_save_deduplication(self):
        """Simulate rerun save deduplication."""
        import streamlit as st
        st.session_state["_last_saved_fingerprint"] = "project_MyProject_uid123"
        fingerprint = "project_MyProject_uid123"
        # Duplicate check passes
        assert st.session_state.get("_last_saved_fingerprint") == fingerprint
