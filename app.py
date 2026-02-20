"""Flask application entry point for the CureHelp+ medical assistant."""
from __future__ import annotations

import os
import logging
import threading
import time
import sys
import json
import secrets
from io import BytesIO
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, List
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import joblib
import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, send_file, send_from_directory, session, url_for
from PIL import Image

from admin import admin_bp
from auth_manager import auth_manager
from chatbot import get_chatbot_response
from consultant import get_consultant_directory, search_providers
from helper import fetch_gemini_recommendations
from makepdf import generate_pdf_report
from profile_manager import profile_manager
from report_parser import REPORT_ALLOWED_EXTENSIONS, parse_medical_report
from user_data_store import user_data_store

load_dotenv()

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SECRET_KEY"] = os.environ.get("CUREHELP_SECRET_KEY", "curehelp-secret-key")
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.permanent_session_lifetime = timedelta(hours=12)
app.register_blueprint(admin_bp)
logger = logging.getLogger(__name__)

MODEL_WARMUP_ENABLED = (os.getenv("MODEL_WARMUP_ENABLED", "true") or "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
MODEL_HEALTH_CHECK_INTERVAL_SECONDS = max(30, int(os.getenv("MODEL_HEALTH_CHECK_INTERVAL_SECONDS", "300") or "300"))
MODEL_HEALTH_STATUS: Dict[str, Any] = {
    "tabular_ready": False,
    "pneumonia_ready": False,
    "tb_ready": False,
    "last_check": None,
}
_MODEL_HEALTH_LOCK = threading.Lock()
_BACKGROUND_SERVICES_STARTED = False


def profile_latency(metric_name: str):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


def load_models() -> Dict[str, Any]:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "models")

    return {
        "diabetes_model": joblib.load(os.path.join(model_dir, "diabetes_model.pkl")),
        "diabetes_scaler": joblib.load(os.path.join(model_dir, "diabetes_scaler.pkl")),
        "heart_model": joblib.load(os.path.join(model_dir, "heart_model.pkl")),
        "heart_scaler": joblib.load(os.path.join(model_dir, "heart_scaler.pkl")),
        "anemia_risk_model": joblib.load(os.path.join(model_dir, "anemia_risk_model.pkl")),
        "anemia_type_model": joblib.load(os.path.join(model_dir, "anemia_type_model.pkl")),
        "anemia_scaler": joblib.load(os.path.join(model_dir, "feature_scaler.pkl")),
        "anemia_label_encoder": joblib.load(os.path.join(model_dir, "label_encoder.pkl")),
    }


MODELS: Dict[str, Any] = {}
_MODEL_LOAD_LOCK = threading.Lock()

PNEUMONIA_ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
PNEUMONIA_IMAGE_SIZE = (224, 224)
PNEUMONIA_THRESHOLD = 0.82
TB_ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
TB_IMAGE_SIZE = (224, 224)
TB_THRESHOLD = float(os.environ.get("TB_THRESHOLD", "0.50"))
MAX_XRAY_IMAGE_SIZE_BYTES = 10 * 1024 * 1024


def _load_pneumonia_artifacts():
    try:
        from tensorflow.keras.applications.resnet import preprocess_input
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing.image import img_to_array, load_img
    except Exception:
        return None, None, None, None

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "pneumonia_model.keras")
    if not os.path.isfile(model_path):
        return None, None, None, None

    try:
        model = load_model(model_path)
    except Exception:
        return None, None, None, None

    return model, preprocess_input, load_img, img_to_array


PNEUMONIA_MODEL = None
PNEUMONIA_PREPROCESS_INPUT = None
PNEUMONIA_LOAD_IMG = None
PNEUMONIA_IMG_TO_ARRAY = None
_PNEUMONIA_ARTIFACTS_LOADED = False


def _load_tuberculosis_artifacts():
    try:
        import torch
        import torch.nn as nn
        from torchvision import models
    except Exception:
        return None, None

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "tb_model.pth")
    if not os.path.isfile(model_path):
        return None, None

    try:
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 1)
        try:
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        model.eval()
    except Exception:
        return None, None

    return model, torch


TB_MODEL = None
TB_TORCH = None
_TB_ARTIFACTS_LOADED = False


def _update_model_health_status() -> None:
    tabular_ready = False
    pneumonia_ready = False
    tb_ready = False

    try:
        _get_models()
        tabular_ready = True
    except Exception:
        tabular_ready = False

    try:
        pneumonia_ready = _ensure_pneumonia_artifacts()
    except Exception:
        pneumonia_ready = False

    try:
        tb_ready = _ensure_tb_artifacts()
    except Exception:
        tb_ready = False

    with _MODEL_HEALTH_LOCK:
        MODEL_HEALTH_STATUS.update(
            {
                "tabular_ready": tabular_ready,
                "pneumonia_ready": pneumonia_ready,
                "tb_ready": tb_ready,
                "last_check": datetime.now().isoformat(timespec="seconds"),
            }
        )


def _startup_warmup() -> None:
    if not MODEL_WARMUP_ENABLED:
        return
    try:
        _update_model_health_status()
    except Exception as exc:
        logger.warning("Startup model warmup failed: %s", exc)


def _periodic_model_health_loop() -> None:
    while True:
        try:
            _update_model_health_status()
        except Exception as exc:
            logger.warning("Periodic model health check failed: %s", exc)
        time.sleep(MODEL_HEALTH_CHECK_INTERVAL_SECONDS)


def _start_background_services() -> None:
    global _BACKGROUND_SERVICES_STARTED
    if _BACKGROUND_SERVICES_STARTED:
        return
    if os.getenv("PYTEST_CURRENT_TEST") or "pytest" in sys.modules:
        return

    _BACKGROUND_SERVICES_STARTED = True

    warmup_thread = threading.Thread(target=_startup_warmup, name="model-warmup", daemon=True)
    warmup_thread.start()

    health_thread = threading.Thread(target=_periodic_model_health_loop, name="model-health", daemon=True)
    health_thread.start()


def _get_models() -> Dict[str, Any]:
    if MODELS:
        return MODELS

    with _MODEL_LOAD_LOCK:
        if not MODELS:
            MODELS.update(load_models())
    return MODELS


def _ensure_pneumonia_artifacts() -> bool:
    global PNEUMONIA_MODEL, PNEUMONIA_PREPROCESS_INPUT, PNEUMONIA_LOAD_IMG, PNEUMONIA_IMG_TO_ARRAY, _PNEUMONIA_ARTIFACTS_LOADED

    if PNEUMONIA_MODEL is not None and PNEUMONIA_PREPROCESS_INPUT is not None:
        return True

    if _PNEUMONIA_ARTIFACTS_LOADED:
        return PNEUMONIA_MODEL is not None and PNEUMONIA_PREPROCESS_INPUT is not None

    with _MODEL_LOAD_LOCK:
        if not _PNEUMONIA_ARTIFACTS_LOADED:
            (
                PNEUMONIA_MODEL,
                PNEUMONIA_PREPROCESS_INPUT,
                PNEUMONIA_LOAD_IMG,
                PNEUMONIA_IMG_TO_ARRAY,
            ) = _load_pneumonia_artifacts()
            _PNEUMONIA_ARTIFACTS_LOADED = True

    return PNEUMONIA_MODEL is not None and PNEUMONIA_PREPROCESS_INPUT is not None


def _ensure_tb_artifacts() -> bool:
    global TB_MODEL, TB_TORCH, _TB_ARTIFACTS_LOADED

    if TB_MODEL is not None and TB_TORCH is not None:
        return True

    if _TB_ARTIFACTS_LOADED:
        return TB_MODEL is not None and TB_TORCH is not None

    with _MODEL_LOAD_LOCK:
        if not _TB_ARTIFACTS_LOADED:
            TB_MODEL, TB_TORCH = _load_tuberculosis_artifacts()
            _TB_ARTIFACTS_LOADED = True

    return TB_MODEL is not None and TB_TORCH is not None


def _crop_lung_region(image_array: np.ndarray) -> np.ndarray:
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        return image_array

    gray = image_array.mean(axis=2)
    threshold = np.percentile(gray, 35)
    mask = gray > threshold

    if not np.any(mask):
        return image_array

    coords = np.argwhere(mask)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0) + 1

    height, width = image_array.shape[:2]
    y_pad = max(2, int((y_max - y_min) * 0.08))
    x_pad = max(2, int((x_max - x_min) * 0.08))

    y_min = max(0, y_min - y_pad)
    x_min = max(0, x_min - x_pad)
    y_max = min(height, y_max + y_pad)
    x_max = min(width, x_max + x_pad)

    cropped = image_array[y_min:y_max, x_min:x_max]
    if cropped.size == 0:
        return image_array
    return cropped


def _tb_preprocess_image(image_bytes: bytes) -> np.ndarray:
    image_stream = BytesIO(image_bytes)
    image = Image.open(image_stream).convert("RGB")

    image_array = np.asarray(image, dtype=np.uint8)
    cropped = _crop_lung_region(image_array)
    cropped_image = Image.fromarray(cropped).resize(TB_IMAGE_SIZE)

    array = np.asarray(cropped_image, dtype=np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    return np.expand_dims(array, axis=0)


def _tb_confidence_category(probability: float) -> str:
    if probability < 0.20:
        return "Very Low Risk"
    if probability < 0.40:
        return "Low Risk"
    if probability < 0.60:
        return "Borderline"
    if probability < 0.80:
        return "High Risk"
    return "Very High Risk"

MAX_REPORT_SIZE_BYTES = 200 * 1024 * 1024
MAX_PROFILE_IMAGE_SIZE_BYTES = 5 * 1024 * 1024
PROFILE_IMAGE_ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
UPLOADS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
PROFILE_UPLOAD_DIR = os.path.join(UPLOADS_ROOT, "profile_images")
REPORT_UPLOAD_DIR = os.path.join(UPLOADS_ROOT, "reports")
XRAY_HISTORY_UPLOAD_DIR = os.path.join(UPLOADS_ROOT, "xray_history")


def _ensure_upload_dirs() -> None:
    for directory in (UPLOADS_ROOT, PROFILE_UPLOAD_DIR, REPORT_UPLOAD_DIR, XRAY_HISTORY_UPLOAD_DIR):
        os.makedirs(directory, exist_ok=True)


def _build_user_upload_filename(user_id: str, source_name: str) -> str:
    extension = os.path.splitext(source_name or "")[1].lower().lstrip(".")
    safe_extension = extension or "bin"
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    return f"{user_id}_{timestamp}.{safe_extension}"


def _is_allowed_extension(filename: str, allowed_extensions: set[str]) -> bool:
    extension = os.path.splitext(filename or "")[1].lower().lstrip(".")
    return extension in allowed_extensions


def _file_size_bytes(file_storage) -> int:
    stream = file_storage.stream
    current_pos = stream.tell()
    stream.seek(0, os.SEEK_END)
    size = int(stream.tell())
    stream.seek(current_pos, os.SEEK_SET)
    return size


def _save_upload(file_storage, target_dir: str, filename: str) -> str:
    _ensure_upload_dirs()
    abs_path = os.path.join(target_dir, filename)
    file_storage.save(abs_path)
    return abs_path


def _save_upload_bytes(image_bytes: bytes, target_dir: str, filename: str) -> str:
    _ensure_upload_dirs()
    abs_path = os.path.join(target_dir, filename)
    with open(abs_path, "wb") as fh:
        fh.write(image_bytes)
    return abs_path


def _public_upload_path(subdir: str, filename: str) -> str:
    return f"/uploads/{subdir}/{filename}"


def _delete_upload_if_exists(public_path: str | None) -> None:
    if not public_path:
        return
    prefix = "/uploads/"
    if not public_path.startswith(prefix):
        return
    relative_path = public_path[len(prefix) :].replace("/", os.sep)
    abs_path = os.path.abspath(os.path.join(UPLOADS_ROOT, relative_path))
    uploads_root = os.path.abspath(UPLOADS_ROOT)
    if not abs_path.startswith(uploads_root):
        return
    if os.path.isfile(abs_path):
        try:
            os.remove(abs_path)
        except OSError:
            pass

DIABETES_NORMALS = {
    "Pregnancies": 3,
    "Glucose": 100,
    "Blood Pressure": 120,
    "Skin Thickness": 20,
    "Insulin": 80,
    "BMI": 22.0,
    "Diabetes Pedigree Function": 0.4,
    "Age": 40,
}

HEART_NORMALS = {
    "Age": 50,
    "Sex": 1,
    "Chest Pain Type": 4,
    "Resting BP": 120,
    "Cholesterol": 200,
    "Fasting BS > 120?": 0,
    "Resting ECG": 0,
    "Max Heart Rate": 150,
    "Exercise Angina": 0,
    "ST Depression": 0.0,
    "Slope of ST": 1,
    "Major Vessels (ca)": 0,
    "Thal": 3,
}

DIABETES_INPUT_LABELS = {
    "gender": "Gender",
    "age": "Age",
    "bmi": "BMI",
    "glucose": "Glucose",
    "blood_pressure": "Blood Pressure",
    "pregnancies": "Pregnancies",
    "skin_thickness": "Skin Thickness",
    "insulin": "Insulin",
    "diabetes_pedigree_function": "Diabetes Pedigree Function",
}

HEART_INPUT_LABELS = {
    "gender": "Sex",
    "age": "Age",
    "resting_bp": "Resting BP",
    "cholesterol": "Cholesterol",
    "chest_pain_type": "Chest Pain Type",
    "fasting_bs": "Fasting BS > 120?",
    "resting_ecg": "Resting ECG",
    "max_heart_rate": "Max Heart Rate",
    "exercise_angina": "Exercise Angina",
    "st_depression": "ST Depression",
    "slope": "Slope of ST",
    "major_vessels": "Major Vessels (ca)",
    "thal": "Thal",
}

TYPE2_DIABETES_LABEL = "Type-2 Diabetes"
LEGACY_DIABETES_LABEL = "Diabetes"


def _canonical_disease_label(disease: str) -> str:
    lowered = (disease or "").strip().lower()
    if lowered in {"diabetes", "type-2 diabetes", "type 2 diabetes", "type-2 diabetes mellitus", "type 2 diabetes mellitus"}:
        return TYPE2_DIABETES_LABEL
    return disease


def _normalise_prediction_labels(predictions: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for disease, payload in (predictions or {}).items():
        canonical = _canonical_disease_label(str(disease))
        normalized[canonical] = payload
    return normalized

ANEMIA_INPUT_LABELS = {
    "gender": "Gender",
    "rbc": "RBC",
    "hemoglobin": "Hemoglobin (Hb)",
    "hematocrit": "Hematocrit (HCT)",
    "mcv": "MCV",
    "mch": "MCH",
    "mchc": "MCHC",
    "wbc": "WBC",
    "platelets": "Platelets",
    "rdw": "RDW",
    "pdw": "PDW",
    "pct": "PCT",
    "lymphocytes": "Lymphocytes",
    "neutrophils_pct": "Neutrophils %",
    "neutrophils_num": "Neutrophils #",
}


def _convert_to_float(payload: Dict[str, Any], key: str) -> float:
    if key not in payload:
        raise ValueError(f"Missing field: {key}")
    try:
        return float(payload[key])
    except (TypeError, ValueError):
        raise ValueError(f"Invalid value for {key}")


def _map_display_inputs(payload: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
    return {friendly: payload.get(raw) for raw, friendly in mapping.items() if raw in payload}


def _current_predictions() -> Dict[str, Any]:
    return _normalise_prediction_labels(session.get("predictions", {}))


def _save_predictions(predictions: Dict[str, Any]) -> None:
    session["predictions"] = predictions
    session.modified = True


def _sync_predictions_to_profile() -> None:
    profile_id = session.get("current_profile_id")
    if profile_id:
        profile_manager.update_predictions(profile_id, _current_predictions())


def _sync_auth_profile_to_patients(user_id: str, profile_payload: Dict[str, Any] | None) -> None:
    if not user_id:
        return
    if not isinstance(profile_payload, dict):
        profile_payload = {}

    profile_id = f"auth_{user_id}"
    name = str(profile_payload.get("name") or "").strip()
    mobile = str(profile_payload.get("mobile") or profile_payload.get("contact") or "").strip()
    address = str(profile_payload.get("address") or "").strip()
    gender = str(profile_payload.get("gender") or "").strip()

    updates = {
        "name": name,
        "contact": mobile,
        "address": address,
        "gender": gender,
    }
    updates = {key: value for key, value in updates.items() if value}

    if not updates:
        return

    existing = profile_manager.get_profile(profile_id)
    if existing:
        profile_manager.update_profile(profile_id, updates)
    else:
        profile_manager.add_profile(
            {
                "id": profile_id,
                "name": name,
                "contact": mobile,
                "address": address,
                "gender": gender,
                "marital_status": "",
                "age": profile_payload.get("age"),
            }
        )


def _store_prediction(
    disease: str,
    payload: Dict[str, Any],
    history_inputs: Dict[str, Any] | None = None,
    history_image_path: str | None = None,
) -> None:
    predictions = _current_predictions().copy()
    predictions[_canonical_disease_label(disease)] = payload
    _save_predictions(predictions)
    _sync_predictions_to_profile()
    user_id = session.get("auth_user_id")
    if user_id:
        disease_key = _canonical_disease_label(disease).strip().lower().replace(" ", "_")
        user_data_store.append_prediction(
            str(user_id),
            {
                "disease": _canonical_disease_label(disease),
                "payload": payload,
                "at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            },
        )
        user_data_store.append_health_history(
            str(user_id),
            disease_type=disease_key,
            input_data=history_inputs if isinstance(history_inputs, dict) else payload.get("inputs", {}),
            image_path=history_image_path,
        )


def _anemia_normals(gender: str) -> Dict[str, float]:
    male = gender.lower() == "male"
    return {
        "Hemoglobin (Hb)": 13.5 if male else 12.0,
        "RBC": 5.0 if male else 4.5,
        "Hematocrit (HCT)": 41.0 if male else 36.0,
        "MCV": 90.0,
        "MCH": 30.0,
        "MCHC": 34.0,
        "RDW": 14.0,
        "Platelets": 250.0,
        "WBC": 7.0,
        "PDW": 12.0,
        "PCT": 0.22,
        "Lymphocytes": 30.0,
        "Neutrophils %": 60.0,
        "Neutrophils #": 4.2,
    }


AUTH_EXEMPT_ENDPOINTS = {
    "index",
    "blog_page",
    "get_config",
    "verify_email",
    "get_auth_status",
    "login_auth",
    "signup_auth",
    "verify_otp_auth",
    "forgot_password_auth",
    "reset_password_auth",
    "resend_verification_auth",
    "static",
    "admin.login",
    "admin.logout",
    "admin.dashboard",
    "admin.delete_patient",
}


def _auth_available() -> bool:
    try:
        auth_manager.ensure_schema()
        return True
    except Exception as exc:
        logger.warning("Auth subsystem unavailable: %s", exc)
        return False


def _clear_auth_session() -> None:
    session.pop("auth_user_id", None)
    session.pop("auth_email", None)
    session.pop("auth_session_id", None)
    session.pop("current_profile_id", None)
    session.pop("current_profile_name", None)
    session.pop("current_profile_gender", None)
    session.pop("predictions", None)
    session.modified = True


def _set_auth_session(user: Dict[str, Any], session_row: Dict[str, Any]) -> None:
    session["auth_user_id"] = user.get("id")
    session["auth_email"] = user.get("email")
    session["auth_session_id"] = session_row.get("id")
    session.permanent = True
    session.modified = True


def _is_authenticated() -> bool:
    return bool(session.get("auth_user_id") and session.get("auth_session_id"))


def _require_auth(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not _is_authenticated():
            return jsonify({"success": False, "error": "Authentication required."}), 401
        return view(*args, **kwargs)

    return wrapped


@app.before_request
def _enforce_server_session_guard():
    endpoint = request.endpoint or ""
    if endpoint in AUTH_EXEMPT_ENDPOINTS or endpoint.startswith("admin."):
        return None
    if not _is_authenticated():
        return None
    if not _auth_available():
        _clear_auth_session()
        return jsonify({"success": False, "error": "Authentication service unavailable."}), 503

    auth_user_id = session.get("auth_user_id")
    auth_session_id = session.get("auth_session_id")
    try:
        valid = auth_manager.validate_session(str(auth_session_id), str(auth_user_id))
    except Exception as exc:
        logger.warning("Session validation failed: %s", exc)
        _clear_auth_session()
        return jsonify({"success": False, "error": "Session validation failed."}), 401

    if not valid:
        _clear_auth_session()
        return jsonify({"success": False, "error": "Session expired. Please log in again."}), 401
    return None


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/blog")
def blog_page() -> str:
    return render_template("blog.html")


@app.route("/uploads/<path:subpath>", methods=["GET"])
@_require_auth
def serve_upload(subpath: str):
    safe_subpath = os.path.normpath(subpath).replace("\\", "/")
    if safe_subpath.startswith("../") or safe_subpath == "..":
        return jsonify({"success": False, "error": "Invalid upload path."}), 400
    directory = os.path.dirname(safe_subpath)
    filename = os.path.basename(safe_subpath)
    return send_from_directory(os.path.join(UPLOADS_ROOT, directory), filename)


@app.route("/api/config", methods=["GET"])
def get_config():
    return jsonify(
        {
            "success": True,
            "normals": {
                "diabetes": DIABETES_NORMALS,
                "heart": HEART_NORMALS,
            },
        }
    )


def _google_client_id() -> str:
    return (os.getenv("GOOGLE_CLIENT_ID") or "").strip()


def _google_client_secret() -> str:
    return (os.getenv("GOOGLE_CLIENT_SECRET") or "").strip()


def _google_redirect_uri() -> str:
    configured_redirect = (os.getenv("GOOGLE_REDIRECT_URI") or "").strip()
    if configured_redirect:
        return configured_redirect

    configured_base_url = (os.getenv("APP_BASE_URL") or "").strip().rstrip("/")
    if configured_base_url:
        return f"{configured_base_url}/api/auth/google/callback"

    return url_for("google_auth_callback", _external=True)


def _google_is_configured() -> bool:
    return bool(_google_client_id() and _google_client_secret())


def _google_request_json(url: str, method: str = "GET", data: Dict[str, Any] | None = None) -> Dict[str, Any]:
    encoded_data = None
    headers = {}

    if data is not None:
        encoded_data = urlencode(data).encode("utf-8")
        headers["Content-Type"] = "application/x-www-form-urlencoded"

    request = Request(url=url, data=encoded_data, method=method, headers=headers)
    try:
        with urlopen(request, timeout=15) as response:
            payload = response.read().decode("utf-8")
            return json.loads(payload or "{}")
    except HTTPError as exc:
        try:
            message = exc.read().decode("utf-8")
        except Exception:
            message = str(exc)
        raise RuntimeError(f"Google request failed: {message}") from exc
    except URLError as exc:
        raise RuntimeError(f"Google request failed: {exc}") from exc


@app.route("/api/auth/google/start", methods=["GET"])
def google_auth_start():
    if not _google_is_configured():
        return redirect(url_for("index", google_auth="config_error"))

    state_token = secrets.token_urlsafe(24)
    redirect_uri = _google_redirect_uri()
    session["google_oauth_state"] = state_token
    session["google_oauth_redirect_uri"] = redirect_uri

    params = {
        "client_id": _google_client_id(),
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state_token,
        "prompt": "select_account",
    }
    auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"
    return redirect(auth_url)


@app.route("/api/auth/google/callback", methods=["GET"])
def google_auth_callback():
    if not _auth_available():
        return redirect(url_for("index", google_auth="service_unavailable"))

    if not _google_is_configured():
        return redirect(url_for("index", google_auth="config_error"))

    request_state = (request.args.get("state") or "").strip()
    expected_state = str(session.pop("google_oauth_state", "") or "")
    redirect_uri = str(session.pop("google_oauth_redirect_uri", "") or "").strip() or _google_redirect_uri()
    if not request_state or not expected_state or request_state != expected_state:
        return redirect(url_for("index", google_auth="state_error"))

    code = (request.args.get("code") or "").strip()
    if not code:
        return redirect(url_for("index", google_auth="access_denied"))

    try:
        token_payload = _google_request_json(
            "https://oauth2.googleapis.com/token",
            method="POST",
            data={
                "code": code,
                "client_id": _google_client_id(),
                "client_secret": _google_client_secret(),
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            },
        )

        id_token = str(token_payload.get("id_token") or "")
        if not id_token:
            return redirect(url_for("index", google_auth="token_error"))

        token_info = _google_request_json(
            f"https://oauth2.googleapis.com/tokeninfo?id_token={id_token}",
            method="GET",
        )

        if str(token_info.get("aud") or "") != _google_client_id():
            return redirect(url_for("index", google_auth="audience_error"))

        email = (token_info.get("email") or "").strip().lower()
        email_verified = str(token_info.get("email_verified") or "").lower() == "true"
        if not email or not email_verified:
            return redirect(url_for("index", google_auth="email_error"))

        auth_result = auth_manager.login_with_google_email(email)
        if not auth_result.success:
            return redirect(url_for("index", google_auth="signin_error"))

        user = auth_result.user or {}
        session_row = auth_manager.create_session(
            user_id=str(user.get("id")),
            user_agent=request.headers.get("User-Agent", ""),
            ip_address=request.headers.get("X-Forwarded-For", request.remote_addr or ""),
        )
        _set_auth_session(user, session_row)
    except Exception as exc:
        logger.exception("Google authentication failed")
        return redirect(url_for("index", google_auth="signin_error"))

    return redirect(url_for("index", google_auth="success"))


@app.route("/api/auth/status", methods=["GET"])
def get_auth_status():
    if not _is_authenticated():
        return jsonify({"success": True, "authenticated": False, "user": None})

    user_id = str(session.get("auth_user_id"))
    if not _auth_available():
        _clear_auth_session()
        return jsonify({"success": True, "authenticated": False, "user": None})

    try:
        user = auth_manager.get_user_by_id(user_id)
    except Exception:
        _clear_auth_session()
        return jsonify({"success": True, "authenticated": False, "user": None})

    if not user:
        _clear_auth_session()
        return jsonify({"success": True, "authenticated": False, "user": None})

    profile_payload = user_data_store.get_user_profile(user_id)
    if not isinstance(profile_payload, dict):
        profile_payload = {}

    db_full_name = str(user.get("full_name") or "").strip()
    if db_full_name:
        profile_payload = {**profile_payload, "name": db_full_name}

    _sync_auth_profile_to_patients(user_id, profile_payload)

    return jsonify(
        {
            "success": True,
            "authenticated": True,
            "user": {
                "id": user.get("id"),
                "email": user.get("email"),
                "full_name": db_full_name,
                "is_verified": user.get("is_verified", False),
                "is_active": user.get("is_active", False),
                "profile": profile_payload,
            },
        }
    )


@app.route("/api/auth/signup", methods=["POST"])
def signup_auth():
    payload = request.get_json(force=True, silent=True) or {}
    email = (payload.get("email") or "").strip().lower()
    password = payload.get("password") or ""
    name = (payload.get("name") or "").strip()

    if not _auth_available():
        return jsonify({"success": False, "error": "Authentication service is unavailable."}), 503

    try:
        result = auth_manager.signup(email=email, password=password, full_name=name)
    except Exception as exc:
        logger.exception("Signup failed")
        return jsonify({"success": False, "error": str(exc)}), 500

    if not result.success:
        return jsonify({"success": False, "error": result.error}), 400

    return jsonify(
        {
            "success": True,
            "message": "Resent OTP",
        }
    )


@app.route("/api/auth/verify-otp", methods=["POST"])
def verify_otp_auth():
    payload = request.get_json(force=True, silent=True) or {}
    email = (payload.get("email") or "").strip().lower()
    otp = (payload.get("otp") or "").strip().upper()

    if not _auth_available():
        return jsonify({"success": False, "error": "Authentication service is unavailable."}), 503

    try:
        result = auth_manager.verify_email_otp(email=email, otp=otp)
    except Exception as exc:
        logger.exception("OTP verification failed")
        return jsonify({"success": False, "error": str(exc)}), 500

    if not result.success:
        return jsonify({"success": False, "error": result.error}), 400

    return jsonify({"success": True, "message": "OTP verified successfully. Please login."})


@app.route("/api/auth/login", methods=["POST"])
def login_auth():
    payload = request.get_json(force=True, silent=True) or {}
    email = (payload.get("email") or "").strip().lower()
    password = payload.get("password") or ""

    if not _auth_available():
        return jsonify({"success": False, "error": "Authentication service is unavailable."}), 503

    try:
        result = auth_manager.login(email=email, password=password)
    except Exception as exc:
        logger.exception("Login failed")
        return jsonify({"success": False, "error": str(exc)}), 500

    if not result.success:
        return jsonify({"success": False, "error": result.error}), 401

    user = result.user or {}
    session_row = auth_manager.create_session(
        user_id=str(user.get("id")),
        user_agent=request.headers.get("User-Agent", ""),
        ip_address=request.headers.get("X-Forwarded-For", request.remote_addr or ""),
    )
    _set_auth_session(user, session_row)

    return jsonify(
        {
            "success": True,
            "message": "Login successful.",
            "user": {
                "id": user.get("id"),
                "email": user.get("email"),
                "is_verified": user.get("is_verified", False),
            },
        }
    )


@app.route("/verify-email", methods=["GET"])
def verify_email():
    token = (request.args.get("token") or "").strip()
    if not token:
        return redirect(url_for("index"))

    if not _auth_available():
        return redirect(url_for("index"))

    try:
        result = auth_manager.verify_email(token)
        if result.success:
            return redirect(url_for("index", verified="1"))
    except Exception:
        return redirect(url_for("index", verified="0"))
    return redirect(url_for("index", verified="0"))


@app.route("/api/auth/resend-verification", methods=["POST"])
def resend_verification_auth():
    payload = request.get_json(force=True, silent=True) or {}
    email = (payload.get("email") or "").strip().lower()

    if not _auth_available():
        return jsonify({"success": False, "error": "Authentication service is unavailable."}), 503

    try:
        user = auth_manager._get_user_by_email(email)
        if user and not user.get("is_verified", False):
            otp_result = auth_manager.send_verification_otp_email(user_id=user["id"], email=email)
            if not otp_result.success:
                return jsonify({"success": False, "error": otp_result.error or "Unable to send OTP right now."}), 503
    except Exception as exc:
        logger.exception("Resend OTP failed")
        return jsonify({"success": False, "error": str(exc)}), 500
    return jsonify({"success": True, "message": "Resent OTP"})


@app.route("/api/auth/forgot-password", methods=["POST"])
def forgot_password_auth():
    payload = request.get_json(force=True, silent=True) or {}
    email = (payload.get("email") or "").strip().lower()

    if not _auth_available():
        return jsonify({"success": False, "error": "Authentication service is unavailable."}), 503

    try:
        result = auth_manager.create_reset_token(email)
    except Exception as exc:
        logger.exception("Forgot password failed")
        return jsonify({"success": False, "error": str(exc)}), 500

    if not result.success:
        return jsonify({"success": False, "error": result.error}), 400
    return jsonify({"success": True, "message": "Temporary password has been sent to email."})


@app.route("/api/auth/reset-password", methods=["POST"])
def reset_password_auth():
    payload = request.get_json(force=True, silent=True) or {}
    token = (payload.get("token") or "").strip()
    new_password = payload.get("new_password") or ""

    if not _auth_available():
        return jsonify({"success": False, "error": "Authentication service is unavailable."}), 503

    try:
        result = auth_manager.reset_password(token, new_password)
    except Exception as exc:
        logger.exception("Reset password failed")
        return jsonify({"success": False, "error": str(exc)}), 500

    if not result.success:
        return jsonify({"success": False, "error": result.error}), 400
    return jsonify({"success": True, "message": "Password has been reset successfully."})


@app.route("/api/auth/logout", methods=["POST"])
@_require_auth
def logout_auth():
    user_id = str(session.get("auth_user_id"))
    session_id = str(session.get("auth_session_id"))
    if _auth_available():
        try:
            auth_manager.revoke_session(session_id=session_id, user_id=user_id)
        except Exception as exc:
            logger.warning("Failed to revoke session: %s", exc)
    _clear_auth_session()
    return jsonify({"success": True})


@app.route("/api/auth/logout-all", methods=["POST"])
@_require_auth
def logout_all_auth():
    user_id = str(session.get("auth_user_id"))
    if _auth_available():
        try:
            auth_manager.revoke_all_sessions(user_id=user_id)
        except Exception as exc:
            logger.warning("Failed to revoke all sessions: %s", exc)
    _clear_auth_session()
    return jsonify({"success": True})


@app.route("/api/auth/profile", methods=["GET"])
@_require_auth
def get_auth_profile():
    user_id = str(session.get("auth_user_id"))
    profile_payload = user_data_store.get_user_profile(user_id)
    _sync_auth_profile_to_patients(user_id, profile_payload)
    medical_payload = user_data_store.get_medical_data(user_id)
    return jsonify(
        {
            "success": True,
            "profile": profile_payload,
            "medical": medical_payload,
            "reports": user_data_store.list_reports(user_id),
            "predictions": user_data_store.list_predictions(user_id),
            "health_history": user_data_store.list_health_history(user_id),
        }
    )


@app.route("/api/auth/profile", methods=["PATCH"])
@_require_auth
def update_auth_profile():
    payload = request.get_json(force=True, silent=True) or {}
    allowed = {"name", "dob", "address", "mobile", "gender", "abha_id", "photo_path"}
    updates = {key: payload.get(key) for key in allowed if key in payload}

    user_id = str(session.get("auth_user_id"))

    if payload.get("current_password") and payload.get("new_password"):
        if not _auth_available():
            return jsonify({"success": False, "error": "Authentication service unavailable."}), 503
        password_result = auth_manager.update_password(
            user_id=user_id,
            current_password=str(payload.get("current_password")),
            new_password=str(payload.get("new_password")),
        )
        if not password_result.success:
            return jsonify({"success": False, "error": password_result.error}), 400
        _clear_auth_session()
        return jsonify({"success": True, "message": "Password updated. Please login again.", "require_relogin": True})

    profile_payload = user_data_store.set_user_profile(user_id, updates)
    _sync_auth_profile_to_patients(user_id, profile_payload)
    return jsonify({"success": True, "profile": profile_payload})


@app.route("/api/profile/upload-photo", methods=["POST"])
@_require_auth
def upload_profile_photo():
    if request.content_type is None or "multipart/form-data" not in request.content_type.lower():
        return jsonify({"success": False, "error": "Use multipart/form-data for image upload."}), 400

    image_file = request.files.get("photo")
    if image_file is None or not (image_file.filename or "").strip():
        return jsonify({"success": False, "error": "Profile image is required."}), 400

    filename = image_file.filename or ""
    if not _is_allowed_extension(filename, PROFILE_IMAGE_ALLOWED_EXTENSIONS):
        return jsonify({"success": False, "error": "Unsupported image type. Allowed: JPG, JPEG, PNG."}), 400

    try:
        size = _file_size_bytes(image_file)
        image_file.stream.seek(0)
    except OSError:
        size = 0

    if size > MAX_PROFILE_IMAGE_SIZE_BYTES:
        return jsonify({"success": False, "error": "Profile image size must be 5MB or smaller."}), 400

    user_id = str(session.get("auth_user_id"))
    profile_payload = user_data_store.get_user_profile(user_id)
    old_path = profile_payload.get("photo_path") if isinstance(profile_payload, dict) else None

    saved_name = _build_user_upload_filename(user_id, filename)
    _save_upload(image_file, PROFILE_UPLOAD_DIR, saved_name)
    public_path = _public_upload_path("profile_images", saved_name)

    user_data_store.set_user_profile(user_id, {"photo_path": public_path})
    _delete_upload_if_exists(old_path)

    return jsonify({"success": True, "photo_path": public_path})


@app.route("/api/auth/reports", methods=["POST"])
@_require_auth
def append_auth_report():
    if request.content_type is None or "multipart/form-data" not in request.content_type.lower():
        return jsonify({"success": False, "error": "Use multipart/form-data for report upload."}), 400

    report_file = request.files.get("file")
    if report_file is None or not (report_file.filename or "").strip():
        return jsonify({"success": False, "error": "Report file is required."}), 400

    filename = report_file.filename or ""
    extension = os.path.splitext(filename)[1].lower().lstrip(".")
    allowed_extensions = {ext.lower().lstrip(".") for ext in set(REPORT_ALLOWED_EXTENSIONS)} | PROFILE_IMAGE_ALLOWED_EXTENSIONS
    if extension not in allowed_extensions:
        return jsonify({"success": False, "error": "Unsupported file type."}), 400

    try:
        size = _file_size_bytes(report_file)
        report_file.stream.seek(0)
    except OSError:
        size = 0

    if size > MAX_REPORT_SIZE_BYTES:
        return jsonify({"success": False, "error": "Report exceeds the maximum size of 200 MB."}), 400

    disease_type = (request.form.get("disease_type") or "pneumonia").strip().lower()
    status = (request.form.get("status") or "Unknown").strip()
    report_type = (report_file.content_type or "application/octet-stream").strip()
    user_id = str(session.get("auth_user_id"))

    saved_name = _build_user_upload_filename(user_id, filename)
    _save_upload(report_file, REPORT_UPLOAD_DIR, saved_name)
    public_path = _public_upload_path("reports", saved_name)

    user_data_store.append_report(
        user_id,
        {
            "id": secrets.token_hex(8),
            "name": filename,
            "status": status,
            "type": report_type,
            "path": public_path,
            "at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        },
    )
    user_data_store.append_health_history(
        user_id,
        disease_type=disease_type,
        input_data={"status": status, "filename": filename, "type": report_type},
        image_path=public_path if extension in PROFILE_IMAGE_ALLOWED_EXTENSIONS else None,
    )
    return jsonify({"success": True, "file_path": public_path})


@app.route("/api/auth/reports/<report_id>", methods=["DELETE"])
@_require_auth
def delete_auth_report(report_id: str):
    user_id = str(session.get("auth_user_id"))
    removed = user_data_store.remove_report(user_id, report_id)
    if not removed:
        return jsonify({"success": False, "error": "Report not found."}), 404

    removed_path = str((removed or {}).get("path") or "").strip()
    if removed_path:
        _delete_upload_if_exists(removed_path)

    return jsonify({"success": True})


@app.route("/api/dashboard/overview", methods=["GET"])
@_require_auth
def dashboard_overview():
    def _parse_iso_timestamp(value: Any) -> datetime | None:
        text = str(value or "").strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return None

    user_id = str(session.get("auth_user_id"))
    profile_payload = user_data_store.get_user_profile(user_id)
    prediction_history = user_data_store.list_predictions(user_id)
    reports = user_data_store.list_reports(user_id)
    health_history = user_data_store.list_health_history(user_id)

    disease_counts: Dict[str, int] = {}
    progression = {"improved": 0, "deteriorated": 0}
    probability_samples: List[float] = []
    timeline_by_disease: Dict[str, List[Dict[str, Any]]] = {}

    sorted_history = sorted(
        prediction_history,
        key=lambda entry: (_parse_iso_timestamp(entry.get("at")) or datetime.min),
    )
    for entry in sorted_history:
        disease = str(entry.get("disease") or "Unknown")
        disease_counts[disease] = disease_counts.get(disease, 0) + 1
        payload = entry.get("payload") if isinstance(entry.get("payload"), dict) else {}
        probability = payload.get("prob", payload.get("probability"))
        try:
            score = float(probability)
        except (TypeError, ValueError):
            score = 0.0

        if score <= 1.0 and probability is not None:
            score = score * 100.0
        score = max(0.0, min(100.0, score))

        if score > 0:
            probability_samples.append(score)
        if score >= 50:
            progression["deteriorated"] += 1
        else:
            progression["improved"] += 1

        timestamp = _parse_iso_timestamp(entry.get("at")) or datetime.utcnow()
        timeline_by_disease.setdefault(disease, []).append(
            {
                "at": timestamp.isoformat(),
                "date": timestamp.strftime("%Y-%m-%d"),
                "value": round(score, 2),
            }
        )

    risk_alerts: List[Dict[str, Any]] = []
    for disease, points in timeline_by_disease.items():
        if len(points) < 2:
            continue
        previous = float(points[-2].get("value") or 0.0)
        latest = float(points[-1].get("value") or 0.0)
        delta = latest - previous
        if delta >= 10.0:
            risk_alerts.append(
                {
                    "disease": disease,
                    "previous": round(previous, 2),
                    "latest": round(latest, 2),
                    "delta": round(delta, 2),
                    "message": f"Latest risk increased by {delta:.1f}% compared to previous result.",
                }
            )

    profile_record = profile_payload if isinstance(profile_payload, dict) else {}
    missing_completeness: List[str] = []
    if not str(profile_record.get("name") or "").strip():
        missing_completeness.append("Name")
    if not str(profile_record.get("dob") or "").strip():
        missing_completeness.append("DOB")
    mobile_value = str(profile_record.get("mobile") or profile_record.get("contact") or "").strip()
    if not mobile_value:
        missing_completeness.append("Mobile")
    if not str(profile_record.get("address") or "").strip():
        missing_completeness.append("Address")
    gender_value = str(profile_record.get("gender") or "").strip()
    if not gender_value:
        missing_completeness.append("Gender")
    abha_value = str(
        profile_record.get("abha_id")
        or profile_record.get("abha")
        or profile_record.get("abha_number")
        or ""
    ).strip()
    if not abha_value or abha_value == "-":
        missing_completeness.append("ABHA ID")

    completeness_parts = 6
    completeness_score = int(round(((completeness_parts - len(missing_completeness)) / completeness_parts) * 100))

    average_risk = (sum(probability_samples) / len(probability_samples)) if probability_samples else 0.0

    latest_prediction_entry = sorted_history[-1] if sorted_history else None
    latest_prediction_at = str((latest_prediction_entry or {}).get("at") or "").strip() if latest_prediction_entry else ""
    recent_slice = sorted_history[-10:] if sorted_history else []
    recent_highest = 0.0
    recent_highest_disease = "-"
    for entry in recent_slice:
        payload = entry.get("payload") if isinstance(entry.get("payload"), dict) else {}
        probability = payload.get("prob", payload.get("probability"))
        try:
            score = float(probability)
        except (TypeError, ValueError):
            score = 0.0
        if score <= 1.0 and probability is not None:
            score = score * 100.0
        score = max(0.0, min(100.0, score))
        if score >= recent_highest:
            recent_highest = score
            recent_highest_disease = str(entry.get("disease") or "-")

    active_disease = max(disease_counts.items(), key=lambda item: item[1])[0] if disease_counts else "-"

    data_quality_missing: List[Dict[str, str]] = []
    if not str(profile_record.get("dob") or "").strip():
        data_quality_missing.append({"key": "dob", "label": "DOB", "action": "fix_profile"})
    if not mobile_value:
        data_quality_missing.append({"key": "contact", "label": "Contact", "action": "fix_profile"})
    if not str(profile_record.get("gender") or "").strip():
        data_quality_missing.append({"key": "gender", "label": "Gender", "action": "fix_profile"})
    if not (prediction_history or reports or health_history):
        data_quality_missing.append({"key": "history", "label": "History", "action": "add_history"})

    data_quality_total = 4
    data_quality_score = int(round(((data_quality_total - len(data_quality_missing)) / data_quality_total) * 100))

    bar_metrics = [
        {
            "label": "TS",
            "full_label": "Total Sessions",
            "value": len(prediction_history),
            "display": str(len(prediction_history)),
        },
        {
            "label": "AR",
            "full_label": "Average Risk",
            "value": round(average_risk, 2),
            "display": f"{average_risk:.1f}%",
        },
        {
            "label": "IM",
            "full_label": "Improvements",
            "value": progression["improved"],
            "display": str(progression["improved"]),
        },
        {
            "label": "DT",
            "full_label": "Deteriorations",
            "value": progression["deteriorated"],
            "display": str(progression["deteriorated"]),
        },
        {
            "label": "RU",
            "full_label": "Reports Uploaded",
            "value": len(reports),
            "display": str(len(reports)),
        },
        {
            "label": "DS",
            "full_label": "Diseases Tracked",
            "value": len(disease_counts),
            "display": str(len(disease_counts)),
        },
    ]

    return jsonify(
        {
            "success": True,
            "summary": {
                "total_predictions": len(prediction_history),
                "diseases_tracked": len(disease_counts),
                "profile": profile_payload,
            },
            "profile_completeness": {
                "score": completeness_score,
                "missing": missing_completeness,
            },
            "risk_alerts": risk_alerts,
            "quick_snapshot": {
                "last_prediction_at": latest_prediction_at,
                "highest_recent_risk": round(recent_highest, 2),
                "highest_recent_risk_disease": recent_highest_disease,
                "active_disease": active_disease,
            },
            "data_quality": {
                "score": data_quality_score,
                "missing": data_quality_missing,
            },
            "timeline": {
                "series": [
                    {
                        "disease": disease,
                        "points": points[-30:],
                    }
                    for disease, points in timeline_by_disease.items()
                ]
            },
            "charts": {
                "donut": [
                    {"label": "Improvement", "value": progression["improved"]},
                    {"label": "Deterioration", "value": progression["deteriorated"]},
                ],
                "bar": bar_metrics,
            },
            "reports": reports,
            "health_history": health_history,
        }
    )


@app.route("/api/auth/history/export", methods=["GET"])
@_require_auth
def export_full_history():
    user_id = str(session.get("auth_user_id"))
    payload = {
        "success": True,
        "exported_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "profile": user_data_store.get_user_profile(user_id),
        "reports": user_data_store.list_reports(user_id),
        "predictions": user_data_store.list_predictions(user_id),
        "health_history": user_data_store.list_health_history(user_id),
    }

    buffer = BytesIO(json.dumps(payload, indent=2).encode("utf-8"))
    buffer.seek(0)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"CureHelp_History_{timestamp}.json",
        mimetype="application/json",
    )


@app.route("/api/profile", methods=["POST"])
@profile_latency("api.profile.create")
@_require_auth
def create_profile():
    file_storage = None
    uploaded_report_name = ""
    if request.content_type and "multipart/form-data" in request.content_type.lower():
        payload = {key: request.form.get(key, "") for key in request.form}
        file_storage = request.files.get("medical_report")
    else:
        payload = request.get_json(force=True, silent=True) or {}

    # Normalize whitespace for string fields
    for key, value in list(payload.items()):
        if isinstance(value, str):
            payload[key] = value.strip()

    required_fields = ["name", "age", "contact", "address", "gender", "marital_status"]
    missing = [field for field in required_fields if not str(payload.get(field, "")).strip()]
    if missing:
        return jsonify({"success": False, "error": f"Missing fields: {', '.join(missing)}"}), 400

    try:
        age_value = int(payload["age"])
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "Age must be a valid integer."}), 400

    profile_data = {
        "name": payload["name"].strip(),
        "age": age_value,
        "contact": payload["contact"].strip(),
        "address": payload["address"].strip(),
        "gender": payload["gender"],
        "marital_status": payload["marital_status"],
        "predictions": {},
    }

    autofill_data: Dict[str, Dict[str, Any]] = {}

    if file_storage and (file_storage.filename or "").strip():
        filename = file_storage.filename or ""
        uploaded_report_name = filename
        extension = os.path.splitext(filename)[1].lower()
        if extension not in REPORT_ALLOWED_EXTENSIONS:
            return (
                jsonify({
                    "success": False,
                    "error": "Unsupported report format. Allowed formats: CSV, PDF, XLS, XLSX.",
                }),
                400,
            )

        try:
            file_storage.stream.seek(0, os.SEEK_END)
            report_size = file_storage.stream.tell()
            file_storage.stream.seek(0)
        except OSError:
            report_size = None

        if report_size is None:
            content_length = request.content_length
        else:
            content_length = report_size

        if content_length is not None and content_length > MAX_REPORT_SIZE_BYTES:
            return (
                jsonify({
                    "success": False,
                    "error": "Report exceeds the maximum size of 200 MB.",
                }),
                400,
            )

        try:
            autofill_data = parse_medical_report(file_storage)
        except ValueError as exc:
            return jsonify({"success": False, "error": str(exc)}), 400

    profile = profile_manager.add_profile(profile_data)
    session["current_profile_id"] = profile["id"]
    session["current_profile_name"] = profile.get("name")
    session["current_profile_gender"] = profile.get("gender", "")
    session["predictions"] = {}
    session.modified = True

    response_payload: Dict[str, Any] = {"success": True, "profile": profile}

    auth_user_id = session.get("auth_user_id")
    if auth_user_id:
        user_id = str(auth_user_id)
        user_data_store.set_user_profile(
            user_id,
            {
                "name": profile.get("name", ""),
                "address": profile.get("address", ""),
                "mobile": profile.get("contact", ""),
                "gender": profile.get("gender", ""),
                "dob": payload.get("dob", ""),
            },
        )
        if uploaded_report_name:
            user_data_store.append_report(
                user_id,
                {
                    "name": uploaded_report_name,
                    "status": "uploaded",
                    "source": "profile_form",
                    "at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
                },
            )
            user_data_store.append_health_history(
                user_id,
                disease_type="profile_upload",
                input_data={"filename": uploaded_report_name, "source": "profile_form"},
                image_path=None,
            )

    if autofill_data:
        response_payload["autofill"] = autofill_data

    return jsonify(response_payload)


@app.route("/api/profile", methods=["GET"])
@_require_auth
def get_current_profile():
    profile_id = session.get("current_profile_id")
    if not profile_id:
        return jsonify({"success": True, "profile": None})
    profile = profile_manager.get_profile(profile_id)
    return jsonify({"success": True, "profile": profile})


@app.route("/api/profiles", methods=["GET"])
@_require_auth
def list_profiles():
    search = request.args.get("q")
    if search:
        profiles = profile_manager.search_profiles(search)
    else:
        profiles = profile_manager.list_profiles()
    return jsonify({"success": True, "profiles": profiles})


@app.route("/api/profiles/<profile_id>", methods=["DELETE"])
@_require_auth
def delete_profile(profile_id: str):
    if session.get("current_profile_id") == profile_id:
        return jsonify({"success": False, "error": "Cannot delete the active profile."}), 400

    if not profile_manager.delete_profile(profile_id):
        return jsonify({"success": False, "error": "Profile not found"}), 404

    return jsonify({"success": True})


@app.route("/api/diabetes", methods=["POST"])
@profile_latency("api.predict.diabetes")
@_require_auth
def predict_diabetes():
    data = request.get_json(force=True, silent=True) or {}
    try:
        models = _get_models()
    except Exception:
        return jsonify({"success": False, "error": "Prediction models are unavailable."}), 503

    try:
        gender = data.get("gender", "Female")
        pregnancies = _convert_to_float(data, "pregnancies") if gender.lower() == "female" else 0.0
        inputs = {
            "Pregnancies": pregnancies,
            "Glucose": _convert_to_float(data, "glucose"),
            "Blood Pressure": _convert_to_float(data, "blood_pressure"),
            "Skin Thickness": _convert_to_float(data, "skin_thickness"),
            "Insulin": _convert_to_float(data, "insulin"),
            "BMI": _convert_to_float(data, "bmi"),
            "Diabetes Pedigree Function": _convert_to_float(data, "diabetes_pedigree_function"),
            "Age": _convert_to_float(data, "age"),
        }
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

    arr = np.array([[
        inputs["Pregnancies"],
        inputs["Glucose"],
        inputs["Blood Pressure"],
        inputs["Skin Thickness"],
        inputs["Insulin"],
        inputs["BMI"],
        inputs["Diabetes Pedigree Function"],
        inputs["Age"],
    ]], dtype=np.float64)
    arr_scaled = models["diabetes_scaler"].transform(arr)
    probability = float(models["diabetes_model"].predict_proba(arr_scaled)[0][1] * 100)

    display_inputs = _map_display_inputs({**data, **{"pregnancies": pregnancies}}, DIABETES_INPUT_LABELS)
    history_inputs = {**data, **{"pregnancies": pregnancies}}
    _store_prediction(
        TYPE2_DIABETES_LABEL,
        {"prob": probability, "inputs": display_inputs},
        history_inputs=history_inputs,
    )

    recommendations = fetch_gemini_recommendations(TYPE2_DIABETES_LABEL, probability)

    return jsonify(
        {
            "success": True,
            "disease": TYPE2_DIABETES_LABEL,
            "probability": probability,
            "inputs": display_inputs,
            "normal_values": DIABETES_NORMALS,
            "recommendations": recommendations,
        }
    )


@app.route("/api/heart", methods=["POST"])
@profile_latency("api.predict.heart")
@_require_auth
def predict_heart():
    data = request.get_json(force=True, silent=True) or {}
    try:
        models = _get_models()
    except Exception:
        return jsonify({"success": False, "error": "Prediction models are unavailable."}), 503

    try:
        gender = data.get("gender", "Male")
        sex_code = 1 if gender.lower() == "male" else 0
        cp_value = str(data.get("chest_pain_type", "1"))
        fbs_value = data.get("fasting_bs", "No")
        restecg_value = str(data.get("resting_ecg", "0"))
        exang_value = data.get("exercise_angina", "No")
        slope_value = str(data.get("slope", "1"))
        thal_value = str(data.get("thal", "3"))

        cp_code = int(cp_value.split(" ")[0]) if " " in cp_value else int(cp_value)
        restecg_code = int(restecg_value.split(" ")[0]) if " " in restecg_value else int(restecg_value)
        slope_code = int(slope_value.split(" ")[0]) if " " in slope_value else int(slope_value)
        thal_code = int(thal_value.split(" ")[0]) if " " in thal_value else int(thal_value)

        inputs = {
            "Age": _convert_to_float(data, "age"),
            "Sex": sex_code,
            "Chest Pain Type": cp_code,
            "Resting BP": _convert_to_float(data, "resting_bp"),
            "Cholesterol": _convert_to_float(data, "cholesterol"),
            "Fasting BS > 120?": 1 if str(fbs_value).lower() in {"yes", "1", "true"} else 0,
            "Resting ECG": restecg_code,
            "Max Heart Rate": _convert_to_float(data, "max_heart_rate"),
            "Exercise Angina": 1 if str(exang_value).lower() in {"yes", "1", "true"} else 0,
            "ST Depression": _convert_to_float(data, "st_depression"),
            "Slope of ST": slope_code,
            "Major Vessels (ca)": _convert_to_float(data, "major_vessels"),
            "Thal": thal_code,
        }
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

    arr = np.array([[
        inputs["Age"],
        inputs["Sex"],
        inputs["Chest Pain Type"],
        inputs["Resting BP"],
        inputs["Cholesterol"],
        inputs["Fasting BS > 120?"],
        inputs["Resting ECG"],
        inputs["Max Heart Rate"],
        inputs["Exercise Angina"],
        inputs["ST Depression"],
        inputs["Slope of ST"],
        inputs["Major Vessels (ca)"],
        inputs["Thal"],
    ]], dtype=np.float64)
    arr_scaled = models["heart_scaler"].transform(arr)
    probability = float(models["heart_model"].predict_proba(arr_scaled)[0][1] * 100)

    display_inputs = _map_display_inputs({**data, **{"gender": gender}}, HEART_INPUT_LABELS)
    display_inputs.update({
        "Sex": inputs["Sex"],
        "Chest Pain Type": inputs["Chest Pain Type"],
        "Fasting BS > 120?": inputs["Fasting BS > 120?"],
        "Exercise Angina": inputs["Exercise Angina"],
        "Slope of ST": inputs["Slope of ST"],
        "Thal": inputs["Thal"],
    })

    _store_prediction(
        "Coronary Artery Disease",
        {"prob": probability, "inputs": display_inputs},
        history_inputs=data,
    )

    recommendations = fetch_gemini_recommendations("Coronary Artery Disease", probability)

    return jsonify(
        {
            "success": True,
            "disease": "Coronary Artery Disease",
            "probability": probability,
            "inputs": display_inputs,
            "normal_values": HEART_NORMALS,
            "recommendations": recommendations,
        }
    )


@app.route("/api/anemia", methods=["POST"])
@profile_latency("api.predict.anemia")
@_require_auth
def predict_anemia():
    data = request.get_json(force=True, silent=True) or {}
    try:
        models = _get_models()
    except Exception:
        return jsonify({"success": False, "error": "Prediction models are unavailable."}), 503

    try:
        gender = data.get("gender", "Female")
        input_array = np.array([
            _convert_to_float(data, "rbc"),
            _convert_to_float(data, "hemoglobin"),
            _convert_to_float(data, "mcv"),
            _convert_to_float(data, "mch"),
            _convert_to_float(data, "mchc"),
            _convert_to_float(data, "hematocrit"),
            _convert_to_float(data, "wbc"),
            _convert_to_float(data, "platelets"),
            _convert_to_float(data, "pdw"),
            _convert_to_float(data, "pct"),
            _convert_to_float(data, "lymphocytes"),
            _convert_to_float(data, "neutrophils_pct"),
            _convert_to_float(data, "neutrophils_num"),
        ]).reshape(1, -1)
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

    input_scaled = models["anemia_scaler"].transform(input_array)
    risk_prob = float(models["anemia_risk_model"].predict_proba(input_scaled)[0][1] * 100)

    try:
        type_pred = models["anemia_type_model"].predict(input_scaled)[0]
        anemia_type_label = models["anemia_label_encoder"].inverse_transform([type_pred])[0]
    except Exception:
        mcv_value = _convert_to_float(data, "mcv")
        anemia_type_label = "Microcytic" if mcv_value < 80 else ("Normocytic" if mcv_value <= 100 else "Macrocytic")

    display_inputs = _map_display_inputs(data, ANEMIA_INPUT_LABELS)
    _store_prediction(
        "Anemia",
        {"prob": risk_prob, "inputs": display_inputs, "severity": anemia_type_label},
        history_inputs=data,
    )

    recommendations = fetch_gemini_recommendations("Anemia", risk_prob)

    return jsonify(
        {
            "success": True,
            "disease": "Anemia",
            "probability": risk_prob,
            "severity": anemia_type_label,
            "inputs": display_inputs,
            "normal_values": _anemia_normals(gender),
            "recommendations": recommendations,
        }
    )


@app.route("/api/pneumonia", methods=["POST"])
@profile_latency("api.predict.pneumonia")
@_require_auth
def predict_pneumonia():
    if request.content_type is None or "multipart/form-data" not in request.content_type.lower():
        return jsonify({"success": False, "error": "Use multipart/form-data for image upload."}), 400

    if not _ensure_pneumonia_artifacts():
        return jsonify({"success": False, "error": "Pneumonia model is unavailable."}), 503

    image_file = request.files.get("image")
    if image_file is None or not (image_file.filename or "").strip():
        return jsonify({"success": False, "error": "X-ray image is required."}), 400

    extension = os.path.splitext(image_file.filename)[1].lower().lstrip(".")
    if extension not in PNEUMONIA_ALLOWED_EXTENSIONS:
        return jsonify({"success": False, "error": "Unsupported image type. Allowed: JPG, JPEG, PNG."}), 400

    try:
        from PIL import Image

        image_bytes = image_file.read()
        if len(image_bytes) > MAX_XRAY_IMAGE_SIZE_BYTES:
            return jsonify({"success": False, "error": "Image size must be 10MB or smaller."}), 400
        image_stream = BytesIO(image_bytes)
        image = Image.open(image_stream).convert("RGB")
        image = image.resize(PNEUMONIA_IMAGE_SIZE)

        if PNEUMONIA_IMG_TO_ARRAY is not None:
            image_array = PNEUMONIA_IMG_TO_ARRAY(image)
        else:
            image_array = np.asarray(image, dtype=np.float32)

        input_batch = np.expand_dims(image_array, axis=0)
        input_batch = PNEUMONIA_PREPROCESS_INPUT(input_batch)
        try:
            raw_prediction = PNEUMONIA_MODEL.predict(input_batch, verbose=0)
        except TypeError:
            raw_prediction = PNEUMONIA_MODEL.predict(input_batch)
    except Exception:
        return jsonify({"success": False, "error": "Unable to process the uploaded X-ray image."}), 400

    probability = float(np.squeeze(raw_prediction))
    probability = float(np.clip(probability, 0.0, 1.0))
    result = "Pneumonia" if probability >= PNEUMONIA_THRESHOLD else "Normal"
    probability_percent = probability * 100.0

    recommendations = fetch_gemini_recommendations("Pneumonia", probability_percent)

    user_id = str(session.get("auth_user_id") or "")
    xray_public_path = None
    if user_id:
        xray_filename = _build_user_upload_filename(user_id, f"pneumonia.{extension}")
        _save_upload_bytes(image_bytes, XRAY_HISTORY_UPLOAD_DIR, xray_filename)
        xray_public_path = _public_upload_path("xray_history", xray_filename)

    payload = {
        "prob": probability_percent,
        "probability": probability,
        "result": result,
        "inputs": {"Pneumonia Score": round(probability_percent, 2)},
        "normal_values": {"Pneumonia Score": PNEUMONIA_THRESHOLD * 100},
        "xray_image_path": xray_public_path,
    }
    _store_prediction(
        "Pneumonia",
        payload,
        history_inputs={"result": result, "probability": round(probability_percent, 2)},
        history_image_path=xray_public_path,
    )

    return jsonify(
        {
            "success": True,
            "disease": "Pneumonia",
            "prob": probability_percent,
            "probability": probability,
            "threshold": PNEUMONIA_THRESHOLD,
            "result": result,
            "inputs": payload["inputs"],
            "normal_values": payload["normal_values"],
            "recommendations": recommendations,
        }
    )


@app.route("/api/tuberculosis", methods=["POST"])
@profile_latency("api.predict.tuberculosis")
@_require_auth
def predict_tuberculosis():
    if request.content_type is None or "multipart/form-data" not in request.content_type.lower():
        return jsonify({"success": False, "error": "Use multipart/form-data for image upload."}), 400

    if not _ensure_tb_artifacts():
        return jsonify({"success": False, "error": "Tuberculosis model is unavailable."}), 503

    image_file = request.files.get("image")
    if image_file is None or not (image_file.filename or "").strip():
        return jsonify({"success": False, "error": "X-ray image is required."}), 400

    extension = os.path.splitext(image_file.filename)[1].lower().lstrip(".")
    if extension not in TB_ALLOWED_EXTENSIONS:
        return jsonify({"success": False, "error": "Unsupported image type. Allowed: JPG, JPEG, PNG."}), 400

    try:
        image_bytes = image_file.read()
        if len(image_bytes) > MAX_XRAY_IMAGE_SIZE_BYTES:
            return jsonify({"success": False, "error": "Image size must be 10MB or smaller."}), 400

        input_batch = _tb_preprocess_image(image_bytes)
        input_tensor = TB_TORCH.tensor(input_batch, dtype=TB_TORCH.float32)
        with TB_TORCH.no_grad():
            raw_prediction = TB_MODEL(input_tensor)
            probability = float(np.squeeze(raw_prediction.detach().cpu().numpy()))
    except Exception:
        return jsonify({"success": False, "error": "Unable to process the uploaded X-ray image."}), 400

    if not np.isfinite(probability):
        probability = 0.0

    if probability < 0.0 or probability > 1.0:
        probability = float(1.0 / (1.0 + np.exp(-probability)))

    probability = float(np.clip(probability, 0.0, 1.0))
    prediction = "Tuberculosis" if probability >= TB_THRESHOLD else "Normal"
    confidence = _tb_confidence_category(probability)
    probability_percent = probability * 100.0

    recommendations = fetch_gemini_recommendations("Tuberculosis", probability_percent)

    user_id = str(session.get("auth_user_id") or "")
    xray_public_path = None
    if user_id:
        xray_filename = _build_user_upload_filename(user_id, f"tuberculosis.{extension}")
        _save_upload_bytes(image_bytes, XRAY_HISTORY_UPLOAD_DIR, xray_filename)
        xray_public_path = _public_upload_path("xray_history", xray_filename)

    payload = {
        "prob": probability_percent,
        "probability": probability,
        "result": prediction,
        "prediction": prediction,
        "confidence": confidence,
        "inputs": {"Tuberculosis Score": round(probability_percent, 2)},
        "normal_values": {"Tuberculosis Score": TB_THRESHOLD * 100},
        "xray_image_path": xray_public_path,
    }
    _store_prediction(
        "Tuberculosis",
        payload,
        history_inputs={"prediction": prediction, "confidence": confidence, "probability": round(probability_percent, 2)},
        history_image_path=xray_public_path,
    )

    return jsonify(
        {
            "success": True,
            "disease": "Tuberculosis",
            "prob": probability_percent,
            "probability": probability,
            "threshold": TB_THRESHOLD,
            "result": prediction,
            "prediction": prediction,
            "confidence": confidence,
            "inputs": payload["inputs"],
            "normal_values": payload["normal_values"],
            "recommendations": recommendations,
        }
    )


@app.route("/api/report", methods=["GET"])
@_require_auth
def get_report_summary():
    return jsonify({"success": True, "predictions": _normalise_prediction_labels(_current_predictions())})


@app.route("/api/report/pdf", methods=["GET"])
@_require_auth
def download_report():
    predictions = _normalise_prediction_labels(_current_predictions())
    if not predictions:
        return jsonify({"success": False, "error": "No predictions available."}), 400
    disease_param = request.args.get("disease", "")
    if disease_param:
        requested = [_canonical_disease_label(item.strip()) for item in disease_param.split(",") if item.strip()]
    else:
        requested = list(predictions.keys())

    selected = [d for d in requested if d in predictions]
    if not selected:
        return jsonify({"success": False, "error": "Selected disease not found."}), 400

    pdf_buffer = generate_pdf_report(predictions, selected)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    return send_file(pdf_buffer, as_attachment=True, download_name=f"CureHelp_Report_{timestamp}.pdf", mimetype="application/pdf")


@app.route("/api/chat", methods=["POST"])
@_require_auth
def chat():
    payload = request.get_json(force=True, silent=True) or {}
    message = payload.get("message", "").strip()
    if not message:
        return jsonify({"success": False, "error": "Message cannot be empty."}), 400

    try:
        response = get_chatbot_response(message)
    except RuntimeError as exc:
        return jsonify({"success": False, "error": str(exc)}), 500

    return jsonify({"success": True, "response": response})


@app.route("/api/consultants", methods=["GET"])
@_require_auth
def consultants():
    query = request.args.get("q")
    if query:
        results = search_providers(query)
    else:
        results = get_consultant_directory()
    return jsonify({"success": True, "data": results})


@app.route("/api/reset", methods=["POST"])
@_require_auth
def reset_session():
    user_id = session.get("auth_user_id")
    session.pop("current_profile_id", None)
    session.pop("current_profile_name", None)
    session.pop("current_profile_gender", None)
    session.pop("predictions", None)
    session.modified = True
    return jsonify({"success": True})


@app.route("/api/metrics/latency", methods=["GET"])
def get_latency_metrics():
    with _MODEL_HEALTH_LOCK:
        model_health = dict(MODEL_HEALTH_STATUS)
    return jsonify(
        {
            "success": True,
            "model_health": model_health,
        }
    )


@app.errorhandler(404)
def handle_not_found(_):
    return jsonify({"success": False, "error": "Endpoint not found."}), 404


@app.errorhandler(500)
def handle_server_error(error):
    return jsonify({"success": False, "error": str(error)}), 500


_start_background_services()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
