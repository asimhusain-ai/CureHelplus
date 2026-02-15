"""Flask application entry point for the CureHelp+ medical assistant."""
from __future__ import annotations

import os
import logging
import threading
import time
import sys
from io import BytesIO
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict

import joblib
import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_file, session
from PIL import Image

from admin import admin_bp
from chatbot import get_chatbot_response
from consultant import get_consultant_directory, search_providers
from helper import fetch_gemini_recommendations
from makepdf import generate_pdf_report
from profile_manager import profile_manager
from report_parser import REPORT_ALLOWED_EXTENSIONS, parse_medical_report

load_dotenv()

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SECRET_KEY"] = os.environ.get("CUREHELP_SECRET_KEY", "curehelp-secret-key")
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


def _store_prediction(disease: str, payload: Dict[str, Any]) -> None:
    predictions = _current_predictions().copy()
    predictions[_canonical_disease_label(disease)] = payload
    _save_predictions(predictions)
    _sync_predictions_to_profile()


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


@app.route("/")
def index() -> str:
    return render_template("index.html")


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


@app.route("/api/profile", methods=["POST"])
@profile_latency("api.profile.create")
def create_profile():
    file_storage = None
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
    if autofill_data:
        response_payload["autofill"] = autofill_data

    return jsonify(response_payload)


@app.route("/api/profile", methods=["GET"])
def get_current_profile():
    profile_id = session.get("current_profile_id")
    if not profile_id:
        return jsonify({"success": True, "profile": None})
    profile = profile_manager.get_profile(profile_id)
    return jsonify({"success": True, "profile": profile})


@app.route("/api/profiles", methods=["GET"])
def list_profiles():
    search = request.args.get("q")
    if search:
        profiles = profile_manager.search_profiles(search)
    else:
        profiles = profile_manager.list_profiles()
    return jsonify({"success": True, "profiles": profiles})


@app.route("/api/profiles/<profile_id>", methods=["DELETE"])
def delete_profile(profile_id: str):
    if session.get("current_profile_id") == profile_id:
        return jsonify({"success": False, "error": "Cannot delete the active profile."}), 400

    if not profile_manager.delete_profile(profile_id):
        return jsonify({"success": False, "error": "Profile not found"}), 404

    return jsonify({"success": True})


@app.route("/api/diabetes", methods=["POST"])
@profile_latency("api.predict.diabetes")
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
    _store_prediction(
        TYPE2_DIABETES_LABEL,
        {"prob": probability, "inputs": display_inputs},
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

    payload = {
        "prob": probability_percent,
        "probability": probability,
        "result": result,
        "inputs": {"Pneumonia Score": round(probability_percent, 2)},
        "normal_values": {"Pneumonia Score": PNEUMONIA_THRESHOLD * 100},
    }
    _store_prediction("Pneumonia", payload)

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

    payload = {
        "prob": probability_percent,
        "probability": probability,
        "result": prediction,
        "prediction": prediction,
        "confidence": confidence,
        "inputs": {"Tuberculosis Score": round(probability_percent, 2)},
        "normal_values": {"Tuberculosis Score": TB_THRESHOLD * 100},
    }
    _store_prediction("Tuberculosis", payload)

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
def get_report_summary():
    return jsonify({"success": True, "predictions": _normalise_prediction_labels(_current_predictions())})


@app.route("/api/report/pdf", methods=["GET"])
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
def consultants():
    query = request.args.get("q")
    if query:
        results = search_providers(query)
    else:
        results = get_consultant_directory()
    return jsonify({"success": True, "data": results})


@app.route("/api/reset", methods=["POST"])
def reset_session():
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
