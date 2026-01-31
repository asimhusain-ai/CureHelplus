"""Flask application entry point for the CureHelp+ medical assistant."""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict

import joblib
import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_file, session

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


def load_models() -> Dict[str, Any]:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "models")

    return {
        "diabetes_model": joblib.load(os.path.join(model_dir, "diabetes_model.pkl")),
        "diabetes_scaler": joblib.load(os.path.join(model_dir, "diabetes_scaler.pkl")),
        "heart_model": joblib.load(os.path.join(model_dir, "heart_model.pkl")),
        "heart_scaler": joblib.load(os.path.join(model_dir, "heart_scaler.pkl")),
        "fever_severity_model": joblib.load(os.path.join(model_dir, "fever_severity_model.pkl")),
        "fever_risk_model": joblib.load(os.path.join(model_dir, "fever_risk_model.pkl")),
        "fever_scaler": joblib.load(os.path.join(model_dir, "fever_scaler.pkl")),
        "fever_target_le": joblib.load(os.path.join(model_dir, "fever_target_encoder.pkl")),
        "fever_le_dict": joblib.load(os.path.join(model_dir, "fever_label_encoders.pkl")),
        "anemia_risk_model": joblib.load(os.path.join(model_dir, "anemia_risk_model.pkl")),
        "anemia_type_model": joblib.load(os.path.join(model_dir, "anemia_type_model.pkl")),
        "anemia_scaler": joblib.load(os.path.join(model_dir, "feature_scaler.pkl")),
        "anemia_label_encoder": joblib.load(os.path.join(model_dir, "label_encoder.pkl")),
    }


MODELS = load_models()

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

FEVER_NORMALS = {
    "Temperature (°C)": 37.0,
    "Age": 40,
    "BMI": 22.0,
    "Humidity (%)": 50,
    "Air Quality Index": 50,
    "Heart Rate": 75,
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

FEVER_CATEGORICAL_ENCODERS = {
    "gender": "Gender",
    "headache": "Headache",
    "body_ache": "Body_Ache",
    "fatigue": "Fatigue",
    "chronic_conditions": "Chronic_Conditions",
    "allergies": "Allergies",
    "smoking_history": "Smoking_History",
    "alcohol_consumption": "Alcohol_Consumption",
    "physical_activity": "Physical_Activity",
    "diet_type": "Diet_Type",
    "blood_pressure": "Blood_Pressure",
    "previous_medication": "Previous_Medication",
}

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
    return session.get("predictions", {})


def _save_predictions(predictions: Dict[str, Any]) -> None:
    session["predictions"] = predictions
    session.modified = True


def _sync_predictions_to_profile() -> None:
    profile_id = session.get("current_profile_id")
    if profile_id:
        profile_manager.update_predictions(profile_id, _current_predictions())


def _store_prediction(disease: str, payload: Dict[str, Any]) -> None:
    predictions = _current_predictions().copy()
    predictions[disease] = payload
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
                "fever": FEVER_NORMALS,
            },
        }
    )


@app.route("/api/profile", methods=["POST"])
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
def predict_diabetes():
    data = request.get_json(force=True, silent=True) or {}
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
    arr_scaled = MODELS["diabetes_scaler"].transform(arr)
    probability = float(MODELS["diabetes_model"].predict_proba(arr_scaled)[0][1] * 100)

    display_inputs = _map_display_inputs({**data, **{"pregnancies": pregnancies}}, DIABETES_INPUT_LABELS)
    _store_prediction(
        "Diabetes",
        {"prob": probability, "inputs": display_inputs},
    )

    recommendations = fetch_gemini_recommendations("Diabetes", probability)

    return jsonify(
        {
            "success": True,
            "disease": "Diabetes",
            "probability": probability,
            "inputs": display_inputs,
            "normal_values": DIABETES_NORMALS,
            "recommendations": recommendations,
        }
    )


@app.route("/api/heart", methods=["POST"])
def predict_heart():
    data = request.get_json(force=True, silent=True) or {}
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
    arr_scaled = MODELS["heart_scaler"].transform(arr)
    probability = float(MODELS["heart_model"].predict_proba(arr_scaled)[0][1] * 100)

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
        "Heart Disease",
        {"prob": probability, "inputs": display_inputs},
    )

    recommendations = fetch_gemini_recommendations("Heart Disease", probability)

    return jsonify(
        {
            "success": True,
            "disease": "Heart Disease",
            "probability": probability,
            "inputs": display_inputs,
            "normal_values": HEART_NORMALS,
            "recommendations": recommendations,
        }
    )


@app.route("/api/fever", methods=["POST"])
def predict_fever():
    data = request.get_json(force=True, silent=True) or {}
    try:
        numeric_inputs = {
            "Temperature (C)": _convert_to_float(data, "temperature"),
            "Age": _convert_to_float(data, "age"),
            "BMI": _convert_to_float(data, "bmi"),
            "Humidity (%)": _convert_to_float(data, "humidity"),
            "Air Quality Index": _convert_to_float(data, "air_quality"),
            "Heart Rate": _convert_to_float(data, "heart_rate"),
        }
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

    numeric_array = np.array([list(numeric_inputs.values())], dtype=np.float64)
    numeric_scaled = MODELS["fever_scaler"].transform(numeric_array)

    encoded = []
    categorical_display: Dict[str, Any] = {}
    try:
        for key, encoder_name in FEVER_CATEGORICAL_ENCODERS.items():
            value = data.get(key)
            if value is None:
                raise ValueError(f"Missing field: {key}")
            categorical_display[encoder_name.replace("_", " ")] = value
            encoder = MODELS["fever_le_dict"][encoder_name]
            if value in encoder.classes_:
                encoded.append(encoder.transform([value])[0])
            else:
                encoded.append(encoder.transform([encoder.classes_[0]])[0])
    except KeyError as exc:
        return jsonify({"success": False, "error": f"Unknown categorical field: {exc}"}), 400
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

    final_input = np.hstack([numeric_scaled, np.array([encoded])])
    severity_idx = int(MODELS["fever_severity_model"].predict(final_input)[0])
    severity_label = MODELS["fever_target_le"].inverse_transform([severity_idx])[0]
    risk_percent = float(np.clip(MODELS["fever_risk_model"].predict(final_input)[0], 0, 100))

    display_inputs = {**numeric_inputs, **categorical_display}
    _store_prediction(
        "Fever",
        {"prob": risk_percent, "inputs": display_inputs, "severity": severity_label},
    )

    recommendations = fetch_gemini_recommendations("Fever", risk_percent)

    return jsonify(
        {
            "success": True,
            "disease": "Fever",
            "probability": risk_percent,
            "severity": severity_label,
            "inputs": display_inputs,
            "normal_values": FEVER_NORMALS,
            "recommendations": recommendations,
        }
    )


@app.route("/api/anemia", methods=["POST"])
def predict_anemia():
    data = request.get_json(force=True, silent=True) or {}
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

    input_scaled = MODELS["anemia_scaler"].transform(input_array)
    risk_prob = float(MODELS["anemia_risk_model"].predict_proba(input_scaled)[0][1] * 100)

    try:
        type_pred = MODELS["anemia_type_model"].predict(input_scaled)[0]
        anemia_type_label = MODELS["anemia_label_encoder"].inverse_transform([type_pred])[0]
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


@app.route("/api/report", methods=["GET"])
def get_report_summary():
    return jsonify({"success": True, "predictions": _current_predictions()})


@app.route("/api/report/pdf", methods=["GET"])
def download_report():
    predictions = _current_predictions()
    if not predictions:
        return jsonify({"success": False, "error": "No predictions available."}), 400
    disease_param = request.args.get("disease", "")
    if disease_param:
        requested = [item.strip() for item in disease_param.split(",") if item.strip()]
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


@app.errorhandler(404)
def handle_not_found(_):
    return jsonify({"success": False, "error": "Endpoint not found."}), 404


@app.errorhandler(500)
def handle_server_error(error):
    return jsonify({"success": False, "error": str(error)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
