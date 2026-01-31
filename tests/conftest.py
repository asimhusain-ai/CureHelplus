import importlib
import io
import sys
from typing import Any, Dict

import joblib
import numpy as np
import pytest

from profile_manager import ProfileManager


class DummyScaler:
    def transform(self, arr):
        return arr


class DummyProbModel:
    def __init__(self, prob: float):
        self.prob = prob

    def predict_proba(self, arr):
        return np.array([[1 - self.prob, self.prob]])


class DummyPredictModel:
    def __init__(self, label: int):
        self.label = label

    def predict(self, arr):
        return np.array([self.label])


class DummyLabelEncoder:
    def __init__(self, mapping: Dict[int, str]):
        self.mapping = mapping

    def inverse_transform(self, indices):
        return [self.mapping.get(int(index), "Unknown") for index in indices]


class DummyEncoder:
    def __init__(self, classes):
        self.classes_ = np.array(classes)

    def transform(self, values):
        value = values[0]
        if value not in self.classes_:
            value = self.classes_[0]
        return np.array([int(np.where(self.classes_ == value)[0][0])])


@pytest.fixture()
def app_client(monkeypatch, tmp_path):
    dummy_models: Dict[str, Any] = {
        "diabetes_model.pkl": DummyProbModel(0.72),
        "diabetes_scaler.pkl": DummyScaler(),
        "heart_model.pkl": DummyProbModel(0.81),
        "heart_scaler.pkl": DummyScaler(),
        "fever_severity_model.pkl": DummyPredictModel(1),
        "fever_risk_model.pkl": DummyPredictModel(65),
        "fever_scaler.pkl": DummyScaler(),
        "fever_target_encoder.pkl": DummyLabelEncoder({0: "Mild", 1: "Moderate", 2: "Severe"}),
        "fever_label_encoders.pkl": {
            "Gender": DummyEncoder(["Male", "Female"]),
            "Headache": DummyEncoder(["No", "Yes"]),
            "Body_Ache": DummyEncoder(["No", "Yes"]),
            "Fatigue": DummyEncoder(["No", "Yes"]),
            "Chronic_Conditions": DummyEncoder(["No", "Yes"]),
            "Allergies": DummyEncoder(["No", "Yes"]),
            "Smoking_History": DummyEncoder(["No", "Yes"]),
            "Alcohol_Consumption": DummyEncoder(["No", "Yes"]),
            "Physical_Activity": DummyEncoder(["Sedentary", "Moderate", "Active"]),
            "Diet_Type": DummyEncoder(["Vegetarian", "Non-Vegetarian"]),
            "Blood_Pressure": DummyEncoder(["Normal", "High", "Low"]),
            "Previous_Medication": DummyEncoder(["None", "Ibuprofen", "Other"]),
        },
        "anemia_risk_model.pkl": DummyProbModel(0.66),
        "anemia_type_model.pkl": DummyPredictModel(0),
        "feature_scaler.pkl": DummyScaler(),
        "label_encoder.pkl": DummyLabelEncoder({0: "Iron Deficiency"}),
    }

    def fake_load(path):
        filename = path.split("\\")[-1]
        filename = filename.split("/")[-1]
        if filename not in dummy_models:
            raise FileNotFoundError(filename)
        return dummy_models[filename]

    monkeypatch.setattr(joblib, "load", fake_load)

    if "app" in sys.modules:
        del sys.modules["app"]
    app_module = importlib.import_module("app")
    app_module.app.config["TESTING"] = True

    temp_profiles = tmp_path / "profiles.json"
    manager = ProfileManager(str(temp_profiles))
    app_module.profile_manager = manager
    import profile_manager as profile_module

    profile_module.profile_manager = manager

    def fake_recommendations(disease, risk):
        return {
            "Risk Level": "mock",
            "prevention_measures": ["Stay hydrated"],
            "medicine_suggestions": ["Consult a doctor"],
        }

    monkeypatch.setattr(app_module, "fetch_gemini_recommendations", fake_recommendations)

    def fake_pdf(predictions, selected):
        return io.BytesIO(b"%PDF-1.4 test")

    monkeypatch.setattr(app_module, "generate_pdf_report", fake_pdf)
    monkeypatch.setattr(app_module, "get_chatbot_response", lambda message: {"message": "ok"})

    with app_module.app.test_client() as client:
        yield app_module, client

    app_module.MODELS.clear()
