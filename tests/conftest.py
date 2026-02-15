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

    def predict(self, arr, verbose=0):
        return np.array([[self.prob]])


class DummyPredictModel:
    def __init__(self, label: int):
        self.label = label

    def predict(self, arr):
        return np.array([self.label])


class DummyTensorOutput:
    def __init__(self, value: float):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.array([[self.value]], dtype=np.float32)


class DummyTBModel:
    def __init__(self, prob: float):
        self.prob = prob

    def __call__(self, _tensor):
        return DummyTensorOutput(self.prob)


class DummyNoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class DummyTorch:
    float32 = "float32"

    def tensor(self, arr, dtype=None):
        return np.asarray(arr, dtype=np.float32)

    def no_grad(self):
        return DummyNoGrad()


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
    app_module.PNEUMONIA_MODEL = DummyProbModel(0.9)
    app_module.PNEUMONIA_PREPROCESS_INPUT = lambda arr: arr
    app_module.PNEUMONIA_LOAD_IMG = lambda file_obj, target_size, color_mode="rgb": np.zeros((target_size[0], target_size[1], 3), dtype=np.float32)
    app_module.PNEUMONIA_IMG_TO_ARRAY = lambda image: np.asarray(image, dtype=np.float32)
    app_module.TB_MODEL = DummyTBModel(0.85)
    app_module.TB_TORCH = DummyTorch()

    with app_module.app.test_client() as client:
        yield app_module, client

    app_module.MODELS.clear()
