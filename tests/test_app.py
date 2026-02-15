import json
import io

import pytest
from PIL import Image


def _valid_png_bytes() -> bytes:
    image = Image.new("RGB", (2, 2), color=(128, 128, 128))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _post_json(client, url, payload):
    return client.post(url, data=json.dumps(payload), content_type="application/json")


def test_create_profile_and_report_flow(app_client):
    app_module, client = app_client

    profile_payload = {
        "name": "Test User",
        "age": 30,
        "contact": "999",
        "address": "123 Street",
        "gender": "Female",
        "marital_status": "Single",
    }
    response = _post_json(client, "/api/profile", profile_payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    profile_id = data["profile"]["id"]

    diabetes_payload = {
        "gender": "Female",
        "pregnancies": 2,
        "glucose": 150,
        "blood_pressure": 85,
        "skin_thickness": 20,
        "insulin": 80,
        "bmi": 26,
        "diabetes_pedigree_function": 0.5,
        "age": 40,
    }
    diabetes_resp = _post_json(client, "/api/diabetes", diabetes_payload)
    assert diabetes_resp.status_code == 200
    diabetes_data = diabetes_resp.get_json()
    assert diabetes_data["probability"] == pytest.approx(72.0)

    report_resp = client.get("/api/report")
    assert report_resp.status_code == 200
    report_data = report_resp.get_json()
    assert "Type-2 Diabetes" in report_data["predictions"]

    pdf_resp = client.get("/api/report/pdf")
    assert pdf_resp.status_code == 200
    assert pdf_resp.mimetype == "application/pdf"

    reset_resp = client.post("/api/reset")
    assert reset_resp.status_code == 200
    assert client.get("/api/report").get_json()["predictions"] == {}


def test_heart_prediction(app_client):
    app_module, client = app_client

    _post_json(
        client,
        "/api/profile",
        {
            "name": "User",
            "age": 35,
            "contact": "111",
            "address": "Main Road",
            "gender": "Male",
            "marital_status": "Married",
        },
    )

    heart_payload = {
        "gender": "Male",
        "age": 55,
        "chest_pain_type": 2,
        "resting_bp": 130,
        "cholesterol": 200,
        "fasting_bs": "Yes",
        "resting_ecg": 1,
        "max_heart_rate": 150,
        "exercise_angina": "No",
        "st_depression": 1.2,
        "slope": 2,
        "major_vessels": 1,
        "thal": 3,
    }
    heart_resp = _post_json(client, "/api/heart", heart_payload)
    assert heart_resp.status_code == 200
    heart_data = heart_resp.get_json()
    assert heart_data["probability"] == pytest.approx(81.0)


def test_anemia_prediction_and_misc_endpoints(app_client):
    app_module, client = app_client

    _post_json(
        client,
        "/api/profile",
        {
            "name": "User",
            "age": 45,
            "contact": "222",
            "address": "Lane",
            "gender": "Female",
            "marital_status": "Single",
        },
    )

    anemia_payload = {
        "gender": "Female",
        "rbc": 4.2,
        "hemoglobin": 11.5,
        "mcv": 82,
        "mch": 27,
        "mchc": 33,
        "hematocrit": 38,
        "wbc": 7,
        "platelets": 220,
        "pdw": 14,
        "pct": 0.22,
        "lymphocytes": 30,
        "neutrophils_pct": 60,
        "neutrophils_num": 4.5,
    }
    anemia_resp = _post_json(client, "/api/anemia", anemia_payload)
    assert anemia_resp.status_code == 200
    anemia_data = anemia_resp.get_json()
    assert anemia_data["probability"] == pytest.approx(66.0)
    assert anemia_data["severity"] == "Iron Deficiency"

    chat_resp = _post_json(client, "/api/chat", {"message": "Hello"})
    assert chat_resp.status_code == 200
    assert chat_resp.get_json()["response"]["message"] == "ok"

    consultants_resp = client.get("/api/consultants?q=Apollo")
    assert consultants_resp.status_code == 200
    assert consultants_resp.get_json()["data"]["hospitals"]

    not_found = client.get("/missing")
    assert not_found.status_code == 404
    assert not_found.get_json()["success"] is False


def test_profile_creation_requires_fields(app_client):
    app_module, client = app_client
    resp = _post_json(client, "/api/profile", {"name": ""})
    assert resp.status_code == 400
    assert resp.get_json()["success"] is False


def test_chat_requires_message(app_client):
    app_module, client = app_client
    resp = _post_json(client, "/api/chat", {"message": ""})
    assert resp.status_code == 400
    assert resp.get_json()["success"] is False


def test_pneumonia_prediction(app_client):
    app_module, client = app_client

    resp = client.post(
        "/api/pneumonia",
        data={"image": (io.BytesIO(_valid_png_bytes()), "scan.png")},
        content_type="multipart/form-data",
    )

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["success"] is True
    assert payload["disease"] == "Pneumonia"
    assert payload["result"] == "Pneumonia"
    assert payload["probability"] == pytest.approx(0.9)


def test_pneumonia_rejects_invalid_extension(app_client):
    app_module, client = app_client

    resp = client.post(
        "/api/pneumonia",
        data={"image": (io.BytesIO(b"fake-xray"), "scan.gif")},
        content_type="multipart/form-data",
    )

    assert resp.status_code == 400
    payload = resp.get_json()
    assert payload["success"] is False


def test_tuberculosis_prediction(app_client):
    app_module, client = app_client

    resp = client.post(
        "/api/tuberculosis",
        data={"image": (io.BytesIO(_valid_png_bytes()), "scan.png")},
        content_type="multipart/form-data",
    )

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["success"] is True
    assert payload["disease"] == "Tuberculosis"
    assert payload["prediction"] == "Tuberculosis"
    assert payload["result"] == "Tuberculosis"
    assert payload["confidence"] == "Very High Risk"
    assert payload["probability"] == pytest.approx(0.85)


def test_tuberculosis_rejects_invalid_extension(app_client):
    app_module, client = app_client

    resp = client.post(
        "/api/tuberculosis",
        data={"image": (io.BytesIO(b"fake-xray"), "scan.gif")},
        content_type="multipart/form-data",
    )

    assert resp.status_code == 400
    payload = resp.get_json()
    assert payload["success"] is False
