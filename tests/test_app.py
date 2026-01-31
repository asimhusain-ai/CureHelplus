import json

import pytest


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
    assert "Diabetes" in report_data["predictions"]

    pdf_resp = client.get("/api/report/pdf")
    assert pdf_resp.status_code == 200
    assert pdf_resp.mimetype == "application/pdf"

    reset_resp = client.post("/api/reset")
    assert reset_resp.status_code == 200
    assert client.get("/api/report").get_json()["predictions"] == {}


def test_heart_and_fever_predictions(app_client):
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

    fever_payload = {
        "temperature": 38.5,
        "age": 28,
        "bmi": 24.0,
        "humidity": 60,
        "air_quality": 80,
        "heart_rate": 90,
        "gender": "Male",
        "headache": "Yes",
        "body_ache": "No",
        "fatigue": "Yes",
        "chronic_conditions": "No",
        "allergies": "No",
        "smoking_history": "No",
        "alcohol_consumption": "No",
        "physical_activity": "Active",
        "diet_type": "Vegetarian",
        "blood_pressure": "Normal",
        "previous_medication": "None",
    }
    fever_resp = _post_json(client, "/api/fever", fever_payload)
    assert fever_resp.status_code == 200
    fever_data = fever_resp.get_json()
    assert fever_data["probability"] == pytest.approx(65.0)
    assert fever_data["severity"] == "Moderate"


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
