from __future__ import annotations

import pandas as pd
import pytest
import zipfile

import chatbot
from chatbot import (
    classify_input_type,
    find_question_answer,
    format_chatbot_reply,
    get_disease_precautions,
    get_disease_symptoms,
    predict_disease_from_symptoms,
    process_user_input,
)


@pytest.fixture()
def sample_datasets():
    precautions_df = pd.DataFrame({
        "Disease": ["Diabetes"],
        "Precaution_1": ["Exercise regularly"],
        "Precaution_2": ["Monitor glucose"],
    })
    precautions_df["Disease_clean"] = precautions_df["Disease"].str.lower()

    symptoms_df = pd.DataFrame({
        "Disease": ["Diabetes"],
        "Symptom_1": ["frequent urination"],
    })
    symptoms_df["Disease_clean"] = symptoms_df["Disease"].str.lower()

    faq_df = pd.DataFrame({
        "question": ["What are the symptoms of diabetes?"],
        "answer": ["Increased thirst and urination among others."],
    })
    faq_df["question_clean"] = faq_df["question"].str.lower()

    augmented_df = pd.DataFrame({
        "diseases": ["Diabetes"],
        "diseases_clean": ["diabetes"],
        "frequent_urination": [1],
        "blurred_vision": [0],
    })

    humanqa_df = pd.DataFrame({
        "question": ["How to manage diabetes?"],
        "answer": ["Follow diet, exercise, and medication."],
    })
    humanqa_df["question_clean"] = humanqa_df["question"].str.lower()

    return precautions_df, symptoms_df, faq_df, augmented_df, humanqa_df


def test_classify_input_type_detection():
    assert classify_input_type("What are the symptoms?") == "question"
    assert classify_input_type("fever, cough") == "symptoms"
    assert classify_input_type("malaria") == "disease"


def test_get_disease_details_from_datasets(sample_datasets):
    precautions_df, symptoms_df, faq_df, augmented_df, _ = sample_datasets

    precautions = get_disease_precautions("diabetes", precautions_df)
    symptoms = get_disease_symptoms("diabetes", symptoms_df, augmented_df)

    assert "Exercise regularly" in precautions
    assert "frequent urination" in symptoms


def test_predict_disease_from_symptoms(sample_datasets):
    _, _, _, augmented_df, _ = sample_datasets

    prediction = predict_disease_from_symptoms(["frequent urination"], augmented_df)
    assert prediction is not None
    disease, confidence, _ = prediction
    assert disease == "Diabetes"
    assert 0.0 <= confidence <= 1.0


def test_find_question_answer_prioritises_overlap(sample_datasets):
    _, _, faq_df, _, humanqa_df = sample_datasets
    match = find_question_answer("What are the symptoms of diabetes?", faq_df, humanqa_df)
    assert match is not None
    assert "symptoms of diabetes" in match["question"].lower()


def test_process_user_input_question(sample_datasets):
    precautions_df, symptoms_df, faq_df, augmented_df, humanqa_df = sample_datasets

    analysis = process_user_input(
        "What are the symptoms of diabetes?",
        precautions_df,
        symptoms_df,
        faq_df,
        augmented_df,
        humanqa_df,
    )

    assert analysis["type"] == "question"
    assert analysis["faq_answer"]


def test_process_user_input_symptoms_list(sample_datasets):
    precautions_df, symptoms_df, faq_df, augmented_df, humanqa_df = sample_datasets

    analysis = process_user_input(
        "frequent urination, blurred vision",
        precautions_df,
        symptoms_df,
        faq_df,
        augmented_df,
        humanqa_df,
    )

    assert analysis["type"] == "symptoms"
    assert analysis["disease"] == "Diabetes"
    assert analysis["precautions"]


def test_format_chatbot_reply_fallback_message():
    payload = {"type": "question", "faq_answer": None}
    reply = format_chatbot_reply("question", payload)
    assert "rephrase" in reply["message"].lower()


def test_get_chatbot_response_with_mocked_datasets(monkeypatch, sample_datasets):
    chatbot.load_datasets.cache_clear()
    monkeypatch.setattr(chatbot, "load_datasets", lambda: sample_datasets)
    response = chatbot.get_chatbot_response("What are the symptoms of diabetes?")
    assert response["analysis"]["faq_answer"]


def test_get_chatbot_response_raises_when_datasets_missing(monkeypatch):
    chatbot.load_datasets.cache_clear()
    monkeypatch.setattr(chatbot, "load_datasets", lambda: (None, None, None, None, None))
    with pytest.raises(RuntimeError):
        chatbot.get_chatbot_response("Hello")


def test_load_datasets_from_zip(tmp_path):
    chatbot.load_datasets.cache_clear()
    zip_path = tmp_path / "bot_data.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("bot_data/Disease precaution.csv", "Disease,Precaution_1\nDiabetes,Exercise\n")
        archive.writestr("bot_data/DiseaseAndSymptoms.csv", "Disease,Symptom_1\nDiabetes,Thirst\n")
        archive.writestr("bot_data/medquad.csv", "question,answer\nWhat is diabetes?,Chronic condition\n")
        archive.writestr(
            "bot_data/Final_Augmented_dataset_Diseases_and_Symptoms.csv",
            "diseases,frequent_urination\nDiabetes,1\n",
        )
        archive.writestr("bot_data/humanqa.csv", "question,answer\nHow to treat?,Consult doctor\n")

    datasets = chatbot.load_datasets(str(zip_path))
    precautions_df, symptoms_df, _, _, _ = datasets
    assert precautions_df is not None and not precautions_df.empty
    assert symptoms_df is not None and not symptoms_df.empty
    chatbot.load_datasets.cache_clear()
