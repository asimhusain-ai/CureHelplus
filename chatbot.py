"""Rule-based medical chatbot utilities for the Flask UI."""
from __future__ import annotations

import io
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_datasets(
    data_dir: str = "bot_data",
) -> Tuple[
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
]:
    """Load and preprocess datasets used by the chatbot."""

    base_path = Path(data_dir)
    zip_loader: Optional[ZipFile] = None

    if base_path.is_file() and base_path.suffix == ".zip":
        zip_loader = ZipFile(base_path)
        base_path = Path(base_path.stem)
    elif not base_path.exists():
        zip_candidate = base_path.with_suffix(".zip")
        if zip_candidate.exists():
            zip_loader = ZipFile(zip_candidate)
            base_path = Path(base_path.name)
        else:
            logger.error("bot_data dataset not found at %s or %s", base_path, zip_candidate)
            return None, None, None, None, None

    try:
        precautions_df = load_csv_flexible(base_path / "Disease precaution.csv", zip_loader)
        symptoms_df = load_csv_flexible(base_path / "DiseaseAndSymptoms.csv", zip_loader)
        faq_df = load_csv_flexible(base_path / "medquad.csv", zip_loader)
        augmented_df = load_csv_flexible(base_path / "Final_Augmented.csv", zip_loader)
        humanqa_df = load_csv_flexible(base_path / "humanqa.csv", zip_loader)
    finally:
        if zip_loader is not None:
            zip_loader.close()

    return preprocess_datasets(precautions_df, symptoms_df, faq_df, augmented_df, humanqa_df)


def load_csv_flexible(file_path: Path, zip_loader: Optional[ZipFile] = None) -> Optional[pd.DataFrame]:
    """Load CSV from disk with flexible encoding handling."""

    if not file_path.exists() and zip_loader is None:
        logger.warning("Dataset missing: %s", file_path)
        return None

    encodings = ("utf-8", "latin-1")
    for encoding in encodings:
        try:
            if zip_loader is None:
                df = pd.read_csv(file_path, encoding=encoding, on_bad_lines="skip")
            else:
                relative_path = str(file_path).replace("\\", "/")
                try:
                    data = zip_loader.read(relative_path)
                except KeyError:
                    continue
                df = pd.read_csv(io.BytesIO(data), encoding=encoding, on_bad_lines="skip")
            return clean_dataframe(df)
        except Exception:
            continue

    try:
        if zip_loader is None:
            df = pd.read_csv(file_path, encoding="utf-8", error_bad_lines=False, warn_bad_lines=False)
        else:
            relative_path = str(file_path).replace("\\", "/")
            data = zip_loader.read(relative_path)
            df = pd.read_csv(io.BytesIO(data), encoding="utf-8", error_bad_lines=False, warn_bad_lines=False)
        return clean_dataframe(df)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Could not load %s: %s", file_path, exc)
        return None


def clean_dataframe(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None:
        return None

    df = df.dropna(how="all")
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("")

    return df


def _normalise_qa_columns(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None:
        return None

    rename: Dict[str, str] = {}
    for column in df.columns:
        key = str(column).strip().lower()
        if key == "question":
            rename[column] = "question"
        elif key == "answer":
            rename[column] = "answer"

    if rename:
        df = df.rename(columns=rename)

    return df


def preprocess_datasets(
    precautions_df: Optional[pd.DataFrame],
    symptoms_df: Optional[pd.DataFrame],
    faq_df: Optional[pd.DataFrame],
    augmented_df: Optional[pd.DataFrame],
    humanqa_df: Optional[pd.DataFrame],
) -> Tuple[
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
]:
    """Preprocess datasets for better matching."""

    try:
        faq_df = _normalise_qa_columns(faq_df)
        humanqa_df = _normalise_qa_columns(humanqa_df)

        for df in (precautions_df, symptoms_df, faq_df):
            if df is not None and "Disease" in df.columns:
                df["Disease_clean"] = df["Disease"].str.lower().str.strip()

        if augmented_df is not None and "diseases" in augmented_df.columns:
            augmented_df["diseases_clean"] = augmented_df["diseases"].str.lower().str.strip()

        if faq_df is not None and "question" in faq_df.columns:
            faq_df["question_clean"] = faq_df["question"].str.lower().str.strip()

        if humanqa_df is not None and "question" in humanqa_df.columns:
            humanqa_df["question_clean"] = humanqa_df["question"].str.lower().str.strip()
    except Exception as exc:
        logger.warning("Dataset preprocessing issue: %s", exc)

    return precautions_df, symptoms_df, faq_df, augmented_df, humanqa_df


def find_question_answer(question: str, *qa_sources: Optional[pd.DataFrame]) -> Optional[pd.Series]:
    if not qa_sources:
        return None

    question_clean = question.lower().strip()
    if not question_clean:
        return None

    question_patterns = [
        r"what (are|is) (the )?(symptoms|signs) of",
        r"what (are|is) (the )?(causes|reason) of",
        r"what (are|is) (the )?(treatment|remedy) for",
        r"how (to|do) (treat|handle|manage)",
        r"what (is|are)",
    ]

    is_symptom_question = any(re.search(pattern, question_clean) for pattern in question_patterns)

    best_match: Optional[pd.Series] = None
    best_score = 0.0

    for dataset in qa_sources:
        if dataset is None or dataset.empty:
            continue

        for _, row in dataset.iterrows():
            qa_question = str(row.get("question", "")).lower().strip()
            if not qa_question:
                continue

            score = 0.0

            if is_symptom_question and ("symptom" in qa_question or "sign" in qa_question):
                score += 0.3

            question_words = set(question_clean.split())
            qa_words = set(qa_question.split())
            if question_words and qa_words:
                overlap = len(question_words.intersection(qa_words))
                score += overlap / len(question_words)

            disease_terms = re.findall(r"[a-zA-Z]+", question_clean)
            for term in disease_terms:
                if len(term) > 4 and term in qa_question:
                    score += 0.2

            if score > best_score:
                best_score = score
                best_match = row

    return best_match if best_score > 0.4 else None


def predict_disease_from_symptoms(
    symptoms_list: List[str], augmented_df: Optional[pd.DataFrame]
) -> Optional[Tuple[str, float, np.ndarray]]:
    if augmented_df is None:
        return None

    try:
        symptom_columns = [col for col in augmented_df.columns if col not in ("diseases", "diseases_clean")]
        query_vector = np.zeros(len(symptom_columns))

        for symptom in symptoms_list:
            symptom_clean = symptom.lower().strip().replace(" ", "_")
            if symptom_clean in symptom_columns:
                idx = symptom_columns.index(symptom_clean)
                query_vector[idx] = 1

        similarities = []
        for _, row in augmented_df.iterrows():
            try:
                disease_vector = row[symptom_columns].values.astype(float)
                similarity = cosine_similarity([query_vector], [disease_vector])[0][0]
                similarities.append((row["diseases"], similarity, disease_vector))
            except Exception:
                continue

        if similarities:
            return max(similarities, key=lambda item: item[1])
        return None
    except Exception as exc:
        logger.warning("Disease prediction failed: %s", exc)
        return None


def get_disease_symptoms(
    disease_name: Optional[str],
    symptoms_df: Optional[pd.DataFrame],
    augmented_df: Optional[pd.DataFrame],
) -> List[str]:
    if not disease_name:
        return []

    disease_clean = disease_name.lower().strip()
    symptoms: List[str] = []

    if augmented_df is not None and "diseases_clean" in augmented_df.columns:
        try:
            aug_match = augmented_df[augmented_df["diseases_clean"] == disease_clean]
            if not aug_match.empty:
                symptom_columns = [col for col in augmented_df.columns if col not in ("diseases", "diseases_clean")]
                for col in symptom_columns:
                    if col in aug_match.columns and not aug_match.empty and aug_match[col].values[0] == 1:
                        symptoms.append(col.replace("_", " "))
        except Exception as exc:
            logger.debug("Augmented dataset symptom lookup failed: %s", exc)

    if not symptoms and symptoms_df is not None and "Disease_clean" in symptoms_df.columns:
        try:
            symptom_match = symptoms_df[symptoms_df["Disease_clean"] == disease_clean]
            if not symptom_match.empty:
                for col in symptom_match.columns:
                    if col.startswith("Symptom_"):
                        value = symptom_match[col].values[0]
                        if pd.notna(value):
                            symptom = str(value).strip()
                            if symptom and symptom.lower() != "nan":
                                symptoms.append(symptom)
        except Exception as exc:
            logger.debug("Symptoms dataset lookup failed: %s", exc)

    return symptoms


def get_disease_precautions(disease_name: Optional[str], precautions_df: Optional[pd.DataFrame]) -> List[str]:
    if precautions_df is None or not disease_name:
        return []

    disease_clean = disease_name.lower().strip()
    precautions: List[str] = []

    try:
        if "Disease_clean" in precautions_df.columns:
            match = precautions_df[precautions_df["Disease_clean"] == disease_clean]
            if not match.empty:
                for col in ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]:
                    value = match[col].values[0] if col in match.columns else None
                    if pd.notna(value):
                        precaution = str(value).strip()
                        if precaution and precaution.lower() != "nan":
                            precautions.append(precaution)
    except Exception as exc:
        logger.debug("Precaution lookup failed: %s", exc)

    return precautions


def get_disease_description(disease_name: Optional[str], faq_df: Optional[pd.DataFrame]) -> Optional[str]:
    if faq_df is None or not disease_name:
        return None

    disease_clean = disease_name.lower().strip()
    try:
        if "question_clean" in faq_df.columns:
            relevant = faq_df[faq_df["question_clean"].str.contains(disease_clean, na=False)]
            if not relevant.empty:
                return relevant.iloc[0]["answer"]
    except Exception as exc:
        logger.debug("Description lookup failed: %s", exc)
    return None


def classify_input_type(user_input: str) -> str:
    if not user_input:
        return "question"

    user_input_lower = user_input.lower().strip()

    question_patterns = [
        r"what (are|is)",
        r"how (to|do|can)",
        r"why (is|are)",
        r"when (should|do)",
        r"where (can|do)",
        r"who (should|can)",
        r"can you",
        r"could you",
        r"would you",
        r"explain",
        r"tell me about",
    ]

    is_question = any(re.search(pattern, user_input_lower) for pattern in question_patterns)
    has_question_mark = "?" in user_input
    has_commas = "," in user_input
    is_short_phrase = len(user_input.split()) <= 5 and not is_question

    if has_question_mark or is_question:
        return "question"
    if has_commas and is_short_phrase:
        return "symptoms"
    if not is_question and len(user_input.split()) <= 3:
        return "disease"
    return "question"


def process_user_input(
    user_input: str,
    precautions_df: Optional[pd.DataFrame],
    symptoms_df: Optional[pd.DataFrame],
    faq_df: Optional[pd.DataFrame],
    augmented_df: Optional[pd.DataFrame],
    humanqa_df: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    response: Dict[str, Any] = {
        "type": None,
        "disease": None,
        "confidence": 0.0,
        "symptoms": [],
        "precautions": [],
        "description": None,
        "faq_question": None,
        "faq_answer": None,
    }

    if not user_input:
        return response

    input_type = classify_input_type(user_input)
    response["type"] = input_type

    try:
        if input_type == "question":
            disease_match = re.search(r"(?:symptoms|signs|causes|treatment|of|for)\s+([^?]+)", user_input.lower())
            if disease_match:
                potential_disease = disease_match.group(1).strip()
                if "symptom" in user_input.lower() or "sign" in user_input.lower():
                    symptoms = get_disease_symptoms(potential_disease, symptoms_df, augmented_df)
                    if symptoms:
                        response.update(
                            {
                                "type": "disease",
                                "disease": potential_disease,
                                "confidence": 0.95,
                                "symptoms": symptoms,
                                "precautions": get_disease_precautions(potential_disease, precautions_df),
                                "description": get_disease_description(potential_disease, faq_df),
                            }
                        )
                        return response

            faq_match = find_question_answer(user_input, faq_df, humanqa_df)
            if faq_match is not None:
                response["faq_question"] = faq_match.get("question")
                response["faq_answer"] = faq_match.get("answer")

        elif input_type == "symptoms":
            symptoms_list = [symptom.strip() for symptom in user_input.split(",")]
            prediction = predict_disease_from_symptoms(symptoms_list, augmented_df)
            if prediction:
                disease_name, confidence, _ = prediction
                response.update(
                    {
                        "disease": disease_name,
                        "confidence": confidence,
                        "symptoms": get_disease_symptoms(disease_name, symptoms_df, augmented_df),
                        "precautions": get_disease_precautions(disease_name, precautions_df),
                        "description": get_disease_description(disease_name, faq_df),
                    }
                )

        elif input_type == "disease":
            response.update(
                {
                    "disease": user_input,
                    "confidence": 0.95,
                    "symptoms": get_disease_symptoms(user_input, symptoms_df, augmented_df),
                    "precautions": get_disease_precautions(user_input, precautions_df),
                    "description": get_disease_description(user_input, faq_df),
                }
            )
    except Exception as exc:
        logger.exception("Error processing chatbot input: %s", exc)

    return response


def format_chatbot_reply(user_input: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Create a serialisable payload for the frontend UI."""

    payload: Dict[str, Any] = {
        "input": user_input,
        "analysis": analysis,
    }

    if analysis.get("type") == "question" and not analysis.get("faq_answer"):
        payload["message"] = (
            "I could not find a specific answer in the knowledge base. "
            "Please rephrase your question or provide more clinical detail."
        )
    elif analysis.get("type") in {"symptoms", "disease"} and not analysis.get("disease"):
        payload["message"] = (
            "I was unable to map those details to a known condition. Consider "
            "providing additional symptoms or verify the spelling."
        )

    return payload


def get_chatbot_response(user_input: str) -> Dict[str, Any]:
    """Public entry-point used by the Flask routes."""

    datasets = load_datasets()
    if not datasets or all(df is None for df in datasets):
        raise RuntimeError("Chatbot currently on the upgradation/maintenance")

    precautions_df, symptoms_df, faq_df, augmented_df, humanqa_df = datasets
    analysis = process_user_input(user_input, precautions_df, symptoms_df, faq_df, augmented_df, humanqa_df)
    return format_chatbot_reply(user_input, analysis)


__all__ = ["get_chatbot_response", "load_datasets"]


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    for sample in [
        "What are the symptoms of malaria?",
        "fever, headache, nausea",
        "diabetes",
    ]:
        reply = get_chatbot_response(sample)
        print(f"\nUser: {sample}\nBot: {reply}\n")
