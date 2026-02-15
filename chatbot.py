"""Rule-based medical chatbot utilities for the Flask UI."""
from __future__ import annotations

import io
import hashlib
import logging
import os
import re
import threading
import time
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zipfile import ZipFile

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CHAT_CACHE_TTL_SECONDS = int(os.getenv("CHAT_CACHE_TTL_SECONDS", "300"))
_CHAT_CACHE_LOCK = threading.Lock()
_CHAT_RESPONSE_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}


def _normalise_query(user_input: str) -> str:
    return " ".join((user_input or "").lower().strip().split())


@lru_cache(maxsize=1)
def load_datasets(
    data_dir: str = "bot_data",
    dataset_version: str | None = None,
) -> Tuple[
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
]:
    """Load and preprocess datasets used by the chatbot."""

    _ = dataset_version

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


def _dataset_version_token(data_dir: str = "bot_data") -> str:
    base_path = Path(data_dir)
    hash_input: List[str] = [str(base_path)]

    if base_path.is_file():
        try:
            stat = base_path.stat()
            hash_input.append(f"file:{base_path.name}:{stat.st_mtime_ns}:{stat.st_size}")
        except OSError:
            hash_input.append(f"missing:{base_path.name}")
    else:
        source_root = base_path if base_path.exists() else base_path.with_suffix(".zip")
        if source_root.exists() and source_root.is_dir():
            for file_path in sorted(source_root.rglob("*")):
                if not file_path.is_file():
                    continue
                try:
                    stat = file_path.stat()
                    rel = file_path.relative_to(source_root)
                    hash_input.append(f"dir:{rel.as_posix()}:{stat.st_mtime_ns}:{stat.st_size}")
                except OSError:
                    continue
        elif source_root.exists() and source_root.is_file():
            try:
                stat = source_root.stat()
                hash_input.append(f"zip:{source_root.name}:{stat.st_mtime_ns}:{stat.st_size}")
            except OSError:
                hash_input.append(f"zip-missing:{source_root.name}")
        else:
            hash_input.append("missing")

    joined = "|".join(hash_input)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


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


def _build_augmented_from_symptoms(symptoms_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if symptoms_df is None or symptoms_df.empty or "Disease" not in symptoms_df.columns:
        return None

    symptom_cols = [col for col in symptoms_df.columns if str(col).startswith("Symptom_")]
    if not symptom_cols:
        return None

    disease_to_symptoms: Dict[str, set] = {}
    for _, row in symptoms_df.iterrows():
        disease_name = str(row.get("Disease", "")).strip()
        if not disease_name:
            continue

        symptoms_set = disease_to_symptoms.setdefault(disease_name, set())
        for column in symptom_cols:
            value = row.get(column)
            if pd.notna(value):
                symptom = str(value).strip().lower().replace(" ", "_")
                if symptom and symptom != "nan":
                    symptoms_set.add(symptom)

    if not disease_to_symptoms:
        return None

    all_symptoms = sorted({symptom for values in disease_to_symptoms.values() for symptom in values})
    if not all_symptoms:
        return None

    rows: List[Dict[str, Any]] = []
    for disease, symptom_set in disease_to_symptoms.items():
        row: Dict[str, Any] = {"diseases": disease}
        for symptom in all_symptoms:
            row[symptom] = 1 if symptom in symptom_set else 0
        rows.append(row)

    return pd.DataFrame(rows)


@lru_cache(maxsize=4)
def _build_chat_indexes(data_dir: str = "bot_data", dataset_version: str | None = None) -> Dict[str, Any]:
    _ = dataset_version
    try:
        precautions_df, symptoms_df, faq_df, augmented_df, humanqa_df = load_datasets(data_dir, dataset_version)
    except TypeError:
        precautions_df, symptoms_df, faq_df, augmented_df, humanqa_df = load_datasets()

    if augmented_df is None:
        augmented_df = _build_augmented_from_symptoms(symptoms_df)
        if augmented_df is not None:
            augmented_df = clean_dataframe(augmented_df)
            if augmented_df is not None and "diseases" in augmented_df.columns:
                augmented_df["diseases_clean"] = augmented_df["diseases"].astype(str).str.lower().str.strip()

    qa_entries: List[Dict[str, Any]] = []
    for dataset in (faq_df, humanqa_df):
        if dataset is None or dataset.empty:
            continue

        for _, row in dataset.iterrows():
            question = str(row.get("question", "")).strip()
            answer = str(row.get("answer", "")).strip()
            if not question or not answer:
                continue

            question_clean = question.lower()
            qa_entries.append(
                {
                    "question": question,
                    "answer": answer,
                    "question_clean": question_clean,
                    "word_set": set(question_clean.split()),
                    "is_symptom": ("symptom" in question_clean or "sign" in question_clean),
                }
            )

    symptom_columns: List[str] = []
    disease_names: List[str] = []
    disease_matrix = np.empty((0, 0), dtype=np.float32)
    disease_matrix_norm = np.empty(0, dtype=np.float32)
    symptom_to_index: Dict[str, int] = {}

    if augmented_df is not None and not augmented_df.empty and "diseases" in augmented_df.columns:
        symptom_columns = [col for col in augmented_df.columns if col not in ("diseases", "diseases_clean")]
        disease_names = augmented_df["diseases"].astype(str).tolist()

        if symptom_columns:
            matrix = augmented_df[symptom_columns].fillna(0).to_numpy(dtype=np.float32, copy=True)
            disease_matrix = matrix
            disease_matrix_norm = np.linalg.norm(disease_matrix, axis=1)
            symptom_to_index = {name: index for index, name in enumerate(symptom_columns)}

    symptoms_map: Dict[str, List[str]] = {}
    if symptoms_df is not None and "Disease_clean" in symptoms_df.columns:
        symptom_dataset_columns = [col for col in symptoms_df.columns if col.startswith("Symptom_")]
        for _, row in symptoms_df.iterrows():
            disease_clean = str(row.get("Disease_clean", "")).strip()
            if not disease_clean:
                continue

            values = symptoms_map.setdefault(disease_clean, [])
            for col in symptom_dataset_columns:
                value = row.get(col)
                if pd.notna(value):
                    symptom = str(value).strip()
                    if symptom and symptom.lower() != "nan" and symptom not in values:
                        values.append(symptom)

    if augmented_df is not None and "diseases_clean" in augmented_df.columns and symptom_columns:
        for _, row in augmented_df.iterrows():
            disease_clean = str(row.get("diseases_clean", "")).strip()
            if not disease_clean:
                continue
            values = symptoms_map.setdefault(disease_clean, [])
            for column in symptom_columns:
                try:
                    is_present = float(row.get(column, 0)) == 1.0
                except Exception:
                    is_present = False
                if is_present:
                    symptom_name = column.replace("_", " ")
                    if symptom_name not in values:
                        values.append(symptom_name)

    precautions_map: Dict[str, List[str]] = {}
    if precautions_df is not None and "Disease_clean" in precautions_df.columns:
        for _, row in precautions_df.iterrows():
            disease_clean = str(row.get("Disease_clean", "")).strip()
            if not disease_clean:
                continue

            values = precautions_map.setdefault(disease_clean, [])
            for col in ("Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"):
                value = row.get(col)
                if pd.notna(value):
                    precaution = str(value).strip()
                    if precaution and precaution.lower() != "nan" and precaution not in values:
                        values.append(precaution)

    descriptions_map: Dict[str, str] = {}
    if faq_df is not None and "question_clean" in faq_df.columns and "answer" in faq_df.columns:
        for _, row in faq_df.iterrows():
            question_clean = str(row.get("question_clean", "")).strip()
            answer = str(row.get("answer", "")).strip()
            if question_clean and answer:
                descriptions_map[question_clean] = answer

    return {
        "datasets": (precautions_df, symptoms_df, faq_df, augmented_df, humanqa_df),
        "qa_entries": qa_entries,
        "symptom_columns": symptom_columns,
        "disease_names": disease_names,
        "disease_matrix": disease_matrix,
        "disease_matrix_norm": disease_matrix_norm,
        "symptom_to_index": symptom_to_index,
        "symptoms_map": symptoms_map,
        "precautions_map": precautions_map,
        "descriptions_map": descriptions_map,
    }


def _build_indexes_from_datasets(
    precautions_df: Optional[pd.DataFrame],
    symptoms_df: Optional[pd.DataFrame],
    faq_df: Optional[pd.DataFrame],
    augmented_df: Optional[pd.DataFrame],
    humanqa_df: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    if augmented_df is None:
        augmented_df = _build_augmented_from_symptoms(symptoms_df)
        if augmented_df is not None:
            augmented_df = clean_dataframe(augmented_df)
            if augmented_df is not None and "diseases" in augmented_df.columns:
                augmented_df["diseases_clean"] = augmented_df["diseases"].astype(str).str.lower().str.strip()

    qa_entries: List[Dict[str, Any]] = []
    for dataset in (faq_df, humanqa_df):
        if dataset is None or dataset.empty:
            continue
        for _, row in dataset.iterrows():
            question = str(row.get("question", "")).strip()
            answer = str(row.get("answer", "")).strip()
            if not question or not answer:
                continue
            question_clean = question.lower()
            qa_entries.append(
                {
                    "question": question,
                    "answer": answer,
                    "question_clean": question_clean,
                    "word_set": set(question_clean.split()),
                    "is_symptom": ("symptom" in question_clean or "sign" in question_clean),
                }
            )

    symptom_columns: List[str] = []
    disease_names: List[str] = []
    disease_matrix = np.empty((0, 0), dtype=np.float32)
    disease_matrix_norm = np.empty(0, dtype=np.float32)
    symptom_to_index: Dict[str, int] = {}
    if augmented_df is not None and not augmented_df.empty and "diseases" in augmented_df.columns:
        symptom_columns = [col for col in augmented_df.columns if col not in ("diseases", "diseases_clean")]
        disease_names = augmented_df["diseases"].astype(str).tolist()
        if symptom_columns:
            disease_matrix = augmented_df[symptom_columns].fillna(0).to_numpy(dtype=np.float32, copy=True)
            disease_matrix_norm = np.linalg.norm(disease_matrix, axis=1)
            symptom_to_index = {name: index for index, name in enumerate(symptom_columns)}

    symptoms_map: Dict[str, List[str]] = {}
    if symptoms_df is not None and "Disease_clean" in symptoms_df.columns:
        symptom_dataset_columns = [col for col in symptoms_df.columns if col.startswith("Symptom_")]
        for _, row in symptoms_df.iterrows():
            disease_clean = str(row.get("Disease_clean", "")).strip()
            if not disease_clean:
                continue
            values = symptoms_map.setdefault(disease_clean, [])
            for col in symptom_dataset_columns:
                value = row.get(col)
                if pd.notna(value):
                    symptom = str(value).strip()
                    if symptom and symptom.lower() != "nan" and symptom not in values:
                        values.append(symptom)

    if augmented_df is not None and "diseases_clean" in augmented_df.columns and symptom_columns:
        for _, row in augmented_df.iterrows():
            disease_clean = str(row.get("diseases_clean", "")).strip()
            if not disease_clean:
                continue
            values = symptoms_map.setdefault(disease_clean, [])
            for column in symptom_columns:
                try:
                    is_present = float(row.get(column, 0)) == 1.0
                except Exception:
                    is_present = False
                if is_present:
                    symptom_name = column.replace("_", " ")
                    if symptom_name not in values:
                        values.append(symptom_name)

    precautions_map: Dict[str, List[str]] = {}
    if precautions_df is not None and "Disease_clean" in precautions_df.columns:
        for _, row in precautions_df.iterrows():
            disease_clean = str(row.get("Disease_clean", "")).strip()
            if not disease_clean:
                continue
            values = precautions_map.setdefault(disease_clean, [])
            for col in ("Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"):
                value = row.get(col)
                if pd.notna(value):
                    precaution = str(value).strip()
                    if precaution and precaution.lower() != "nan" and precaution not in values:
                        values.append(precaution)

    descriptions_map: Dict[str, str] = {}
    if faq_df is not None and "question_clean" in faq_df.columns and "answer" in faq_df.columns:
        for _, row in faq_df.iterrows():
            question_clean = str(row.get("question_clean", "")).strip()
            answer = str(row.get("answer", "")).strip()
            if question_clean and answer:
                descriptions_map[question_clean] = answer

    return {
        "datasets": (precautions_df, symptoms_df, faq_df, augmented_df, humanqa_df),
        "qa_entries": qa_entries,
        "symptom_columns": symptom_columns,
        "disease_names": disease_names,
        "disease_matrix": disease_matrix,
        "disease_matrix_norm": disease_matrix_norm,
        "symptom_to_index": symptom_to_index,
        "symptoms_map": symptoms_map,
        "precautions_map": precautions_map,
        "descriptions_map": descriptions_map,
    }


def find_question_answer(
    question: str,
    *qa_sources: Optional[pd.DataFrame],
    qa_entries: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    if qa_entries is None and qa_sources:
        first = qa_sources[0]
        if isinstance(first, list):
            qa_entries = first
        else:
            qa_entries = _build_indexes_from_datasets(None, None, qa_sources[0] if len(qa_sources) > 0 else None, None, qa_sources[1] if len(qa_sources) > 1 else None).get("qa_entries")

    if not qa_entries:
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

    best_match: Optional[Dict[str, Any]] = None
    best_score = 0.0

    question_words = set(question_clean.split())
    disease_terms = [term for term in re.findall(r"[a-zA-Z]+", question_clean) if len(term) > 4]

    for entry in qa_entries:
        qa_question = entry["question_clean"]
        if not qa_question:
            continue

        score = 0.0

        if is_symptom_question and entry["is_symptom"]:
            score += 0.3

        qa_words = entry["word_set"]
        if question_words and qa_words:
            overlap = len(question_words.intersection(qa_words))
            score += overlap / len(question_words)

        for term in disease_terms:
            if term in qa_question:
                score += 0.2

        if score > best_score:
            best_score = score
            best_match = entry

    return best_match if best_score > 0.4 else None


def predict_disease_from_symptoms(
    symptoms_list: List[str], indexes: Optional[Dict[str, Any]]
) -> Optional[Tuple[str, float, np.ndarray]]:
    if isinstance(indexes, pd.DataFrame):
        indexes = _build_indexes_from_datasets(None, None, None, indexes, None)

    if not indexes:
        return None

    try:
        symptom_columns = indexes.get("symptom_columns") or []
        disease_names = indexes.get("disease_names") or []
        disease_matrix = indexes.get("disease_matrix")
        disease_matrix_norm = indexes.get("disease_matrix_norm")
        symptom_to_index = indexes.get("symptom_to_index") or {}

        if not symptom_columns or disease_matrix is None or len(disease_names) == 0:
            return None

        query_vector = np.zeros(len(symptom_columns), dtype=np.float32)

        for symptom in symptoms_list:
            symptom_clean = symptom.lower().strip().replace(" ", "_")
            idx = symptom_to_index.get(symptom_clean)
            if idx is not None:
                query_vector[idx] = 1

        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            return None

        denominator = np.maximum(disease_matrix_norm * query_norm, 1e-8)
        similarities = np.dot(disease_matrix, query_vector) / denominator

        if similarities.size > 0:
            best_idx = int(np.argmax(similarities))
            return disease_names[best_idx], float(similarities[best_idx]), disease_matrix[best_idx]
        return None
    except Exception as exc:
        logger.warning("Disease prediction failed: %s", exc)
        return None


def get_disease_symptoms(
    disease_name: Optional[str],
    symptoms_map: Optional[Dict[str, List[str]]],
    augmented_df: Optional[pd.DataFrame] = None,
) -> List[str]:
    if not disease_name:
        return []

    if isinstance(symptoms_map, pd.DataFrame):
        indexes = _build_indexes_from_datasets(None, symptoms_map, None, augmented_df, None)
        symptoms_map = indexes.get("symptoms_map")

    disease_clean = disease_name.lower().strip()
    return list((symptoms_map or {}).get(disease_clean, []))


def get_disease_precautions(disease_name: Optional[str], precautions_map: Optional[Dict[str, List[str]]]) -> List[str]:
    if not disease_name:
        return []
    if isinstance(precautions_map, pd.DataFrame):
        precautions_map = _build_indexes_from_datasets(precautions_map, None, None, None, None).get("precautions_map")
    disease_clean = disease_name.lower().strip()
    return list((precautions_map or {}).get(disease_clean, []))


def get_disease_description(disease_name: Optional[str], descriptions_map: Optional[Dict[str, str]]) -> Optional[str]:
    if not disease_name:
        return None

    if isinstance(descriptions_map, pd.DataFrame):
        descriptions_map = _build_indexes_from_datasets(None, None, descriptions_map, None, None).get("descriptions_map")

    disease_clean = disease_name.lower().strip()
    for question, answer in (descriptions_map or {}).items():
        if disease_clean in question:
            return answer
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
    precautions_df: Optional[pd.DataFrame] = None,
    symptoms_df: Optional[pd.DataFrame] = None,
    faq_df: Optional[pd.DataFrame] = None,
    augmented_df: Optional[pd.DataFrame] = None,
    humanqa_df: Optional[pd.DataFrame] = None,
    indexes: Optional[Dict[str, Any]] = None,
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
        if indexes is None:
            indexes = _build_indexes_from_datasets(precautions_df, symptoms_df, faq_df, augmented_df, humanqa_df)

        qa_entries = (indexes or {}).get("qa_entries")
        symptoms_map = (indexes or {}).get("symptoms_map")
        precautions_map = (indexes or {}).get("precautions_map")
        descriptions_map = (indexes or {}).get("descriptions_map")

        if input_type == "question":
            disease_match = re.search(r"(?:symptoms|signs|causes|treatment|of|for)\s+([^?]+)", user_input.lower())
            if disease_match:
                potential_disease = disease_match.group(1).strip()
                if "symptom" in user_input.lower() or "sign" in user_input.lower():
                    symptoms = get_disease_symptoms(potential_disease, symptoms_map)
                    if symptoms:
                        response.update(
                            {
                                "type": "disease",
                                "disease": potential_disease,
                                "confidence": 0.95,
                                "symptoms": symptoms,
                                "precautions": get_disease_precautions(potential_disease, precautions_map),
                                "description": get_disease_description(potential_disease, descriptions_map),
                            }
                        )
                        return response

            faq_match = find_question_answer(user_input, qa_entries=qa_entries)
            if faq_match is not None:
                response["faq_question"] = faq_match.get("question")
                response["faq_answer"] = faq_match.get("answer")

        elif input_type == "symptoms":
            symptoms_list = [symptom.strip() for symptom in user_input.split(",")]
            prediction = predict_disease_from_symptoms(symptoms_list, indexes)
            if prediction:
                disease_name, confidence, _ = prediction
                response.update(
                    {
                        "disease": disease_name,
                        "confidence": confidence,
                        "symptoms": get_disease_symptoms(disease_name, symptoms_map),
                        "precautions": get_disease_precautions(disease_name, precautions_map),
                        "description": get_disease_description(disease_name, descriptions_map),
                    }
                )

        elif input_type == "disease":
            response.update(
                {
                    "disease": user_input,
                    "confidence": 0.95,
                    "symptoms": get_disease_symptoms(user_input, symptoms_map),
                    "precautions": get_disease_precautions(user_input, precautions_map),
                    "description": get_disease_description(user_input, descriptions_map),
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

    dataset_version = _dataset_version_token("bot_data")
    try:
        datasets = load_datasets("bot_data", dataset_version)
    except TypeError:
        datasets = load_datasets()
    if not datasets or all(df is None for df in datasets):
        raise RuntimeError("Chatbot currently on the upgradation/maintenance")

    key = _normalise_query(user_input)
    if key and CHAT_CACHE_TTL_SECONDS > 0:
        with _CHAT_CACHE_LOCK:
            cached = _CHAT_RESPONSE_CACHE.get(key)
            if cached and (time.time() - cached[0] <= CHAT_CACHE_TTL_SECONDS):
                return deepcopy(cached[1])

    try:
        indexes = _build_chat_indexes("bot_data", dataset_version)
    except TypeError:
        indexes = _build_chat_indexes()

    analysis = process_user_input(user_input, indexes=indexes)
    reply = format_chatbot_reply(user_input, analysis)

    if key and CHAT_CACHE_TTL_SECONDS > 0:
        with _CHAT_CACHE_LOCK:
            _CHAT_RESPONSE_CACHE[key] = (time.time(), deepcopy(reply))
            if len(_CHAT_RESPONSE_CACHE) > 500:
                oldest_key = min(_CHAT_RESPONSE_CACHE.items(), key=lambda item: item[1][0])[0]
                _CHAT_RESPONSE_CACHE.pop(oldest_key, None)

    return reply


__all__ = ["get_chatbot_response", "load_datasets"]


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    for sample in [
        "What are the symptoms of malaria?",
        "diabetes",
    ]:
        reply = get_chatbot_response(sample)
        print(f"\nUser: {sample}\nBot: {reply}\n")
