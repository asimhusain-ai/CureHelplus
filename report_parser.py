"""Utilities for extracting structured data from uploaded medical reports."""
from __future__ import annotations

import os
import re
import tempfile
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
from PyPDF2 import PdfReader

REPORT_ALLOWED_EXTENSIONS = {".csv", ".pdf", ".xls", ".xlsx"}

FIELD_ALIASES: Dict[str, Dict[str, Iterable[str]]] = {
    "diabetes": {
        "gender": ["gender", "sex"],
        "age": ["age", "patientage"],
        "bmi": ["bmi", "bodymassindex", "body_mass_index"],
        "glucose": ["glucose", "fastingglucose", "bloodglucose"],
        "blood_pressure": ["bloodpressure", "systolic", "bp"],
        "pregnancies": ["pregnancies", "pregnancycount"],
        "skin_thickness": ["skinthickness", "skinfold"],
        "insulin": ["insulin", "fastinginsulin"],
        "diabetes_pedigree_function": ["diabetespedigreefunction", "dpf", "pedigree"],
    },
    "heart": {
        "gender": ["gender", "sex"],
        "age": ["age", "patientage"],
        "resting_bp": ["restingbp", "restingbloodpressure", "bloodpressure"],
        "cholesterol": ["cholesterol", "serumcholesterol", "chol"],
        "chest_pain_type": ["chestpaintype", "cpt", "cp"],
        "fasting_bs": ["fastingbs", "fastingbloodsugar", "fbs"],
        "resting_ecg": ["restingecg", "ecg"],
        "max_heart_rate": ["maxheartrate", "maxhr", "heartrate"],
        "exercise_angina": ["exerciseangina", "exang"],
        "st_depression": ["stdepression", "oldpeak"],
        "slope": ["slope", "stslope"],
        "major_vessels": ["majorvessels", "ca"],
        "thal": ["thal", "thalassemia"],
    },
    "anemia": {
        "gender": ["gender", "sex"],
        "rbc": ["rbc", "redbloodcells"],
        "hemoglobin": ["hemoglobin", "hb"],
        "hematocrit": ["hematocrit", "hct"],
        "mcv": ["mcv"],
        "mch": ["mch"],
        "mchc": ["mchc"],
        "wbc": ["wbc", "whitebloodcells"],
        "platelets": ["platelets", "plateletcount"],
        "rdw": ["rdw"],
        "pdw": ["pdw"],
        "pct": ["pct", "plateletcrit"],
        "lymphocytes": ["lymphocytes", "lymphs"],
        "neutrophils_pct": ["neutrophilspct", "neutrophilspercent"],
        "neutrophils_num": ["neutrophilsnum", "neutrophilsabsolute", "neutrophils"],
    },
}

NUMERIC_FIELDS = {
    "diabetes": {
        "age",
        "bmi",
        "glucose",
        "blood_pressure",
        "pregnancies",
        "skin_thickness",
        "insulin",
        "diabetes_pedigree_function",
    },
    "heart": {
        "age",
        "resting_bp",
        "cholesterol",
        "chest_pain_type",
        "max_heart_rate",
        "st_depression",
        "slope",
        "major_vessels",
        "thal",
    },
    "anemia": {
        "rbc",
        "hemoglobin",
        "hematocrit",
        "mcv",
        "mch",
        "mchc",
        "wbc",
        "platelets",
        "rdw",
        "pdw",
        "pct",
        "lymphocytes",
        "neutrophils_pct",
        "neutrophils_num",
    },
}

INTEGER_FIELDS = {
    "pregnancies",
    "chest_pain_type",
    "major_vessels",
    "thal",
}

BOOLEAN_FIELDS = {
    "fasting_bs",
    "exercise_angina",
}

BLOOD_PRESSURE_CHOICES = {
    "normal": "Normal",
    "low": "Low",
    "high": "High",
}

MEDICATION_CHOICES = {
    "none": "None",
    "na": "None",
    "ibuprofen": "Ibuprofen",
    "paracetamol": "Paracetamol",
    "acetaminophen": "Paracetamol",
    "other": "Other",
}

FIELD_LOOKUP: Dict[str, List[Tuple[str, str]]] = {}
for disease, field_map in FIELD_ALIASES.items():
    for field_name, aliases in field_map.items():
        for alias in set(aliases) | {field_name}:
            normalized = re.sub(r"[^a-z0-9]", "", alias.lower())
            if normalized:
                FIELD_LOOKUP.setdefault(normalized, []).append((disease, field_name))


def parse_medical_report(file_storage) -> Dict[str, Dict[str, Any]]:
    """Parse an uploaded medical report and return structured values per disease."""

    filename = (file_storage.filename or "").strip()
    extension = os.path.splitext(filename)[1].lower()
    if not extension:
        raise ValueError("Unable to determine report format.")
    if extension not in REPORT_ALLOWED_EXTENSIONS:
        raise ValueError("Unsupported report format. Allowed formats: CSV, PDF, XLS, XLSX.")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as handle:
            temp_path = handle.name
            file_storage.stream.seek(0)
            handle.write(file_storage.read())

        if extension == ".pdf":
            raw_records = _parse_pdf(temp_path)
        elif extension in {".xls", ".xlsx"}:
            raw_records = _parse_excel(temp_path)
        else:
            raw_records = _parse_csv(temp_path)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Failed to parse medical report: {exc}") from exc
    finally:
        try:
            file_storage.stream.seek(0)
        except OSError:
            pass
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

    return _map_records_to_forms(raw_records)


def _parse_csv(path: str) -> Dict[str, Any]:
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, sep=None, engine="python")
    return _dataframe_to_records(df)


def _parse_excel(path: str) -> Dict[str, Any]:
    df = pd.read_excel(path)
    return _dataframe_to_records(df)


def _parse_pdf(path: str) -> Dict[str, Any]:
    reader = PdfReader(path)
    text_blocks = []
    for page in reader.pages:
        extracted = page.extract_text() or ""
        if extracted:
            text_blocks.append(extracted)
    combined = "\n".join(text_blocks)
    return _text_to_records(combined)


def _dataframe_to_records(df: pd.DataFrame) -> Dict[str, Any]:
    records: Dict[str, Any] = {}
    if df.empty:
        return records

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join(str(part) for part in col if str(part) != "nan").strip() for col in df.columns]

    first_row = df.iloc[0]
    for column in df.columns:
        value = first_row[column]
        if pd.isna(value):
            continue
        normalized = _normalize_key(column)
        if normalized and normalized not in records:
            records[normalized] = value

    for _, row in df.iterrows():
        if len(row) < 2:
            continue
        key_candidate = str(row.iloc[0]).strip()
        value_candidate = row.iloc[1]
        if not key_candidate or pd.isna(value_candidate):
            continue
        normalized = _normalize_key(key_candidate)
        if normalized and normalized not in records:
            records[normalized] = value_candidate

    return records


def _text_to_records(text: str) -> Dict[str, Any]:
    records: Dict[str, Any] = {}
    pending_key: str | None = None

    def store_record(raw_key: str, raw_value: str) -> None:
        normalized = _normalize_key(raw_key)
        value = raw_value.strip()
        if not normalized or not value or normalized in records:
            return
        records[normalized] = value

    for line in text.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue

        if ":" in cleaned or "=" in cleaned:
            if ":" in cleaned:
                key, value = cleaned.split(":", 1)
            else:
                key, value = cleaned.split("=", 1)
            key = key.strip()
            value = value.strip()
            if value:
                store_record(key, value)
                pending_key = None
            else:
                pending_key = key
            continue

        match_inline = re.match(r"([A-Za-z][A-Za-z0-9 /_%()-]+)\s+([-+]?\d+[\d.,]*)", cleaned)
        if match_inline:
            store_record(match_inline.group(1), match_inline.group(2))
            pending_key = None
            continue

        if pending_key is not None:
            store_record(pending_key, cleaned)
            pending_key = None
            continue

        if re.match(r"^[A-Za-z][A-Za-z0-9 /_%()-]+$", cleaned):
            pending_key = cleaned
            continue

    return records


def _map_records_to_forms(records: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    mapped: Dict[str, Dict[str, Any]] = {disease: {} for disease in FIELD_ALIASES}
    for normalized_key, raw_value in records.items():
        targets = _resolve_field_targets(normalized_key)
        if not targets:
            continue
        for disease, field_name in targets:
            processed_value = _normalize_field_value(disease, field_name, raw_value)
            if processed_value is not None:
                mapped[disease][field_name] = processed_value
    return {disease: values for disease, values in mapped.items() if values}


def _normalize_field_value(disease: str, field: str, value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, float) and pd.isna(value):
        return None

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None

    if field == "gender":
        return _normalize_gender(value)

    if field in BOOLEAN_FIELDS:
        return _normalize_boolean(value)

    if field in {"chest_pain_type", "resting_ecg", "slope", "thal"}:
        numeric = _coerce_numeric(value)
        if numeric is None:
            return None
        return str(int(round(numeric)))

    if disease in NUMERIC_FIELDS and field in NUMERIC_FIELDS[disease]:
        numeric_value = _coerce_numeric(value)
        if numeric_value is None:
            return None
        numeric_float = float(numeric_value)
        if field in INTEGER_FIELDS or numeric_float.is_integer():
            return int(round(numeric_float))
        return round(numeric_float, 4)

    return str(value)


def _normalize_boolean(value: Any) -> Any:
    normalized = str(value).strip().lower()
    if normalized in {"yes", "y", "true", "1", "present"}:
        return "Yes"
    if normalized in {"no", "n", "false", "0", "absent"}:
        return "No"
    return None


def _normalize_gender(value: Any) -> Any:
    normalized = str(value).strip().lower()
    if normalized in {"male", "m", "1"}:
        return "Male"
    if normalized in {"female", "f", "0"}:
        return "Female"
    if normalized:
        return normalized.title()
    return None


def _normalize_choice(value: Any, choices: Dict[str, str]) -> Any:
    normalized = _normalize_key(value)
    if not normalized:
        return None
    for key, label in choices.items():
        if normalized == key:
            return label
    return None


def _coerce_numeric(value: Any) -> Any:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(",", ".")
    matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not matches:
        return None
    try:
        return float(matches[0])
    except ValueError:
        return None


def _normalize_key(value: Any) -> str:
    text = str(value).lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = text.replace("µ", "u")
    text = text.replace("°", " ")
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"[^a-z0-9]", "", text)


def _resolve_field_targets(normalized_key: str) -> List[Tuple[str, str]]:
    direct = FIELD_LOOKUP.get(normalized_key)
    if direct:
        return direct

    best_match: List[Tuple[str, str]] = []
    best_len = 0
    for alias_key, targets in FIELD_LOOKUP.items():
        if alias_key in normalized_key and len(alias_key) > best_len:
            best_match = targets
            best_len = len(alias_key)
    return best_match