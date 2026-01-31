import io

import pytest
from werkzeug.datastructures import FileStorage

from report_parser import (
    _normalize_boolean,
    _normalize_field_value,
    _normalize_key,
    _resolve_field_targets,
    parse_medical_report,
)


def _make_filestorage(content: str, filename: str) -> FileStorage:
    return FileStorage(stream=io.BytesIO(content.encode("utf-8")), filename=filename)


def test_parse_medical_report_csv():
    csv_content = "Gender,Age,Glucose,BloodPressure,BMI\nMale,45,150,130,28.5\n"
    storage = _make_filestorage(csv_content, "report.csv")

    result = parse_medical_report(storage)
    diabetes_data = result["diabetes"]
    assert diabetes_data["age"] == 45
    assert diabetes_data["glucose"] == 150
    assert diabetes_data["blood_pressure"] == 130
    assert diabetes_data["bmi"] == 28.5


def test_parse_medical_report_invalid_extension():
    storage = _make_filestorage("", "report.txt")
    with pytest.raises(ValueError):
        parse_medical_report(storage)


def test_normalization_helpers():
    assert _normalize_key("Blood Pressure (mm Hg)") == "bloodpressure"
    assert _normalize_boolean("Yes") == "Yes"
    assert _normalize_boolean("0") == "No"

    targets = _resolve_field_targets("glucose")
    assert any(field == "glucose" for _, field in targets)

    value = _normalize_field_value("diabetes", "glucose", "150 mg/dL")
    assert value == 150

    unknown = _normalize_field_value("diabetes", "glucose", "not a number")
    assert unknown is None
