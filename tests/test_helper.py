import pytest

from helper import fetch_gemini_recommendations


@pytest.mark.parametrize(
    "disease,risk,expected_level",
    [
        ("Diabetes", 10, "low"),
        ("heart disease", 55, "medium"),
        ("FEVER", 90, "high"),
    ],
)
def test_fetch_gemini_recommendations_levels(disease, risk, expected_level):
    result = fetch_gemini_recommendations(disease, risk)
    assert result["Risk Level"] == expected_level
    assert len(result["prevention_measures"]) > 0
    assert len(result["medicine_suggestions"]) > 0


def test_fetch_gemini_recommendations_unknown_disease():
    result = fetch_gemini_recommendations("unknown", 40)
    assert result == {
        "Risk Level": "medium",
        "prevention_measures": [],
        "medicine_suggestions": [],
    }
