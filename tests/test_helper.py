import pytest

from helper import fetch_gemini_recommendations


@pytest.mark.parametrize(
    "disease,risk,expected_level",
    [
        ("Diabetes", 10, "low"),
        ("coronary artery disease", 55, "medium"),
        ("Anemia", 90, "high"),
        ("Tuberculosis", 42, "medium"),
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
