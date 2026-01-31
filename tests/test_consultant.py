from consultant import (
    DOCTORS_DATA,
    HOSPITALS_DATA,
    get_consultant_directory,
    get_doctors_data,
    get_hospitals_data,
    search_providers,
)


def test_get_hospitals_data_returns_copy():
    hospitals = get_hospitals_data()
    assert hospitals is not HOSPITALS_DATA
    original_len = len(HOSPITALS_DATA)
    hospitals.pop()
    assert len(HOSPITALS_DATA) == original_len


def test_get_doctors_data_returns_copy():
    doctors = get_doctors_data()
    assert doctors is not DOCTORS_DATA
    original_len = len(DOCTORS_DATA)
    doctors.pop()
    assert len(DOCTORS_DATA) == original_len


def test_get_consultant_directory_combines_sources():
    directory = get_consultant_directory()
    assert set(directory.keys()) == {"hospitals", "doctors"}
    assert directory["hospitals"][0]["name"] == HOSPITALS_DATA[0]["name"]
    assert directory["doctors"][0]["name"] == DOCTORS_DATA[0]["name"]


def test_search_providers_matches_case_insensitive():
    results = search_providers("apollo")
    assert any("apollo" in hospital["name"].lower() for hospital in results["hospitals"])
    assert any("apollo" in doctor["name"].lower() for doctor in results["doctors"]) is False
