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
    assert directory["doctors"]
    assert directory["doctors"][0]["name"] == DOCTORS_DATA[0]["name"]
    assert directory["doctors"][0]["image_url"].startswith("/static/assets/doctors/")


def test_search_providers_matches_case_insensitive():
    results = search_providers("tmu")
    assert any("tmu" in hospital["name"].lower() for hospital in results["hospitals"])
    assert any("tmu" in doctor["name"].lower() for doctor in results["doctors"]) is False


def test_doctors_loaded_from_images_have_core_fields():
    doctors = get_doctors_data()
    assert doctors
    sample = doctors[0]
    assert sample["name"].lower().startswith("dr")
    assert sample["contact"]
    assert sample["specialization"]
    assert sample["image_url"].startswith("/static/assets/doctors/")
