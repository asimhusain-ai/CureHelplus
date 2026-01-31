import json

import numpy as np
import pytest

from profile_manager import ProfileManager


def test_profile_manager_add_and_get_profile(tmp_path):
    storage_path = tmp_path / "profiles.json"
    manager = ProfileManager(str(storage_path))

    profile = manager.add_profile({
        "name": "Alice",
        "age": 30,
        "contact": "1234567890",
        "address": "123 Street",
        "gender": "Female",
        "marital_status": "Single",
        "predictions": {"Diabetes": {"prob": np.float64(42.5)}},
    })

    assert profile["id"].startswith("user_")
    stored = manager.get_profile(profile["id"])
    assert stored["name"] == "Alice"
    assert stored["predictions"]["Diabetes"]["prob"] == 42.5

    with storage_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    assert data[0]["name"] == "Alice"


def test_profile_manager_update_and_delete(tmp_path):
    manager = ProfileManager(str(tmp_path / "profiles.json"))

    profile = manager.add_profile({
        "name": "Bob",
        "age": 40,
        "contact": "999999",
    })

    updated = manager.update_profile(profile["id"], {"contact": "111111"})
    assert updated["contact"] == "111111"
    assert manager.get_profile(profile["id"])["contact"] == "111111"

    manager.update_predictions(profile["id"], {"Heart": {"prob": 70}})
    assert manager.get_profile(profile["id"])["predictions"]["Heart"]["prob"] == 70

    assert manager.delete_profile(profile["id"])
    assert manager.get_profile(profile["id"]) is None


def test_profile_manager_search_and_upsert(tmp_path):
    manager = ProfileManager(str(tmp_path / "profiles.json"))

    first = manager.add_profile({"name": "Charlie", "age": 50})
    second = manager.add_profile({"name": "Charlotte", "age": 35})

    results = manager.search_profiles("char")
    assert {entry["id"] for entry in results} == {first["id"], second["id"]}

    upserted = manager.upsert_profile(first["id"], {"contact": "888"})
    assert upserted["contact"] == "888"

    brand_new = manager.upsert_profile(None, {"name": "Dave", "age": 28})
    assert brand_new["id"] != first["id"]
