"""Consultant directory utilities for CureHelp+."""
from __future__ import annotations

import os
import hashlib
import re
from functools import lru_cache
from typing import Dict, List
from urllib.parse import quote

def _build_maps_url(query: str) -> str:
    return f"https://maps.google.com/?q={quote(query)}"


def _hospital_image_url(filename: str) -> str:
    return f"/static/assets/hospitals/{quote(filename, safe='')}"


def _doctor_image_url(filename: str) -> str:
    return f"/static/assets/doctors/{quote(filename, safe='')}"


def _normalize_asset_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _resolve_image_filename(filenames: List[str], *candidates: str) -> str:
    indexed = [
        (_normalize_asset_label(os.path.splitext(filename)[0]), filename)
        for filename in filenames
    ]
    for candidate in candidates:
        token = _normalize_asset_label(candidate or "")
        if not token:
            continue
        for normalized_name, filename in indexed:
            if token == normalized_name or token in normalized_name or normalized_name in token:
                return filename
    return ""


@lru_cache(maxsize=1)
def _hospital_image_filenames(asset_version: str = "") -> List[str]:
    _ = asset_version
    base_dir = os.path.dirname(os.path.abspath(__file__))
    hospitals_dir = os.path.join(base_dir, "static", "assets", "hospitals")
    if not os.path.isdir(hospitals_dir):
        return []
    return [entry for entry in os.listdir(hospitals_dir) if entry and not entry.startswith(".")]


def _resolve_hospital_image_url(*keywords: str, asset_version: str = "") -> str:
    filename = _resolve_image_filename(_hospital_image_filenames(asset_version), *keywords)
    return _hospital_image_url(filename) if filename else ""


@lru_cache(maxsize=1)
def _doctor_image_filenames(asset_version: str = "") -> List[str]:
    _ = asset_version
    base_dir = os.path.dirname(os.path.abspath(__file__))
    doctors_dir = os.path.join(base_dir, "static", "assets", "doctors")
    if not os.path.isdir(doctors_dir):
        return []
    return [entry for entry in os.listdir(doctors_dir) if entry and not entry.startswith(".")]


def _resolve_doctor_image_url(*candidates: str, asset_version: str = "") -> str:
    filename = _resolve_image_filename(_doctor_image_filenames(asset_version), *candidates)
    return _doctor_image_url(filename) if filename else ""


def _asset_version_token() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    asset_dirs = [
        os.path.join(base_dir, "static", "assets", "hospitals"),
        os.path.join(base_dir, "static", "assets", "doctors"),
    ]

    rows: List[str] = []
    for directory in asset_dirs:
        if not os.path.isdir(directory):
            rows.append(f"missing:{directory}")
            continue
        for entry in sorted(os.listdir(directory), key=str.lower):
            if not entry or entry.startswith("."):
                continue
            path = os.path.join(directory, entry)
            try:
                stat = os.stat(path)
                rows.append(f"{os.path.basename(directory)}:{entry}:{stat.st_mtime_ns}:{stat.st_size}")
            except OSError:
                continue

    digest_source = "|".join(rows)
    return hashlib.sha1(digest_source.encode("utf-8")).hexdigest()


@lru_cache(maxsize=8)
def _consultant_directory_cached(asset_version: str) -> Dict[str, List[Dict[str, str]]]:
    hospitals = _build_hospitals_data(asset_version)
    doctors = _build_doctors_data(asset_version)
    return {"hospitals": hospitals, "doctors": doctors}


_HOSPITALS_BASE_DATA: List[Dict[str, str]] = [
    {
        "name": "Cosmos Hospital",
        "address": "00, Cosmos Hospital, Kanth Road Moradabad-244504, Uttar Praesh",
        "contact": "08460223572",
        "speciality": "Multi-Speciality",
        "location_url": _build_maps_url("Cosmos Hospital Kanth Road Moradabad"),
        "website_url": "",
        "image_match": ["Cosmos Hospital"],
    },
    {
        "name": "Crest Hospital",
        "address": "Pream Nagar, Kanth Road, Civil Lines, Moradabad-244001, Uttar Pradesh",
        "contact": "07942694997",
        "speciality": "Multi-Speciality",
        "location_url": _build_maps_url("Crest Hospital Prem Nagar Kanth Road Civil Lines Moradabad"),
        "website_url": "",
        "image_match": ["Crest Hospital"],
    },
    {
        "name": "Dhanwantri Superspecialty Hospital Moradabad",
        "address": "behind Head Post Office, Sarai Khalsa, Civil Lines, Moradabad, Uttar Pradesh 244001, India",
        "contact": "07302960464",
        "speciality": "Super Speciality",
        "location_url": _build_maps_url("Dhanwantri Superspecialty Hospital Moradabad"),
        "website_url": "",
        "image_match": ["Dhanwantri Superspecialty Hospital"],
    },
    {
        "name": "DMR Hospital",
        "address": "DM (Neuro) Fellow Child Neurology",
        "contact": "07411855875",
        "speciality": "Neurology",
        "location_url": _build_maps_url("DMR Hospital Moradabad"),
        "website_url": "",
        "image_match": ["DMR Hospital"],
    },
    {
        "name": "GEM Hospital (Superspeciality)",
        "address": "Rampur Rd, near Prem Wonderland, Moradabad, Uttar Pradesh 244001, India",
        "contact": "09068621241",
        "speciality": "Super Speciality",
        "location_url": _build_maps_url("GEM Hospital Superspeciality Rampur Road Moradabad"),
        "website_url": "",
        "image_match": ["GEM Hospital", "GEM Hospital (Superspeciality)"],
    },
    {
        "name": "Kalra Dental and Facial Surgery Centre",
        "address": "B 71, Gandhi Nagar, Behind Old Road Ways Bus Stand, Shani Mandir Road, Gandhi Nagar, Moradabad-244001, Uttar Pradesh",
        "contact": "09307253738",
        "speciality": "Dental & Facial Surgery",
        "location_url": _build_maps_url("Kalra Dental and Facial Surgery Centre Moradabad"),
        "website_url": "",
        "image_match": ["Kalra Dental and Facial Surgery Centre"],
    },
    {
        "name": "M I Hospital",
        "address": "Sirswan Gour Sirswan Doraha, Near Indian Petrol Pump, Kashipur Road,Pipalsana, Moradabad-244402,Uttar Pradesh",
        "contact": "09972850979",
        "speciality": "Multi-Speciality",
        "location_url": _build_maps_url("M I Hospital Kashipur Road Pipalsana Moradabad"),
        "website_url": "",
        "image_match": ["M I Hospital"],
    },
    {
        "name": "Moradabad Medicity Hospital",
        "address": "Gandhi Nagar, Moradabad, Uttar Pradesh 244001, India",
        "contact": "9719145333",
        "speciality": "Multi-Speciality",
        "location_url": _build_maps_url("Moradabad Medicity Hospital Gandhi Nagar Moradabad"),
        "website_url": "",
        "image_match": ["Moradabad Medicity Hospital"],
    },
    {
        "name": "RSD Hospital and Research Center",
        "address": "Ram Ganga Vihar Phase 2, Moradabad, Uttar Pradesh 24410",
        "contact": "5912970349",
        "speciality": "Multispecialty Hospital, Knee and Hip Replacement, Dental Hospital, Dialysis Center",
        "location_url": _build_maps_url("RSD Hospital and Research Center Ram Ganga Vihar Phase 2 Moradabad"),
        "website_url": "",
        "image_match": ["RSD Hospital and Research Center", "RSD Hospital and Research Center  Multispecialty Hospital"],
    },
    {
        "name": "Surgimed Hospital",
        "address": "Ekta Vihar, Near State Bank, Rampur Road, Civil Lines, Moradabad-244001, Uttar Pradesh",
        "contact": "08460520087",
        "speciality": "Multi-Speciality",
        "location_url": _build_maps_url("Surgimed Hospital Ekta Vihar Rampur Road Civil Lines Moradabad"),
        "website_url": "",
        "image_match": ["Surgimed Hospital"],
    },
    {
        "name": "Synergy Superspeciality Hospital",
        "address": "Plot no 4, Kanth Rd, near Government polytechnic college, Ashiyana Colony, Moradabad, Uttar Pradesh 244105, India",
        "contact": "08923474942",
        "speciality": "Super Speciality",
        "location_url": _build_maps_url("Synergy Superspeciality Hospital Kanth Road Moradabad"),
        "website_url": "",
        "image_match": ["Synergy Superspeciality Hospital"],
    },
    {
        "name": "Tmu Hospital",
        "address": "NH-9,Delhi,Moradabad,Tmu Hospital,Near,Delhi Road,Pakbara,Moradabad-244102,Uttar Pradesh",
        "contact": "08401606581",
        "speciality": "Multi-Speciality",
        "location_url": _build_maps_url("TMU Hospital Delhi Road Pakbara Moradabad"),
        "website_url": "",
        "image_match": ["Tmu Hospital"],
    },
]


_DOCTORS_BASE_DATA: List[Dict[str, str]] = [
    {
        "name": "Dr. Abhijeet Sinha",
        "contact": "919389808035",
        "address": "New Moradabad UP 244102",
        "qualification": "",
        "specialization": "ENT Specialist",
        "experience": "",
        "rating": "",
        "location_url": _build_maps_url("Dr. Abhijeet Sinha New Moradabad UP 244102"),
        "website_url": "",
        "image_match": ["Dr. Abhijeet Sinha"],
    },
    {
        "name": "Dr. Amit Rastogi",
        "contact": "918881293178",
        "address": "Near Raj Mahal Hotel Civil Lines Moradabad UP 244001",
        "qualification": "",
        "specialization": "Diabetologist Endocrine Specialist",
        "experience": "",
        "rating": "",
        "location_url": _build_maps_url("Dr. Amit Rastogi Near Raj Mahal Hotel Civil Lines Moradabad UP 244001"),
        "website_url": "",
        "image_match": ["Dr. Amit Rastogi"],
    },
    {
        "name": "Dr. Anurag Agarwal",
        "contact": "9389808035",
        "address": "Sector 16 New Moradabad UP 244102",
        "qualification": "",
        "specialization": "Orthopedic Team",
        "experience": "",
        "rating": "",
        "location_url": _build_maps_url("Dr. Anurag Agarwal Sector 16 New Moradabad UP 244102"),
        "website_url": "",
        "image_match": ["Dr. Anurag Agarwal"],
    },
    {
        "name": "Dr. Anurag Agarwal1",
        "contact": "918460223572",
        "address": "Kanth Road Moradabad UP 244001",
        "qualification": "",
        "specialization": "Urologist Urinary Male Health Specialist",
        "experience": "",
        "rating": "",
        "location_url": _build_maps_url("Dr. Anurag Agarwal1 Kanth Road Moradabad UP 244001"),
        "website_url": "",
        "image_match": ["Dr. Anurag Agarwal1"],
    },
    {
        "name": "Dr. Arpit Bansal",
        "contact": "919259071497",
        "address": "Ram Ganga Vihar Phase Moradabad UP 244105",
        "qualification": "",
        "specialization": "Gastroenterologist Digestive Liver Specialist",
        "experience": "",
        "rating": "",
        "location_url": _build_maps_url("Dr. Arpit Bansal Ram Ganga Vihar Phase Moradabad UP 244105"),
        "website_url": "",
        "image_match": ["Dr. Arpit Bansal"],
    },
    {
        "name": "Dr. Arpit Bansal",
        "contact": "9259071497",
        "address": "Navjeevan Super Specialty, Moradabad UP 244105",
        "qualification": "",
        "specialization": "Gastroenterologist",
        "experience": "",
        "rating": "",
        "location_url": _build_maps_url("Dr. Arpit Bansal Navjeevan Super Specialty Moradabad UP 244105"),
        "website_url": "",
        "image_match": ["Dr. Arpit Bansal"],
    },
    {
        "name": "Dr. Geetesh Manik",
        "contact": "915912551100",
        "address": "Kanth Road Moradabad UP 244001",
        "qualification": "",
        "specialization": "Oncologist Cancer Specialist",
        "experience": "",
        "rating": "",
        "location_url": _build_maps_url("Dr. Geetesh Manik Kanth Road Moradabad UP 244001"),
        "website_url": "",
        "image_match": ["Dr. Geetesh Manik"],
    },
    {
        "name": "Dr. Indranil Halder",
        "contact": "919389808035",
        "address": "Kanth Road Moradabad UP 244001",
        "qualification": "",
        "specialization": "Radiologist CT MRI Imaging Specialist",
        "experience": "",
        "rating": "",
        "location_url": _build_maps_url("Dr. Indranil Halder Kanth Road Moradabad UP 244001"),
        "website_url": "",
        "image_match": ["Dr. Indranil Halder"],
    },
    {
        "name": "Dr. Maj Vishesh Chauhan",
        "contact": "917454801788",
        "address": "Gagan Tiraha Sambhal Chandausi Road Moradabad UP 244001",
        "qualification": "",
        "specialization": "Neuro Psychiatrist",
        "experience": "",
        "rating": "",
        "location_url": _build_maps_url("Dr. Maj Vishesh Chauhan Gagan Tiraha Sambhal Chandausi Road Moradabad UP 244001"),
        "website_url": "",
        "image_match": ["Dr. Maj Vishesh Chauhan"],
    },
    {
        "name": "Dr. Mohd Avais",
        "contact": "9512551100",
        "address": "Asian Vivekanand Super Speciality Hospital Moradabad",
        "qualification": "",
        "specialization": "Pediatrician Childrens Doctor",
        "experience": "",
        "rating": "",
        "location_url": _build_maps_url("Dr. Mohd Avais Asian Vivekanand Super Speciality Hospital Moradabad"),
        "website_url": "",
        "image_match": ["Dr. Mohd Avais"],
    },
    {
        "name": "Dr. Nitin Agarwal",
        "contact": "919927218765",
        "address": "Gandhi Nagar Moradabad UP 244001",
        "qualification": "",
        "specialization": "Gynecologist Obstetrician",
        "experience": "",
        "rating": "",
        "location_url": _build_maps_url("Dr. Nitin Agarwal Gandhi Nagar Moradabad UP 244001"),
        "website_url": "",
        "image_match": ["Dr. Nitin Agarwal"],
    },
    {
        "name": "Dr. Praveen Kumar Jain",
        "contact": "9412244777",
        "address": "Civil Lines Moradabad UP 244001",
        "qualification": "",
        "specialization": "Ophthalmologist Eye Specialist",
        "experience": "",
        "rating": "",
        "location_url": _build_maps_url("Dr. Praveen Kumar Jain Civil Lines Moradabad UP 244001"),
        "website_url": "",
        "image_match": ["Dr. Praveen Kumar Jain"],
    },
    {
        "name": "Dr. Rajeev Agarwal",
        "contact": "7942694997",
        "address": "Prem Nagar Civil Lines Moradabad UP 244001",
        "qualification": "",
        "specialization": "Cardiologist",
        "experience": "",
        "rating": "",
        "location_url": _build_maps_url("Dr. Rajeev Agarwal Prem Nagar Civil Lines Moradabad UP 244001"),
        "website_url": "",
        "image_match": ["Dr. Rajeev Agarwal"],
    },
    {
        "name": "Dr. Rakesh Kumar",
        "contact": "7351002610",
        "address": "Kanth Road Prem Nagar Moradabad UP 244001",
        "qualification": "",
        "specialization": "Nephrologist Kidney Specialist",
        "experience": "",
        "rating": "",
        "location_url": _build_maps_url("Dr. Rakesh Kumar Kanth Road Prem Nagar Moradabad UP 244001"),
        "website_url": "",
        "image_match": ["Dr. Rakesh Kumar"],
    },
    {
        "name": "Dr. Ritesh Kumar",
        "contact": "7579559577",
        "address": "Moradabad UP 244105",
        "qualification": "",
        "specialization": "TB and Chest Pulmonology Specialist",
        "experience": "",
        "rating": "",
        "location_url": _build_maps_url("Dr. Ritesh Kumar Moradabad UP 244105"),
        "website_url": "",
        "image_match": ["Dr. Ritesh Kumar"],
    },
    {
        "name": "Dr. Vishesh Chauhan",
        "contact": "7454801788",
        "address": "Anesthesia Critical Care Moradabad UP 244001",
        "qualification": "",
        "specialization": "Anesthesiologist Surgical",
        "experience": "",
        "rating": "",
        "location_url": _build_maps_url("Dr. Vishesh Chauhan Anesthesia Critical Care Moradabad UP 244001"),
        "website_url": "",
        "image_match": ["Dr. Vishesh Chauhan"],
    },
    {
        "name": "Dr.Rakesh Kumar Singh",
        "contact": "8791632831",
        "address": "Banglagaon Chauraha Daulatbag Moradabad UP 244001",
        "qualification": "",
        "specialization": "Neurologist",
        "experience": "",
        "rating": "",
        "location_url": _build_maps_url("Dr.Rakesh Kumar Singh Banglagaon Chauraha Daulatbag Moradabad UP 244001"),
        "website_url": "",
        "image_match": ["Dr.Rakesh Kumar Singh", "Dr. Rakesh Kumar Singh"],
    },
]


def _build_hospitals_data(asset_version: str) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    for item in _HOSPITALS_BASE_DATA:
        record = dict(item)
        image_match = record.pop("image_match", [])
        record["image_url"] = _resolve_hospital_image_url(*image_match, asset_version=asset_version)
        records.append(record)
    return records


def _build_doctors_data(asset_version: str) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    for item in _DOCTORS_BASE_DATA:
        record = dict(item)
        image_match = record.pop("image_match", [])
        record["image_url"] = _resolve_doctor_image_url(*image_match, asset_version=asset_version)
        records.append(record)
    return records


DOCTORS_DATA: List[Dict[str, str]] = _build_doctors_data(_asset_version_token())
HOSPITALS_DATA: List[Dict[str, str]] = _build_hospitals_data(_asset_version_token())

def get_hospitals_data() -> List[Dict[str, str]]:
    version = _asset_version_token()
    return [dict(item) for item in _consultant_directory_cached(version)["hospitals"]]


def get_doctors_data() -> List[Dict[str, str]]:
    version = _asset_version_token()
    return [dict(item) for item in _consultant_directory_cached(version)["doctors"]]


def get_consultant_directory() -> Dict[str, List[Dict[str, str]]]:
    version = _asset_version_token()
    data = _consultant_directory_cached(version)
    return {
        "hospitals": [dict(item) for item in data["hospitals"]],
        "doctors": [dict(item) for item in data["doctors"]],
    }


def search_providers(query: str) -> Dict[str, List[Dict[str, str]]]:
    directory = get_consultant_directory()
    query_lower = query.lower()
    hospitals = [
        hospital
        for hospital in directory["hospitals"]
        if query_lower in hospital.get("name", "").lower()
        or query_lower in hospital.get("address", "").lower()
        or query_lower in hospital.get("speciality", "").lower()
    ]
    doctors = [doctor for doctor in directory["doctors"] if query_lower in doctor["name"].lower()]
    return {"hospitals": hospitals, "doctors": doctors}


__all__ = [
    "HOSPITALS_DATA",
    "DOCTORS_DATA",
    "get_hospitals_data",
    "get_doctors_data",
    "get_consultant_directory",
    "search_providers",
]
