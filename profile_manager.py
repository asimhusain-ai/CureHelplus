"""Profile persistence utilities for the Flask version of CureHelp+."""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np


class ProfileManager:
    """Lightweight JSON-backed profile storage manager."""

    def __init__(self, profiles_file: str = "user_profiles.json") -> None:
        self.profiles_file = os.path.abspath(profiles_file)
        self._lock = threading.Lock()
        self._ensure_file()

    def _ensure_file(self) -> None:
        directory = os.path.dirname(self.profiles_file)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        if not os.path.exists(self.profiles_file):
            with open(self.profiles_file, "w", encoding="utf-8") as fh:
                json.dump([], fh, indent=4)

    def _load_profiles_unlocked(self) -> List[Dict[str, Any]]:
        with open(self.profiles_file, "r", encoding="utf-8") as fh:
            try:
                data = json.load(fh)
            except json.JSONDecodeError:
                data = []
        return data if isinstance(data, list) else []

    def load_profiles(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [profile.copy() for profile in self._load_profiles_unlocked()]

    def convert_numpy_types(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {key: self.convert_numpy_types(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        return obj

    def _write_profiles_unlocked(self, profiles: List[Dict[str, Any]]) -> None:
        serialisable = self.convert_numpy_types(profiles)
        with open(self.profiles_file, "w", encoding="utf-8") as fh:
            json.dump(serialisable, fh, indent=4)

    def _generate_profile_id(self, profiles: List[Dict[str, Any]]) -> str:
        existing = []
        for entry in profiles:
            profile_id = entry.get("id")
            if isinstance(profile_id, str) and profile_id.startswith("user_"):
                try:
                    existing.append(int(profile_id.split("_")[1]))
                except (IndexError, ValueError):
                    continue
        next_index = max(existing, default=0) + 1
        return f"user_{next_index:03d}"

    def add_profile(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            profiles = self._load_profiles_unlocked()
            profile_id = profile_data.get("id") or self._generate_profile_id(profiles)
            timestamp = datetime.now().strftime("%d-%b-%Y %H:%M")

            profile = {
                "id": profile_id,
                "name": profile_data.get("name", ""),
                "age": profile_data.get("age"),
                "contact": profile_data.get("contact", ""),
                "address": profile_data.get("address", ""),
                "gender": profile_data.get("gender", ""),
                "marital_status": profile_data.get("marital_status", ""),
                "predictions": self.convert_numpy_types(profile_data.get("predictions", {})),
                "created_at": profile_data.get("created_at", timestamp),
                "last_updated": profile_data.get("last_updated", timestamp),
            }

            # Remove empty keys for cleaner storage
            profile = {k: v for k, v in profile.items() if v not in (None, "")}

            profiles.append(profile)
            self._write_profiles_unlocked(profiles)
            return profile

    def update_profile(self, profile_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        with self._lock:
            profiles = self._load_profiles_unlocked()
            for profile in profiles:
                if profile.get("id") == profile_id:
                    profile.update(self.convert_numpy_types(updates))
                    profile["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self._write_profiles_unlocked(profiles)
                    return profile.copy()
        return None

    def update_predictions(self, profile_id: str, predictions: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        payload = {"predictions": self.convert_numpy_types(predictions)}
        return self.update_profile(profile_id, payload)

    def get_profile(self, profile_id: str) -> Optional[Dict[str, Any]]:
        for profile in self.load_profiles():
            if profile.get("id") == profile_id:
                return profile
        return None

    def list_profiles(self) -> List[Dict[str, Any]]:
        return self.load_profiles()

    def search_profiles(self, query: str) -> List[Dict[str, Any]]:
        query_lower = query.lower()
        return [profile for profile in self.load_profiles() if query_lower in profile.get("name", "").lower()]

    def delete_profile(self, profile_id: str) -> bool:
        with self._lock:
            profiles = self._load_profiles_unlocked()
            updated = [profile for profile in profiles if profile.get("id") != profile_id]
            if len(updated) == len(profiles):
                return False
            self._write_profiles_unlocked(updated)
            return True

    def upsert_profile(self, profile_id: Optional[str], payload: Dict[str, Any]) -> Dict[str, Any]:
        if profile_id:
            updated = self.update_profile(profile_id, payload)
            if updated is not None:
                return updated
        return self.add_profile(payload)


profile_manager = ProfileManager()

__all__ = ["ProfileManager", "profile_manager"]
