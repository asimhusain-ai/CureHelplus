"""Structured JSON storage for authenticated user domain data."""

from __future__ import annotations

import json
import os
import threading
import secrets
from datetime import datetime
from typing import Any, Dict, List

try:
    import psycopg2
    from psycopg2.extras import Json
except Exception:  # pragma: no cover - optional in environments without psycopg2
    psycopg2 = None
    Json = None


class UserDataStore:
    """Stores non-auth user data in JSON files keyed by user id."""

    def __init__(self, base_dir: str = "user_data") -> None:
        self.base_dir = os.path.abspath(base_dir)
        self._lock = threading.Lock()
        self._files = {
            "profiles": os.path.join(self.base_dir, "profiles.json"),
            "medical": os.path.join(self.base_dir, "medical_data.json"),
            "predictions": os.path.join(self.base_dir, "predictions.json"),
            "reports": os.path.join(self.base_dir, "reports.json"),
            "health_history": os.path.join(self.base_dir, "health_history.json"),
        }
        self.database_url = (os.getenv("DATABASE_URL") or "").strip()
        self._postgres_available = bool(self.database_url and psycopg2 is not None)
        self._ensure_files()
        self._ensure_health_history_table()

    def _ensure_files(self) -> None:
        os.makedirs(self.base_dir, exist_ok=True)
        for path in self._files.values():
            if not os.path.exists(path):
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump({}, fh, indent=2)

    def _read_map(self, category: str) -> Dict[str, Any]:
        path = self._files[category]
        with open(path, "r", encoding="utf-8") as fh:
            try:
                payload = json.load(fh)
            except json.JSONDecodeError:
                payload = {}
        return payload if isinstance(payload, dict) else {}

    def _write_map(self, category: str, payload: Dict[str, Any]) -> None:
        path = self._files[category]
        temp_path = f"{path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(temp_path, path)

    def _db_connect(self):
        if not self._postgres_available or psycopg2 is None:
            raise RuntimeError("PostgreSQL unavailable")
        return psycopg2.connect(self.database_url, connect_timeout=3)

    def _ensure_health_history_table(self) -> None:
        if not self._postgres_available:
            return
        statement = """
            CREATE TABLE IF NOT EXISTS health_history (
                id BIGSERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                disease_type TEXT NOT NULL,
                input_data JSONB NOT NULL DEFAULT '{}'::jsonb,
                image_path TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_health_history_user_id ON health_history(user_id);
            CREATE INDEX IF NOT EXISTS idx_health_history_disease_type ON health_history(disease_type);
            CREATE INDEX IF NOT EXISTS idx_health_history_created_at ON health_history(created_at DESC);
        """
        try:
            with self._db_connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(statement)
                conn.commit()
        except Exception:
            self._postgres_available = False

    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        with self._lock:
            profiles = self._read_map("profiles")
            entry = profiles.get(user_id, {})
            return entry if isinstance(entry, dict) else {}

    def set_user_profile(self, user_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            profiles = self._read_map("profiles")
            current = profiles.get(user_id, {})
            if not isinstance(current, dict):
                current = {}
            merged = {**current, **(updates or {})}
            merged["updated_at"] = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
            profiles[user_id] = merged
            self._write_map("profiles", profiles)
            return merged

    def append_prediction(self, user_id: str, prediction: Dict[str, Any]) -> None:
        with self._lock:
            predictions = self._read_map("predictions")
            current = predictions.get(user_id, [])
            if not isinstance(current, list):
                current = []
            current.append(prediction)
            predictions[user_id] = current
            self._write_map("predictions", predictions)

    def list_predictions(self, user_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            predictions = self._read_map("predictions")
            current = predictions.get(user_id, [])
            return current if isinstance(current, list) else []

    def append_report(self, user_id: str, report_entry: Dict[str, Any]) -> None:
        with self._lock:
            reports = self._read_map("reports")
            current = reports.get(user_id, [])
            if not isinstance(current, list):
                current = []
            normalized = dict(report_entry or {})
            if not str(normalized.get("id") or "").strip():
                normalized["id"] = secrets.token_hex(8)
            current.append(normalized)
            reports[user_id] = current
            self._write_map("reports", reports)

    def list_reports(self, user_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            reports = self._read_map("reports")
            current = reports.get(user_id, [])
            if not isinstance(current, list):
                return []

            changed = False
            normalized: List[Dict[str, Any]] = []
            for entry in current:
                item = dict(entry or {})
                if not str(item.get("id") or "").strip():
                    item["id"] = secrets.token_hex(8)
                    changed = True
                normalized.append(item)

            if changed:
                reports[user_id] = normalized
                self._write_map("reports", reports)

            return normalized

    def remove_report(self, user_id: str, report_id: str) -> Dict[str, Any] | None:
        target = str(report_id or "").strip()
        if not target:
            return None

        with self._lock:
            reports = self._read_map("reports")
            current = reports.get(user_id, [])
            if not isinstance(current, list):
                return None

            kept: List[Dict[str, Any]] = []
            removed: Dict[str, Any] | None = None
            for entry in current:
                entry_id = str((entry or {}).get("id") or "").strip()
                if removed is None and entry_id == target:
                    removed = entry
                    continue
                kept.append(entry)

            if removed is None:
                return None

            reports[user_id] = kept
            self._write_map("reports", reports)
            return removed

    def set_medical_data(self, user_id: str, medical_payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            medical = self._read_map("medical")
            current = medical.get(user_id, {})
            if not isinstance(current, dict):
                current = {}
            merged = {**current, **(medical_payload or {})}
            merged["updated_at"] = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
            medical[user_id] = merged
            self._write_map("medical", medical)
            return merged

    def get_medical_data(self, user_id: str) -> Dict[str, Any]:
        with self._lock:
            medical = self._read_map("medical")
            payload = medical.get(user_id, {})
            return payload if isinstance(payload, dict) else {}

    def append_health_history(
        self,
        user_id: str,
        disease_type: str,
        input_data: Dict[str, Any] | None = None,
        image_path: str | None = None,
    ) -> None:
        normalized_type = (disease_type or "").strip() or "unknown"
        normalized_input = input_data if isinstance(input_data, dict) else {}
        created_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

        if self._postgres_available:
            statement = """
                INSERT INTO health_history (user_id, disease_type, input_data, image_path, created_at)
                VALUES (%s, %s, %s, %s, %s);
            """
            try:
                with self._db_connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            statement,
                            (
                                user_id,
                                normalized_type,
                                Json(normalized_input) if Json is not None else normalized_input,
                                image_path,
                                created_at,
                            ),
                        )
                    conn.commit()
                return
            except Exception:
                self._postgres_available = False

        with self._lock:
            history = self._read_map("health_history")
            current = history.get(user_id, [])
            if not isinstance(current, list):
                current = []
            current.append(
                {
                    "id": len(current) + 1,
                    "user_id": user_id,
                    "disease_type": normalized_type,
                    "input_data": normalized_input,
                    "image_path": image_path,
                    "created_at": created_at,
                }
            )
            history[user_id] = current[-500:]
            self._write_map("health_history", history)

    def list_health_history(self, user_id: str) -> List[Dict[str, Any]]:
        if self._postgres_available:
            statement = """
                SELECT id, user_id, disease_type, input_data, image_path, created_at
                FROM health_history
                WHERE user_id = %s
                ORDER BY created_at DESC;
            """
            try:
                with self._db_connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute(statement, (user_id,))
                        rows = cur.fetchall()
                return [
                    {
                        "id": row[0],
                        "user_id": row[1],
                        "disease_type": row[2],
                        "input_data": row[3] if isinstance(row[3], dict) else {},
                        "image_path": row[4],
                        "created_at": row[5].isoformat() if hasattr(row[5], "isoformat") else str(row[5]),
                    }
                    for row in rows
                ]
            except Exception:
                self._postgres_available = False

        with self._lock:
            history = self._read_map("health_history")
            current = history.get(user_id, [])
            return current if isinstance(current, list) else []

    def remove_user(self, user_id: str) -> None:
        with self._lock:
            for category in self._files:
                payload = self._read_map(category)
                if user_id in payload:
                    payload.pop(user_id, None)
                    self._write_map(category, payload)


user_data_store = UserDataStore()

__all__ = ["UserDataStore", "user_data_store"]
