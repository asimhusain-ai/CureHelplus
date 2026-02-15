"""Profile persistence utilities for the Flask version of CureHelp+."""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv

try:
    import psycopg2
    from psycopg2.extras import Json
except Exception:  # pragma: no cover - optional in JSON-only environments
    psycopg2 = None
    Json = None


load_dotenv()
logger = logging.getLogger(__name__)


class ProfileManager:
    """Profile storage manager supporting JSON file and PostgreSQL backends."""

    def __init__(
        self,
        profiles_file: str = "user_profiles.json",
        backend: Optional[str] = None,
        database_url: Optional[str] = None,
        table_name: str = "user_profiles",
    ) -> None:
        self.profiles_file = os.path.abspath(profiles_file)
        self._lock = threading.Lock()
        self.table_name = table_name
        self._json_cache: Optional[List[Dict[str, Any]]] = None
        self._json_cache_mtime: Optional[float] = None

        env_database_url = (database_url or os.getenv("DATABASE_URL") or "").strip()
        self.database_url = self._normalise_database_url(env_database_url)

        explicit_backend = (backend or "").strip().lower()
        env_backend = (os.getenv("PROFILE_STORAGE_BACKEND") or "").strip().lower()
        default_storage_file = os.path.abspath("user_profiles.json")
        is_default_storage_target = self.profiles_file == default_storage_file

        if explicit_backend in {"json", "postgres"}:
            requested_backend = explicit_backend
        elif is_default_storage_target:
            requested_backend = env_backend
        else:
            requested_backend = "json"

        inferred_backend = "postgres" if self.database_url else "json"
        self.backend = requested_backend if requested_backend in {"json", "postgres"} else inferred_backend
        self._strict_postgres = (os.getenv("PROFILE_STORAGE_STRICT", "false") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._json_primary_mode = self.backend == "postgres" and not self._strict_postgres
        self._runtime_json_fallback = (os.getenv("PROFILE_RUNTIME_JSON_FALLBACK", "true") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._db_connect_timeout_seconds = max(1, int(os.getenv("PROFILE_DB_CONNECT_TIMEOUT", "2") or "2"))
        self._db_statement_timeout_ms = max(500, int(os.getenv("PROFILE_DB_STATEMENT_TIMEOUT_MS", "3000") or "3000"))
        self._sync_queue_maxsize = max(100, int(os.getenv("PROFILE_SYNC_QUEUE_MAXSIZE", "2000") or "2000"))
        self._postgres_runtime_failed = False
        self._postgres_sync_queue: Optional[queue.Queue] = None
        self._postgres_sync_worker_started = False

        # Keep JSON storage ready as a safety net for runtime DB outages.
        self._ensure_file()

        if self.backend == "postgres":
            self._init_postgres_or_fallback()

        if self._json_primary_mode and self.backend == "postgres":
            self._start_postgres_sync_worker()

    def _use_postgres(self) -> bool:
        return self.backend == "postgres" and not self._postgres_runtime_failed

    def _activate_runtime_fallback(self, error: Optional[Exception] = None) -> None:
        if not self._runtime_json_fallback:
            if error is not None:
                raise error
            raise RuntimeError("PostgreSQL runtime failure and JSON fallback is disabled")
        self._postgres_runtime_failed = True

    def _generate_profile_id_from_list(self, profiles: List[Dict[str, Any]]) -> str:
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

    @staticmethod
    def _normalise_database_url(url: str) -> str:
        if not url:
            return ""
        if url.startswith("postgresql+psycopg2://"):
            url = "postgresql://" + url[len("postgresql+psycopg2://") :]
        if url.startswith("postgres+psycopg2://"):
            url = "postgres://" + url[len("postgres+psycopg2://") :]

        scheme_sep = "://"
        if scheme_sep not in url:
            return url

        scheme, remainder = url.split(scheme_sep, 1)
        authority, slash, tail = remainder.partition("/")
        if authority.count("@") >= 2:
            last_at = authority.rfind("@")
            userinfo = authority[:last_at]
            hostinfo = authority[last_at + 1 :]
            if ":" in userinfo:
                username, password = userinfo.split(":", 1)
                username = username.replace("@", "%40")
                authority = f"{username}:{password}@{hostinfo}"
                url = f"{scheme}{scheme_sep}{authority}{slash}{tail}"

        return url

    def _init_postgres_or_fallback(self) -> None:
        if not self.database_url or psycopg2 is None:
            if self._strict_postgres:
                raise RuntimeError("PostgreSQL backend requested but DATABASE_URL/psycopg2 is unavailable")
            self.backend = "json"
            self._ensure_file()
            return

        try:
            self._ensure_table_postgres()
            self._migrate_json_once_if_enabled()
        except Exception:
            if self._strict_postgres:
                raise
            self.backend = "json"
            self._ensure_file()

    def _migrate_json_once_if_enabled(self) -> None:
        should_migrate = (os.getenv("PROFILE_AUTO_MIGRATE_JSON", "true") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if not should_migrate:
            return
        self.migrate_from_json(self.profiles_file)

    def _ensure_file(self) -> None:
        directory = os.path.dirname(self.profiles_file)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        if not os.path.exists(self.profiles_file):
            with open(self.profiles_file, "w", encoding="utf-8") as fh:
                json.dump([], fh, indent=4)
        self._json_cache = None
        self._json_cache_mtime = None

    def _file_mtime_unlocked(self) -> Optional[float]:
        try:
            return os.path.getmtime(self.profiles_file)
        except OSError:
            return None

    def _load_profiles_unlocked(self) -> List[Dict[str, Any]]:
        current_mtime = self._file_mtime_unlocked()
        if self._json_cache is not None and self._json_cache_mtime == current_mtime:
            return deepcopy(self._json_cache)

        with open(self.profiles_file, "r", encoding="utf-8") as fh:
            try:
                data = json.load(fh)
            except json.JSONDecodeError:
                data = []
        profiles = data if isinstance(data, list) else []
        self._json_cache = deepcopy(profiles)
        self._json_cache_mtime = current_mtime
        return profiles

    def _db_connect(self):
        if psycopg2 is None:
            raise RuntimeError("psycopg2 is not installed")
        return psycopg2.connect(
            self.database_url,
            connect_timeout=self._db_connect_timeout_seconds,
            options=f"-c statement_timeout={self._db_statement_timeout_ms}",
        )

    def _start_postgres_sync_worker(self) -> None:
        if self._postgres_sync_worker_started:
            return
        self._postgres_sync_queue = queue.Queue(maxsize=self._sync_queue_maxsize)
        worker = threading.Thread(target=self._postgres_sync_worker_loop, name="profile-postgres-sync", daemon=True)
        worker.start()
        self._postgres_sync_worker_started = True

    def _enqueue_postgres_sync(self, operation: str, payload: Any) -> None:
        if not self._postgres_sync_queue or not self._use_postgres():
            return
        try:
            self._postgres_sync_queue.put_nowait((operation, payload))
        except queue.Full:
            try:
                self._postgres_sync_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._postgres_sync_queue.put_nowait((operation, payload))
            except queue.Full:
                logger.warning("PostgreSQL sync queue is full; dropping operation: %s", operation)

    def _postgres_sync_worker_loop(self) -> None:
        while True:
            if not self._postgres_sync_queue:
                return
            try:
                operation, payload = self._postgres_sync_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                if operation == "upsert":
                    self._sync_profile_to_postgres(payload)
                elif operation == "delete":
                    self._sync_delete_to_postgres(payload)
            except Exception as exc:
                logger.warning("Asynchronous PostgreSQL sync failed: %s", exc)
            finally:
                self._postgres_sync_queue.task_done()

    def _sync_profile_to_postgres(self, profile: Dict[str, Any]) -> None:
        if not self._use_postgres():
            return
        statement = f"""
            INSERT INTO {self.table_name} (
                id, name, age, contact, address, gender, marital_status, predictions, created_at, last_updated
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                name = EXCLUDED.name,
                age = EXCLUDED.age,
                contact = EXCLUDED.contact,
                address = EXCLUDED.address,
                gender = EXCLUDED.gender,
                marital_status = EXCLUDED.marital_status,
                predictions = EXCLUDED.predictions,
                created_at = EXCLUDED.created_at,
                last_updated = EXCLUDED.last_updated;
        """
        try:
            with self._db_connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        statement,
                        (
                            profile.get("id"),
                            profile.get("name"),
                            profile.get("age"),
                            profile.get("contact"),
                            profile.get("address"),
                            profile.get("gender"),
                            profile.get("marital_status"),
                            Json(profile.get("predictions", {})) if Json is not None else profile.get("predictions", {}),
                            profile.get("created_at"),
                            profile.get("last_updated"),
                        ),
                    )
                conn.commit()
        except Exception as exc:
            self._activate_runtime_fallback(exc)

    def _sync_delete_to_postgres(self, profile_id: str) -> None:
        if not self._use_postgres():
            return
        statement = f"DELETE FROM {self.table_name} WHERE id = %s;"
        try:
            with self._db_connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(statement, (profile_id,))
                conn.commit()
        except Exception as exc:
            self._activate_runtime_fallback(exc)

    def _ensure_table_postgres(self) -> None:
        statement = f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id TEXT PRIMARY KEY,
                name TEXT,
                age INTEGER,
                contact TEXT,
                address TEXT,
                gender TEXT,
                marital_status TEXT,
                predictions JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                created_at TEXT,
                last_updated TEXT
            );
        """
        with self._db_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(statement)
            conn.commit()

    @staticmethod
    def _row_to_profile(row: Any) -> Dict[str, Any]:
        return {
            "id": row[0],
            "name": row[1],
            "age": row[2],
            "contact": row[3],
            "address": row[4],
            "gender": row[5],
            "marital_status": row[6],
            "predictions": row[7] if isinstance(row[7], dict) else {},
            "created_at": row[8],
            "last_updated": row[9],
        }

    def load_profiles(self) -> List[Dict[str, Any]]:
        with self._lock:
            if self._json_primary_mode:
                return [profile.copy() for profile in self._load_profiles_unlocked()]
            if self._use_postgres():
                try:
                    statement = f"""
                        SELECT id, name, age, contact, address, gender, marital_status, predictions, created_at, last_updated
                        FROM {self.table_name}
                        ORDER BY id ASC;
                    """
                    with self._db_connect() as conn:
                        with conn.cursor() as cur:
                            cur.execute(statement)
                            rows = cur.fetchall()
                    return [self._row_to_profile(row) for row in rows]
                except Exception as exc:
                    self._activate_runtime_fallback(exc)
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
        temp_file = f"{self.profiles_file}.tmp"
        with open(temp_file, "w", encoding="utf-8") as fh:
            json.dump(serialisable, fh, indent=4)
        os.replace(temp_file, self.profiles_file)
        self._json_cache = deepcopy(serialisable)
        self._json_cache_mtime = self._file_mtime_unlocked()

    def _generate_profile_id(self, profiles: List[Dict[str, Any]]) -> str:
        if self._use_postgres():
            statement = f"""
                SELECT COALESCE(MAX(CAST(SUBSTRING(id FROM 6) AS INTEGER)), 0)
                FROM {self.table_name}
                WHERE id ~ '^user_[0-9]+$';
            """
            with self._db_connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(statement)
                    current_max = cur.fetchone()[0] or 0
            return f"user_{int(current_max) + 1:03d}"
        return self._generate_profile_id_from_list(profiles)

    def add_profile(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            json_profiles = self._load_profiles_unlocked()
            if profile_data.get("id"):
                profile_id = profile_data.get("id")
            elif self._use_postgres() and not self._json_primary_mode:
                try:
                    profile_id = self._generate_profile_id([])
                except Exception as exc:
                    self._activate_runtime_fallback(exc)
                    profile_id = self._generate_profile_id_from_list(json_profiles)
            else:
                profile_id = self._generate_profile_id_from_list(json_profiles)
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

            if self._json_primary_mode:
                json_profiles.append(profile)
                self._write_profiles_unlocked(json_profiles)
                self._enqueue_postgres_sync("upsert", profile)
                return profile

            if self._use_postgres():
                try:
                    self._sync_profile_to_postgres(profile)
                    return profile
                except Exception as exc:
                    self._activate_runtime_fallback(exc)

            json_profiles.append(profile)
            self._write_profiles_unlocked(json_profiles)
            return profile

    def update_profile(self, profile_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        with self._lock:
            if self._json_primary_mode:
                profiles = self._load_profiles_unlocked()
                for profile in profiles:
                    if profile.get("id") == profile_id:
                        profile.update(self.convert_numpy_types(updates))
                        profile["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        self._write_profiles_unlocked(profiles)
                        self._enqueue_postgres_sync("upsert", profile)
                        return profile.copy()
                return None

            if self._use_postgres():
                try:
                    current = self.get_profile(profile_id)
                    if current is None:
                        return None

                    current.update(self.convert_numpy_types(updates))
                    current["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    statement = f"""
                        UPDATE {self.table_name}
                        SET
                            name = %s,
                            age = %s,
                            contact = %s,
                            address = %s,
                            gender = %s,
                            marital_status = %s,
                            predictions = %s,
                            created_at = %s,
                            last_updated = %s
                        WHERE id = %s;
                    """
                    with self._db_connect() as conn:
                        with conn.cursor() as cur:
                            cur.execute(
                                statement,
                                (
                                    current.get("name"),
                                    current.get("age"),
                                    current.get("contact"),
                                    current.get("address"),
                                    current.get("gender"),
                                    current.get("marital_status"),
                                    Json(current.get("predictions", {})) if Json is not None else current.get("predictions", {}),
                                    current.get("created_at"),
                                    current.get("last_updated"),
                                    profile_id,
                                ),
                            )
                        conn.commit()
                    return current.copy()
                except Exception as exc:
                    self._activate_runtime_fallback(exc)

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
        if self._json_primary_mode:
            for profile in self.load_profiles():
                if profile.get("id") == profile_id:
                    return profile
            return None

        if self._use_postgres():
            try:
                statement = f"""
                    SELECT id, name, age, contact, address, gender, marital_status, predictions, created_at, last_updated
                    FROM {self.table_name}
                    WHERE id = %s;
                """
                with self._db_connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute(statement, (profile_id,))
                        row = cur.fetchone()
                return self._row_to_profile(row) if row else None
            except Exception as exc:
                self._activate_runtime_fallback(exc)

        for profile in self.load_profiles():
            if profile.get("id") == profile_id:
                return profile
        return None

    def list_profiles(self) -> List[Dict[str, Any]]:
        return self.load_profiles()

    def search_profiles(self, query: str) -> List[Dict[str, Any]]:
        if self._json_primary_mode:
            query_lower = query.lower()
            return [profile for profile in self.load_profiles() if query_lower in profile.get("name", "").lower()]

        if self._use_postgres():
            try:
                statement = f"""
                    SELECT id, name, age, contact, address, gender, marital_status, predictions, created_at, last_updated
                    FROM {self.table_name}
                    WHERE name ILIKE %s
                    ORDER BY id ASC;
                """
                with self._db_connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute(statement, (f"%{query}%",))
                        rows = cur.fetchall()
                return [self._row_to_profile(row) for row in rows]
            except Exception as exc:
                self._activate_runtime_fallback(exc)

        query_lower = query.lower()
        return [profile for profile in self.load_profiles() if query_lower in profile.get("name", "").lower()]

    def delete_profile(self, profile_id: str) -> bool:
        with self._lock:
            if self._json_primary_mode:
                profiles = self._load_profiles_unlocked()
                updated = [profile for profile in profiles if profile.get("id") != profile_id]
                if len(updated) == len(profiles):
                    return False
                self._write_profiles_unlocked(updated)
                self._enqueue_postgres_sync("delete", profile_id)
                return True

            if self._use_postgres():
                try:
                    statement = f"DELETE FROM {self.table_name} WHERE id = %s;"
                    with self._db_connect() as conn:
                        with conn.cursor() as cur:
                            cur.execute(statement, (profile_id,))
                            deleted = cur.rowcount > 0
                        conn.commit()
                    return deleted
                except Exception as exc:
                    self._activate_runtime_fallback(exc)

            profiles = self._load_profiles_unlocked()
            updated = [profile for profile in profiles if profile.get("id") != profile_id]
            if len(updated) == len(profiles):
                return False
            self._write_profiles_unlocked(updated)
            return True

    def migrate_from_json(self, json_path: Optional[str] = None) -> int:
        if self.backend != "postgres":
            return 0

        source_path = os.path.abspath(json_path or self.profiles_file)
        if not os.path.exists(source_path):
            return 0

        try:
            with open(source_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return 0

        if not isinstance(data, list):
            return 0

        migrated = 0
        for entry in data:
            if not isinstance(entry, dict):
                continue
            existing_id = entry.get("id")
            self.upsert_profile(existing_id if isinstance(existing_id, str) else None, entry)
            migrated += 1
        return migrated

    def upsert_profile(self, profile_id: Optional[str], payload: Dict[str, Any]) -> Dict[str, Any]:
        if profile_id:
            updated = self.update_profile(profile_id, payload)
            if updated is not None:
                return updated
        return self.add_profile(payload)


profile_manager = ProfileManager()

__all__ = ["ProfileManager", "profile_manager"]
