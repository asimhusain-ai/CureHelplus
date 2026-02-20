"""Authentication and session persistence service for CureHelp+."""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
import re
import secrets
import smtplib
import string
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
import ssl
from typing import Any, Dict, List, Optional

import bcrypt
from dotenv import load_dotenv

try:
    import psycopg2
except Exception:  # pragma: no cover - optional in environments without psycopg2
    psycopg2 = None


PROJECT_ROOT = Path(__file__).resolve().parent
DOTENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=DOTENV_PATH)
logger = logging.getLogger(__name__)

EMAIL_PATTERN = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _parse_ts(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    try:
        return datetime.fromisoformat(str(value)).astimezone(timezone.utc)
    except ValueError:
        return None


@dataclass
class AuthResult:
    success: bool
    error: Optional[str] = None
    user: Optional[Dict[str, Any]] = None
    token: Optional[str] = None


class AuthManager:
    """Auth storage manager backed by PostgreSQL with secure tokens and sessions."""

    def __init__(self, database_url: Optional[str] = None) -> None:
        self._dotenv_path = DOTENV_PATH
        load_dotenv(dotenv_path=self._dotenv_path)
        self.database_url = (database_url or os.getenv("DATABASE_URL") or "").strip()
        self.database_url = self._normalise_database_url(self.database_url)
        self._schema_ready = False
        self._schema_lock = threading.Lock()
        self._db_timeout_seconds = max(2, int(os.getenv("AUTH_DB_CONNECT_TIMEOUT", "5") or "5"))
        self._smtp_email = ""
        self._smtp_password = ""
        self._smtp_host = "smtp.gmail.com"
        self._smtp_port = 465
        self._smtp_use_ssl = True
        self._smtp_use_tls = False
        self._smtp_timeout_seconds = 20
        self._otp_email_async = True
        self._refresh_smtp_settings()
        self._base_url = (os.getenv("APP_BASE_URL") or "http://127.0.0.1:5000").rstrip("/")

    def _refresh_smtp_settings(self) -> None:
        load_dotenv(dotenv_path=self._dotenv_path)
        self._smtp_email = (os.getenv("SMTP_EMAIL_ADDRESS") or os.getenv("SMTP_EMAIL") or "").strip()
        self._smtp_password = (os.getenv("SMTP_APP_PASSWORD") or os.getenv("SMTP_PASSWORD") or "").strip()
        self._smtp_host = (os.getenv("SMTP_HOST") or "smtp.gmail.com").strip()
        self._smtp_port = int((os.getenv("SMTP_PORT") or "465").strip() or "465")
        self._smtp_use_ssl = (os.getenv("SMTP_USE_SSL") or "true").strip().lower() in {"1", "true", "yes", "on"}
        self._smtp_use_tls = (os.getenv("SMTP_USE_TLS") or "false").strip().lower() in {"1", "true", "yes", "on"}
        self._smtp_timeout_seconds = max(5, int((os.getenv("SMTP_TIMEOUT_SECONDS") or "20").strip() or "20"))
        self._otp_email_async = (os.getenv("OTP_EMAIL_ASYNC") or "true").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    @staticmethod
    def _normalise_database_url(url: str) -> str:
        if not url:
            return ""
        if url.startswith("postgresql+psycopg2://"):
            url = "postgresql://" + url[len("postgresql+psycopg2://") :]
        if url.startswith("postgres+psycopg2://"):
            url = "postgres://" + url[len("postgres+psycopg2://") :]
        return url

    @property
    def available(self) -> bool:
        return bool(self.database_url and psycopg2 is not None)

    def _connect(self):
        if not self.available:
            raise RuntimeError("Authentication database is not configured.")
        return psycopg2.connect(self.database_url, connect_timeout=self._db_timeout_seconds)

    def ensure_schema(self) -> None:
        if self._schema_ready:
            return
        with self._schema_lock:
            if self._schema_ready:
                return
            if not self.available:
                raise RuntimeError("Authentication database is not configured.")

            statement = """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    full_name TEXT,
                    password_hash TEXT NOT NULL,
                    is_verified BOOLEAN NOT NULL DEFAULT FALSE,
                    is_active BOOLEAN NOT NULL DEFAULT TRUE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS verification_tokens (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    token_hash TEXT UNIQUE NOT NULL,
                    expires_at TIMESTAMPTZ NOT NULL,
                    used_at TIMESTAMPTZ,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS reset_tokens (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    token_hash TEXT UNIQUE NOT NULL,
                    expires_at TIMESTAMPTZ NOT NULL,
                    used_at TIMESTAMPTZ,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    expires_at TIMESTAMPTZ NOT NULL,
                    revoked_at TIMESTAMPTZ,
                    last_seen_at TIMESTAMPTZ,
                    user_agent TEXT,
                    ip_address TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
                CREATE INDEX IF NOT EXISTS idx_verification_user_id ON verification_tokens(user_id);
                CREATE INDEX IF NOT EXISTS idx_reset_user_id ON reset_tokens(user_id);
                CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
                CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at);
            """

            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(statement)
                    cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS full_name TEXT;")
                conn.commit()
            self._schema_ready = True

    @staticmethod
    def validate_email(email: str) -> bool:
        return bool(EMAIL_PATTERN.match((email or "").strip()))

    @staticmethod
    def _token_hash(raw_token: str) -> str:
        return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()

    @staticmethod
    def _hash_password(password: str) -> str:
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")

    @staticmethod
    def _check_password(password: str, password_hash: str) -> bool:
        try:
            return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
        except ValueError:
            return False

    @staticmethod
    def _generate_verification_otp(length: int = 6) -> str:
        otp_length = max(6, int(length or 6))
        alphabet = string.ascii_uppercase + string.digits
        while True:
            otp = "".join(secrets.choice(alphabet) for _ in range(otp_length))
            has_alpha = any(char.isalpha() for char in otp)
            has_digit = any(char.isdigit() for char in otp)
            if has_alpha and has_digit:
                return otp

    def _send_email(self, to_email: str, subject: str, body: str) -> None:
        self._refresh_smtp_settings()
        if not (self._smtp_email and self._smtp_password):
            raise RuntimeError("SMTP is not configured. Set SMTP_EMAIL_ADDRESS and SMTP_APP_PASSWORD.")

        message = EmailMessage()
        message["From"] = self._smtp_email
        message["To"] = to_email
        message["Subject"] = subject
        message.set_content(body)

        try:
            if self._smtp_use_ssl:
                with smtplib.SMTP_SSL(self._smtp_host, self._smtp_port, timeout=self._smtp_timeout_seconds) as server:
                    server.login(self._smtp_email, self._smtp_password)
                    server.send_message(message)
            else:
                with smtplib.SMTP(self._smtp_host, self._smtp_port, timeout=self._smtp_timeout_seconds) as server:
                    server.ehlo()
                    if self._smtp_use_tls:
                        server.starttls(context=ssl.create_default_context())
                        server.ehlo()
                    server.login(self._smtp_email, self._smtp_password)
                    server.send_message(message)
        except Exception as exc:
            raise RuntimeError(f"SMTP delivery failed: {exc}") from exc

    def _send_email_async(self, to_email: str, subject: str, body: str) -> None:
        def _worker() -> None:
            try:
                self._send_email(to_email, subject, body)
            except Exception:
                logger.exception("Async SMTP delivery failed for %s", to_email)

        thread = threading.Thread(target=_worker, name="otp-email-dispatch", daemon=True)
        thread.start()

    def send_verification_otp_email(self, user_id: str, email: str, valid_for_minutes: int = 15) -> AuthResult:
        normalized_email = (email or "").strip().lower()
        if not self.validate_email(normalized_email):
            return AuthResult(success=False, error="Please provide a valid email address.")

        otp = self.create_verification_otp(user_id=user_id, valid_for_minutes=valid_for_minutes)
        subject = "CureHelp+ verification OTP"
        body = (
            "Welcome to CureHelp+\n\n"
            f"Your verification OTP is: {otp}\n\n"
            "This OTP expires in 15 minutes."
        )

        if self._otp_email_async:
            self._send_email_async(normalized_email, subject, body)
            return AuthResult(success=True, token=otp)

        try:
            self._send_email(normalized_email, subject, body)
        except Exception as exc:
            logger.exception("Failed to send verification OTP to %s", normalized_email)
            return AuthResult(success=False, error=str(exc))

        return AuthResult(success=True, token=otp)

    def _get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        self.ensure_schema()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, email, full_name, password_hash, is_verified, is_active, created_at, updated_at
                    FROM users
                    WHERE email = %s
                    LIMIT 1;
                    """,
                    (email,),
                )
                row = cur.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "email": row[1],
            "full_name": row[2],
            "password_hash": row[3],
            "is_verified": bool(row[4]),
            "is_active": bool(row[5]),
            "created_at": _to_iso(row[6]) if row[6] else None,
            "updated_at": _to_iso(row[7]) if row[7] else None,
        }

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        self.ensure_schema()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, email, full_name, is_verified, is_active, created_at, updated_at
                    FROM users
                    WHERE id = %s
                    LIMIT 1;
                    """,
                    (user_id,),
                )
                row = cur.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "email": row[1],
            "full_name": row[2],
            "is_verified": bool(row[3]),
            "is_active": bool(row[4]),
            "created_at": _to_iso(row[5]) if row[5] else None,
            "updated_at": _to_iso(row[6]) if row[6] else None,
        }

    def signup(self, email: str, password: str, full_name: str = "") -> AuthResult:
        normalized_email = (email or "").strip().lower()
        raw_password = password or ""
        normalized_full_name = (full_name or "").strip()

        if not self.validate_email(normalized_email):
            return AuthResult(success=False, error="Please provide a valid email address.")

        if len(raw_password) < 8:
            return AuthResult(success=False, error="Password must be at least 8 characters long.")

        self.ensure_schema()
        existing = self._get_user_by_email(normalized_email)
        if existing is not None:
            return AuthResult(success=False, error="An account with this email already exists.")

        user_id = str(uuid.uuid4())
        password_hash = self._hash_password(raw_password)

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO users (id, email, full_name, password_hash, is_verified, is_active)
                    VALUES (%s, %s, %s, %s, FALSE, TRUE);
                    """,
                    (user_id, normalized_email, normalized_full_name, password_hash),
                )
            conn.commit()

        otp_result = self.send_verification_otp_email(user_id=user_id, email=normalized_email)
        if not otp_result.success:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM users WHERE id = %s;", (user_id,))
                conn.commit()
            return AuthResult(success=False, error=otp_result.error or "Unable to send OTP email.")

        return AuthResult(success=True, user={"id": user_id, "email": normalized_email, "is_verified": False})

    def login_with_google_email(self, email: str) -> AuthResult:
        normalized_email = (email or "").strip().lower()
        if not self.validate_email(normalized_email):
            return AuthResult(success=False, error="Please provide a valid email address.")

        self.ensure_schema()
        user = self._get_user_by_email(normalized_email)

        if user is None:
            user_id = str(uuid.uuid4())
            random_password_hash = self._hash_password(secrets.token_urlsafe(24))
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO users (id, email, password_hash, is_verified, is_active)
                        VALUES (%s, %s, %s, TRUE, TRUE);
                        """,
                        (user_id, normalized_email, random_password_hash),
                    )
                conn.commit()

            return AuthResult(
                success=True,
                user={
                    "id": user_id,
                    "email": normalized_email,
                    "is_verified": True,
                    "is_active": True,
                },
            )

        if not user.get("is_active", False):
            return AuthResult(success=False, error="This account is deactivated.")

        if not user.get("is_verified", False):
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("UPDATE users SET is_verified = TRUE, updated_at = NOW() WHERE id = %s;", (user["id"],))
                conn.commit()

        return AuthResult(
            success=True,
            user={
                "id": user["id"],
                "email": user["email"],
                "is_verified": True,
                "is_active": True,
            },
        )

    def create_verification_otp(self, user_id: str, valid_for_minutes: int = 15) -> str:
        self.ensure_schema()
        raw_token = self._generate_verification_otp(6)
        token_hash = self._token_hash(raw_token)
        token_id = str(uuid.uuid4())
        expires_at = _utc_now() + timedelta(minutes=max(5, valid_for_minutes))

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM verification_tokens WHERE user_id = %s AND used_at IS NULL;", (user_id,))
                cur.execute(
                    """
                    INSERT INTO verification_tokens (id, user_id, token_hash, expires_at)
                    VALUES (%s, %s, %s, %s);
                    """,
                    (token_id, user_id, token_hash, expires_at),
                )
            conn.commit()
        return raw_token

    def create_verification_token(self, user_id: str, valid_for_hours: int = 24) -> str:
        valid_minutes = max(60, int(valid_for_hours or 24) * 60)
        return self.create_verification_otp(user_id=user_id, valid_for_minutes=valid_minutes)

    def verify_email(self, raw_token: str) -> AuthResult:
        token_hash = self._token_hash((raw_token or "").strip())
        if not token_hash:
            return AuthResult(success=False, error="Invalid verification token.")

        self.ensure_schema()
        now = _utc_now()

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, user_id, expires_at, used_at
                    FROM verification_tokens
                    WHERE token_hash = %s
                    LIMIT 1;
                    """,
                    (token_hash,),
                )
                token_row = cur.fetchone()

                if token_row is None:
                    return AuthResult(success=False, error="Invalid or expired verification link.")

                token_id, user_id, expires_at, used_at = token_row
                parsed_expiry = _parse_ts(expires_at)

                if used_at is not None or parsed_expiry is None or parsed_expiry <= now:
                    return AuthResult(success=False, error="Invalid or expired verification link.")

                cur.execute("UPDATE users SET is_verified = TRUE, updated_at = NOW() WHERE id = %s;", (user_id,))
                cur.execute("UPDATE verification_tokens SET used_at = NOW() WHERE id = %s;", (token_id,))
            conn.commit()

        return AuthResult(success=True)

    def verify_email_otp(self, email: str, otp: str) -> AuthResult:
        normalized_email = (email or "").strip().lower()
        raw_otp = (otp or "").strip().upper()

        if not self.validate_email(normalized_email):
            return AuthResult(success=False, error="Please provide a valid email address.")

        if len(raw_otp) != 6:
            return AuthResult(success=False, error="Please enter a valid 6-character OTP.")

        user = self._get_user_by_email(normalized_email)
        if user is None:
            return AuthResult(success=False, error="Invalid email or OTP.")

        token_hash = self._token_hash(raw_otp)
        now = _utc_now()
        self.ensure_schema()

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, expires_at, used_at
                    FROM verification_tokens
                    WHERE user_id = %s AND token_hash = %s
                    LIMIT 1;
                    """,
                    (user["id"], token_hash),
                )
                token_row = cur.fetchone()

                if token_row is None:
                    return AuthResult(success=False, error="Invalid or expired OTP.")

                token_id, expires_at, used_at = token_row
                parsed_expiry = _parse_ts(expires_at)

                if used_at is not None or parsed_expiry is None or parsed_expiry <= now:
                    return AuthResult(success=False, error="Invalid or expired OTP.")

                cur.execute("UPDATE users SET is_verified = TRUE, updated_at = NOW() WHERE id = %s;", (user["id"],))
                cur.execute("UPDATE verification_tokens SET used_at = NOW() WHERE id = %s;", (token_id,))
            conn.commit()

        return AuthResult(success=True)

    def login(self, email: str, password: str) -> AuthResult:
        normalized_email = (email or "").strip().lower()
        raw_password = password or ""

        if not self.validate_email(normalized_email):
            return AuthResult(success=False, error="Please provide a valid email address.")

        user = self._get_user_by_email(normalized_email)
        if user is None or not self._check_password(raw_password, user.get("password_hash", "")):
            return AuthResult(success=False, error="Invalid email or password.")
        if not user.get("is_active", False):
            return AuthResult(success=False, error="This account is deactivated.")
        if not user.get("is_verified", False):
            return AuthResult(success=False, error="Email is not verified. Please verify before login.")

        return AuthResult(
            success=True,
            user={
                "id": user["id"],
                "email": user["email"],
                "is_verified": user["is_verified"],
                "is_active": user["is_active"],
            },
        )

    def create_session(self, user_id: str, user_agent: str = "", ip_address: str = "", ttl_hours: int = 12) -> Dict[str, Any]:
        self.ensure_schema()
        session_id = secrets.token_urlsafe(48)
        expires_at = _utc_now() + timedelta(hours=max(1, ttl_hours))

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO sessions (id, user_id, expires_at, user_agent, ip_address, last_seen_at)
                    VALUES (%s, %s, %s, %s, %s, NOW());
                    """,
                    (session_id, user_id, expires_at, (user_agent or "")[:512], (ip_address or "")[:128]),
                )
            conn.commit()

        return {"id": session_id, "user_id": user_id, "expires_at": _to_iso(expires_at)}

    def validate_session(self, session_id: str, user_id: str) -> bool:
        self.ensure_schema()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT expires_at, revoked_at
                    FROM sessions
                    WHERE id = %s AND user_id = %s
                    LIMIT 1;
                    """,
                    (session_id, user_id),
                )
                row = cur.fetchone()
                if row is None:
                    return False

                expires_at, revoked_at = row
                parsed_expiry = _parse_ts(expires_at)
                if revoked_at is not None or parsed_expiry is None or parsed_expiry <= _utc_now():
                    return False

                cur.execute("UPDATE sessions SET last_seen_at = NOW() WHERE id = %s;", (session_id,))
            conn.commit()
        return True

    def revoke_session(self, session_id: str, user_id: str) -> None:
        self.ensure_schema()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE sessions SET revoked_at = NOW() WHERE id = %s AND user_id = %s AND revoked_at IS NULL;",
                    (session_id, user_id),
                )
            conn.commit()

    def revoke_all_sessions(self, user_id: str) -> None:
        self.ensure_schema()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE sessions SET revoked_at = NOW() WHERE user_id = %s AND revoked_at IS NULL;",
                    (user_id,),
                )
            conn.commit()

    def create_reset_token(self, email: str, valid_for_minutes: int = 30) -> AuthResult:
        normalized_email = (email or "").strip().lower()
        if not self.validate_email(normalized_email):
            return AuthResult(success=False, error="Please provide a valid email address.")

        user = self._get_user_by_email(normalized_email)
        if user is None:
            return AuthResult(success=True)

        temp_password = self._generate_verification_otp(10)
        new_password_hash = self._hash_password(temp_password)
        old_password_hash = user.get("password_hash", "")

        self.ensure_schema()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE users SET password_hash = %s, updated_at = NOW() WHERE id = %s;",
                    (new_password_hash, user["id"]),
                )
                cur.execute(
                    "UPDATE sessions SET revoked_at = NOW() WHERE user_id = %s AND revoked_at IS NULL;",
                    (user["id"],),
                )
            conn.commit()

        try:
            self._send_email(
                normalized_email,
                "CureHelp+ temporary password",
                (
                    "We received a forgot password request for your CureHelp+ account.\n\n"
                    f"Temporary password: {temp_password}\n\n"
                    "Use this temporary password to login, then change your password from profile settings."
                ),
            )
        except Exception as exc:
            logger.exception("Failed to send temporary password email to %s", normalized_email)
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE users SET password_hash = %s, updated_at = NOW() WHERE id = %s;",
                        (old_password_hash, user["id"]),
                    )
                conn.commit()
            return AuthResult(success=False, error=str(exc))

        return AuthResult(success=True)

    def reset_password(self, raw_token: str, new_password: str) -> AuthResult:
        if len((new_password or "").strip()) < 8:
            return AuthResult(success=False, error="Password must be at least 8 characters long.")

        token_hash = self._token_hash((raw_token or "").strip())
        if not token_hash:
            return AuthResult(success=False, error="Invalid reset token.")

        now = _utc_now()
        self.ensure_schema()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, user_id, expires_at, used_at
                    FROM reset_tokens
                    WHERE token_hash = %s
                    LIMIT 1;
                    """,
                    (token_hash,),
                )
                row = cur.fetchone()
                if row is None:
                    return AuthResult(success=False, error="Invalid or expired reset token.")

                token_id, user_id, expires_at, used_at = row
                parsed_expiry = _parse_ts(expires_at)
                if used_at is not None or parsed_expiry is None or parsed_expiry <= now:
                    return AuthResult(success=False, error="Invalid or expired reset token.")

                password_hash = self._hash_password(new_password)
                cur.execute(
                    "UPDATE users SET password_hash = %s, updated_at = NOW() WHERE id = %s;",
                    (password_hash, user_id),
                )
                cur.execute("UPDATE reset_tokens SET used_at = NOW() WHERE id = %s;", (token_id,))
                cur.execute("UPDATE sessions SET revoked_at = NOW() WHERE user_id = %s AND revoked_at IS NULL;", (user_id,))
            conn.commit()

        return AuthResult(success=True)

    def update_password(self, user_id: str, current_password: str, new_password: str) -> AuthResult:
        if len((new_password or "").strip()) < 8:
            return AuthResult(success=False, error="Password must be at least 8 characters long.")

        self.ensure_schema()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT password_hash FROM users WHERE id = %s LIMIT 1;", (user_id,))
                row = cur.fetchone()
                if row is None:
                    return AuthResult(success=False, error="User not found.")
                if not self._check_password(current_password or "", row[0] or ""):
                    return AuthResult(success=False, error="Current password is incorrect.")

                new_hash = self._hash_password(new_password)
                cur.execute("UPDATE users SET password_hash = %s, updated_at = NOW() WHERE id = %s;", (new_hash, user_id))
                cur.execute("UPDATE sessions SET revoked_at = NOW() WHERE user_id = %s AND revoked_at IS NULL;", (user_id,))
            conn.commit()
        return AuthResult(success=True)

    def list_users(self, search: str = "") -> List[Dict[str, Any]]:
        self.ensure_schema()
        query = (search or "").strip()
        with self._connect() as conn:
            with conn.cursor() as cur:
                if query:
                    cur.execute(
                        """
                        SELECT id, email, is_verified, is_active, created_at, updated_at
                        FROM users
                        WHERE email ILIKE %s
                        ORDER BY created_at DESC;
                        """,
                        (f"%{query}%",),
                    )
                else:
                    cur.execute(
                        """
                        SELECT id, email, is_verified, is_active, created_at, updated_at
                        FROM users
                        ORDER BY created_at DESC;
                        """
                    )
                rows = cur.fetchall()

        users = []
        for row in rows:
            users.append(
                {
                    "id": row[0],
                    "email": row[1],
                    "is_verified": bool(row[2]),
                    "is_active": bool(row[3]),
                    "created_at": _to_iso(row[4]) if row[4] else None,
                    "updated_at": _to_iso(row[5]) if row[5] else None,
                }
            )
        return users

    def set_user_active(self, user_id: str, is_active: bool) -> bool:
        self.ensure_schema()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE users SET is_active = %s, updated_at = NOW() WHERE id = %s;",
                    (bool(is_active), user_id),
                )
                updated = cur.rowcount > 0
                if not is_active:
                    cur.execute("UPDATE sessions SET revoked_at = NOW() WHERE user_id = %s AND revoked_at IS NULL;", (user_id,))
            conn.commit()
        return updated

    def force_password_reset(self, user_id: str) -> bool:
        self.ensure_schema()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT email FROM users WHERE id = %s LIMIT 1;", (user_id,))
                row = cur.fetchone()
                if row is None:
                    return False
                email = row[0]
            conn.commit()

        result = self.create_reset_token(email)
        return result.success

    def delete_user(self, user_id: str) -> bool:
        self.ensure_schema()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM users WHERE id = %s;", (user_id,))
                deleted = cur.rowcount > 0
            conn.commit()
        return deleted


auth_manager = AuthManager()

__all__ = ["AuthManager", "AuthResult", "auth_manager"]
