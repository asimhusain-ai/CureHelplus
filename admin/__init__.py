"""Admin blueprint providing a simple dashboard for CureHelp+."""
from __future__ import annotations

import os
from collections import Counter
from datetime import datetime
from functools import wraps
from typing import Any, Dict, List

from flask import Blueprint, current_app, flash, redirect, render_template, request, session, url_for

from auth_manager import auth_manager
from profile_manager import profile_manager
from user_data_store import user_data_store

admin_bp = Blueprint("admin", __name__, url_prefix="/admin")

_RETURN_TO_KEY = "admin_return_to"
_ALLOWED_DASHBOARD_TABS = {
    "user-dashboard",
    "diabetes",
    "heart",
    "anemia",
    "pneumonia",
    "tuberculosis",
    "consultants",
    "profiles",
}


def _normalise_return_to(value: str | None) -> str | None:
    token = str(value or "").strip().lower()
    if not token:
        return None
    if token in {"landing", "resources"}:
        return token
    if token.startswith("dashboard:"):
        _, tab = token.split(":", 1)
        if tab in _ALLOWED_DASHBOARD_TABS:
            return f"dashboard:{tab}"
    return None


def _capture_return_to_from_request() -> None:
    token = _normalise_return_to(request.args.get("from"))
    if token:
        session[_RETURN_TO_KEY] = token
        session.modified = True


def _home_redirect_url() -> str:
    token = _normalise_return_to(session.get(_RETURN_TO_KEY))
    if not token:
        return url_for("index")
    return url_for("index", return_to=token)


def _admin_credentials() -> Dict[str, str]:
    """Resolve admin credentials from config or environment."""
    username = current_app.config.get("ADMIN_USERNAME") or os.getenv("CUREHELP_ADMIN_USER", "admin")
    password = current_app.config.get("ADMIN_PASSWORD") or os.getenv("CUREHELP_ADMIN_PASS", "curehelp")
    return {"username": username, "password": password}


def _is_admin_authenticated() -> bool:
    return session.get("is_admin", False) is True


def admin_required(view):
    """Decorator to ensure admin login."""

    @wraps(view)
    def wrapped(*args: Any, **kwargs: Any):
        if not _is_admin_authenticated():
            session["admin_next"] = request.full_path if request.query_string else request.path
            _capture_return_to_from_request()
            session.modified = True
            return redirect(url_for("admin.login"))
        return view(*args, **kwargs)

    return wrapped


@admin_bp.route("/login", methods=["GET", "POST"])
def login():
    _capture_return_to_from_request()
    if _is_admin_authenticated():
        return redirect(url_for("admin.dashboard"))

    error = None
    form_values = {"username": "", "password": ""}
    if request.method == "POST":
        credentials = _admin_credentials()
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""
        form_values["username"] = username
        form_values["password"] = password
        if username == credentials["username"] and password == credentials["password"]:
            session["is_admin"] = True
            session["admin_username"] = username
            next_url = session.pop("admin_next", None)
            session.modified = True
            return redirect(next_url or url_for("admin.dashboard"))
        error = "Incorrect Password"

    return render_template("admin/login.html", error=error, form_values=form_values)


@admin_bp.route("/logout", methods=["POST"])
@admin_required
def logout():
    session.pop("is_admin", None)
    session.pop("admin_username", None)
    session.modified = True
    return redirect(url_for("admin.login"))


@admin_bp.route("/home", methods=["GET"])
def go_home():
    return redirect(_home_redirect_url())


def _parse_timestamp(ts: str | None) -> datetime:
    if not ts:
        return datetime.min
    for fmt in ("%d-%b-%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%d/%m/%Y %H:%M"):
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    return datetime.min


def _risk_level(probability: float | None) -> str:
    if probability is None:
        return "unknown"
    if probability >= 75:
        return "critical"
    if probability >= 50:
        return "elevated"
    return "stable"


def _collect_dashboard_metrics(user_search: str = "") -> Dict[str, Any]:
    profiles = profile_manager.list_profiles()
    total_profiles = len(profiles)
    gender_counter = Counter()
    disease_counter = Counter()
    total_predictions = 0
    high_risk_count = 0

    enriched_profiles: List[Dict[str, Any]] = []
    for profile in profiles:
        gender = (profile.get("gender") or "Unknown").title()
        gender_counter[gender] += 1
        predictions = profile.get("predictions") or {}
        total_predictions += len(predictions)
        diseases = []
        highest_prob = None
        for disease, payload in predictions.items():
            disease_counter[disease] += 1
            probability = payload.get("prob") if isinstance(payload, dict) else None
            if isinstance(probability, (int, float)) and probability >= 70:
                high_risk_count += 1
            if isinstance(probability, (int, float)):
                highest_prob = max(highest_prob or probability, probability)
            diseases.append({
                "name": disease,
                "prob": probability,
                "risk": _risk_level(probability if isinstance(probability, (int, float)) else None),
            })
        enriched_profiles.append({
            "id": profile.get("id", "-"),
            "name": profile.get("name", "Unknown"),
            "gender": gender,
            "last_updated": profile.get("last_updated") or profile.get("created_at", ""),
            "diseases": diseases,
            "highest_prob": highest_prob,
        })

    recent_profiles = sorted(enriched_profiles, key=lambda p: _parse_timestamp(p.get("last_updated")), reverse=True)

    total_predictions = max(total_predictions, 0)
    disease_breakdown = []
    if disease_counter:
        for disease, count in disease_counter.most_common():
            percent = round((count / sum(disease_counter.values())) * 100)
            disease_breakdown.append({
                "name": disease,
                "count": count,
                "percent": percent,
            })

    gender_breakdown = [
        {"label": gender, "count": count, "percent": round((count / total_profiles) * 100) if total_profiles else 0}
        for gender, count in gender_counter.most_common()
    ]

    metrics = {
        "total_profiles": total_profiles,
        "total_predictions": total_predictions,
        "high_risk": high_risk_count,
        "gender_breakdown": gender_breakdown,
        "disease_breakdown": disease_breakdown,
        "recent_profiles": recent_profiles,
        "last_refresh": datetime.now().strftime("%d %b %Y, %H:%M"),
    }

    auth_users: List[Dict[str, Any]] = []
    try:
        if auth_manager.available:
            auth_manager.ensure_schema()
            auth_users = auth_manager.list_users(user_search)
    except Exception:
        auth_users = []

    metrics["auth_users"] = auth_users
    metrics["auth_users_count"] = len(auth_users)
    metrics["auth_search"] = user_search

    return metrics


@admin_bp.route("/")
@admin_required
def dashboard():
    _capture_return_to_from_request()
    user_search = (request.args.get("user_q") or "").strip()
    metrics = _collect_dashboard_metrics(user_search)
    return render_template(
        "admin/dashboard.html",
        metrics=metrics,
        admin_username=session.get("admin_username", "Administrator"),
    )


@admin_bp.route("/patients/<profile_id>/delete", methods=["POST"])
@admin_required
def delete_patient(profile_id: str):
    profile_manager.delete_profile(profile_id)
    return redirect(url_for("admin.dashboard"))


@admin_bp.route("/users/<user_id>/activate", methods=["POST"])
@admin_required
def activate_user(user_id: str):
    try:
        if auth_manager.set_user_active(user_id, True):
            flash("User activated.", "success")
        else:
            flash("User not found.", "error")
    except Exception as exc:
        flash(f"Unable to activate user: {exc}", "error")
    return redirect(url_for("admin.dashboard", user_q=request.args.get("user_q", "")))


@admin_bp.route("/users/<user_id>/deactivate", methods=["POST"])
@admin_required
def deactivate_user(user_id: str):
    try:
        if auth_manager.set_user_active(user_id, False):
            flash("User deactivated.", "success")
        else:
            flash("User not found.", "error")
    except Exception as exc:
        flash(f"Unable to deactivate user: {exc}", "error")
    return redirect(url_for("admin.dashboard", user_q=request.args.get("user_q", "")))


@admin_bp.route("/users/<user_id>/force-reset", methods=["POST"])
@admin_required
def force_reset_user(user_id: str):
    try:
        if auth_manager.force_password_reset(user_id):
            flash("Password reset email sent.", "success")
        else:
            flash("User not found.", "error")
    except Exception as exc:
        flash(f"Unable to force reset: {exc}", "error")
    return redirect(url_for("admin.dashboard", user_q=request.args.get("user_q", "")))


@admin_bp.route("/users/<user_id>/delete", methods=["POST"])
@admin_required
def delete_user(user_id: str):
    try:
        deleted = auth_manager.delete_user(user_id)
        if deleted:
            user_data_store.remove_user(user_id)
            flash("User and related data deleted.", "success")
        else:
            flash("User not found.", "error")
    except Exception as exc:
        flash(f"Unable to delete user: {exc}", "error")
    return redirect(url_for("admin.dashboard", user_q=request.args.get("user_q", "")))
