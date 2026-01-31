"""Admin blueprint providing a simple dashboard for CureHelp+."""
from __future__ import annotations

import os
from collections import Counter
from datetime import datetime
from functools import wraps
from typing import Any, Dict, List

from flask import Blueprint, current_app, redirect, render_template, request, session, url_for

from profile_manager import profile_manager

admin_bp = Blueprint("admin", __name__, url_prefix="/admin")


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
            session["admin_next"] = request.path
            session.modified = True
            return redirect(url_for("admin.login"))
        return view(*args, **kwargs)

    return wrapped


@admin_bp.route("/login", methods=["GET", "POST"])
def login():
    if _is_admin_authenticated():
        return redirect(url_for("admin.dashboard"))

    error = None
    if request.method == "POST":
        credentials = _admin_credentials()
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""
        if username == credentials["username"] and password == credentials["password"]:
            session["is_admin"] = True
            session["admin_username"] = username
            next_url = session.pop("admin_next", None)
            session.modified = True
            return redirect(next_url or url_for("admin.dashboard"))
        error = "Invalid username or password"

    return render_template("admin/login.html", error=error)


@admin_bp.route("/logout", methods=["POST"])
@admin_required
def logout():
    session.pop("is_admin", None)
    session.pop("admin_username", None)
    session.modified = True
    return redirect(url_for("admin.login"))


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


def _collect_dashboard_metrics() -> Dict[str, Any]:
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

    return metrics


@admin_bp.route("/")
@admin_required
def dashboard():
    metrics = _collect_dashboard_metrics()
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
