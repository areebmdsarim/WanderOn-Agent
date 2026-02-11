"""
Lookup tools for things like visa info and per diem rates.
These use mock data / local databases.
Inputs get validated with Pydantic first.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from loguru import logger

from src.schemas import (
    ApprovalRequest,
    FlightPolicyRequest,
    PerDiemRequest,
    ToolResponse,
    VisaCheckRequest,
)


# --- Lookup data (simulated databases) --- #

_VISA_DB: Dict[str, Dict[str, Any]] = {
    ("india", "united kingdom"): {
        "requires_visa": True,
        "visa_type": "Business",
        "processing_days": 15,
        "notes": "Business visa required; submit application at least 60 days before travel.",
    },
    ("india", "united states"): {
        "requires_visa": True,
        "visa_type": "B1/B2",
        "processing_days": 30,
        "notes": "B1/B2 visa required. Schedule interview at US consulate.",
    },
    ("india", "singapore"): {
        "requires_visa": False,
        "visa_type": None,
        "processing_days": 0,
        "notes": "Visa-free for up to 30 days for Indian passport holders.",
    },
    ("united states", "united kingdom"): {
        "requires_visa": False,
        "visa_type": None,
        "processing_days": 0,
        "notes": "ESTA/Visa Waiver — no visa needed for stays under 90 days.",
    },
}

_PER_DIEM_DB: Dict[str, Dict[str, Any]] = {
    ("bangalore", "india"): {
        "daily_rate": 3500,
        "currency": "INR",
        "includes": "meals and incidentals",
    },
    ("mumbai", "india"): {
        "daily_rate": 4000,
        "currency": "INR",
        "includes": "meals and incidentals",
    },
    ("new delhi", "india"): {
        "daily_rate": 3800,
        "currency": "INR",
        "includes": "meals and incidentals",
    },
    ("london", "united kingdom"): {
        "daily_rate": 150,
        "currency": "GBP",
        "includes": "meals and incidentals",
    },
    ("new york", "united states"): {
        "daily_rate": 200,
        "currency": "USD",
        "includes": "meals and incidentals",
    },
    ("san francisco", "united states"): {
        "daily_rate": 220,
        "currency": "USD",
        "includes": "meals and incidentals",
    },
    ("singapore", "singapore"): {
        "daily_rate": 250,
        "currency": "SGD",
        "includes": "meals and incidentals",
    },
    ("tokyo", "japan"): {
        "daily_rate": 18000,
        "currency": "JPY",
        "includes": "meals and incidentals",
    },
    ("dubai", "uae"): {
        "daily_rate": 700,
        "currency": "AED",
        "includes": "meals and incidentals",
    },
}

_FLIGHT_POLICY: Dict[str, Dict[str, Any]] = {
    "economy": {"max_cost": 50000, "advance_booking_days": 7, "refundable": False},
    "premium_economy": {
        "max_cost": 80000,
        "advance_booking_days": 14,
        "refundable": False,
    },
    "business": {
        "max_cost": 200000,
        "advance_booking_days": 21,
        "refundable": True,
        "requires_approval": "VP",
    },
    "first": {
        "max_cost": 500000,
        "advance_booking_days": 30,
        "refundable": True,
        "requires_approval": "CXO",
    },
}

_APPROVAL_THRESHOLDS = {
    "domestic": {
        "auto_approve_limit": 25000,
        "manager_limit": 100000,
        "vp_limit": 500000,
    },
    "international": {
        "auto_approve_limit": 50000,
        "manager_limit": 200000,
        "vp_limit": 1000000,
    },
    "high_risk": {"auto_approve_limit": 0, "manager_limit": 50000, "vp_limit": 200000},
}


# --- Tool functions --- #


def check_visa_requirements(req: VisaCheckRequest) -> ToolResponse:
    key = (req.passport_country.lower(), req.destination_country.lower())
    data = _VISA_DB.get(key)
    if data is None:
        data = {
            "requires_visa": True,
            "notes": f"Couldn't find specific visa info for {req.passport_country} to {req.destination_country}. "
            "Suggest checking with the embassy or consulate directly.",
        }
    return ToolResponse(tool="check_visa_requirements", data=data, source="visa-db-v1")


def get_per_diem_rate(req: PerDiemRequest) -> ToolResponse:
    target_city = req.city.lower()

    # Direct lookup if country is provided and valid
    valid_country = req.country and req.country.lower() not in [
        "<unknown>",
        "unknown",
        "none",
        "n/a",
        "null",
        "",
    ]

    if valid_country:
        key = (target_city, req.country.lower())
        data = _PER_DIEM_DB.get(key)
    else:
        # Fuzzy lookup: find city in DB (assuming unique cities for now)
        data = None
        for (city, country), info in _PER_DIEM_DB.items():
            if city == target_city:
                # IMPORTANT: use a copy to avoid mutating the DB record
                data = info.copy()
                # Add country to data for clarity
                data = {**data, "country": country.title(), "inferred": True}
                break

    if data is None:
        country_str = (
            f", {req.country}"
            if req.country and req.country.lower() != "<unknown>"
            else ""
        )
        data = {"error": f"No per-diem data found for {req.city}{country_str}."}

    return ToolResponse(tool="get_per_diem_rate", data=data, source="per-diem-db-v1")


def check_flight_policy(req: FlightPolicyRequest) -> ToolResponse:
    policy = _FLIGHT_POLICY.get(req.cabin_class)
    if policy is None:
        policy = {"error": f"Unknown cabin class: {req.cabin_class}"}
    data = {
        "origin": req.origin,
        "destination": req.destination,
        "cabin_class": req.cabin_class,
        **policy,
    }
    return ToolResponse(
        tool="check_flight_policy", data=data, source="flight-policy-v1"
    )


def get_approval_requirements(req: ApprovalRequest) -> ToolResponse:
    thresholds = _APPROVAL_THRESHOLDS.get(req.destination_type)
    if thresholds is None:
        return ToolResponse(
            tool="get_approval_requirements",
            data={"error": f"Unknown destination type: {req.destination_type}"},
            source="approval-policy-v1",
        )

    if req.trip_cost <= thresholds["auto_approve_limit"]:
        approval = "auto_approved"
        approver = "system"
    elif req.trip_cost <= thresholds["manager_limit"]:
        approval = "manager_approval_required"
        approver = "direct manager"
    elif req.trip_cost <= thresholds["vp_limit"]:
        approval = "vp_approval_required"
        approver = "VP"
    else:
        approval = "cxo_approval_required"
        approver = "CXO / CFO"

    data = {
        "trip_cost": req.trip_cost,
        "destination_type": req.destination_type,
        "approval_status": approval,
        "approver": approver,
        "thresholds": thresholds,
    }
    return ToolResponse(
        tool="get_approval_requirements", data=data, source="approval-policy-v1"
    )


# --- Allow-list registry --- #

TOOL_REGISTRY: Dict[str, Any] = {
    "check_visa_requirements": {
        "fn": check_visa_requirements,
        "schema": VisaCheckRequest,
    },
    "get_per_diem_rate": {
        "fn": get_per_diem_rate,
        "schema": PerDiemRequest,
    },
    "check_flight_policy": {
        "fn": check_flight_policy,
        "schema": FlightPolicyRequest,
    },
    "get_approval_requirements": {
        "fn": get_approval_requirements,
        "schema": ApprovalRequest,
    },
}
