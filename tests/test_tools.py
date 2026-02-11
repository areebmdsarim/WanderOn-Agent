"""
Tests for Pydantic tool schemas and tool functions.
"""

import pytest
from pydantic import ValidationError

from src.schemas import (
    ApprovalRequest,
    FlightPolicyRequest,
    PerDiemRequest,
    VisaCheckRequest,
)
from src.tools.travel_tools import (
    check_flight_policy,
    check_visa_requirements,
    get_approval_requirements,
    get_per_diem_rate,
)


# ── Schema validation tests ────────────────────────────────────────────────


class TestSchemaValidation:

    def test_visa_valid(self):
        req = VisaCheckRequest(
            passport_country="India", destination_country="United Kingdom"
        )
        assert req.passport_country == "India"

    def test_visa_too_short(self):
        with pytest.raises(ValidationError):
            VisaCheckRequest(passport_country="I", destination_country="UK")

    def test_per_diem_valid(self):
        req = PerDiemRequest(city="Bangalore", country="India")
        assert req.city == "Bangalore"

    def test_flight_valid_classes(self):
        for cls in ("economy", "premium_economy", "business", "first"):
            req = FlightPolicyRequest(origin="BLR", destination="LHR", cabin_class=cls)
            assert req.cabin_class == cls

    def test_flight_invalid_class(self):
        with pytest.raises(ValidationError):
            FlightPolicyRequest(
                origin="BLR", destination="LHR", cabin_class="ultra_luxury"
            )

    def test_approval_valid(self):
        req = ApprovalRequest(trip_cost=50000, destination_type="domestic")
        assert req.trip_cost == 50000

    def test_approval_invalid_type(self):
        with pytest.raises(ValidationError):
            ApprovalRequest(trip_cost=100, destination_type="unknown")

    def test_approval_negative_cost(self):
        with pytest.raises(ValidationError):
            ApprovalRequest(trip_cost=-100, destination_type="domestic")


# ── Tool execution tests ───────────────────────────────────────────────────


class TestToolExecution:

    def test_visa_known_pair(self):
        req = VisaCheckRequest(
            passport_country="India", destination_country="United Kingdom"
        )
        result = check_visa_requirements(req)
        assert result.ok
        assert result.data["requires_visa"] is True

    def test_visa_unknown_pair(self):
        req = VisaCheckRequest(passport_country="Brazil", destination_country="Finland")
        result = check_visa_requirements(req)
        assert result.ok
        assert (
            "embassy" in result.data["notes"].lower()
            or "check" in result.data["notes"].lower()
        )

    def test_per_diem_known_city(self):
        req = PerDiemRequest(city="Bangalore", country="India")
        result = get_per_diem_rate(req)
        assert result.ok
        assert result.data["daily_rate"] == 3500
        assert result.data["currency"] == "INR"

    def test_per_diem_unknown_city(self):
        req = PerDiemRequest(city="Unknown", country="Nowhere")
        result = get_per_diem_rate(req)
        assert result.ok
        assert "error" in result.data

    def test_flight_policy_economy(self):
        req = FlightPolicyRequest(
            origin="BLR", destination="DEL", cabin_class="economy"
        )
        result = check_flight_policy(req)
        assert result.ok
        assert result.data["max_cost"] == 50000

    def test_flight_policy_business(self):
        req = FlightPolicyRequest(
            origin="BLR", destination="LHR", cabin_class="business"
        )
        result = check_flight_policy(req)
        assert result.ok
        assert result.data["requires_approval"] == "VP"

    def test_approval_auto_approved(self):
        req = ApprovalRequest(trip_cost=10000, destination_type="domestic")
        result = get_approval_requirements(req)
        assert result.ok
        assert result.data["approval_status"] == "auto_approved"

    def test_approval_manager_needed(self):
        req = ApprovalRequest(trip_cost=60000, destination_type="domestic")
        result = get_approval_requirements(req)
        assert result.ok
        assert result.data["approval_status"] == "manager_approval_required"

    def test_approval_vp_needed(self):
        req = ApprovalRequest(trip_cost=200000, destination_type="domestic")
        result = get_approval_requirements(req)
        assert result.ok
        assert result.data["approval_status"] == "vp_approval_required"

    def test_approval_cxo_needed(self):
        req = ApprovalRequest(trip_cost=600000, destination_type="domestic")
        result = get_approval_requirements(req)
        assert result.ok
        assert result.data["approval_status"] == "cxo_approval_required"

    def test_approval_high_risk(self):
        req = ApprovalRequest(trip_cost=5000, destination_type="high_risk")
        result = get_approval_requirements(req)
        assert result.ok
        # High-risk always needs at least manager approval
        assert result.data["approval_status"] == "manager_approval_required"
