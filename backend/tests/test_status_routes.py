"""Tests for /api/status and /api/job/cancel endpoints."""

import pytest
from app import create_app
from app.services.pipeline import ServiceState


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True

    # Reset SAM3Service singleton state (may be dirty from other tests)
    from app.services.sam3_service import SAM3Service
    sam = SAM3Service()
    with sam._state_lock:
        sam._service_state = ServiceState()

    with app.test_client() as client:
        yield client


class TestStatusEndpoint:
    def test_returns_idle_by_default(self, client):
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["phase"] == "idle"
        assert data["session_id"] is None
        assert data["boot_id"] is not None

    def test_no_auth_required(self, client):
        # No Authorization header — should still work
        resp = client.get("/api/status")
        assert resp.status_code == 200


class TestCancelEndpoint:
    def test_cancel_nothing_running(self, client):
        resp = client.post("/api/job/cancel")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "nothing_to_cancel"


class TestDismissErrorEndpoint:
    def test_dismiss_when_not_in_error_is_noop(self, client):
        resp = client.post("/api/status/dismiss-error")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "not_in_error"

    def test_dismiss_transitions_error_to_idle(self, client):
        from app.services.sam3_service import SAM3Service
        sam = SAM3Service()
        with sam._state_lock:
            sam._service_state = ServiceState(
                phase="error", session_id="s", video_name="v", error="boom",
            )

        resp = client.post("/api/status/dismiss-error")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "dismissed"

        # Verify status endpoint now reports idle
        status = client.get("/api/status").get_json()
        assert status["phase"] == "idle"
        assert status["session_id"] is None
        assert status["error"] is None
