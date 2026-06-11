"""Shared-password auth: when AUTH_PASSWORD is set, every request must
carry a matching X-Auth-Token header — including /api/health, because the
frontend heartbeat keeps the billed GPU container alive. When unset
(local dev), everything stays open."""

import pytest

from app import create_app
from app.services.pipeline import ServiceState

PASSWORD = "crimson-otter-lantern"


def make_client(monkeypatch, password):
    if password is None:
        monkeypatch.delenv("AUTH_PASSWORD", raising=False)
    else:
        monkeypatch.setenv("AUTH_PASSWORD", password)
    app = create_app()
    app.config["TESTING"] = True

    # Reset SAM3Service singleton state (may be dirty from other tests)
    from app.services.sam3_service import SAM3Service
    sam = SAM3Service()
    with sam._state_lock:
        sam._service_state = ServiceState()

    return app.test_client()


class TestAuthEnabled:
    def test_missing_header_is_401(self, monkeypatch):
        client = make_client(monkeypatch, PASSWORD)
        resp = client.get("/api/health")
        assert resp.status_code == 401
        assert resp.get_json() == {"error": "unauthorized"}

    def test_wrong_password_is_401(self, monkeypatch):
        client = make_client(monkeypatch, PASSWORD)
        resp = client.get("/api/health", headers={"X-Auth-Token": "wrong-guess-here"})
        assert resp.status_code == 401

    def test_correct_password_is_200(self, monkeypatch):
        client = make_client(monkeypatch, PASSWORD)
        resp = client.get("/api/health", headers={"X-Auth-Token": PASSWORD})
        assert resp.status_code == 200
        assert resp.get_json()["ok"] is True

    def test_status_requires_auth_too(self, monkeypatch):
        client = make_client(monkeypatch, PASSWORD)
        assert client.get("/api/status").status_code == 401
        resp = client.get("/api/status", headers={"X-Auth-Token": PASSWORD})
        assert resp.status_code == 200

    def test_options_preflight_passes_without_token(self, monkeypatch):
        # CORS preflights never carry custom headers; they must not 401.
        client = make_client(monkeypatch, PASSWORD)
        resp = client.options("/api/status")
        assert resp.status_code != 401


class TestAuthDisabled:
    def test_no_password_means_open(self, monkeypatch):
        client = make_client(monkeypatch, None)
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/status").status_code == 200
