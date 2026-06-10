"""Tests for H3 review fix part 1: GCSSyncManager.flush() must delete
the `.unsynced/<rel>` staged blob for any file it just uploaded
canonically.

Regression: before H3, a staged blob written by a previous teardown
could linger on GCS even after a fresh container uploaded a newer
canonical. The next cold resume's promotion path would then clobber
the newer canonical with the older staged bytes — the data-loss
window this fix closes.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.services.gcs_sync import GCSSyncManager


@pytest.fixture
def manager(tmp_path):
    return GCSSyncManager("bkt", "sid", str(tmp_path))


@pytest.fixture
def mock_upload():
    upload = MagicMock()
    with patch("app.services.gcs_sync._get_upload_file", return_value=upload):
        yield upload


def test_flush_deletes_staged_blob_after_canonical_upload(
    mock_upload, manager, tmp_path, monkeypatch,
):
    """After upload_file succeeds for rel_path, flush() must call
    gcs_storage.delete_staged_blob(bucket, session_id, rel_path)."""
    (tmp_path / "state.json").write_text("{}")
    manager.mark_dirty("state.json")

    delete_calls: list[tuple[str, str, str]] = []

    def fake_delete_staged(bucket, sid, rel):
        delete_calls.append((bucket, sid, rel))

    import app.services.gcs_storage as gcs_storage
    monkeypatch.setattr(gcs_storage, "delete_staged_blob", fake_delete_staged)

    result = manager.flush()

    assert result.ok
    assert delete_calls == [("bkt", "sid", "state.json")], (
        f"expected staged-blob cleanup after successful canonical upload; "
        f"got {delete_calls}"
    )


def test_flush_does_not_delete_staged_for_failed_uploads(
    mock_upload, manager, tmp_path, monkeypatch,
):
    """If the canonical upload failed, the staged blob is the ONLY copy
    of the bytes — we MUST NOT delete it. Only successful canonical
    uploads trigger staged cleanup."""
    (tmp_path / "masks.json").write_text("{}")
    mock_upload.side_effect = RuntimeError("gcs transient")

    manager.mark_dirty("masks.json")

    delete_calls: list[str] = []

    def fake_delete_staged(bucket, sid, rel):
        delete_calls.append(rel)

    import app.services.gcs_storage as gcs_storage
    monkeypatch.setattr(gcs_storage, "delete_staged_blob", fake_delete_staged)

    result = manager.flush()

    assert not result.ok
    assert delete_calls == [], (
        "staged blob must NOT be deleted when canonical upload failed"
    )


def test_flush_staged_cleanup_is_best_effort(
    mock_upload, manager, tmp_path, monkeypatch,
):
    """If delete_staged_blob raises (e.g. transient GCS), flush still
    returns success for the canonical upload. Cleanup is best-effort."""
    (tmp_path / "state.json").write_text("{}")
    manager.mark_dirty("state.json")

    def fake_delete_raises(bucket, sid, rel):
        raise RuntimeError("delete blew up")

    import app.services.gcs_storage as gcs_storage
    monkeypatch.setattr(gcs_storage, "delete_staged_blob", fake_delete_raises)

    result = manager.flush()
    assert result.ok
    assert "state.json" not in manager._dirty


def test_flush_no_staged_cleanup_when_nothing_uploaded(
    mock_upload, manager, monkeypatch,
):
    """If flush() uploads nothing (empty dirty set OR all files missing
    on disk), delete_staged_blob is not called."""
    delete_calls: list[str] = []

    def fake_delete(bucket, sid, rel):
        delete_calls.append(rel)

    import app.services.gcs_storage as gcs_storage
    monkeypatch.setattr(gcs_storage, "delete_staged_blob", fake_delete)

    # Empty dirty set
    result = manager.flush()
    assert result.ok
    assert delete_calls == []

    # Dirty rel_path but local file missing — no upload → no staged cleanup
    manager.mark_dirty("missing.json")
    result = manager.flush()
    assert result.ok
    assert delete_calls == [], "no upload happened, so no staged cleanup"
