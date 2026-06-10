"""Unit tests for the .unsynced marker module."""
import json
import os
import pytest

from app.services.unsynced_marker import write_marker, read_marker, clear_marker, MARKER_FILENAME


def test_write_and_read_roundtrip(tmp_path):
    write_marker(str(tmp_path), ["state.json", "masks.json"], reason="test")
    files = read_marker(str(tmp_path))
    assert files == ["masks.json", "state.json"]  # sorted


def test_read_returns_none_when_absent(tmp_path):
    assert read_marker(str(tmp_path)) is None


def test_write_with_empty_list_is_noop(tmp_path):
    write_marker(str(tmp_path), [], reason="test")
    assert read_marker(str(tmp_path)) is None


def test_clear_removes_marker(tmp_path):
    write_marker(str(tmp_path), ["state.json"])
    assert read_marker(str(tmp_path)) == ["state.json"]
    clear_marker(str(tmp_path))
    assert read_marker(str(tmp_path)) is None


def test_clear_is_idempotent(tmp_path):
    clear_marker(str(tmp_path))  # no-op when no marker
    clear_marker(str(tmp_path))  # still no-op


def test_read_handles_corrupt_marker(tmp_path):
    path = os.path.join(str(tmp_path), MARKER_FILENAME)
    with open(path, "w") as f:
        f.write("not-json{{")
    assert read_marker(str(tmp_path)) is None


def test_write_deduplicates_files(tmp_path):
    write_marker(str(tmp_path), ["a", "a", "b"], reason="test")
    assert read_marker(str(tmp_path)) == ["a", "b"]


# ------------------------------------------------------------------
# B2: persist_unsynced — GCS staging + marker upload
# ------------------------------------------------------------------

def test_persist_unsynced_stages_files_to_gcs(tmp_path, monkeypatch):
    """persist_unsynced uploads each failed file to the staging prefix
    so scale-to-zero recovery has the bytes even if the canonical upload
    couldn't complete."""
    from app.services.unsynced_marker import persist_unsynced

    sdir = tmp_path / "sess"
    sdir.mkdir()
    (sdir / "state.json").write_text('{"real": true}')
    (sdir / "masks.json").write_text("{}")

    staged: list[tuple[str, str, str, str]] = []

    def fake_stage(bucket, session_id, rel_path, local_path):
        staged.append((bucket, session_id, rel_path, local_path))

    def fake_upload(bucket, session_id, rel_path, local_path):
        pass  # marker upload

    import app.services.gcs_storage as gcs_storage
    monkeypatch.setattr(gcs_storage, "stage_unsynced_file", fake_stage, raising=False)
    monkeypatch.setattr(gcs_storage, "upload_file", fake_upload, raising=False)

    result = persist_unsynced(
        "bkt", "sess", str(sdir),
        ["state.json", "masks.json"], reason="test",
    )
    assert set(result) == {"state.json", "masks.json"}
    assert {s[2] for s in staged} == {"state.json", "masks.json"}


def test_persist_unsynced_writes_local_marker(tmp_path, monkeypatch):
    """Even when GCS staging succeeds, the local marker is still
    written (warm-container recovery path)."""
    from app.services.unsynced_marker import persist_unsynced, read_marker

    sdir = tmp_path / "sess"
    sdir.mkdir()
    (sdir / "state.json").write_text("{}")

    import app.services.gcs_storage as gcs_storage
    monkeypatch.setattr(gcs_storage, "stage_unsynced_file",
                        lambda *a, **kw: None, raising=False)
    monkeypatch.setattr(gcs_storage, "upload_file",
                        lambda *a, **kw: None, raising=False)

    persist_unsynced("bkt", "sess", str(sdir), ["state.json"], reason="t")
    assert read_marker(str(sdir)) == ["state.json"]


def test_persist_unsynced_uploads_marker_to_gcs(tmp_path, monkeypatch):
    """The marker itself must be uploaded to GCS so cold containers
    (scale-to-zero) can still find the recovery index."""
    from app.services.unsynced_marker import persist_unsynced, MARKER_FILENAME

    sdir = tmp_path / "sess"
    sdir.mkdir()
    (sdir / "state.json").write_text("{}")

    uploaded: list[str] = []

    def fake_upload(bucket, session_id, rel_path, local_path):
        uploaded.append(rel_path)

    import app.services.gcs_storage as gcs_storage
    monkeypatch.setattr(gcs_storage, "stage_unsynced_file",
                        lambda *a, **kw: None, raising=False)
    monkeypatch.setattr(gcs_storage, "upload_file", fake_upload, raising=False)

    persist_unsynced("bkt", "sess", str(sdir), ["state.json"], reason="t")
    assert MARKER_FILENAME in uploaded


def test_persist_unsynced_tolerates_stage_failure(tmp_path, monkeypatch):
    """If staging fails, the local marker is still written with the
    original failed list so warm-container recovery still works."""
    from app.services.unsynced_marker import persist_unsynced, read_marker

    sdir = tmp_path / "sess"
    sdir.mkdir()
    (sdir / "state.json").write_text("{}")

    def failing_stage(*a, **kw):
        raise RuntimeError("gcs staging down")

    import app.services.gcs_storage as gcs_storage
    monkeypatch.setattr(gcs_storage, "stage_unsynced_file",
                        failing_stage, raising=False)
    monkeypatch.setattr(gcs_storage, "upload_file",
                        lambda *a, **kw: None, raising=False)

    result = persist_unsynced("bkt", "sess", str(sdir),
                              ["state.json"], reason="t")
    assert result == []  # nothing staged
    assert read_marker(str(sdir)) == ["state.json"]


def test_persist_unsynced_no_bucket_still_writes_local_marker(tmp_path):
    """Local-only mode (bucket=None): skip GCS calls but still write
    the local marker so the next resume's recovery loop can upload."""
    from app.services.unsynced_marker import persist_unsynced, read_marker

    sdir = tmp_path / "sess"
    sdir.mkdir()
    (sdir / "state.json").write_text("{}")

    result = persist_unsynced(None, "sess", str(sdir),
                              ["state.json"], reason="t")
    assert result == []
    assert read_marker(str(sdir)) == ["state.json"]


def test_persist_unsynced_skips_missing_local_files(tmp_path, monkeypatch):
    """A file listed as failed but not actually on disk is not staged —
    nothing to upload. The marker still records the intent."""
    from app.services.unsynced_marker import persist_unsynced, read_marker

    sdir = tmp_path / "sess"
    sdir.mkdir()
    # state.json exists, masks.json does NOT

    staged: list[str] = []

    def fake_stage(bucket, session_id, rel_path, local_path):
        staged.append(rel_path)

    import app.services.gcs_storage as gcs_storage
    monkeypatch.setattr(gcs_storage, "stage_unsynced_file", fake_stage, raising=False)
    monkeypatch.setattr(gcs_storage, "upload_file", lambda *a, **kw: None, raising=False)
    (sdir / "state.json").write_text("{}")

    persist_unsynced("bkt", "sess", str(sdir),
                     ["state.json", "masks.json"], reason="t")

    assert staged == ["state.json"]
    marker = read_marker(str(sdir))
    # Only successfully-staged files end up in the marker when any
    # were staged; masks.json (not on disk) isn't recoverable.
    assert marker == ["state.json"]


def test_persist_unsynced_noop_when_failed_empty(tmp_path):
    from app.services.unsynced_marker import persist_unsynced, read_marker
    sdir = tmp_path / "sess"
    sdir.mkdir()
    result = persist_unsynced("bkt", "sess", str(sdir), [], reason="t")
    assert result == []
    assert read_marker(str(sdir)) is None


def test_persist_unsynced_partial_staging_records_all_dirty_files(tmp_path, monkeypatch):
    """Mixed-success staging: some files upload to the .unsynced/ prefix,
    others raise (transient GCS 5xx). The marker must list BOTH sets so
    warm-container resume can recover the failed-to-stage files from
    local disk (they passed isfile) while cold-container resume pulls
    the staged files from GCS.

    Previous behavior dropped failed-to-stage files entirely when any
    other file staged successfully (`marker_files = staged if staged
    else sorted(failed)`), silently losing their recoverability.
    """
    from app.services.unsynced_marker import persist_unsynced, read_marker

    sdir = tmp_path / "sess"
    sdir.mkdir()
    # All four files exist on disk; half will raise during staging.
    (sdir / "state.json").write_text('{"a": 1}')
    (sdir / "masks.json").write_text("{}")
    (sdir / "prompts.json").write_text("{}")
    (sdir / "classes.json").write_text("[]")

    stage_ok = {"state.json", "prompts.json"}
    stage_fail = {"masks.json", "classes.json"}
    staged_calls: list[str] = []

    def flaky_stage(bucket, session_id, rel_path, local_path):
        staged_calls.append(rel_path)
        if rel_path in stage_fail:
            raise RuntimeError("transient gcs 5xx")

    import app.services.gcs_storage as gcs_storage
    monkeypatch.setattr(gcs_storage, "stage_unsynced_file", flaky_stage, raising=False)
    monkeypatch.setattr(gcs_storage, "upload_file",
                        lambda *a, **kw: None, raising=False)

    result = persist_unsynced(
        "bkt", "sess", str(sdir),
        sorted(stage_ok | stage_fail), reason="partial",
    )

    # Returned list: only the staged files (caller-visible success set).
    assert set(result) == stage_ok

    # Marker: EVERY dirty file that is still recoverable from somewhere —
    # staged (via .unsynced/ blobs) OR failed-to-stage (via local disk).
    marker = read_marker(str(sdir))
    assert set(marker) == stage_ok | stage_fail, (
        f"marker {marker!r} must contain all dirty files (staged "
        f"∪ failed), not just staged"
    )
