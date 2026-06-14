"""delete_session_gcs must be idempotent and race-safe.

A blob can disappear between the `list_blobs` snapshot and the per-blob
`delete()` — a concurrent GCSSyncManager flush, or a repeated DELETE of the
same session. The per-blob delete must treat that as already-gone, not raise
(regression 2026-06-14: double-DELETE of a just-closed session 500'd because
`blob.delete()` raised NotFound mid-loop).
"""

from google.api_core.exceptions import NotFound


def test_delete_session_gcs_tolerates_blob_vanishing_between_list_and_delete(monkeypatch):
    import app.services.gcs_storage as gcs_storage

    class _Blob:
        def __init__(self, name, vanish=False):
            self.name = name
            self._vanish = vanish

        def delete(self):
            if self._vanish:
                raise NotFound(f"{self.name} already gone")

    blobs = [
        _Blob("sid/meta.json"),
        _Blob("sid/state.json", vanish=True),  # deleted by a concurrent writer
        _Blob("sid/masks.json"),
    ]

    class _Client:
        def bucket(self, name):
            return object()

        def list_blobs(self, bucket, prefix=None):
            return iter(blobs)

    monkeypatch.setattr(gcs_storage, "_get_client", lambda: _Client())

    # Must not raise; returns the count actually deleted (the vanished one is
    # tolerated as already-gone).
    deleted = gcs_storage.delete_session_gcs("test-bucket", "sid")
    assert deleted == 2


def test_delete_session_gcs_empty_prefix_is_noop(monkeypatch):
    import app.services.gcs_storage as gcs_storage

    class _Client:
        def bucket(self, name):
            return object()

        def list_blobs(self, bucket, prefix=None):
            return iter([])

    monkeypatch.setattr(gcs_storage, "_get_client", lambda: _Client())

    assert gcs_storage.delete_session_gcs("test-bucket", "missing") == 0
