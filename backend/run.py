import os
import signal
import sys
import logging

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

logger = logging.getLogger(__name__)


def _handle_sigterm(signum, frame):
    """Flush dirty GCS files before Cloud Run kills the container."""
    logger.info("SIGTERM received — flushing GCS sync")
    try:
        from app.services.sam3_service import SAM3Service
        sam = SAM3Service()
        state = sam.get_service_state()
        if state.session_id:
            sam.cancel_propagation(state.session_id)
        sam.cancel_pipeline()
    except Exception:
        logger.warning("sigterm | cancel failed", exc_info=True)

    try:
        from app.config import get_sync_manager
        sm = get_sync_manager()
        if sm:
            result = sm.flush_with_retry(max_attempts=3, initial_backoff_s=0.5, backoff_factor=2.0)
            if not result.ok:
                from app.services.unsynced_marker import write_marker
                write_marker(sm.session_dir, sorted(result.failed), reason="sigterm_flush_failed")
            sm.stop()
    except Exception:
        logger.error("sigterm | flush failed", exc_info=True)

    sys.exit(0)


signal.signal(signal.SIGTERM, _handle_sigterm)

from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5555, debug=True)
