"""Structured JSON logging for Cloud Run.

In cloud mode, logs are emitted as single-line JSON objects that Cloud Logging
parses automatically. Each log line includes:
- severity (INFO, WARNING, ERROR, DEBUG)
- message
- logger name
- userId, orgId, sessionId (when available from request context)

In standalone mode, logs use the standard human-readable format.
"""

import json
import logging
import os
import time

from flask import g, has_request_context


class CloudRunFormatter(logging.Formatter):
    """JSON formatter compatible with Google Cloud Logging.

    Cloud Logging auto-parses JSON log lines with a "severity" field.
    Extra context (userId, orgId, sessionId) is pulled from Flask's g
    object when available.
    """

    def format(self, record: logging.LogRecord) -> str:
        # Map Python log levels to Cloud Logging severity
        severity_map = {
            "DEBUG": "DEBUG",
            "INFO": "INFO",
            "WARNING": "WARNING",
            "ERROR": "ERROR",
            "CRITICAL": "CRITICAL",
        }

        entry: dict[str, object] = {
            "severity": severity_map.get(record.levelname, record.levelname),
            "message": record.getMessage(),
            "logger": record.name,
            "timestamp": time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)
            )
            + f".{int(record.msecs):03d}Z",
        }

        # Inject request context from Flask's g object
        if has_request_context():
            val = getattr(g, "session_id", None)
            if val is not None:
                entry["session_id"] = val

        # Include exception info if present
        if record.exc_info and record.exc_info[1] is not None:
            entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(entry, default=str)


def configure_logging() -> None:
    """Set up logging based on SEGMENT_MODE.

    Cloud mode: JSON lines to stdout (parsed by Cloud Logging).
    Standalone: human-readable format for local development.
    """
    is_cloud = os.environ.get("SEGMENT_MODE") == "cloud"
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Remove any existing handlers (e.g., from basicConfig)
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.StreamHandler()

    if is_cloud:
        handler.setFormatter(CloudRunFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        )

    root.addHandler(handler)
