"""Request-safe JSON logs and stage-duration helpers."""
import json
import logging
import time
import uuid
from contextlib import contextmanager

logger = logging.getLogger("rrr.support")


def new_request_id() -> str:
    return uuid.uuid4().hex


def log_event(event: str, request_id: str, **safe_fields: object) -> None:
    """Log metadata only: callers must not pass prompts, transcripts, or secrets."""
    logger.info(json.dumps({"event": event, "request_id": request_id, **safe_fields}))


@contextmanager
def timed(timings: dict[str, float], stage: str):
    started = time.perf_counter()
    try:
        yield
    finally:
        timings[stage] = round((time.perf_counter() - started) * 1000, 2)
