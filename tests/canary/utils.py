# utils.py

import logging
import time
from collections.abc import Callable
from typing import Any

import replicate
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------
# Input helpers
# -----------------------------------------------------

def normalize_output(text: str) -> str:
    return text.strip()


def clean_input(payload: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in payload.items() if v is not None}


# -----------------------------------------------------
# Exceptions
# -----------------------------------------------------

class InferenceTimeoutError(RuntimeError):
    """Inference exceeded allowed latency."""


class InvalidModelOutputError(RuntimeError):
    """Model output was invalid or unexpected."""


# -----------------------------------------------------
# Core execution (single attempt)
# -----------------------------------------------------

def _run_once(
    deployment_id: str,
    payload: dict[str, Any],
    timeout_s: float,
    *,
    min_chars: int = 10,
    validator: Callable[[str], None] | None = None,
) -> tuple[str, float]:
    cleaned = clean_input(payload)

    start = time.time()
    raw = replicate.run(deployment_id, input=cleaned)
    elapsed = time.time() - start

    if elapsed > timeout_s:
        raise InferenceTimeoutError(
            f"Inference exceeded time budget: {elapsed:.2f}s"
        )

    if not isinstance(raw, str):
        raise InvalidModelOutputError(
            f"Expected text output, got {type(raw)}"
        )

    text = normalize_output(raw)

    if len(text) < min_chars:
        raise InvalidModelOutputError(
            f"Output too short ({len(text)} chars)"
        )

    if validator is not None:
        validator(text)

    return text, elapsed


# -----------------------------------------------------
# Tenacity-wrapped execution
# -----------------------------------------------------

@retry(
    retry=retry_if_exception_type(
        (InferenceTimeoutError, replicate.exceptions.ReplicateError)
    ),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=5, max=40),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def run_and_time(
    deployment_id: str,
    payload: dict[str, Any],
    *,
    timeout_s: float = 90.0,
    min_chars: int = 10,
    validator: Callable[[str], None] | None = None,
) -> tuple[str, float]:
    """
    Run a Replicate deployment with retries.

    Suitable for:
    - text models
    - multimodal models that return text
    """
    return _run_once(
        deployment_id,
        payload,
        timeout_s,
        min_chars=min_chars,
        validator=validator,
    )
