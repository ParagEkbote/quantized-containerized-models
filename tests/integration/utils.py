# utils.py

import logging
import time
from collections.abc import Iterable
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


def clean_input(payload: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in payload.items() if v is not None}


def normalize_string_bools(payload: dict[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    out = dict(payload)
    for k in keys:
        if k in out:
            v = out[k]
            if isinstance(v, bool):
                out[k] = "true" if v else "false"
            else:
                out[k] = str(v).lower()
    return out


# -----------------------------------------------------
# Exceptions
# -----------------------------------------------------


class InferenceTimeoutError(RuntimeError):
    """Inference exceeded allowed latency."""


# -----------------------------------------------------
# Core execution (single attempt)
# -----------------------------------------------------


def _run_once(
    deployment_id: str,
    payload: dict[str, Any],
    timeout_s: float,
) -> tuple[str, float]:
    cleaned = clean_input(payload)

    start = time.time()
    raw = replicate.run(deployment_id, input=cleaned)
    elapsed = time.time() - start

    if elapsed > timeout_s:
        raise InferenceTimeoutError(f"Inference exceeded time budget: {elapsed:.2f}s")

    text = normalize_output(raw)

    if len(text.strip()) < 10:
        raise AssertionError("Model output is empty or too short")

    return text, elapsed


# -----------------------------------------------------
# Tenacity-wrapped execution
# -----------------------------------------------------


@retry(
    retry=retry_if_exception_type((InferenceTimeoutError, replicate.exceptions.ReplicateError)),
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
) -> tuple[str, float]:
    """
    Run a Replicate deployment with retries.

    Retries on:
    - slow cold starts
    - transient Replicate errors

    Fails hard only after retries are exhausted.
    """
    return _run_once(deployment_id, payload, timeout_s)
