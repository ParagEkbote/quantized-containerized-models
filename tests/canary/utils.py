import logging
import time
from collections.abc import Iterable
from typing import Any, Tuple

import replicate
import requests
from io import BytesIO
from PIL import Image
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


class InvalidOutputError(RuntimeError):
    """Model output was invalid or unexpected."""


# -----------------------------------------------------
# Core execution (text)
# -----------------------------------------------------

def _run_text_once(
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

    if not isinstance(raw, str):
        raise InvalidOutputError(f"Expected text output, got {type(raw)}")

    text = normalize_output(raw)

    if len(text) < 10:
        raise InvalidOutputError("Model output is empty or too short")

    return text, elapsed


# -----------------------------------------------------
# Core execution (image)
# -----------------------------------------------------

def _run_image_once(
    deployment_id: str,
    payload: dict[str, Any],
    timeout_s: float,
) -> tuple[Image.Image, float]:
    cleaned = clean_input(payload)

    start = time.time()
    raw = replicate.run(deployment_id, input=cleaned)
    elapsed = time.time() - start

    if elapsed > timeout_s:
        raise InferenceTimeoutError(f"Inference exceeded time budget: {elapsed:.2f}s")

    if not isinstance(raw, str) or not raw.startswith("http"):
        raise InvalidOutputError(f"Expected image URL, got {raw}")

    resp = requests.get(raw, timeout=60)
    resp.raise_for_status()

    img = Image.open(BytesIO(resp.content)).convert("RGB")
    return img, elapsed


# -----------------------------------------------------
# Tenacity-wrapped execution (text)
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
    Run a text Replicate deployment with retries.
    """
    return _run_text_once(deployment_id, payload, timeout_s)


# -----------------------------------------------------
# Tenacity-wrapped execution (image)
# -----------------------------------------------------

@retry(
    retry=retry_if_exception_type((InferenceTimeoutError, replicate.exceptions.ReplicateError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=5, max=40),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def run_image_and_time(
    deployment_id: str,
    payload: dict[str, Any],
    *,
    timeout_s: float = 180.0,
) -> tuple[Image.Image, float]:
    """
    Run an image Replicate deployment with retries.
    """
    return _run_image_once(deployment_id, payload, timeout_s)
