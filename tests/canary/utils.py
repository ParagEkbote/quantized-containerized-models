import logging
import time
from collections.abc import Callable, Iterable
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
# Exceptions
# -----------------------------------------------------

class InferenceTimeoutError(RuntimeError):
    """Inference exceeded allowed latency."""


class InvalidModelOutputError(RuntimeError):
    """Unexpected or invalid model output."""


# -----------------------------------------------------
# Helpers
# -----------------------------------------------------

def clean_input(payload: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in payload.items() if v is not None}


def normalize_string_bools(
    payload: dict[str, Any],
    keys: Iterable[str],
) -> dict[str, Any]:
    """
    Convert boolean-like inputs to lowercase strings for Cog schema compatibility.
    """
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
# Text / Multimodal execution
# -----------------------------------------------------

def _run_text_once(
    deployment_id: str,
    payload: dict[str, Any],
    timeout_s: float,
    *,
    min_chars: int = 10,
    validator: Callable[[str], None] | None = None,
    output_type: str = "text",
) -> tuple[str, float]:
    """
    Execute a text or multimodal-text (VLM) deployment once.
    """
    cleaned = clean_input(payload)

    start = time.time()
    cleaned = normalize_string_bools(
        cleaned,
        keys=("use_quantization", "use_sparsity"),
    )
    raw = replicate.run(deployment_id, input=cleaned)
    elapsed = time.time() - start

    if elapsed > timeout_s:
        raise InferenceTimeoutError(
            f"Inference exceeded time budget: {elapsed:.2f}s"
        )

    # VLMs still normalize to text
    if not isinstance(raw, str):
        raise InvalidModelOutputError(
            f"Expected text output for {output_type}, got {type(raw)}"
        )

    text = raw.strip()

    if len(text) < min_chars:
        raise InvalidModelOutputError(
            f"Output too short ({len(text)} chars)"
        )

    if validator is not None:
        validator(text)

    return text, elapsed


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
    output_type: str = "text",
    validator: Callable[[str], None] | None = None,
) -> tuple[str, float]:
    """
    Run a text or multimodal-text Replicate deployment with retries.

    Parameters
    ----------
    deployment_id : str
        Full Replicate deployment ID (r8.im/owner/model:version).
    payload : dict
        Model input payload.
    timeout_s : float
        Maximum allowed inference time.
    min_chars : int
        Minimum required output length.
    output_type : {"text", "vlm"}
        Output modality. "vlm" is normalized to text.
    validator : callable, optional
        Optional validation function applied to the output text.
    """
    return _run_text_once(
        deployment_id,
        payload,
        timeout_s,
        min_chars=min_chars,
        validator=validator,
        output_type=output_type,
    )


# -----------------------------------------------------
# Image execution (unchanged)
# -----------------------------------------------------

def _run_image_once(
    deployment_id: str,
    payload: dict[str, Any],
    timeout_s: float,
) -> tuple[str, float]:
    cleaned = clean_input(payload)

    start = time.time()
    raw = replicate.run(deployment_id, input=cleaned)
    elapsed = time.time() - start

    if elapsed > timeout_s:
        raise InferenceTimeoutError(
            f"Inference exceeded time budget: {elapsed:.2f}s"
        )

    # Accept common Replicate image formats
    if isinstance(raw, list):
        if not raw:
            raise InvalidModelOutputError("Empty image output list")
        raw = raw[0]

    if not isinstance(raw, str):
        raise InvalidModelOutputError(
            f"Expected image URL string, got {type(raw)}"
        )

    if not (raw.startswith("http") or raw.endswith(".png")):
        raise InvalidModelOutputError(
            f"Unexpected image output: {raw}"
        )

    return raw, elapsed


@retry(
    retry=retry_if_exception_type(
        (InferenceTimeoutError, replicate.exceptions.ReplicateError)
    ),
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
) -> tuple[str, float]:
    """
    Run an image-producing Replicate deployment with retries.
    """
    return _run_image_once(deployment_id, payload, timeout_s)
