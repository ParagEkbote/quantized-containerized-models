import logging
import os
import time
from collections.abc import Callable, Iterable
from typing import Any

import httpx
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


class DeploymentNotReadyError(RuntimeError):
    """Deployment is not active/ready."""


# -----------------------------------------------------
# Helpers
# -----------------------------------------------------


def resolve_latest_version_httpx(model_base: str) -> str:
    """
    Resolve the latest *model version* (not deployment) via Replicate REST API.

    IMPORTANT:
    - model_base must be owner/name (no :version, no deployment alias)
    """

    if ":" in model_base:
        raise ValueError("resolve_latest_version_httpx expects owner/name only (no version or deployment alias)")

    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        raise RuntimeError("REPLICATE_API_TOKEN not set")

    owner, name = model_base.split("/", 1)

    url = f"https://api.replicate.com/v1/models/{owner}/{name}/versions"
    headers = {
        "Authorization": f"Token {token}",
        "Accept": "application/json",
    }

    with httpx.Client(timeout=10.0) as client:
        resp = client.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    versions = data.get("results", [])
    if not versions:
        raise RuntimeError(f"No versions found for model {model_base}")

    versions.sort(key=lambda v: v["created_at"], reverse=True)
    latest_id = versions[0]["id"]

    return f"{model_base}:{latest_id}"


def clean_input(payload: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in payload.items() if v is not None}


def normalize_string_bools(payload: dict[str, Any], keys: Iterable[str]) -> dict[str, Any]:
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


def _extract_text_from_output(raw: Any) -> str:
    """
    Extract text from various Replicate output types.

    Handles:
    - FileOutput objects (have .read() method)
    - Direct string output
    - Iterator/generator output
    - List of strings

    Returns:
        Extracted text as string

    Raises:
        InvalidModelOutputError: If output type is unrecognized
    """
    # Case 1: FileOutput object (has .read() method)
    if hasattr(raw, "read"):
        content = raw.read()
        if isinstance(content, bytes):
            return content.decode("utf-8")
        return str(content)

    # Case 2: Direct string output
    if isinstance(raw, str):
        return raw

    # Case 3: List output (take first element or join)
    if isinstance(raw, list):
        if not raw:
            raise InvalidModelOutputError("Empty list output")
        if len(raw) == 1:
            return _extract_text_from_output(raw[0])
        # Multiple items - join them
        return "\n".join(str(item) for item in raw)

    # Case 4: Iterator/generator (some models stream output)
    if hasattr(raw, "__iter__") and not isinstance(raw, (str, bytes)):
        try:
            return "".join(str(chunk) for chunk in raw)
        except Exception as e:
            raise InvalidModelOutputError(f"Failed to iterate output: {e}")

    raise InvalidModelOutputError(f"Unexpected output type: {type(raw)}")


# -----------------------------------------------------
# Text execution
# -----------------------------------------------------


@retry(
    retry=retry_if_exception_type((InferenceTimeoutError, replicate.exceptions.ReplicateError)),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=10, max=180),
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
    Run a text or multimodal-text Replicate deployment with retries.

    Handles various output types:
    - FileOutput objects (read and decode)
    - Direct string output
    - Iterator/streaming output
    - List output

    Args:
        deployment_id: Model version or deployment name
        payload: Input parameters for the model
        timeout_s: Maximum allowed inference time
        min_chars: Minimum expected output length
        validator: Optional validation function

    Returns:
        Tuple of (output_text, elapsed_time)

    Raises:
        InferenceTimeoutError: If inference exceeds timeout
        InvalidModelOutputError: If output is invalid or too short
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
        raise InferenceTimeoutError(f"Inference exceeded time budget: {elapsed:.2f}s")

    # Normalize output to text
    text = _extract_text_from_output(raw).strip()

    if len(text) < min_chars:
        raise InvalidModelOutputError(f"Output too short ({len(text)} chars)")

    if validator is not None:
        validator(text)

    return text, elapsed


# -----------------------------------------------------
# Image execution
# -----------------------------------------------------


@retry(
    retry=retry_if_exception_type((InferenceTimeoutError, replicate.exceptions.ReplicateError)),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=10, max=60),
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

    Args:
        deployment_id: Model version or deployment name
        payload: Input parameters for the model
        timeout_s: Maximum allowed inference time

    Returns:
        Tuple of (image_url, elapsed_time)

    Raises:
        InferenceTimeoutError: If inference exceeds timeout
        InvalidModelOutputError: If output is invalid or not an image URL
    """
    cleaned = clean_input(payload)

    start = time.time()
    raw = replicate.run(deployment_id, input=cleaned)
    elapsed = time.time() - start

    if elapsed > timeout_s:
        raise InferenceTimeoutError(f"Inference exceeded time budget: {elapsed:.2f}s")

    # Accept common Replicate image formats
    if isinstance(raw, list):
        if not raw:
            raise InvalidModelOutputError("Empty image output list")
        raw = raw[0]

    if not isinstance(raw, str):
        raise InvalidModelOutputError(f"Expected image URL string, got {type(raw)}")

    if not (raw.startswith("http") or raw.endswith(".png")):
        raise InvalidModelOutputError(f"Unexpected image output: {raw}")

    return raw, elapsed
