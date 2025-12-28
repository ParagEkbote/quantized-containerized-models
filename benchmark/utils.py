import logging
import json
import requests
import replicate
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Replicate retry wrapper
# ---------------------------------------------------------
@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=20),
    retry=retry_if_exception_type(replicate.exceptions.ReplicateError),
)
def safe_replicate_run(deployment_id, params):
    logger.debug("Calling replicate.run: %s", deployment_id)
    return replicate.run(deployment_id, input=params)


# ---------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------
def _fetch_url_bytes(url: str) -> bytes:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.content


# ---------------------------------------------------------
# Output normalization
# ---------------------------------------------------------
def normalize_output(output, *, output_type: str = "text"):
    """
    Normalize Replicate outputs.

    output_type:
        - "text": LLM output
        - "vlm": multimodal output normalized to text
        - "image": binary image output
    """
    if output_type == "vlm":
        output_type = "text"

    logger.debug(
        "normalize_output: type=%s output_type=%s",
        type(output),
        output_type,
    )

    # --------------------------------------------------
    # File-like outputs
    # --------------------------------------------------
    if hasattr(output, "read"):
        content = output.read()
        if output_type == "image":
            return content
        return content.decode("utf-8", errors="replace")

    # --------------------------------------------------
    # URL string
    # --------------------------------------------------
    if isinstance(output, str):
        if output.startswith(("http://", "https://")):
            if output_type == "image":
                return _fetch_url_bytes(output)
            return requests.get(output, timeout=60).text
        return output

    # --------------------------------------------------
    # Iterable (token streams, lists)
    # --------------------------------------------------
    if hasattr(output, "__iter__") and not isinstance(output, (str, bytes, dict)):
        items = list(output)
        if not items:
            return "" if output_type == "text" else b""

        first = items[0]

        # Image outputs often come as list of URLs
        if isinstance(first, str) and first.startswith(("http://", "https://")):
            if output_type == "image":
                return _fetch_url_bytes(first)
            return requests.get(first, timeout=60).text

        if output_type == "text":
            return "".join(
                chunk if isinstance(chunk, str) else str(chunk)
                for chunk in items
            )

        raise TypeError("Iterable output incompatible with image output_type")

    # --------------------------------------------------
    # Dictionary (common for VLMs)
    # --------------------------------------------------
    if isinstance(output, dict):
        for key in ("text", "output", "response", "generated_text", "result"):
            if key in output:
                return normalize_output(output[key], output_type=output_type)

        logger.warning(
            "Unrecognized dict output keys: %s",
            list(output.keys()),
        )
        return json.dumps(output, indent=2)

    # --------------------------------------------------
    # Fallback
    # --------------------------------------------------
    if output_type == "image":
        raise TypeError(f"Unsupported image output type: {type(output)}")

    return str(output)
