import logging
import replicate
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Replicate retry wrapper for rate limits + network issues
# ---------------------------------------------------------
@retry(
    reraise=True,
    stop=stop_after_attempt(5),  # 5 retries
    wait=wait_exponential(multiplier=2, min=2, max=20),
    retry=retry_if_exception_type(replicate.exceptions.ReplicateError),
)
def safe_replicate_run(deployment_id, params):
    """Retry-safe wrapper for Replicate's run method."""
    return replicate.run(deployment_id, input=params)

# ---------------------------------------------------------
# Normalize output across text / file / streaming outputs
# ---------------------------------------------------------
def normalize_output(output):
    """Return either text or bytes, depending on output type."""
    # FileOutput -> .read()
    if hasattr(output, "read"):
        content = output.read()
        try:
            return content.decode("utf-8")  # text file
        except Exception:
            return content  # binary image or binary file

    # Raw string text
    if isinstance(output, str):
        return output

    # Streaming list of chunks
    if isinstance(output, list):
        return "".join(str(chunk) for chunk in output)

    # Fallback: string representation
    return str(output)
