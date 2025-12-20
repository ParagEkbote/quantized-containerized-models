import logging
import requests
import replicate
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

logger = logging.getLogger(__name__)

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

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
    logger.debug(f"Calling replicate.run with deployment: {deployment_id}")
    logger.debug(f"Input params: {params}")
    result = replicate.run(deployment_id, input=params)
    logger.debug(f"Raw result type: {type(result)}")
    logger.debug(f"Raw result preview: {str(result)[:200]}")
    return result

# ---------------------------------------------------------
# Normalize output across text / file / streaming outputs
# ---------------------------------------------------------
def normalize_output(output):
    """
    Return either text or bytes, depending on output type.
    Handles file URLs returned when Cog predict() returns Path objects.
    """
    logger.debug(f"normalize_output called with type: {type(output)}")

    # Case 1: FileOutput object with .read() method
    if hasattr(output, "read"):
        logger.info("Output has read() method - treating as FileOutput")
        try:
            content = output.read()
            logger.debug(f"Read content type: {type(content)}, length: {len(content)}")
            try:
                decoded = content.decode("utf-8")
                logger.debug(f"Successfully decoded as UTF-8, length: {len(decoded)}")
                return decoded
            except Exception as e:
                logger.warning(f"Could not decode as UTF-8: {e}, returning bytes")
                return content
        except Exception as e:
            logger.error(f"Error reading FileOutput: {e}")
            raise

    # Case 2: URL string pointing to a file (common when Cog returns Path)
    if isinstance(output, str):
        logger.debug(f"Output is string, length: {len(output)}")

        # Check if it's a URL
        if output.startswith('http://') or output.startswith('https://'):
            logger.info(f"Detected URL output: {output}")
            try:
                logger.info("Fetching content from URL...")
                response = requests.get(output, timeout=60)
                response.raise_for_status()
                logger.info(f"HTTP {response.status_code}, Content-Type: {response.headers.get('content-type')}")
                logger.debug(f"Response length: {len(response.content)} bytes")

                # Try to decode as text, fallback to bytes
                try:
                    text = response.text
                    logger.debug(f"Successfully got text, length: {len(text)}")
                    logger.debug(f"Text preview: {text[:200]}")
                    return text
                except Exception as e:
                    logger.warning(f"Could not decode response as text: {e}")
                    return response.content

            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to fetch URL {output}: {e}")
                logger.warning("Returning URL string as fallback")
                return output

        # Check if it looks like a file path (might be returned in some cases)
        elif '/tmp/' in output or 'output_' in output:
            logger.warning(f"Output looks like a file path but not a URL: {output}")
            logger.warning("Cannot access local paths from client - model may be misconfigured")
            return output

        # If not a URL or path, return as-is (direct text output)
        logger.debug("String output is not URL or path - treating as direct text")
        logger.debug(f"Text preview: {output[:200]}")
        return output

    # Case 3: Streaming list of chunks or iterator
    if hasattr(output, '__iter__') and not isinstance(output, (str, bytes, dict)):
        logger.info("Output is iterable (list/generator)")
        try:
            # Convert to list if it's an iterator
            if not isinstance(output, list):
                logger.debug("Converting iterator to list")
                output = list(output)

            logger.debug(f"List length: {len(output)}")

            if not output:
                logger.warning("Empty list/iterator")
                return ""

            # Check if list contains URLs (multiple file outputs)
            first_item = output[0]
            logger.debug(f"First item type: {type(first_item)}, value: {str(first_item)[:100]}")

            if isinstance(first_item, str) and (
                first_item.startswith('http://') or first_item.startswith('https://')
            ):
                logger.info(f"List contains URLs, fetching from first: {first_item}")
                try:
                    response = requests.get(first_item, timeout=60)
                    response.raise_for_status()
                    logger.info(f"Successfully fetched URL from list")
                    try:
                        return response.text
                    except Exception:
                        return response.content
                except requests.exceptions.RequestException as e:
                    logger.error(f"Failed to fetch URL from list: {e}")
                    return str(first_item)

            # Otherwise treat as streaming text chunks
            logger.debug("Joining list items as text chunks")
            result = "".join(str(chunk) for chunk in output)
            logger.debug(f"Joined result length: {len(result)}")
            logger.debug(f"Joined result preview: {result[:200]}")
            return result

        except Exception as e:
            logger.error(f"Error processing iterable: {e}")
            return str(output)

    # Case 4: Dictionary output
    if isinstance(output, dict):
        logger.info("Output is dictionary")
        logger.debug(f"Dict keys: {list(output.keys())}")

        # Try common keys that might contain the actual output
        for key in ['output', 'text', 'response', 'generated_text', 'result']:
            if key in output:
                logger.info(f"Found '{key}' in dict, recursively normalizing")
                return normalize_output(output[key])

        logger.warning("No standard output key found in dict, stringifying")
        logger.debug(f"Dict contents: {output}")
        return str(output)

    # Case 5: Fallback for any other type
    logger.warning(f"Unknown output type: {type(output)}, converting to string")
    result = str(output)
    logger.debug(f"Stringified result: {result[:200]}")
    return result
