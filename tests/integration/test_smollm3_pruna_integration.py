import logging
import os

import pytest

from integration.utils import (
    resolve_latest_version_httpx,
    run_and_time,
)

# ----------------------------------------------------
# Configure logger
# ----------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ----------------------------------------------------
# Deployment ID (PINNED)
# ----------------------------------------------------
MODEL_ID = "paragekbote/smollm3-3b-smashed"
TARGET_MODEL = "smollm3-pruna"

# ----------------------------------------------------
# Base input (TEXT SAFE)
# ----------------------------------------------------
BASE_INPUT = {
    "prompt": "Explain quantum computing in simple words.",
    "seed": 123,
    "max_new_tokens": 90,
}


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("MODEL_NAME") != TARGET_MODEL,
    reason="Not the target model for this integration test",
)
@pytest.mark.skipif(
    "REPLICATE_API_TOKEN" not in os.environ,
    reason="REPLICATE_API_TOKEN missing",
)
def test_replicate_two_modes():
    """
    Integration test:
    - mode='no_think'
    - mode='think'

    Verifies:
    - both execution modes run successfully
    - output is valid text
    - latency characteristics are reasonable
    """
    logger.info("Resolving latest model version: %s", MODEL_ID)
    resolved_model_id = resolve_latest_version_httpx(MODEL_ID)
    logger.info("Resolved model version: %s", resolved_model_id)

    logger.info("Starting integration test")
    logger.info("Deployment ID: %s", MODEL_ID)

    results = []

    for mode in ("no_think", "think"):
        params = {**BASE_INPUT, "mode": mode}

        logger.info("Calling model with mode=%s", mode)
        logger.info("Input parameters: %s", params)

        text, elapsed = run_and_time(
            resolved_model_id,
            params,
            timeout_s=180.0,  # retries handled internally
        )

        logger.info("Mode=%s completed in %.2fs", mode, elapsed)
        logger.info("Output preview (%s): %r", mode, text[:120])

        # Basic validity checks
        assert isinstance(text, str)
        assert len(text.strip()) > 20, f"Output too short for mode={mode}"

        results.append((mode, text, elapsed))

    # ----------------------------------------------------
    # Cross-run sanity checks
    # ----------------------------------------------------
    (_, text1, time1), (_, text2, time2) = results

    # Do NOT require different text (not guaranteed)
    assert len(text1.strip()) > 20
    assert len(text2.strip()) > 20

    # Latency ratio sanity check (very loose by design)
    ratio = time1 / time2 if time2 > 0 else float("inf")
    logger.info("Latency ratio (no_think / think) = %.2f", ratio)

    assert 0.2 < ratio < 5.0, f"Unexpected latency ratio: {ratio:.2f}"

    logger.info("Integration test completed successfully.")
