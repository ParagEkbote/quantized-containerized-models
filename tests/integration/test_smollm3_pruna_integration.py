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
# Model configuration
# ----------------------------------------------------
MODEL_BASE_ID = "paragekbote/smollm3-3b-smashed"
TARGET_MODEL = "smollm3-3b-smashed"

# ----------------------------------------------------
# Base input (TEXT SAFE)
# ----------------------------------------------------
BASE_INPUT = {
    "prompt": "Explain quantum computing in simple words.",
    "seed": 123,
    "max_new_tokens": 90,
}


# ----------------------------------------------------
# Helpers
# ----------------------------------------------------
def get_candidate_model_id() -> str:
    """
    Resolve the model ID for integration testing.

    Priority:
      1. Explicit CANDIDATE_MODEL_ID (CI/CD)
      2. Latest published version (local fallback)
    """
    cid = os.environ.get("CANDIDATE_MODEL_ID")
    if cid:
        return cid

    logger.info(
        "CANDIDATE_MODEL_ID not set; resolving latest version for %s",
        MODEL_BASE_ID,
    )
    return resolve_latest_version_httpx(MODEL_BASE_ID)


# ----------------------------------------------------
# Integration test
# ----------------------------------------------------
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

    # --------------------------------------------------
    # CI safety: enforce deployment-scoped testing
    # --------------------------------------------------
    if os.environ.get("CI"):
        assert os.environ.get("CANDIDATE_MODEL_ID"), "CANDIDATE_MODEL_ID must be set in CI integration tests"

    resolved_model_id = get_candidate_model_id()
    logger.info("Using model ID for integration test: %s", resolved_model_id)

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

        assert isinstance(text, str)
        assert len(text.strip()) > 20, f"Output too short for mode={mode}"

        results.append((mode, text, elapsed))

    # ----------------------------------------------------
    # Cross-run sanity checks
    # ----------------------------------------------------
    (_, text1, time1), (_, text2, time2) = results

    assert len(text1.strip()) > 20
    assert len(text2.strip()) > 20

    ratio = time1 / time2 if time2 > 0 else float("inf")
    logger.info("Latency ratio (no_think / think) = %.2f", ratio)

    assert 0.2 < ratio < 5.0, f"Unexpected latency ratio: {ratio:.2f}"

    logger.info("SmolLM3 integration test completed successfully.")
