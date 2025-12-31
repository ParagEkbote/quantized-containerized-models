from __future__ import annotations

import logging
import os

import pytest

from integration.utils import resolve_latest_version_httpx, run_and_time

# ----------------------------------------------------
# Logging configuration
# ----------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ----------------------------------------------------
# Model configuration
# ----------------------------------------------------
MODEL_BASE_ID = "paragekbote/smollm3-3b-smashed"
TARGET_MODEL = "smollm3-3b-smashed"

# ----------------------------------------------------
# Base input (schema-correct, deterministic)
# ----------------------------------------------------
BASE_INPUT = {
    "prompt": "Explain quantum computing in simple words.",
    "seed": 123,
    "max_new_tokens": 90,
}


# ----------------------------------------------------
# Helpers
# ----------------------------------------------------
def resolve_candidate_model_id() -> str:
    """
    Resolve the model ID for integration testing.

    Contract:
      - CI: CANDIDATE_MODEL_ID MUST be provided
      - Local: fallback to latest published version
    """
    cid = os.environ.get("CANDIDATE_MODEL_ID")

    if os.environ.get("CI"):
        assert cid, "CANDIDATE_MODEL_ID must be set in CI integration tests"
        return cid

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
    reason="Missing REPLICATE_API_TOKEN",
)
def test_smollm3_think_and_no_think_modes():
    """
    Integration test for SmolLM3 smashed deployment.

    Validates:
      - Predictor boots and accepts requests
      - `no_think` and `think` execution modes both function
      - Outputs are valid text
      - Latency characteristics are sane between modes

    This test is STRICT and BLOCKING by design.
    """

    model_id = resolve_candidate_model_id()
    logger.info("Using model ID for integration test: %s", model_id)

    results: list[tuple[str, str, float]] = []

    for mode in ("no_think", "think"):
        params = {**BASE_INPUT, "mode": mode}

        logger.info("Invoking model with mode=%s", mode)
        logger.debug("Input parameters: %s", params)

        text, elapsed = run_and_time(
            model_id,
            params,
            timeout_s=180.0,
            min_chars=20,
        )

        logger.info("Mode=%s completed in %.2fs", mode, elapsed)
        logger.info("Output preview (%s): %s...", mode, text[:120])

        # --------------------------------------------------
        # Output validation (STRICT)
        # --------------------------------------------------
        assert isinstance(text, str), "Model output must be a string"
        assert len(text.strip()) >= 20, f"Output too short for mode={mode}"

        results.append((mode, text, elapsed))

    # ----------------------------------------------------
    # Cross-run sanity checks
    # ----------------------------------------------------
    (_, _, t_no_think), (_, _, t_think) = results

    ratio = t_no_think / t_think if t_think > 0 else float("inf")
    logger.info("Latency ratio (no_think / think) = %.2f", ratio)

    # Coarse sanity only — not a performance benchmark
    assert 0.2 < ratio < 5.0, f"Unexpected latency ratio: {ratio:.2f}"

    logger.info("SmolLM3 integration test passed successfully.")
