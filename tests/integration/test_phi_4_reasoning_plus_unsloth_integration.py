from __future__ import annotations

import logging
import os

import pytest

from integration.utils import resolve_latest_version_httpx, run_and_time

# -----------------------------------------------------
# Logging configuration
# -----------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# -----------------------------------------------------
# Model configuration
# -----------------------------------------------------
MODEL_BASE_ID = "paragekbote/phi-4-reasoning-plus-unsloth"
TARGET_MODEL = "phi-4-reasoning-plus-unsloth"

# -----------------------------------------------------
# Base input (schema-correct, deterministic)
# -----------------------------------------------------
BASE_INPUT = {
    "prompt": "Summarize the plot of Alice in Wonderland.",
    "seed": 42,
    "max_new_tokens": 40,
}


# -----------------------------------------------------
# Helpers
# -----------------------------------------------------
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


# -----------------------------------------------------
# Integration test
# -----------------------------------------------------
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
def test_phi4_stable_and_creative_sampling():
    """
    Integration test for Phi-4 reasoning deployment.

    Validates:
      - Predictor boots and accepts requests
      - Stable and creative sampling paths both execute
      - Outputs are valid text
      - Latency characteristics are sane between modes

    This test is STRICT and BLOCKING by design.
    """

    model_id = resolve_candidate_model_id()
    logger.info("Using model ID for integration test: %s", model_id)

    calls = [
        (
            "stable",
            {
                **BASE_INPUT,
                "temperature": 0.3,
                "top_p": 0.95,
                "top_k": 60,
            },
        ),
        (
            "creative",
            {
                **BASE_INPUT,
                "temperature": 0.9,
                "top_p": 0.5,
                "top_k": 35,
            },
        ),
    ]

    results: list[tuple[str, str, float]] = []

    for label, params in calls:
        logger.info("Invoking Phi-4 with %s sampling", label)
        logger.debug("Input parameters: %s", params)

        text, elapsed = run_and_time(
            model_id,
            params,
            timeout_s=180.0,
            min_chars=20,
        )

        logger.info("%s sampling completed in %.2fs", label, elapsed)
        logger.info("Output preview (%s): %s...", label, text[:120])

        # --------------------------------------------------
        # Output validation (STRICT)
        # --------------------------------------------------
        assert isinstance(text, str), "Model output must be a string"
        assert len(text.strip()) >= 20, f"Output too short for {label} sampling"

        results.append((label, text, elapsed))

    # -----------------------------------------------------
    # Cross-run sanity checks
    # -----------------------------------------------------
    (_, _, t_stable), (_, _, t_creative) = results

    ratio = t_stable / t_creative if t_creative > 0 else float("inf")
    logger.info("Latency ratio (stable / creative) = %.2f", ratio)

    # Coarse sanity only; not performance benchmarking
    assert 0.2 < ratio < 5.0, f"Unexpected latency ratio: {ratio:.2f}"

    logger.info("Phi-4 integration test passed successfully.")
