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
# Base input (TEXT SAFE)
# -----------------------------------------------------
BASE_INPUT = {
    "prompt": "Summarize the plot of Alice in Wonderland.",
    "seed": 42,
    "max_new_tokens": 40,
}


# -----------------------------------------------------
# Helpers
# -----------------------------------------------------
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
    reason="Replicate API token not available",
)
def test_phi4_two_sampling_modes():
    """
    Integration test for Unsloth Phi-4 reasoning predictor.

    Verifies:
      - stable vs creative sampling both execute
      - valid text output is produced
      - latency characteristics are sane
    """

    # --------------------------------------------------
    # CI safety: enforce deployment-scoped testing
    # --------------------------------------------------
    if os.environ.get("CI"):
        assert os.environ.get("CANDIDATE_MODEL_ID"), "CANDIDATE_MODEL_ID must be set in CI integration tests"

    resolved_model_id = get_candidate_model_id()
    logger.info("Using model ID for integration test: %s", resolved_model_id)

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

    results = []

    for label, cfg in calls:
        logger.info("Calling Phi-4 with %s sampling", label)
        logger.info("Input params: %s", cfg)

        text, elapsed = run_and_time(
            resolved_model_id,
            cfg,
            timeout_s=180.0,
        )

        logger.info("%s sampling completed in %.2fs", label, elapsed)
        logger.info("Output preview (%s): %r", label, text[:120])

        assert isinstance(text, str)
        assert len(text.strip()) > 20, f"Output too short for {label} sampling"

        results.append((label, text, elapsed))

    # -----------------------------------------------------
    # Cross-run sanity checks
    # -----------------------------------------------------
    (_, out1, t1), (_, out2, t2) = results

    assert len(out1.strip()) > 20
    assert len(out2.strip()) > 20

    ratio = t1 / t2 if t2 > 0 else float("inf")
    logger.info("Latency ratio (stable / creative) = %.2f", ratio)

    assert 0.2 < ratio < 5.0, f"Unexpected latency ratio: {ratio:.2f}"

    logger.info("Phi-4 sampling integration test completed successfully.")
