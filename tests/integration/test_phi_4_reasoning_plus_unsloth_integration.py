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
# Model ID (not deployment)
# -----------------------------------------------------
MODEL_ID = "paragekbote/phi-4-reasoning-plus-unsloth"

# -----------------------------------------------------
# Base input (TEXT SAFE)
# -----------------------------------------------------
BASE_INPUT = {
    "prompt": "Summarize the plot of Alice in Wonderland.",
    "seed": 42,
    "max_new_tokens": 40,
}


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    "REPLICATE_API_TOKEN" not in os.environ,
    reason="Replicate API token not available",
)
def test_phi4_two_sampling_modes():
    """
    Integration test for Unsloth Phi-4 reasoning predictor:
    - Call 1: stable sampling (low temperature, high top_p)
    - Call 2: creative sampling (high temperature, low top_p)

    Verifies:
    - both configurations run successfully
    - valid text output is produced
    - latency characteristics are sane
    """
    # Resolve the latest model version (not deployment)
    logger.info("Resolving latest model version: %s", MODEL_ID)
    resolved_model_id = resolve_latest_version_httpx(MODEL_ID)
    logger.info("Resolved model version: %s", resolved_model_id)

    calls = [
        {
            **BASE_INPUT,
            "temperature": 0.3,
            "top_p": 0.95,
            "label": "stable",
        },
        {
            **BASE_INPUT,
            "temperature": 0.9,
            "top_p": 0.5,
            "label": "creative",
        },
    ]

    results = []

    for cfg in calls:
        label = cfg.pop("label")

        logger.info("Calling Phi-4 with %s sampling", label)
        logger.info("Input params: %s", cfg)

        # Use the resolved model version (with hash)
        text, elapsed = run_and_time(
            resolved_model_id,  # This includes the version hash
            cfg,
            timeout_s=120.0,
        )

        logger.info("%s sampling completed in %.2fs", label, elapsed)
        logger.info("Output preview (%s): %r", label, text[:120])

        # Basic validity checks
        assert isinstance(text, str)
        assert len(text.strip()) > 20, f"Output too short for {label} sampling"

        results.append((label, text, elapsed))

    # -----------------------------------------------------
    # Cross-run sanity checks
    # -----------------------------------------------------
    (_, out1, t1), (_, out2, t2) = results

    # Do NOT require different text (not guaranteed)
    assert len(out1.strip()) > 20
    assert len(out2.strip()) > 20

    # Latency ratio sanity check (loose by design)
    ratio = t1 / t2 if t2 > 0 else float("inf")
    logger.info("Latency ratio (stable / creative) = %.2f", ratio)

    assert 0.2 < ratio < 5.0, f"Unexpected latency ratio: {ratio:.2f}"

    logger.info("Phi-4 sampling integration test completed successfully.")