import logging
import os

import pytest

from integration.utils import (
    resolve_latest_version_httpx,
    run_image_and_time,
)

# -----------------------------------------------------
# Logging configuration
# -----------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# -----------------------------------------------------
# Model configuration
# -----------------------------------------------------
MODEL_BASE_ID = "paragekbote/flux-fast-lora-hotswap"
TARGET_MODEL = "flux-fast-lora-hotswap"

# -----------------------------------------------------
# Base request (IMG2IMG SAFE)
# -----------------------------------------------------
BASE_REQUEST = {
    "prompt": "A serene mountain landscape during sunrise.",
    "guidance_scale": 7.0,
    "num_inference_steps": 25,
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
    reason="Missing REPLICATE_API_TOKEN; skipping integration test.",
)
def test_flux_fast_lora_two_modes():
    """
    Integration test for Flux Fast LoRA model:
      - Adapter 1: open-image-preferences (trigger='Anime')
      - Adapter 2: flux-ghibsky (trigger='GHIBSKY')

    Verifies:
      - LoRA adapter switching works
      - Valid image outputs are produced
      - Latency characteristics are reasonable
    """

    # --------------------------------------------------
    # CI safety: enforce deployment-scoped testing
    # --------------------------------------------------
    if os.environ.get("CI"):
        assert os.environ.get("CANDIDATE_MODEL_ID"), "CANDIDATE_MODEL_ID must be set in CI integration tests"

    resolved_model_id = get_candidate_model_id()
    logger.info("Using model ID for integration test: %s", resolved_model_id)

    requests = [
        {**BASE_REQUEST, "trigger_word": "Anime"},
        {**BASE_REQUEST, "trigger_word": "GHIBSKY"},
    ]

    results = []

    for req in requests:
        trigger = req["trigger_word"]
        logger.info("Calling model with trigger_word=%s", trigger)
        logger.info("Input params: %s", req)

        output, elapsed = run_image_and_time(
            resolved_model_id,
            req,
            timeout_s=180.0,  # image generation can be slow
        )

        logger.info("Completed trigger_word=%s in %.2fs", trigger, elapsed)
        logger.info("Output: %s", output)

        assert isinstance(output, str)
        assert output.startswith("http") or output.endswith(".png"), f"Unexpected output format: {output}"

        results.append((trigger, output, elapsed))

    # -----------------------------------------------------
    # Cross-run checks
    # -----------------------------------------------------
    (mode1, img1, t1), (mode2, img2, t2) = results

    # Adapter outputs should generally differ
    assert img1 != img2, "LoRA adapter switching did not produce distinct outputs"

    ratio = t1 / t2 if t2 > 0 else float("inf")
    logger.info("Latency ratio (%s / %s) = %.2f", mode1, mode2, ratio)

    assert 0.3 < ratio < 3.0, f"Latency ratio out of bounds: {ratio:.2f}"

    logger.info(
        "Flux Fast LoRA integration test passed | %s=%.2fs %s=%.2fs",
        mode1,
        t1,
        mode2,
        t2,
    )
