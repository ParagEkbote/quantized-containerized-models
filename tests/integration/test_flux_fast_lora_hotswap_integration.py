import logging
import os

import pytest

from integration.utils import run_image_and_time, resolve_latest_version_httpx

# -----------------------------------------------------
# Logging configuration
# -----------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# -----------------------------------------------------
# Deployment ID (FULLY QUALIFIED & PINNED)
# -----------------------------------------------------
MODEL_ID = "paragekbote/flux-fast-lora-hotswap"

# -----------------------------------------------------
# Base request (IMG2IMG SAFE)
# -----------------------------------------------------
BASE_REQUEST = {
    "prompt": "A serene mountain landscape during sunrise.",
    "guidance_scale": 7.0,
    "num_inference_steps": 25,
}


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    "REPLICATE_API_TOKEN" not in os.environ,
    reason="Missing REPLICATE_API_TOKEN; skipping integration test.",
)
def test_flux_fast_lora_two_modes():
    """
    Integration test for Flux Fast LoRA model:
    - First call triggers open-image-preferences (via 'Anime')
    - Second call triggers flux-ghibsky (via 'GHIBSKY')

    Verifies:
    - both adapters run successfully
    - valid image outputs are produced
    - latency characteristics are reasonable
    """
    logger.info("Resolving latest model version: %s", MODEL_ID)
    resolved_model_id = resolve_latest_version_httpx(MODEL_ID)
    logger.info("Resolved model version: %s", resolved_model_id)

    requests = [
        {**BASE_REQUEST, "trigger_word": "Anime"},
        {**BASE_REQUEST, "trigger_word": "GHIBSKY"},
    ]

    results = []

    for req in requests:
        logger.info("Calling model with trigger_word=%s", req["trigger_word"])
        logger.info("Input params: %s", req)

        # Tenacity-wrapped execution
        output, elapsed = run_image_and_time(
            resolved_model_id,
            req,
            timeout_s=180.0,  # image generation can be slow
        )

        logger.info(
            "Completed trigger_word=%s in %.2fs",
            req["trigger_word"],
            elapsed,
        )
        logger.info("Output: %s", output)

        # Basic output validation
        assert isinstance(output, str)
        assert output.startswith("http") or output.endswith(".png"), f"Unexpected output format: {output}"

        results.append((req["trigger_word"], output, elapsed))

    # -----------------------------------------------------
    # Cross-run checks
    # -----------------------------------------------------
    (mode1, img1, t1), (mode2, img2, t2) = results

    # Do NOT require images to differ byte-for-byte,
    # but URLs should generally differ across adapters.
    assert img1 != img2, "LoRA adapter switching did not produce distinct outputs (unexpected but indicates possible adapter misrouting)."

    # Latency sanity check (loose by design)
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
