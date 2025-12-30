import logging
import os

import pytest

from integration.utils import resolve_latest_version_httpx, run_image_and_time

# -----------------------------------------------------
# Logging configuration
# -----------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# -----------------------------------------------------
# Deployment ID (PINNED)
# -----------------------------------------------------
MODEL_ID = "paragekbote/flux-fast-lora-hotswap-img2img"
TARGET_MODEL = "flux-fast-lora-hotswap-img2img"

# -----------------------------------------------------
# Base input (SCHEMA-CORRECT)
# -----------------------------------------------------
BASE_INPUT = {
    "prompt": "A serene mountain landscape during sunrise.",
    # Public image URL required by schema
    "init_image": "https://images.pexels.com/photos/33649783/pexels-photo-33649783.jpeg",
    "seed": 42,
    "strength": 0.60,
    "guidance_scale": 7.0,
    "num_inference_steps": 25,
}


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("MODEL_NAME") != TARGET_MODEL,
    reason="Not the target model for this integration test",
)
@pytest.mark.skipif(
    "REPLICATE_API_TOKEN" not in os.environ,
    reason="Replicate API token not found",
)
def test_flux_fast_lora_img2img_two_adapters():
    """
    Integration test for Flux Fast LoRA Img2Img predictor.

    - Call 1: trigger_word='Anime' (open-image-preferences)
    - Call 2: trigger_word='GHIBSKY' (flux-ghibsky)

    Verifies:
    - both adapters run successfully
    - valid image outputs are produced
    - latency characteristics are reasonable
    """
    logger.info("Resolving latest model version: %s", MODEL_ID)
    resolved_model_id = resolve_latest_version_httpx(MODEL_ID)
    logger.info("Resolved model version: %s", resolved_model_id)

    requests = [
        {**BASE_INPUT, "trigger_word": "Anime"},
        {**BASE_INPUT, "trigger_word": "GHIBSKY"},
    ]

    results = []

    for req in requests:
        logger.info("Calling model with trigger_word=%s", req["trigger_word"])
        logger.info("Input params: %s", req)

        output, elapsed = run_image_and_time(
            resolved_model_id,
            req,
            timeout_s=180.0,  # img2img can be slow
        )

        logger.info(
            "Completed trigger_word=%s in %.2fs",
            req["trigger_word"],
            elapsed,
        )
        logger.info("Output: %s", output)

        # Output must be image URL or path
        assert isinstance(output, str)
        assert output.startswith("http") or output.endswith(".png"), f"Unexpected output format: {output}"

        results.append((req["trigger_word"], output, elapsed))

    # -----------------------------------------------------
    # Cross-run checks
    # -----------------------------------------------------
    (mode1, img1, t1), (mode2, img2, t2) = results

    # Adapter switching signal (reasonable for image LoRAs)
    assert img1 != img2, "LoRA adapter switching did not produce distinct outputs (possible adapter misrouting)"

    # Latency sanity check (loose by design)
    ratio = t1 / t2 if t2 > 0 else float("inf")
    logger.info("Latency ratio (%s / %s) = %.2f", mode1, mode2, ratio)

    assert 0.3 < ratio < 3.0, f"Unexpected latency ratio: {ratio:.2f}"

    logger.info(
        "Flux Fast LoRA Img2Img test passed | %s=%.2fs %s=%.2fs",
        mode1,
        t1,
        mode2,
        t2,
    )
