from __future__ import annotations

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
# Model configuration
# -----------------------------------------------------
MODEL_BASE_ID = "paragekbote/flux-fast-lora-hotswap"
TARGET_MODEL = "flux-fast-lora-hotswap"

# -----------------------------------------------------
# Base request (schema-correct, deterministic)
# -----------------------------------------------------
BASE_REQUEST = {
    "prompt": "A serene mountain landscape during sunrise.",
    "guidance_scale": 7.0,
    "num_inference_steps": 25,
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
def test_flux_fast_lora_txt2img_adapter_switching():
    """
    Integration test for Flux Fast LoRA txt2img predictor.

    Validates:
      - Predictor boots and accepts requests
      - Runtime LoRA adapter switching works
      - Outputs are valid image references
      - Latency characteristics are sane

    This test is STRICT and BLOCKING by design.
    """

    model_id = resolve_candidate_model_id()
    logger.info("Using model ID for integration test: %s", model_id)

    requests = [
        {**BASE_REQUEST, "trigger_word": "Anime"},
        {**BASE_REQUEST, "trigger_word": "GHIBSKY"},
    ]

    results: list[tuple[str, str, float]] = []

    for req in requests:
        trigger = req["trigger_word"]
        logger.info("Invoking model with trigger_word=%s", trigger)
        logger.debug("Input params: %s", req)

        output, elapsed = run_image_and_time(
            model_id,
            req,
            timeout_s=180.0,
        )

        logger.info(
            "Completed trigger_word=%s in %.2fs",
            trigger,
            elapsed,
        )

        # --------------------------------------------------
        # Output validation (STRICT)
        # --------------------------------------------------
        assert isinstance(output, str), "Model output must be a string"
        assert output.startswith("http") or output.endswith(".png"), f"Unexpected output format: {output}"

        results.append((trigger, output, elapsed))

    # --------------------------------------------------
    # Cross-call assertions
    # --------------------------------------------------
    (t1_name, img1, t1), (t2_name, img2, t2) = results

    # Adapter switching must produce distinct outputs
    assert img1 != img2, "LoRA adapter switching failed: outputs are identical"

    # Coarse latency sanity (not a benchmark)
    ratio = t1 / t2 if t2 > 0 else float("inf")
    logger.info(
        "Latency ratio (%s / %s): %.2f",
        t1_name,
        t2_name,
        ratio,
    )

    assert 0.3 < ratio < 3.0, f"Unexpected latency ratio between adapters: {ratio:.2f}"

    logger.info(
        "Integration test passed | %s=%.2fs | %s=%.2fs",
        t1_name,
        t1,
        t2_name,
        t2,
    )
