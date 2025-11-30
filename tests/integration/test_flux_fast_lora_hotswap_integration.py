import logging
import os
import time

import pytest
import replicate

# -----------------------------------------------------
# Logging configuration
# -----------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEPLOYMENT_ID = "flux-fast-lora-hotswap:a958687317369721e1ce66e5436fa989bcff2e40a13537d9b4aa4c6af4a34539"


@pytest.mark.integration
@pytest.mark.skipif(
    "REPLICATE_API_TOKEN" not in os.environ,
    reason="Missing REPLICATE_API_TOKEN; skipping integration test.",
)
def test_flux_fast_lora_two_modes():
    """
    Integration test for Flux Fast LoRA model:
    - First call triggers open-image-preferences (via 'Anime')
    - Second call triggers flux-ghibsky (via 'GHIBSKY')
    """

    # First call → LoRA1
    request_one = {
        "prompt": "A serene mountain landscape during sunrise.",
        "trigger_word": "Anime",
    }

    # Second call → LoRA2
    request_two = {
        "prompt": "A serene mountain landscape during sunrise.",
        "trigger_word": "GHIBSKY",
    }

    results = []

    for request_params in (request_one, request_two):
        start_time = time.time()

        # Run the deployed model on Replicate
        prediction_output = replicate.run(DEPLOYMENT_ID, input=request_params)

        elapsed_time = time.time() - start_time

        # Basic sanity checks
        assert isinstance(prediction_output, str), "Output must be a string path or URL"
        assert prediction_output.endswith(".png") or prediction_output.startswith("http"), f"Unexpected output format: {prediction_output}"
        assert elapsed_time < 60, f"Generation too slow: {elapsed_time:.2f} seconds"

        results.append((request_params, prediction_output, elapsed_time))

    # Unpack the two runs
    (req1, image_one, time_one), (req2, image_two, time_two) = results

    # Images should be DIFFERENT across two adapters
    assert image_one != image_two, "Outputs for 'Anime' (LoRA1) and 'GHIBSKY' (LoRA2) should differ, which indicates correct adapter switching."

    # Ensure generation times are within sane ratio
    time_ratio = time_one / time_two
    assert 0.4 < time_ratio < 2.5, f"Time ratio too large: {time_ratio:.2f}"

    logger.info("LoRA1 (Anime):", image_one)
    logger.info("LoRA2 (GHIBSKY):", image_two)
