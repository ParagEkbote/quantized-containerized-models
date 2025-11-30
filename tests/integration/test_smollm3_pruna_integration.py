import logging
import os
import time

import pytest
import replicate

# ----------------------------------------------------
# Configure logger
# ----------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

# ----------------------------------------------------
# Deployment ID (string, not tuple)
# ----------------------------------------------------
DEPLOYMENT_ID = (
    "paragekbote/smollm3-3b-smashed:"
    "232b6f87dac025cb54803cfbc52135ab8366c21bbe8737e11cd1aee4bf3a2423"
)

BASE_INPUT = {
    "prompt": "Explain quantum computing in simple words.",
    "seed": 123,
    "max_new_tokens": 50,
}


@pytest.mark.integration
@pytest.mark.skipif("REPLICATE_API_TOKEN" not in os.environ, reason="REPLICATE_API_TOKEN missing")
def test_replicate_two_modes():
    """
    Integration test:
    - mode='no_think'
    - mode='think'
    Ensures model runs correctly and outputs differ.
    """

    logger.info("Starting integration test for deployment:")
    logger.info(f"Deployment ID: {DEPLOYMENT_ID}")

    results = []

    for mode in ["no_think", "think"]:
        params = {**BASE_INPUT, "mode": mode}

        logger.info(f"Calling model with mode={mode} ...")
        logger.info(f"Input parameters: {params}")

        start = time.time()
        raw_output = replicate.run(DEPLOYMENT_ID, input=params)
        elapsed = time.time() - start

        # Concatenate chunks if needed
        if isinstance(raw_output, list):
            text = "".join(raw_output)
        else:
            text = "".join(chunk for chunk in raw_output)

        logger.info(f"Mode={mode} completed in {elapsed:.2f}s")
        logger.info(f"Output preview ({mode}): {text[:120]!r}")

        results.append((mode, text, elapsed))

        # Assertions
        assert isinstance(text, str), "Output must be text"
        assert len(text) > 10, f"Output too short for mode={mode}"
        assert elapsed < 20, f"Call too slow: {elapsed:.2f}s for mode={mode}"

    # Compare outputs
    mode1, text1, time1 = results[0]
    mode2, text2, time2 = results[1]

    logger.info("Comparing outputs between no_think and think modes...")
    assert text1 != text2, "Outputs for think vs no_think should differ"

    # Timing drift analysis
    ratio = time1 / time2 if time2 > 0 else float("inf")
    logger.info(f"Timing ratio = {ratio:.2f}")

    assert 0.3 < ratio < 3.0, f"Timing drift too large: ratio={ratio:.2f}"

    logger.info("Integration test completed successfully.")
