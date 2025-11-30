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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DEPLOYMENT_ID = (
    "paragekbote/phi-4-reasoning-plus-unsloth:"
    "a6b2aa30b793e79ee4f7e30165dce1636730b20c2798d487fc548427ba6314d7"
)

BASE_INPUT = {
    "prompt": "Summarize the plot of Alice in Wonderland.",
    "seed": 42,
    "max_new_tokens": 40,
}


@pytest.mark.integration
@pytest.mark.skipif(
    "REPLICATE_API_TOKEN" not in os.environ,
    reason="Replicate API token not available",
)
def test_phi4_two_sampling_modes():
    """
    Integration test for Unsloth Phi-4 reasoning predictor:
    - Call 1: stable (low temperature, high top_p)
    - Call 2: creative (high temperature, low top_p)
    """

    # Deterministic sampling
    call1 = {
        **BASE_INPUT,
        "temperature": 0.3,
        "top_p": 0.95,
    }

    # Creative sampling
    call2 = {
        **BASE_INPUT,
        "temperature": 0.9,
        "top_p": 0.5,
    }

    results = []

    for params in (call1, call2):
        logger.info(f"\nCalling model with params: {params}")

        start = time.time()
        raw_output = replicate.run(DEPLOYMENT_ID, input=params)
        elapsed = time.time() - start

        # Handle streamed output or complete text
        if isinstance(raw_output, list):
            text = "".join(raw_output)
        else:
            text = "".join(chunk for chunk in raw_output)

        logger.info(f"Call completed in {elapsed:.2f}s")
        logger.info(f"Output preview: {text[:120]!r}")

        # Assertions
        assert isinstance(text, str), "Model returned non-string output"
        assert len(text) > 20, "Output too short — likely generation failure"
        assert elapsed < 25, f"Call too slow: {elapsed:.2f}s"

        results.append((params, text, elapsed))

    # -----------------------------------------------------
    # Post-run comparison
    # -----------------------------------------------------
    (params1, out1, t1), (params2, out2, t2) = results

    logger.info(f"Stable sampling output preview: {out1[:120]!r}")
    logger.info(f"Creative sampling output preview: {out2[:120]!r}")

    # 1. Content difference
    assert out1 != out2, "High-temp and low-temp outputs should differ"

    # 2. Timing comparison
    ratio = t1 / t2 if t2 > 0 else float("inf")
    logger.info(f"Timing ratio (t1/t2): {ratio:.2f}")
    assert 0.4 < ratio < 2.5, f"Timing drift too large: ratio={ratio:.2f}"

    # 3. Creativity difference heuristic
    unique_words_1 = len(set(out1.split()))
    unique_words_2 = len(set(out2.split()))
    logger.info(f"Unique word counts → stable: {unique_words_1}, creative: {unique_words_2}")

    assert unique_words_2 >= unique_words_1 * 0.7, (
        "Creative mode output does not appear significantly more varied"
    )
