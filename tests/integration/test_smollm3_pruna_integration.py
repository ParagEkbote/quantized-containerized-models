import os
import pytest
import replicate
import time


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
@pytest.mark.skipif(
    "REPLICATE_API_TOKEN" not in os.environ,
    reason="Replicate API token not found in environment",
)
def test_replicate_two_modes():
    """
    Integration test:
    - Call 1 uses mode='no_think'
    - Call 2 uses mode='think'
    - Ensures inference works and outputs differ between modes
    """
    outputs = []

    for mode in ["no_think", "think"]:
        params = {**BASE_INPUT, "mode": mode}

        start = time.time()
        raw_output = replicate.run(DEPLOYMENT_ID, input=params)
        elapsed = time.time() - start

        # Combine streamed chunks
        if isinstance(raw_output, list):
            text = "".join(raw_output)
        else:
            text = "".join(chunk for chunk in raw_output)

        outputs.append((mode, text, elapsed))

        # Basic validations
        assert isinstance(text, str)
        assert len(text) > 10, f"Output too short for mode={mode}"
        assert elapsed < 20, f"Call too slow: {elapsed:.2f}s for mode={mode}"

    # Unpack results
    (_, text1, t1), (_, text2, t2) = outputs

    # Ensure mode differences create different outputs
    assert text1 != text2, "Outputs for think vs no_think should differ"

    # Ensure timings are within reasonable range
    ratio = t1 / t2
    assert 0.3 < ratio < 3.0, f"Timing drift too large: ratio={ratio:.2f}"
