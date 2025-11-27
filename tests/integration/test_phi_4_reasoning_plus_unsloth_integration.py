import os
import pytest
import replicate
import time

DEPLOYMENT_ID = (
    "paragekbote/phi-4-reasoning-plus-unsloth:"
    "a6b2aa30b793e79ee4f7e30165dce1636730b20c2798d487fc548427ba6314d7"
)

BASE_INPUT = {
    "prompt": "Summarize the plot of Alice in Wonderland.",
    "seed": 42,
    "max_new_tokens": 40,  # keep fast for integration testing
}


@pytest.mark.integration
@pytest.mark.skipif(
    "REPLICATE_API_TOKEN" not in os.environ,
    reason="Replicate API token not available",
)
def test_phi4_two_sampling_modes():
    """
    Integration test for Unsloth Phi-4 reasoning predictor:
    - Call 1: lower temperature, higher top_p (more deterministic)
    - Call 2: higher temperature, lower top_p (more creative)
    - Ensures both calls succeed and outputs differ meaningfully.
    """

    # First sampling configuration (more stable / deterministic)
    call1 = {
        **BASE_INPUT,
        "temperature": 0.3,
        "top_p": 0.95,
    }

    # Second sampling configuration (more stochastic / creative)
    call2 = {
        **BASE_INPUT,
        "temperature": 0.9,
        "top_p": 0.5,
    }

    outputs = []

    for params in (call1, call2):
        start = time.time()
        raw_output = replicate.run(DEPLOYMENT_ID, input=params)
        elapsed = time.time() - start

        # Replicate may stream or return a list
        if isinstance(raw_output, list):
            text = "".join(raw_output)
        else:
            text = "".join(chunk for chunk in raw_output)

        outputs.append((params, text, elapsed))

        # Basic text/output validation
        assert isinstance(text, str), "Output must be a string"
        assert len(text) > 20, "Output too short — generation failed"
        assert elapsed < 25, f"Inference too slow ({elapsed:.2f}s) for params={params}"

    # Unpack the results
    (_, out1, t1), (_, out2, t2) = outputs

    # 1. Outputs between sampling configs should differ
    assert out1 != out2, "High-temp and low-temp outputs should not be identical"

    # 2. Latency should be roughly in the same range
    ratio = t1 / t2
    assert 0.4 < ratio < 2.5, f"Latency drift too large: ratio={ratio:.2f}"

    # 3. Ensure temperature and top_p behave as expected:
    #    High temperature usually produces more varied text
    assert len(set(out2.split())) >= len(set(out1.split())) * 0.7, (
        "High-temperature output appears unnaturally similar to low-temperature output"
    )
