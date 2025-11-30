# tests/integration/test_gemma_inference.py
import os
import pytest
import replicate
import time
import logging

# -----------------------------------------------------
# Logging configuration
# -----------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEPLOYMENT_ID = (
    "paragekbote/gemma3-torchao-quant-sparse:"
    "44626bdc478fcfe56ee3d8a5a846b72f1e25abac25f740b2b615c1fcb2b63cb2")


BASE_INPUT = {
    "prompt": "Explain the difference between supervised and unsupervised learning in simple words.",
    "image_url": None,
    "seed": 42,
    "max_new_tokens": 50,
}

@pytest.mark.integration
@pytest.mark.skipif(
    "REPLICATE_API_TOKEN" not in os.environ,
    reason="Replicate API token not found in environment"
)
def test_gemma_two_configs():
    """
    Integration test for Gemma-3-4b-it style predictor:
    - First call: quantization enabled, no sparsity
    - Second call: quantization disabled, sparsity enabled (layer_norm with 0.2)
    """

    # Call one: quantization on, sparsity off
    request_one = {
        **BASE_INPUT,
        "temperature": 0.6,
        "top_p": 0.9,
        "use_quantization": "true",
        "use_sparsity": "false",
    }

    # Call two: quantization off, sparsity on (layer_norm)
    request_two = {
        **BASE_INPUT,
        "temperature": 0.9,
        "top_p": 0.6,
        "use_quantization": "false",
        "use_sparsity": "true",
        "sparsity_type": "layer_norm",
        "sparsity_ratio": 0.2,
    }

    results = []

    for request_params in (request_one, request_two):
        start_time = time.time()
        raw_output = replicate.run(DEPLOYMENT_ID, input=request_params)
        elapsed_time = time.time() - start_time

        # Replicate may return a string or stream/list; coerce to str
        if isinstance(raw_output, list):
            text_output = "".join(raw_output)
        else:
            text_output = str(raw_output)

        # Basic assertions
        assert isinstance(text_output, str)
        assert len(text_output) > 10, "Output too short; generation may have failed"
        assert elapsed_time < 60, f"Inference too slow: {elapsed_time:.2f}s"

        results.append((request_params, text_output, elapsed_time))

    (params_one, out_one, t_one), (params_two, out_two, t_two) = results

    # Outputs should differ because of sampling + optimization differences
    assert out_one != out_two, "Outputs for different configs should not be identical"

    # Latency sanity check
    ratio = t_one / t_two if t_two > 0 else float("inf")
    assert 0.3 < ratio < 3.0, f"Latency ratio suspicious: {ratio:.2f}"

    logger.info("Call 1 params:", params_one)
    logger.info("Call 1 elapsed:", t_one)
    logger.info("Call 2 params:", params_two)
    logger.info("Call 2 elapsed:", t_two)
