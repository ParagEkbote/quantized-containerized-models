# tests/integration/test_gemma_torchao.py
import os
import time
import pytest
import replicate
import logging

# -----------------------------------------------------
# Logging configuration
# -----------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# -----------------------------------------------------------
# Replace with your actual Replicate deployment ID
# -----------------------------------------------------------
DEPLOYMENT_ID = (
    "paragekbote/flux-fast-lora-hotswap-img2img:"
    "e6e00065d5aa5e5dba299ab01b5177db8fa58dc4449849aa0cb3f1edf50430cd")


BASE_INPUT = {
    "prompt": "Compare supervised, unsupervised and reinforcement learning briefly.",
    "image_url": None,
    "seed": 42,
    "max_new_tokens": 50,
    "temperature": 0.7,
    "top_p": 0.9,
}


@pytest.mark.integration
@pytest.mark.skipif(
    "REPLICATE_API_TOKEN" not in os.environ,
    reason="Replicate API token not found"
)
def test_gemma_torchao_two_paths():
    """
    Integration test for the Gemma TorchAO predictor:
    - Call 1: quantization enabled, NO sparsity
    - Call 2: quantization disabled, sparsity enabled
    """

    # ---------------------------------------------------
    # First call — INT8 quantization only
    # ---------------------------------------------------
    request_quant = {
        **BASE_INPUT,
        "use_quantization": "true",
        "use_sparsity": "false",
    }

    # ---------------------------------------------------
    # Second call — Sparsity only (layer_norm, ratio=0.2)
    # ---------------------------------------------------
    request_sparse = {
        **BASE_INPUT,
        "use_quantization": "false",
        "use_sparsity": "true",
        "sparsity_type": "layer_norm",
        "sparsity_ratio": 0.2,
    }

    collected = []

    for request in (request_quant, request_sparse):
        start = time.time()
        raw = replicate.run(DEPLOYMENT_ID, input=request)
        elapsed = time.time() - start

        # Replicate returns either a list of chunks or a single string
        text_out = "".join(raw) if isinstance(raw, list) else str(raw)

        assert isinstance(text_out, str)
        assert len(text_out.strip()) > 10, "Output is too short; model might have failed"
        assert elapsed < 60, f"Inference exceeded time budget: {elapsed:.2f}s"

        collected.append((text_out, elapsed, request))

    # --------------------------
    # Post-run comparison checks
    # --------------------------
    out1, time1, req1 = collected[0]
    out2, time2, req2 = collected[1]

    assert out1 != out2, "Quantized vs sparse outputs should not be identical"

    ratio = time1 / time2 if time2 > 0 else float("inf")
    assert 0.2 < ratio < 5.0, f"Latency ratio unexpected: {ratio:.2f}"

    logger.info("\n--- Test Summary ---")
    logger.info("Request 1 (Quantization):", req1)
    logger.info("Time:", time1)
    logger.info("Output sample:", out1[:120], "...")
    logger.info("\nRequest 2 (Sparsity):", req2)
    logger.info("Time:", time2)
    logger.info("Output sample:", out2[:120], "...")
