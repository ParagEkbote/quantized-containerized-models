import logging
import os

import pytest

from integration.utils import (
    normalize_string_bools,
    run_and_time,
    resolve_latest_version_httpx,
)

# -----------------------------------------------------
# Logging configuration
# -----------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# -----------------------------------------------------
# Deployment ID (PINNED)
# -----------------------------------------------------
MODEL_ID = "paragekbote/gemma3-torchao-quant-sparse"

# -----------------------------------------------------
# Base input
# -----------------------------------------------------
BASE_INPUT = {
    "prompt": "Compare supervised, unsupervised and reinforcement learning briefly.",
    "seed": 42,
    "max_new_tokens": 50,
    "temperature": 0.7,
    "top_p": 0.9,
}


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    "REPLICATE_API_TOKEN" not in os.environ,
    reason="Replicate API token not found",
)
def test_gemma_torchao_two_paths():
    """
    Integration test for Gemma TorchAO predictor:
    - Call 1: quantization enabled, NO sparsity
    - Call 2: quantization disabled, sparsity enabled
    """
    logger.info("Resolving latest model version: %s", MODEL_ID)
    resolved_model_id = resolve_latest_version_httpx(MODEL_ID)
    logger.info("Resolved model version: %s", resolved_model_id)
    # --------------------------
    # Request 1 — Quantization
    # --------------------------
    req_quant = normalize_string_bools(
        {
            **BASE_INPUT,
            "use_quantization": True,
            "use_sparsity": False,
        },
        keys=("use_quantization", "use_sparsity"),
    )

    # --------------------------
    # Request 2 — Sparsity
    # --------------------------
    req_sparse = normalize_string_bools(
        {
            **BASE_INPUT,
            "use_quantization": False,
            "use_sparsity": True,
            "sparsity_type": "layer_norm",
            "sparsity_ratio": 0.2,
        },
        keys=("use_quantization", "use_sparsity"),
    )

    results = []

    for req in (req_quant, req_sparse):
        text, elapsed = run_and_time(
            resolved_model_id,
            req,
            timeout_s=180.0,
            min_chars=20,
        )

        logger.info("Request params: %s", req)
        logger.info("Latency: %.2fs", elapsed)
        logger.info("Output sample: %s...", text[:120])

        results.append((text, elapsed))

    # --------------------------
    # Post-run assertions
    # --------------------------
    (out1, t1), (out2, t2) = results

    assert len(out1.strip()) > 20
    assert len(out2.strip()) > 20

    ratio = t1 / t2 if t2 > 0 else float("inf")
    assert 0.2 < ratio < 5.0, f"Unexpected latency ratio: {ratio:.2f}"

    logger.info(
        "Test completed successfully | quant=%.2fs sparse=%.2fs ratio=%.2f",
        t1,
        t2,
        ratio,
    )
