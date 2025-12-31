import logging
import os

import pytest

from integration.utils import (
    normalize_string_bools,
    resolve_latest_version_httpx,
    run_and_time,
)

# -----------------------------------------------------
# Logging configuration
# -----------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# -----------------------------------------------------
# Model configuration (intentionally pinned)
# -----------------------------------------------------
MODEL_BASE_ID = "paragekbote/gemma3-torchao-quant-sparse"
TARGET_MODEL = "gemma3-torchao-quant-sparse"


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


# -----------------------------------------------------
# Helpers
# -----------------------------------------------------
def get_candidate_model_id() -> str:
    """
    Resolve the model ID for integration testing.

    Priority:
      1. Explicit CANDIDATE_MODEL_ID (CI/CD)
      2. Latest published version (local fallback)
    """
    cid = os.environ.get("CANDIDATE_MODEL_ID")
    if cid:
        return cid

    logger.info("CANDIDATE_MODEL_ID not set; resolving latest version for %s", MODEL_BASE_ID)
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
    reason="Replicate API token not found",
)
def test_gemma_torchao_two_paths():
    """
    Integration test for Gemma TorchAO predictor.

    Validates:
      - Quantization-only execution path
      - Sparsity-only execution path
      - End-to-end inference correctness
      - Reasonable latency behavior between modes
    """

    # --------------------------------------------------
    # Environment contract (CI safety)
    # --------------------------------------------------
    if os.environ.get("CI"):
        assert os.environ.get("CANDIDATE_MODEL_ID"), "CANDIDATE_MODEL_ID must be set in CI integration tests"

    resolved_model_id = get_candidate_model_id()
    logger.info("Using model ID for integration test: %s", resolved_model_id)

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
        "Integration test passed | quant=%.2fs sparse=%.2fs ratio=%.2f",
        t1,
        t2,
        ratio,
    )
