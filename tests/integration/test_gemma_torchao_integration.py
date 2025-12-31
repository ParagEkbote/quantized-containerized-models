from __future__ import annotations

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
# Model configuration
# -----------------------------------------------------
MODEL_BASE_ID = "paragekbote/gemma3-torchao-quant-sparse"
TARGET_MODEL = "gemma3-torchao-quant-sparse"

# -----------------------------------------------------
# Base input (schema-correct, deterministic)
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
def test_gemma_torchao_quant_and_sparse_paths():
    """
    Integration test for Gemma3 TorchAO predictor.

    Validates:
      - Quantization-only execution path
      - Sparsity-only execution path
      - End-to-end inference correctness
      - Reasonable latency behavior between execution modes

    This test is STRICT and BLOCKING by design.
    """

    model_id = resolve_candidate_model_id()
    logger.info("Using model ID for integration test: %s", model_id)

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

    results: list[tuple[str, float]] = []

    for req in (req_quant, req_sparse):
        text, elapsed = run_and_time(
            model_id,
            req,
            timeout_s=180.0,
            min_chars=20,
        )

        logger.info("Request params: %s", req)
        logger.info("Latency: %.2fs", elapsed)
        logger.info("Output sample: %s...", text[:120])

        # ------------------------------
        # Output validation (STRICT)
        # ------------------------------
        assert isinstance(text, str), "Model output must be a string"
        assert len(text.strip()) >= 20, "Model output too short"

        results.append((text, elapsed))

    # --------------------------
    # Cross-run assertions
    # --------------------------
    (_, t1), (_, t2) = results

    ratio = t1 / t2 if t2 > 0 else float("inf")
    logger.info("Latency ratio (quant / sparse): %.2f", ratio)

    # Coarse sanity only; not a benchmark
    assert 0.2 < ratio < 5.0, f"Unexpected latency ratio: {ratio:.2f}"

    logger.info(
        "Integration test passed | quant=%.2fs sparse=%.2fs ratio=%.2f",
        t1,
        t2,
        ratio,
    )
