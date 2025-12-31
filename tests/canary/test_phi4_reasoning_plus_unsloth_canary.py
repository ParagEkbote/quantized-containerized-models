from __future__ import annotations

import os

import numpy as np
import pytest
import replicate
from sentence_transformers import SentenceTransformer
from utils import run_and_time

# ---------------------------------------------------------------------
# Configuration (intentionally pinned)
# ---------------------------------------------------------------------

MODEL_BASE = "paragekbote/phi-4-reasoning-plus-unsloth"
TARGET_MODEL = "phi-4-reasoning-plus-unsloth"

STABLE_MODEL_ID = "paragekbote/phi-4-reasoning-plus-unsloth:22438984324149ef4ecfcea3d631185641c23e46ce526ae4439b8c89c27ac086"

MIN_OUTPUT_CHARS = 120
MIN_LENGTH_RATIO = 0.4
MAX_LENGTH_RATIO = 2.5
MIN_SEMANTIC_SIMILARITY = 0.85

CANARY_CASES: list[dict] = [
    {
        "name": "math_reasoning",
        "input": {
            "prompt": ("If a train travels 120 km in 2 hours and then 180 km in 3 hours, what is its average speed?"),
            "seed": 42,
            "temperature": 0.0,
        },
    },
    {
        "name": "logical_reasoning",
        "input": {
            "prompt": ("Explain why all squares are rectangles but not all rectangles are squares."),
            "seed": 42,
            "temperature": 0.0,
        },
    },
]

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def get_latest_model_id() -> str:
    assert os.environ.get("REPLICATE_API_TOKEN"), "REPLICATE_API_TOKEN must be set to run canary tests"

    model = replicate.models.get(MODEL_BASE)
    versions = list(model.versions.list())
    versions.sort(key=lambda v: v.created_at, reverse=True)

    return f"{MODEL_BASE}:{versions[0].id}"


def get_candidate_model_id() -> str:
    """
    Resolve the candidate model ID for canary testing.

    Priority:
      1. Explicit CANDIDATE_MODEL_ID (CI/CD)
      2. Latest Replicate version (local fallback)
    """
    cid = os.environ.get("CANDIDATE_MODEL_ID")
    if cid:
        return cid

    return get_latest_model_id()


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


# ---------------------------------------------------------------------
# Canary test
# ---------------------------------------------------------------------


@pytest.mark.canary
@pytest.mark.skipif(
    os.environ.get("MODEL_NAME") != TARGET_MODEL,
    reason="Not the target model for this canary test",
)
def test_canary_phi4_reasoning():
    """
    Canary release test for Phi-4 reasoning model.

    Guards against:
      - output collapse
      - verbosity regressions
      - semantic drift

    Contract:
      - CI must pass CANDIDATE_MODEL_ID explicitly
      - Local runs fall back to latest published version
      - Only runtime inference credentials are used
    """

    # --------------------------------------------------
    # Environment hygiene (fail fast)
    # --------------------------------------------------
    assert os.environ.get("REPLICATE_API_TOKEN"), "REPLICATE_API_TOKEN must be set"

    assert os.environ.get("REPLICATE_CLI_AUTH_TOKEN") is None, "CLI auth must not be set during canary tests"

    assert os.environ.get("HF_TOKEN") is None, "HF auth must not be required for canary tests"

    assert STABLE_MODEL_ID, "Stable model ID must be pinned"

    candidate_id = get_candidate_model_id()

    # --------------------------------------------------
    # Explicit no-op canary
    # --------------------------------------------------
    if candidate_id == STABLE_MODEL_ID:
        pytest.skip("No-op canary: candidate model equals stable model")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    for case in CANARY_CASES:
        # ------------------------------
        # Baseline vs candidate
        # ------------------------------
        old_text, _ = run_and_time(
            STABLE_MODEL_ID,
            case["input"],
        )
        new_text, _ = run_and_time(
            candidate_id,
            case["input"],
        )

        old_text = normalize_text(old_text)
        new_text = normalize_text(new_text)

        # --------------------------------------------------
        # Hard blockers
        # --------------------------------------------------
        assert len(new_text) >= MIN_OUTPUT_CHARS, f"{case['name']} output too short"

        # --------------------------------------------------
        # Length sanity
        # --------------------------------------------------
        ratio = len(new_text) / max(len(old_text), 1)
        assert MIN_LENGTH_RATIO <= ratio <= MAX_LENGTH_RATIO, f"{case['name']} length ratio abnormal: {ratio:.2f}"

        # --------------------------------------------------
        # Semantic similarity (primary signal)
        # --------------------------------------------------
        old_emb = embedder.encode(old_text, normalize_embeddings=True)
        new_emb = embedder.encode(new_text, normalize_embeddings=True)

        similarity = float(np.dot(old_emb, new_emb))
        assert similarity >= MIN_SEMANTIC_SIMILARITY, f"{case['name']} semantic drift too high: {similarity:.3f}"
