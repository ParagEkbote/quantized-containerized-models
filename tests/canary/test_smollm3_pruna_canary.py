from __future__ import annotations

import os

import numpy as np
import pytest
import replicate
from sentence_transformers import SentenceTransformer
from utils import run_and_time

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MODEL_BASE = "paragekbote/smollm3-3b-smashed"
STABLE_SMOLLM3_MODEL_ID = os.environ.get("STABLE_SMOLLM3_MODEL_ID")

MIN_OUTPUT_CHARS = 150
MIN_LENGTH_RATIO = 0.4
MAX_LENGTH_RATIO = 2.8
MIN_SEMANTIC_SIMILARITY = 0.84

CANARY_CASES: list[dict] = [
    {
        "name": "no_think_reasoning",
        "input": {
            "prompt": ("Why does increasing batch size sometimes reduce training stability?"),
            "mode": "no_think",
            "seed": 42,
        },
    },
    {
        "name": "think_reasoning",
        "input": {
            "prompt": ("Explain the difference between quantization and pruning in neural networks."),
            "mode": "think",
            "seed": 42,
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


def normalize_text(text: str) -> str:
    """
    Normalize whitespace and remove obvious template noise.
    """
    text = " ".join(text.strip().split())
    for marker in ("/think", "/no_think"):
        text = text.replace(marker, "")
    return text.strip()


def repetition_ratio(text: str) -> float:
    """
    Detect pathological repetition (common failure mode in smashed models).
    """
    tokens = text.split()
    if not tokens:
        return 1.0
    return len(set(tokens)) / len(tokens)


# ---------------------------------------------------------------------
# Canary test
# ---------------------------------------------------------------------


@pytest.mark.canary
def test_canary_smollm3_pruna():
    """
    Canary release test for SmolLM3 + Pruna smashed deployment.

    Guards against:
      - output collapse
      - verbosity regressions
      - smashed-model repetition
      - semantic drift

    Behavior:
      - FAIL if misconfigured
      - PASS explicitly if candidate == stable
      - FAIL on regression
    """

    # --------------------------------------------------
    # Hard requirements (fail fast)
    # --------------------------------------------------
    assert STABLE_SMOLLM3_MODEL_ID, "STABLE_SMOLLM3_MODEL_ID must be set"

    candidate_id = get_latest_model_id()

    # --------------------------------------------------
    # No-op canary (explicit pass)
    # --------------------------------------------------
    if candidate_id == STABLE_SMOLLM3_MODEL_ID:
        assert True, "No-op canary: candidate model equals stable model (nothing new to compare)"
        return

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    for case in CANARY_CASES:
        # ------------------------------
        # Baseline vs candidate
        # ------------------------------
        old_text, _ = run_and_time(
            STABLE_SMOLLM3_MODEL_ID,
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
        # Repetition guard (smashed-model specific)
        # --------------------------------------------------
        rep = repetition_ratio(new_text)
        assert rep > 0.35, f"{case['name']} excessive repetition detected: {rep:.2f}"

        # --------------------------------------------------
        # Semantic similarity (primary signal)
        # --------------------------------------------------
        old_emb = embedder.encode(old_text, normalize_embeddings=True)
        new_emb = embedder.encode(new_text, normalize_embeddings=True)

        similarity = float(np.dot(old_emb, new_emb))
        assert similarity >= MIN_SEMANTIC_SIMILARITY, f"{case['name']} semantic drift too high: {similarity:.3f}"
