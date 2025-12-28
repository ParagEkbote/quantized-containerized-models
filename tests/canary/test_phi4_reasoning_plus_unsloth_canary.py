from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pytest
import replicate
from sentence_transformers import SentenceTransformer

from utils import run_and_time


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MODEL_BASE = "r8.im/paragekbote/phi-4-reasoning-plus-unsloth"
STABLE_MODEL_ID = os.environ.get("STABLE_MODEL_ID")

MIN_OUTPUT_CHARS = 120
MAX_LENGTH_RATIO = 2.5
MIN_LENGTH_RATIO = 0.4
MIN_SEMANTIC_SIMILARITY = 0.85

CANARY_CASES: List[Dict] = [
    {
        "name": "math_reasoning",
        "input": {
            "prompt": "If a train travels 120 km in 2 hours and then 180 km in 3 hours, what is its average speed?",
            "seed": 42,
            "temperature": 0.0,
        },
    },
    {
        "name": "logical_reasoning",
        "input": {
            "prompt": "Explain why all squares are rectangles but not all rectangles are squares.",
            "seed": 42,
            "temperature": 0.0,
        },
    },
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def get_latest_model_id() -> str:
    model = replicate.models.get(MODEL_BASE)
    versions = list(model.versions.list())
    versions.sort(key=lambda v: v.created_at, reverse=True)
    return f"{MODEL_BASE}:{versions[0].id}"


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


# ---------------------------------------------------------------------
# Canary Test
# ---------------------------------------------------------------------

@pytest.mark.canary
def test_canary_phi4_reasoning():
    """
    Canary release test for Phi-4 reasoning model.
    Compares latest candidate against last known stable version.
    """

    if not STABLE_MODEL_ID:
        pytest.skip("STABLE_MODEL_ID not set")

    candidate_id = get_latest_model_id()

    if candidate_id == STABLE_MODEL_ID:
        pytest.skip("Candidate equals stable")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    for case in CANARY_CASES:
        old_text, _ = run_and_time(STABLE_MODEL_ID, case["input"])
        new_text, _ = run_and_time(candidate_id, case["input"])

        old_text = normalize_text(old_text)
        new_text = normalize_text(new_text)

        # --------------------------------------------------
        # Hard blockers
        # --------------------------------------------------
        assert len(new_text) >= MIN_OUTPUT_CHARS, (
            f"{case['name']} output too short"
        )

        # --------------------------------------------------
        # Length sanity
        # --------------------------------------------------
        ratio = len(new_text) / max(len(old_text), 1)
        assert MIN_LENGTH_RATIO <= ratio <= MAX_LENGTH_RATIO, (
            f"{case['name']} length ratio abnormal: {ratio:.2f}"
        )

        # --------------------------------------------------
        # Semantic similarity (core signal)
        # --------------------------------------------------
        old_emb = embedder.encode(old_text, normalize_embeddings=True)
        new_emb = embedder.encode(new_text, normalize_embeddings=True)

        similarity = float(np.dot(old_emb, new_emb))
        assert similarity >= MIN_SEMANTIC_SIMILARITY, (
            f"{case['name']} semantic drift too high: {similarity:.3f}"
        )
