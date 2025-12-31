from __future__ import annotations

import os

import numpy as np
import pytest
from sentence_transformers import SentenceTransformer
from utils import run_and_time

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

OWNER = "paragekbote"
MODEL_NAME = "phi-4-reasoning-plus-unsloth"
MODEL_ALIAS = f"{OWNER}/{MODEL_NAME}"

TARGET_MODEL = "phi-4-reasoning-plus-unsloth"

MIN_OUTPUT_CHARS = 120
MIN_SELF_SIMILARITY = 0.75  # collapse detector, not regression gate

CANARY_CASES = [
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


def resolve_canary_model_ref() -> str:
    """
    Resolve model reference for canary testing.

    Priority:
      1. Best-effort candidate version
      2. Model alias (latest deployment)
    """
    return os.environ.get("CANDIDATE_MODEL_ID") or MODEL_ALIAS


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


# ---------------------------------------------------------------------
# Canary Test
# ---------------------------------------------------------------------


@pytest.mark.canary
@pytest.mark.skipif(
    os.environ.get("MODEL_NAME") != TARGET_MODEL,
    reason="Not the target model for this canary",
)
def test_canary_phi4_reasoning():
    """
    Canary test for Phi-4 reasoning deployment.

    Observes for:
      - output collapse
      - inference failures
      - catastrophic semantic degradation

    Non-goals:
      - regression comparison
      - version identity validation
      - correctness guarantees
    """

    model_ref = resolve_canary_model_ref()

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    for case in CANARY_CASES:
        text, latency = run_and_time(
            model_ref,
            case["input"],
        )

        text = normalize_text(text)

        # --------------------------------------------------
        # Hard collapse guard
        # --------------------------------------------------
        assert len(text) >= MIN_OUTPUT_CHARS, f"[CANARY] {case['name']} output too short ({len(text)} chars)"

        # --------------------------------------------------
        # Semantic sanity (self-similarity)
        # --------------------------------------------------
        emb = embedder.encode(text, normalize_embeddings=True)
        self_sim = float(np.dot(emb, emb))

        assert self_sim >= MIN_SELF_SIMILARITY, f"[CANARY] {case['name']} semantic collapse suspected (score={self_sim:.3f})"
