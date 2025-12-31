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
MODEL_NAME = "smollm3-3b-smashed"
MODEL_ALIAS = f"{OWNER}/{MODEL_NAME}"

TARGET_MODEL = "smollm3-3b-smashed"

MIN_OUTPUT_CHARS = 120
MIN_REPETITION_RATIO = 0.35
MIN_SEMANTIC_SIMILARITY = 0.75  # canary-level: detect collapse, not drift

CANARY_CASES: list[dict] = [
    {
        "name": "no_think_reasoning",
        "input": {
            "prompt": "Why does increasing batch size sometimes reduce training stability?",
            "mode": "no_think",
            "seed": 42,
        },
    },
    {
        "name": "think_reasoning",
        "input": {
            "prompt": "Explain the difference between quantization and pruning in neural networks.",
            "mode": "think",
            "seed": 42,
        },
    },
]

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def resolve_canary_model_ref() -> str:
    """
    Resolve the model reference for canary testing.

    Priority:
      1. Explicit candidate version (best-effort)
      2. Model alias (latest deployed version)
    """
    return os.environ.get("CANDIDATE_MODEL_ID") or MODEL_ALIAS


def normalize_text(text: str) -> str:
    """
    Normalize whitespace and remove template artifacts.
    """
    text = " ".join(text.strip().split())
    for marker in ("/think", "/no_think"):
        text = text.replace(marker, "")
    return text.strip()


def repetition_ratio(text: str) -> float:
    """
    Detect pathological repetition (smashed-model failure mode).
    """
    tokens = text.split()
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


# ---------------------------------------------------------------------
# Canary Test
# ---------------------------------------------------------------------


@pytest.mark.canary
@pytest.mark.skipif(
    os.environ.get("MODEL_NAME") != TARGET_MODEL,
    reason="Not the target model for this canary",
)
def test_canary_smollm3_smashed():
    """
    Canary test for SmolLM3 smashed deployment.

    Observes for:
      - output collapse
      - excessive repetition
      - catastrophic semantic degradation

    Non-goals:
      - exact correctness
      - version identity validation
      - regression-level guarantees
    """

    model_ref = resolve_canary_model_ref()

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    for case in CANARY_CASES:
        # --------------------------------------------------
        # Run inference
        # --------------------------------------------------
        text, latency = run_and_time(
            model_ref,
            case["input"],
        )

        text = normalize_text(text)

        # --------------------------------------------------
        # Canary signals (hard collapse only)
        # --------------------------------------------------
        assert len(text) >= MIN_OUTPUT_CHARS, f"[CANARY] {case['name']} output too short ({len(text)} chars)"

        rep = repetition_ratio(text)
        assert rep >= MIN_REPETITION_RATIO, f"[CANARY] {case['name']} excessive repetition (ratio={rep:.2f})"

        # --------------------------------------------------
        # Semantic sanity (collapse detection only)
        # --------------------------------------------------
        emb = embedder.encode(text, normalize_embeddings=True)
        self_sim = float(np.dot(emb, emb))

        assert self_sim > MIN_SEMANTIC_SIMILARITY, f"[CANARY] {case['name']} semantic collapse suspected (score={self_sim:.3f})"
