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

MODEL_BASE = "r8.im/paragekbote/gemma3-torchao-quant-sparse"
STABLE_MODEL_ID = os.environ.get("STABLE_MODEL_ID")

MIN_OUTPUT_CHARS = 120
MIN_LENGTH_RATIO = 0.4
MAX_LENGTH_RATIO = 3.0
MIN_SEMANTIC_SIMILARITY = 0.82

CANARY_CASES: List[Dict] = [
    {
        "name": "text_only_reasoning",
        "input": {
            "prompt": "Explain why gradient clipping can stabilize training in deep neural networks.",
            "temperature": 0.0,
            "use_quantization": "true",
            "use_sparsity": "false",
            "seed": 42,
        },
    },
    {
        "name": "image_conditioned_reasoning",
        "input": {
            "prompt": "Describe the scene and explain what the image suggests about the environment.",
            "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Fronalpstock_big.jpg/512px-Fronalpstock_big.jpg",
            "temperature": 0.0,
            "use_quantization": "true",
            "use_sparsity": "false",
            "seed": 42,
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
    text = " ".join(text.strip().split())
    # Remove common Gemma artifacts
    for tok in ["<bos>", "<eos>", "<image>"]:
        text = text.replace(tok, "")
    return text.strip()


def repetition_ratio(text: str) -> float:
    tokens = text.split()
    if not tokens:
        return 1.0
    return len(set(tokens)) / len(tokens)

def multimodal_text_validator(text: str) -> None:
    """
    Validator for multimodal text outputs.
    Ensures the vision path actually contributed.
    """
    lowered = text.lower()

    # Very loose but effective signals
    vision_tokens = [
        "image",
        "scene",
        "mountain",
        "landscape",
        "sky",
        "foreground",
        "background",
    ]

    if len(text) < 200:
        # Short outputs must explicitly reference visual content
        assert any(tok in lowered for tok in vision_tokens), (
            "Multimodal output does not reference visual content"
        )


# ---------------------------------------------------------------------
# Canary test
# ---------------------------------------------------------------------

@pytest.mark.canary
def test_canary_gemma_torchao():
    """
    Canary release test for Gemma-3 with torchao INT8 quantization.
    """

    if not STABLE_MODEL_ID:
        pytest.skip("STABLE_MODEL_ID not set")

    candidate_id = get_latest_model_id()

    if candidate_id == STABLE_MODEL_ID:
        pytest.skip("Candidate equals stable")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    for case in CANARY_CASES:
        old_text, _ = run_and_time(
            STABLE_MODEL_ID,
            case["input"],
            timeout_s=120.0 if "image_url" in case["input"] else 90.0,
            min_chars=MIN_OUTPUT_CHARS,
            validator=multimodal_text_validator if "image_url" in case["input"] else None,
    )

        new_text, _ = run_and_time(
            candidate_id,
            case["input"],
            timeout_s=120.0 if "image_url" in case["input"] else 90.0,
            min_chars=MIN_OUTPUT_CHARS,
            validator=multimodal_text_validator if "image_url" in case["input"] else None,
    )


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
        # Degeneration guard (quantization-specific)
        # --------------------------------------------------
        rep = repetition_ratio(new_text)
        assert rep > 0.35, (
            f"{case['name']} excessive repetition detected: {rep:.2f}"
        )

        # --------------------------------------------------
        # Semantic similarity (primary signal)
        # --------------------------------------------------
        old_emb = embedder.encode(old_text, normalize_embeddings=True)
        new_emb = embedder.encode(new_text, normalize_embeddings=True)

        similarity = float(np.dot(old_emb, new_emb))
        assert similarity >= MIN_SEMANTIC_SIMILARITY, (
            f"{case['name']} semantic drift too high: {similarity:.3f}"
        )
