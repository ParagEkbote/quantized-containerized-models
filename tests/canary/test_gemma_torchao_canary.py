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
MODEL_NAME = "gemma3-torchao-quant-sparse"
MODEL_ALIAS = f"{OWNER}/{MODEL_NAME}"

TARGET_MODEL = MODEL_NAME

MIN_OUTPUT_CHARS = 120
MIN_REPETITION_RATIO = 0.35
MIN_SELF_SIMILARITY = 0.75  # collapse detector only

CANARY_CASES = [
    {
        "name": "text_reasoning",
        "input": {
            "prompt": ("Explain why gradient clipping can stabilize training in deep neural networks."),
            "temperature": 1.4,
            "use_quantization": True,
            "use_sparsity": False,
            "seed": 42,
        },
        "is_multimodal": False,
    },
    {
        "name": "image_conditioned_reasoning",
        "input": {
            "prompt": ("Describe the scene and explain what the image suggests about the environment."),
            "image_url": ("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Fronalpstock_big.jpg/512px-Fronalpstock_big.jpg"),
            "temperature": 1.5,
            "use_quantization": True,
            "use_sparsity": False,
            "seed": 42,
        },
        "is_multimodal": True,
    },
]

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def resolve_canary_model_ref() -> str:
    """
    Canary uses best-effort candidate version,
    otherwise the model alias (latest deployment).
    """
    return os.environ.get("CANDIDATE_MODEL_ID") or MODEL_ALIAS


def normalize_text(text: str) -> str:
    text = " ".join(text.strip().split())
    for tok in ("<bos>", "<eos>", "<image>"):
        text = text.replace(tok, "")
    return text.strip()


def repetition_ratio(text: str) -> float:
    tokens = text.split()
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def multimodal_text_validator(text: str) -> None:
    lowered = text.lower()
    vision_tokens = [
        "image",
        "scene",
        "mountain",
        "landscape",
        "sky",
        "foreground",
        "background",
        "terrain",
        "cloud",
    ]

    hits = [tok for tok in vision_tokens if tok in lowered]
    assert hits, "[CANARY] multimodal output lacks visual grounding"


# ---------------------------------------------------------------------
# Canary Test
# ---------------------------------------------------------------------


@pytest.mark.canary
@pytest.mark.skipif(
    os.environ.get("MODEL_NAME") != TARGET_MODEL,
    reason="Not the target model for this canary",
)
def test_canary_gemma3_torchao():
    """
    Canary test for Gemma-3 TorchAO VLM.

    Observes for:
      - inference failures
      - output collapse
      - repetition loops
      - multimodal path breakage

    Non-goals:
      - regression comparison
      - version identity checks
      - semantic equivalence
    """

    model_ref = resolve_canary_model_ref()
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    for case in CANARY_CASES:
        timeout = 120.0 if case["is_multimodal"] else 90.0
        validator = multimodal_text_validator if case["is_multimodal"] else None

        text, _ = run_and_time(
            model_ref,
            case["input"],
            timeout_s=timeout,
            min_chars=MIN_OUTPUT_CHARS,
            output_type="vlm",
            validator=validator,
        )

        text = normalize_text(text)

        # --------------------------------------------------
        # Hard collapse guards
        # --------------------------------------------------
        assert len(text) >= MIN_OUTPUT_CHARS, f"[CANARY] {case['name']} output too short ({len(text)} chars)"

        rep = repetition_ratio(text)
        assert rep >= MIN_REPETITION_RATIO, f"[CANARY] {case['name']} repetition detected (ratio={rep:.2f})"

        # --------------------------------------------------
        # Semantic sanity (self-similarity only)
        # --------------------------------------------------
        emb = embedder.encode(text, normalize_embeddings=True)
        self_sim = float(np.dot(emb, emb))

        assert self_sim >= MIN_SELF_SIMILARITY, f"[CANARY] {case['name']} semantic collapse suspected (score={self_sim:.3f})"
