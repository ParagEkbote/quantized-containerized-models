from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pytest
import replicate
import imagehash
import torch
import clip
from PIL import Image

from utils import run_image_and_time


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MODEL_BASE = "r8.im/paragekbote/flux-fast-lora-hotswap-img2img"
STABLE_MODEL_ID = os.environ.get("STABLE_MODEL_ID")

PHASH_MAX_DISTANCE = 16
CLIP_MIN_SIMILARITY = 0.90

CANARY_CASES: List[Dict] = [
    {
        "name": "open_image_preferences_lora",
        "input": {
            "prompt": "A cinematic portrait of a cyberpunk samurai",
            "trigger_word": "Cinematic",
            "init_image": "https://images.pexels.com/photos/4934914/pexels-photo-4934914.jpeg",
            "seed": 42,
        },
    },
    {
        "name": "ghibsky_lora",
        "input": {
            "prompt": "A peaceful countryside village at sunset",
            "trigger_word": "GHIBSKY",
            "init_image": "https://images.pexels.com/photos/4934914/pexels-photo-4934914.jpeg",
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


def assert_image_sanity(img: Image.Image) -> None:
    arr = np.asarray(img, dtype=np.float32)
    assert np.isfinite(arr).all(), "NaN or Inf detected in image"

    mean_val = float(arr.mean())
    assert 1.0 < mean_val < 254.0, f"Abnormal brightness: {mean_val}"


class ClipEmbedder:
    def __init__(self):
        self.device = "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def embed(self, img: Image.Image) -> np.ndarray:
        with torch.no_grad():
            x = self.preprocess(img).unsqueeze(0).to(self.device)
            emb = self.model.encode_image(x)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            return emb.cpu().numpy()[0]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ---------------------------------------------------------------------
# Canary test
# ---------------------------------------------------------------------

@pytest.mark.canary
def test_canary_release():
    if not STABLE_MODEL_ID:
        pytest.skip("STABLE_MODEL_ID not set")

    candidate_id = get_latest_model_id()

    if candidate_id == STABLE_MODEL_ID:
        pytest.skip("Candidate equals stable")

    clipper = ClipEmbedder()

    for case in CANARY_CASES:
        old_img, _ = run_image_and_time(STABLE_MODEL_ID, case["input"])
        new_img, _ = run_image_and_time(candidate_id, case["input"])

        assert old_img.size == new_img.size
        assert_image_sanity(new_img)

        # Structural regression
        p_dist = imagehash.phash(old_img) - imagehash.phash(new_img)
        assert p_dist <= PHASH_MAX_DISTANCE, (
            f"{case['name']} pHash drift too high: {p_dist}"
        )

        # Semantic regression
        sim = cosine_similarity(
            clipper.embed(old_img),
            clipper.embed(new_img),
        )
        assert sim >= CLIP_MIN_SIMILARITY, (
            f"{case['name']} CLIP similarity too low: {sim:.4f}"
        )
