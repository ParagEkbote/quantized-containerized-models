from __future__ import annotations

import os

import imagehash
import numpy as np
import pytest
import replicate
import timm
import torch
from PIL import Image
from torchvision import transforms
from utils import run_image_and_time

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MODEL_BASE = "paragekbote/flux-fast-lora-hotswap-img2img"
TARGET_MODEL = "flux-fast-lora-hotswap-img2img"
STABLE_FLUX_IMG2IMG_MODEL_ID = os.environ.get("STABLE_FLUX_IMG2IMG_MODEL_ID")

PHASH_MAX_DISTANCE = 16
TIMM_MIN_SIMILARITY = 0.85  # slightly looser than CLIP


CANARY_CASES: list[dict] = [
    {
        "name": "open_image_preferences_lora",
        "input": {
            "prompt": "A cinematic portrait of a cyberpunk samurai",
            "trigger_word": "Cinematic",
            "init_image": ("https://images.pexels.com/photos/4934914/pexels-photo-4934914.jpeg"),
            "seed": 42,
            "guidance_scale": 7.0,
            "num_inference_steps": 20,
        },
    },
    {
        "name": "ghibsky_lora",
        "input": {
            "prompt": "A peaceful countryside village at sunset",
            "trigger_word": "GHIBSKY",
            "init_image": ("https://images.pexels.com/photos/4934914/pexels-photo-4934914.jpeg"),
            "seed": 42,
            "guidance_scale": 7.0,
            "num_inference_steps": 20,
        },
    },
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def get_latest_model_id() -> str:
    if not os.environ.get("REPLICATE_API_TOKEN"):
        raise RuntimeError("REPLICATE_API_TOKEN not set")

    model = replicate.models.get(MODEL_BASE)
    versions = list(model.versions.list())
    versions.sort(key=lambda v: v.created_at, reverse=True)

    return f"{MODEL_BASE}:{versions[0].id}"


def assert_image_sanity(img: Image.Image) -> None:
    arr = np.asarray(img, dtype=np.float32)

    assert np.isfinite(arr).all(), "NaN or Inf detected in image"

    mean_val = float(arr.mean())
    assert 1.0 < mean_val < 254.0, f"Abnormal brightness detected: {mean_val}"


class TimmEmbedder:
    """
    Lightweight semantic image embedder using timm.
    """

    def __init__(self, model_name: str = "resnet50"):
        self.device = "cpu"
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,  # feature extractor
        ).to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def embed(self, img: Image.Image) -> np.ndarray:
        with torch.no_grad():
            x = self.transform(img).unsqueeze(0).to(self.device)
            emb = self.model(x)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            return emb.cpu().numpy()[0]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


# ---------------------------------------------------------------------
# Canary test
# ---------------------------------------------------------------------


@pytest.mark.canary
@pytest.mark.skipif(
    os.environ.get("MODEL_NAME") != TARGET_MODEL,
    reason="Not the target model for this integration test",
)
def test_canary_release_flux_img2img():
    """
    Canary test for Flux img2img deployments.

    Guards against:
      - structural regressions (pHash)
      - semantic drift (timm embeddings)
    """

    # --------------------------------------------------
    # Hard requirements (fail fast)
    # --------------------------------------------------
    assert os.environ.get("REPLICATE_API_TOKEN"), "REPLICATE_API_TOKEN must be set to run canary tests"

    assert STABLE_FLUX_IMG2IMG_MODEL_ID, "STABLE_FLUX_IMG2IMG_MODEL_ID must be set"

    candidate_id = get_latest_model_id()

    # --------------------------------------------------
    # No-op canary (explicit pass)
    # --------------------------------------------------
    if candidate_id == STABLE_FLUX_IMG2IMG_MODEL_ID:
        assert True, "No-op canary: candidate model equals stable model (nothing new to compare)"
        return

    embedder = TimmEmbedder()

    for case in CANARY_CASES:
        old_img, _ = run_image_and_time(
            STABLE_FLUX_IMG2IMG_MODEL_ID,
            case["input"],
        )
        new_img, _ = run_image_and_time(
            candidate_id,
            case["input"],
        )

        # ------------------------------
        # Basic invariants
        # ------------------------------
        assert old_img.size == new_img.size, f"{case['name']} output resolution changed"

        assert_image_sanity(new_img)

        # ------------------------------
        # Structural regression
        # ------------------------------
        p_dist = imagehash.phash(old_img) - imagehash.phash(new_img)
        assert p_dist <= PHASH_MAX_DISTANCE, f"{case['name']} pHash drift too high: {p_dist}"

        # ------------------------------
        # Semantic regression (timm)
        # ------------------------------
        sim = cosine_similarity(
            embedder.embed(old_img),
            embedder.embed(new_img),
        )

        assert sim >= TIMM_MIN_SIMILARITY, f"{case['name']} timm similarity too low: {sim:.4f}"
