from __future__ import annotations

import os

import imagehash
import numpy as np
import pytest
from PIL import Image
from utils import run_image_and_time

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

OWNER = "paragekbote"
MODEL_NAME = "flux-fast-lora-hotswap-img2img"
MODEL_ALIAS = f"{OWNER}/{MODEL_NAME}"

TARGET_MODEL = MODEL_NAME

MIN_PHASH_ENTROPY = 10  # collapse detector, not regression
MIN_PIXEL_VARIANCE = 5.0  # blank / near-constant image guard

CANARY_CASES = [
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


def resolve_canary_model_ref() -> str:
    """
    Canary uses best-effort candidate version,
    otherwise the model alias (latest deployment).
    """
    return os.environ.get("CANDIDATE_MODEL_ID") or MODEL_ALIAS


def assert_image_sanity(img: Image.Image) -> None:
    arr = np.asarray(img, dtype=np.float32)

    assert np.isfinite(arr).all(), "[CANARY] NaN/Inf detected in image"

    mean = float(arr.mean())
    var = float(arr.var())

    assert 1.0 < mean < 254.0, f"[CANARY] abnormal brightness: {mean:.2f}"
    assert var > MIN_PIXEL_VARIANCE, f"[CANARY] image collapse suspected (var={var:.2f})"


# ---------------------------------------------------------------------
# Canary Test
# ---------------------------------------------------------------------


@pytest.mark.canary
@pytest.mark.skipif(
    os.environ.get("MODEL_NAME") != TARGET_MODEL,
    reason="Not the target model for this canary",
)
def test_canary_flux_img2img():
    """
    Canary test for Flux img2img deployment.

    Observes for:
      - inference failures
      - blank / collapsed images
      - NaN / Inf outputs
      - obvious structural collapse

    Non-goals:
      - regression comparison
      - LoRA semantic equivalence
      - version identity validation
    """

    model_ref = resolve_canary_model_ref()

    for case in CANARY_CASES:
        img, latency = run_image_and_time(
            model_ref,
            case["input"],
        )

        # --------------------------------------------------
        # Hard collapse guards
        # --------------------------------------------------
        assert isinstance(img, Image.Image), "[CANARY] output is not an image"

        assert_image_sanity(img)

        # --------------------------------------------------
        # Structural sanity (self pHash entropy)
        # --------------------------------------------------
        ph = imagehash.phash(img)
        entropy = len(set(ph.hash.flatten()))

        assert entropy >= MIN_PHASH_ENTROPY, f"[CANARY] {case['name']} image appears collapsed (pHash entropy={entropy})"
