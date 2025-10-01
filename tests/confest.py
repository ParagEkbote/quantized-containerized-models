from pathlib import Path

import torch
from PIL import Image

import pytest
from models.flux_fast_lora_hotswap_img2img.predict import Predictor


# ------------------------
# Temporary directory fixture
# ------------------------
@pytest.fixture
def tmp_data_dir(tmp_path) -> Path:
    """
    Provides a temporary directory for storing test data/images.
    """
    return tmp_path


# ------------------------
# Automatically set dummy HF_TOKEN
# ------------------------
@pytest.fixture(autouse=True)
def ensure_hf_token(monkeypatch):
    """
    Automatically sets a dummy HF_TOKEN if not set.
    Useful for login_with_env_token tests.
    """
    if not (token := monkeypatch.getenv("HF_TOKEN")):
        monkeypatch.setenv("HF_TOKEN", "dummy_token")


# ------------------------
# Fixture for Predictor with mocks
# ------------------------
@pytest.fixture
def predictor(monkeypatch) -> Predictor:
    """
    Returns a Predictor instance with heavy dependencies mocked:
    - DiffusionPipeline replaced with DummyPipeline
    - torch.compile returns identity
    - torch.cuda.memory_allocated returns 1 GB
    - save_image returns a fake path
    """

    class DummyPipeline:
        def __init__(self):
            self.text_encoder = lambda x: x
            self.text_encoder_2 = lambda x: x
            self.vae = lambda x: x
            self.images = [Image.new("RGB", (64, 64), color="pink")]

        def set_adapters(self, adapters, adapter_weights):
            self.last_adapters = adapters
            self.last_weights = adapter_weights

        def enable_lora_hotswap(self, target_rank):
            pass

        def load_lora_weights(self, *args, **kwargs):
            pass

        def __call__(self, **kwargs):
            return self

    pred = Predictor()
    monkeypatch.setattr(pred, "pipe", DummyPipeline())
    monkeypatch.setattr(torch, "compile", lambda f, **kwargs: f)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 1e9)
    monkeypatch.setattr(
        "models.flux_fast_lora_hotswap_img2img.predict.save_image",
        lambda img: Path("/tmp/fake_image.png"),
    )

    pred.current_adapter = "open-image-preferences"
    pred.lora1_triggers = ["Cinematic", "Anime"]
    pred.lora2_triggers = ["GHIBSKY"]

    return pred
