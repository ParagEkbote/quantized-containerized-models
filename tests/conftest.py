# tests/conftest.py
import os
import sys
import io
import importlib
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import requests
from PIL import Image


# ---------------------------------------------------------------------
# Add src/ to the path
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


# ---------------------------------------------------------------------
# Simple Fixtures
# ---------------------------------------------------------------------
@pytest.fixture
def tmp_data_dir(tmp_path):
    return tmp_path


@pytest.fixture
def dummy_image():
    return Image.new("RGB", (64, 64), "blue")


@pytest.fixture
def test_image_file(tmp_path):
    p = tmp_path / "test.png"
    Image.new("RGB", (100, 100), "red").save(p)
    return p


@pytest.fixture
def sample_image_path(tmp_path):
    p = tmp_path / "sample.png"
    Image.new("RGB", (512, 512), (128, 128, 128)).save(p)
    return str(p)


# ---------------------------------------------------------------------
# HF_TOKEN helpers
# ---------------------------------------------------------------------
@pytest.fixture
def ensure_hf_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test_token")


@pytest.fixture
def clear_hf_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)


# ---------------------------------------------------------------------
# Mock requests.get
# ---------------------------------------------------------------------
@pytest.fixture
def mock_requests(monkeypatch, dummy_image):
    """Return dummy PNG bytes for any HTTP GET."""
    def _mock(*args, **kwargs):
        buf = io.BytesIO()
        dummy_image.save(buf, format="PNG")
        buf.seek(0)

        class Resp:
            status_code = 200
            content = buf.read()
            def raise_for_status(self): pass

        return Resp()

    monkeypatch.setattr(requests, "get", _mock)
    return _mock


# ---------------------------------------------------------------------
# Mock torch.no_grad
# ---------------------------------------------------------------------
@pytest.fixture
def mock_no_grad(monkeypatch):
    ctx = MagicMock()
    monkeypatch.setattr(torch, "no_grad", lambda: ctx)
    return ctx


# ---------------------------------------------------------------------
# Mock diffusion pipeline
# ---------------------------------------------------------------------
@pytest.fixture
def mock_pipeline():
    mock_img = Image.new("RGB", (512, 512), "red")

    pipe = Mock()
    pipe.return_value = Mock(images=[mock_img])
    pipe.to.return_value = pipe
    pipe.enable_model_cpu_offload = Mock()
    pipe.set_adapters = Mock()
    pipe.load_lora_weights = Mock()
    pipe.fuse_lora = Mock()
    pipe.unfuse_lora = Mock()
    pipe.unload_lora_weights = Mock()

    return pipe


@pytest.fixture
def mock_scheduler():
    s = Mock()
    s.config = {}
    return s


# ---------------------------------------------------------------------
# Mock save_image globally for all image model tests
# ---------------------------------------------------------------------
@pytest.fixture(autouse=True)
def mock_save_image(monkeypatch, tmp_path):
    fake = tmp_path / "fake.png"
    def _fake(*args, **kwargs): return fake

    # Patch image model modules
    for mdl in ("flux_fast_lora_hotswap_img2img", "flux_fast_lora_hotswap"):
        try:
            mod = importlib.import_module(f"models.{mdl}.predict")
            if hasattr(mod, "save_image"):
                monkeypatch.setattr(mod, "save_image", _fake)
        except ImportError:
            pass

    return fake


# ---------------------------------------------------------------------
# Predictor fixture (centralized patching)
# ---------------------------------------------------------------------
@pytest.fixture
def predictor(mock_pipeline, mock_scheduler, monkeypatch):
    """Creates a Predictor instance with all dependencies mocked."""
    with (
        patch("models.flux_fast_lora_hotswap_img2img.predict.FluxPipeline") as P,
        patch("models.flux_fast_lora_hotswap_img2img.predict.FlowMatchEulerDiscreteScheduler") as S,
        patch("models.flux_fast_lora_hotswap_img2img.predict.torch.compile") as C,
        patch("models.flux_fast_lora_hotswap_img2img.predict.torch.cuda.is_available", return_value=True),
    ):
        P.from_pretrained.return_value = mock_pipeline
        S.from_config.return_value = mock_scheduler
        C.return_value = mock_pipeline

        # Fix Pydantic FieldInfo issues
        from models.flux_fast_lora_hotswap_img2img import predict as M
        defaults = dict(width=512, height=512, steps=20, seed=42)
        for k, v in defaults.items():
            if k in M.__dict__:
                monkeypatch.setattr(M, k, v)

        from models.flux_fast_lora_hotswap_img2img.predict import Predictor
        pred = Predictor()
        pred.setup()

        pred.mock_pipeline = mock_pipeline
        pred.mock_scheduler = mock_scheduler
        return pred
