from pathlib import Path
from unittest.mock import MagicMock
import io

import pytest
import torch
import requests
from PIL import Image

from models.flux_fast_lora_hotswap_img2img.predict import Predictor
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ------------------------
# Temporary directory fixture
# ------------------------
@pytest.fixture
def tmp_data_dir(tmp_path) -> Path:
    """Provides a temporary directory for storing test data/images."""
    return tmp_path


# ------------------------
# Automatically set dummy HF_TOKEN
# ------------------------
@pytest.fixture(autouse=True)
def ensure_hf_token(monkeypatch):
    """Automatically sets a dummy HF_TOKEN for tests."""
    try:
        existing = monkeypatch.getenv("HF_TOKEN")
        if not existing:
            monkeypatch.setenv("HF_TOKEN", "dummy_token")
    except AttributeError:
        monkeypatch.setenv("HF_TOKEN", "dummy_token")


# ------------------------
# Fixture to clear HF_TOKEN for negative tests
# ------------------------
@pytest.fixture
def clear_hf_token(monkeypatch):
    """Temporarily remove HF_TOKEN."""
    monkeypatch.delenv("HF_TOKEN", raising=False)


# ------------------------
# DummyPipeline class
# ------------------------
class DummyPipeline:
    """Mock pipeline that tracks all method calls for verification in tests."""
    def __init__(self):
        self.text_encoder = lambda x: x
        self.text_encoder_2 = lambda x: x
        self.vae = lambda x: x
        self.images = [Image.new("RGB", (64, 64), color="pink")]

        # Tracking
        self.set_adapters_calls = []
        self.call_kwargs = None
        self.hotswap_enabled = False
        self.target_rank = None
        self.last_adapters = None
        self.last_weights = None

    def set_adapters(self, adapters, adapter_weights):
        self.set_adapters_calls.append({
            'adapters': adapters,
            'weights': adapter_weights
        })
        self.last_adapters = adapters
        self.last_weights = adapter_weights

    def enable_lora_hotswap(self, target_rank):
        self.hotswap_enabled = True
        self.target_rank = target_rank

    def load_lora_weights(self, *args, **kwargs):
        pass

    def __call__(self, **kwargs):
        self.call_kwargs = kwargs
        return self

    def reset_tracking(self):
        self.set_adapters_calls = []
        self.call_kwargs = None
        self.last_adapters = None
        self.last_weights = None


# ------------------------
# Fixture for Predictor with mocks
# ------------------------
@pytest.fixture
def predictor(monkeypatch) -> Predictor:
    """
    Returns a Predictor instance with heavy dependencies mocked.
    """
    pred = Predictor()

    # Patch the predictor's pipe
    dummy_pipe = DummyPipeline()
    pred.pipe = dummy_pipe

    # Track torch.compile calls
    pred._compile_calls = []

    def mock_compile(f, **kwargs):
        pred._compile_calls.append({'func': f, 'kwargs': kwargs})
        return f

    monkeypatch.setattr(torch, "compile", mock_compile)

    # Mock CUDA memory
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 1e9)

    # Mock save_image (matches predict.py import)
    monkeypatch.setattr(
        "models.flux_fast_lora_hotswap_img2img.predict.save_image",
        lambda img, output_dir=None: Path("/tmp/fake_image.png")
    )

    # Set adapter triggers
    pred.current_adapter = "open-image-preferences"
    pred.lora1_triggers = [
        "Cinematic", "Photographic", "Anime", "Manga",
        "Digital art", "Pixel art", "Fantasy art", "Neonpunk",
        "3D Model", "Painting", "Animation", "Illustration",
    ]
    pred.lora2_triggers = ["GHIBSKY"]

    return pred


# ------------------------
# Fixture for dummy image
# ------------------------
@pytest.fixture
def dummy_image() -> Image.Image:
    """Provides a small dummy RGB image for testing."""
    return Image.new("RGB", (64, 64), color="blue")


# ------------------------
# Fixture for mock torch.no_grad
# ------------------------
@pytest.fixture
def mock_no_grad(monkeypatch):
    """Provides a mock for torch.no_grad context manager."""
    mock_context = MagicMock()
    mock_no_grad_func = MagicMock(return_value=mock_context)
    monkeypatch.setattr(torch, "no_grad", mock_no_grad_func)
    return mock_no_grad_func


# ------------------------
# Fixture for test image file
# ------------------------
@pytest.fixture
def test_image_file(tmp_path) -> Path:
    """Creates a temporary test image file and returns its path."""
    img_path = tmp_path / "test_image.png"
    img = Image.new("RGB", (100, 100), color="red")
    img.save(img_path)
    return img_path


# ------------------------
# Fixture to mock requests.get for load_image tests
# ------------------------
@pytest.fixture
def mock_requests(monkeypatch, dummy_image):
    """
    Mocks requests.get to return a dummy image in memory.
    Useful for load_image(url) tests.
    """
    def mock_get(url, *args, **kwargs):
        img_bytes = io.BytesIO()
        dummy_image.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        class MockResponse:
            def __init__(self):
                self.status_code = 200
                self.content = img_bytes.read()
            def raise_for_status(self): pass

        return MockResponse()

    monkeypatch.setattr(requests, "get", mock_get)
    return mock_get


# ------------------------
# Fixture to globally mock save_image for all tests
# ------------------------
@pytest.fixture(autouse=True)
def mock_save_image(monkeypatch, tmp_path):
    """
    Globally mock save_image so tests don’t write files.
    """
    fake_path = tmp_path / "fake_saved.png"
    monkeypatch.setattr(
        "models.flux_fast_lora_hotswap_img2img.predict.save_image",
        lambda img, output_dir=None: fake_path
    )
    return fake_path
