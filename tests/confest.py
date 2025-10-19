# tests/conftest.py
import sys
import os
from pathlib import Path
import io
from unittest.mock import MagicMock
import pytest
import torch
import requests
from PIL import Image

# ------------------------
# Ensure src/ is on sys.path for all tests
# ------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Now all model imports work:
# from models.flux_fast_lora_hotswap_img2img.predict import Predictor, load_image, save_image

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
    if not monkeypatch.getenv("HF_TOKEN"):
        monkeypatch.setenv("HF_TOKEN", "dummy_token")


# ------------------------
# Fixture to clear HF_TOKEN for negative tests
# ------------------------
@pytest.fixture
def clear_hf_token(monkeypatch):
    """Temporarily remove HF_TOKEN."""
    monkeypatch.delenv("HF_TOKEN", raising=False)


# ------------------------
# Fixture for dummy image
# ------------------------
@pytest.fixture
def dummy_image() -> Image.Image:
    """Provides a small dummy RGB image for testing."""
    return Image.new("RGB", (64, 64), color="blue")


# ------------------------
# Fixture for test image file
# ------------------------
@pytest.fixture
def test_image_file(tmp_path) -> Path:
    """Creates a temporary test image file and returns its path."""
    img_path = tmp_path / "test_image.png"
    Image.new("RGB", (100, 100), color="red").save(img_path)
    return img_path


# ------------------------
# Fixture to mock torch.no_grad
# ------------------------
@pytest.fixture
def mock_no_grad(monkeypatch):
    """Provides a mock for torch.no_grad context manager."""
    mock_context = MagicMock()
    mock_no_grad_func = MagicMock(return_value=mock_context)
    monkeypatch.setattr(torch, "no_grad", mock_no_grad_func)
    return mock_no_grad_func


# ------------------------
# Fixture to mock requests.get for load_image tests
# ------------------------
@pytest.fixture
def mock_requests(monkeypatch, dummy_image):
    """Mocks requests.get to return a dummy image in memory."""
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
    """Globally mock save_image so tests don’t write files."""
    fake_path = tmp_path / "fake_saved.png"
    # Patch generically; works for any model module
    def fake_save_image(img, output_dir=None):
        return fake_path

    # Example: patch all known predict modules if needed
    import importlib
    for model in [
        "flux_fast_lora_hotswap_img2img",
        "flux_fast_lora_hotswap",
        "gemma_torchao",
        "phi_4_reasoning_plus_unsloth",
        "smollm3_pruna",
    ]:
        try:
            mod = importlib.import_module(f"models.{model}.predict")
            monkeypatch.setattr(mod, "save_image", fake_save_image)
        except ModuleNotFoundError:
            pass

    return fake_path
