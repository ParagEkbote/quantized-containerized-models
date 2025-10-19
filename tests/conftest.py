# tests/conftest.py
import sys
import os
from pathlib import Path
import io
from unittest.mock import MagicMock, Mock, patch
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
def tmp_data_dir(tmp_path):
    """Create a temporary directory for test data."""
    return tmp_path


# ------------------------
# Fixture to clear HF_TOKEN for negative tests
# ------------------------
@pytest.fixture
def clear_hf_token(monkeypatch):
    """Temporarily remove HF_TOKEN."""
    monkeypatch.delenv("HF_TOKEN", raising=False)


# ------------------------
# Fixture to ensure HF_TOKEN is set
# ------------------------
@pytest.fixture
def ensure_hf_token(monkeypatch):
    """Ensure HF_TOKEN is set for tests that need it."""
    monkeypatch.setenv("HF_TOKEN", "test_token_12345")


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
# Fixture for sample image path (img2img testing)
# ------------------------
@pytest.fixture
def sample_image_path(tmp_path):
    """Create a temporary test image for img2img testing."""
    # Create a simple RGB image
    img = Image.new('RGB', (512, 512), color=(128, 128, 128))
    img_path = tmp_path / "test_input_image.png"
    img.save(img_path)
    return str(img_path)


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
    def mock_get(*args, **kwargs):
        img_bytes = io.BytesIO()
        dummy_image.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        class MockResponse:
            def __init__(self):
                self.status_code = 200
                self.content = img_bytes.read()
            
            def raise_for_status(self):
                pass

        return MockResponse()

    monkeypatch.setattr(requests, "get", mock_get)
    return mock_get


# ------------------------
# Mock pipeline for flux_fast_lora_hotswap_img2img
# ------------------------
@pytest.fixture
def mock_pipeline():
    """Create a mock diffusion pipeline."""
    pipeline = Mock()
    
    # Mock the pipeline call to return a valid result
    mock_image = Image.new('RGB', (512, 512), color=(255, 0, 0))
    pipeline.return_value = Mock(images=[mock_image])
    
    # Mock other pipeline attributes
    pipeline.to = Mock(return_value=pipeline)
    pipeline.enable_model_cpu_offload = Mock()
    pipeline.load_lora_weights = Mock()
    pipeline.set_adapters = Mock()
    pipeline.fuse_lora = Mock()
    pipeline.unfuse_lora = Mock()
    pipeline.unload_lora_weights = Mock()
    
    return pipeline


# ------------------------
# Mock scheduler for flux models
# ------------------------
@pytest.fixture
def mock_scheduler():
    """Create a mock scheduler."""
    scheduler = Mock()
    scheduler.config = {}
    return scheduler


# ------------------------
# Predictor fixture for flux_fast_lora_hotswap_img2img
# ------------------------
@pytest.fixture
def predictor(mock_pipeline, mock_scheduler, monkeypatch):
    """Create a Predictor instance with mocked dependencies."""
    with patch("models.flux_fast_lora_hotswap_img2img.predict.FluxPipeline") as mock_flux_pipeline, \
         patch("models.flux_fast_lora_hotswap_img2img.predict.FlowMatchEulerDiscreteScheduler") as mock_sched_class, \
         patch("models.flux_fast_lora_hotswap_img2img.predict.torch.compile") as mock_compile, \
         patch("models.flux_fast_lora_hotswap_img2img.predict.torch.cuda.is_available", return_value=True):

        # 🩹 Patch: Ensure any Pydantic FieldInfo configs return actual numbers
        from models import flux_fast_lora_hotswap_img2img as module
        predict_mod = module.predict

        for name in ("width", "height", "steps", "seed"):
            if hasattr(predict_mod, name):
                val = getattr(predict_mod, name)
                # Replace FieldInfo objects with ints
                if "FieldInfo" in str(type(val)):
                    monkeypatch.setattr(predict_mod, name, 512 if name != "steps" else 20)

        # Setup mocks
        mock_flux_pipeline.from_pretrained.return_value = mock_pipeline
        mock_sched_class.from_config.return_value = mock_scheduler
        mock_compile.return_value = mock_pipeline

        # Create predictor
        from models.flux_fast_lora_hotswap_img2img.predict import Predictor
        predictor_instance = Predictor()
        predictor_instance.setup()

        # Store mocks for test assertions
        predictor_instance.mock_pipeline = mock_pipeline
        predictor_instance.mock_scheduler = mock_scheduler

        return predictor_instance


# ------------------------
# Fixture to globally mock save_image for all tests
# ------------------------
@pytest.fixture(autouse=True)
def mock_save_image(monkeypatch, tmp_path):
    """Globally mock save_image so tests don't write files (only for image models)."""
    fake_path = tmp_path / "fake_saved.png"
    
    def fake_save_image(*args, **kwargs):
        return fake_path

    # Patch all known image-based model modules
    import importlib
    image_models = [
        "flux_fast_lora_hotswap_img2img",
        "flux_fast_lora_hotswap",
    ]
    
    for model in image_models:
        try:
            mod = importlib.import_module(f"models.{model}.predict")
            if hasattr(mod, "save_image"):
                monkeypatch.setattr(mod, "save_image", fake_save_image)
        except (ModuleNotFoundError, AttributeError):
            pass

    return fake_path