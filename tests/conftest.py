# tests/conftest.py
"""
Pytest configuration and shared fixtures.

Provides common test utilities including:
- Path setup for src/ directory
- Image fixtures for testing image processing
- Mock utilities for external dependencies
- HF token management
- Pipeline mocks for diffusion models
"""
import os
import sys
import io
import importlib
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
from contextlib import contextmanager

import pytest
import torch
import requests
from PIL import Image


# ---------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

# Only add to path if not already present
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ---------------------------------------------------------------------
# Simple Fixtures
# ---------------------------------------------------------------------
@pytest.fixture
def tmp_data_dir(tmp_path):
    """Temporary directory for test data."""
    return tmp_path


@pytest.fixture
def dummy_image():
    """Small blue RGB image for basic tests."""
    return Image.new("RGB", (64, 64), "blue")


@pytest.fixture
def test_image_file(tmp_path):
    """Saved red test image file."""
    p = tmp_path / "test.png"
    Image.new("RGB", (100, 100), "red").save(p)
    return p


@pytest.fixture
def sample_image_path(tmp_path):
    """Saved gray test image at standard resolution."""
    p = tmp_path / "sample.png"
    Image.new("RGB", (512, 512), (128, 128, 128)).save(p)
    return str(p)


# ---------------------------------------------------------------------
# HuggingFace Token Management
# ---------------------------------------------------------------------
@pytest.fixture
def ensure_hf_token(monkeypatch):
    """Set HF_TOKEN environment variable for tests requiring auth."""
    monkeypatch.setenv("HF_TOKEN", "test_token")


@pytest.fixture
def clear_hf_token(monkeypatch):
    """Remove HF_TOKEN from environment."""
    monkeypatch.delenv("HF_TOKEN", raising=False)


# ---------------------------------------------------------------------
# HTTP Mocking
# ---------------------------------------------------------------------
@pytest.fixture
def mock_requests(monkeypatch, dummy_image):
    """
    Mock requests.get to return dummy PNG bytes.
    
    Usage:
        def test_download(mock_requests):
            response = requests.get("http://example.com/image.png")
            assert response.status_code == 200
    """
    def _mock_get(*args, **kwargs):
        buf = io.BytesIO()
        dummy_image.save(buf, format="PNG")
        buf.seek(0)

        class MockResponse:
            status_code = 200
            content = buf.read()
            
            def raise_for_status(self):
                pass
            
            def json(self):
                return {}

        return MockResponse()

    monkeypatch.setattr(requests, "get", _mock_get)
    return _mock_get


# ---------------------------------------------------------------------
# PyTorch Utilities
# ---------------------------------------------------------------------
@pytest.fixture
def mock_no_grad(monkeypatch):
    """Mock torch.no_grad context manager."""
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=ctx)
    ctx.__exit__ = MagicMock(return_value=False)
    monkeypatch.setattr(torch, "no_grad", lambda: ctx)
    return ctx


@pytest.fixture(autouse=True)
def mock_cuda_for_cpu_tests():
    """
    Auto-mock CUDA availability to False for CPU-only testing.
    
    Override in specific tests with:
        @patch("torch.cuda.is_available", return_value=True)
    """
    with patch("torch.cuda.is_available", return_value=False):
        yield


# ---------------------------------------------------------------------
# Diffusion Pipeline Mocks
# ---------------------------------------------------------------------
@pytest.fixture
def mock_pipeline():
    """
    Mock diffusion pipeline with common methods.
    
    Returns a pipeline that:
    - Returns a red 512x512 image when called
    - Has standard methods (to, enable_model_cpu_offload, etc.)
    """
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
    pipe.enable_vae_slicing = Mock()
    pipe.enable_vae_tiling = Mock()

    return pipe


@pytest.fixture
def mock_scheduler():
    """Mock scheduler with empty config."""
    s = Mock()
    s.config = {}
    return s


# ---------------------------------------------------------------------
# Image Saving Mock (Global)
# ---------------------------------------------------------------------
@pytest.fixture(autouse=True)
def mock_save_image(monkeypatch, tmp_path):
    """
    Mock save_image globally to avoid actual file I/O in tests.
    
    Returns a path to a fake file that tests can use.
    Override in specific tests if real file saving is needed.
    """
    fake_path = tmp_path / "fake_output.png"
    
    def _fake_save(*args, **kwargs):
        return fake_path

    # List of modules that have save_image function
    image_modules = [
        "flux_fast_lora_hotswap_img2img",
        "flux_fast_lora_hotswap",
        # Add more modules here as needed
    ]

    for module_name in image_modules:
        try:
            mod = importlib.import_module(f"models.{module_name}.predict")
            if hasattr(mod, "save_image"):
                monkeypatch.setattr(mod, "save_image", _fake_save)
        except (ImportError, AttributeError):
            # Module doesn't exist or doesn't have save_image - skip
            continue

    return fake_path


# ---------------------------------------------------------------------
# Predictor Factory (Centralized Patching)
# ---------------------------------------------------------------------
@pytest.fixture
def predictor(mock_pipeline, mock_scheduler, monkeypatch):
    """
    Creates a fully-mocked Predictor instance for flux models.
    
    All dependencies (FluxPipeline, scheduler, CUDA, etc.) are mocked.
    Use this for unit tests that don't need real model loading.
    
    Returns:
        Predictor instance with:
        - .mock_pipeline: Access to mocked pipeline
        - .mock_scheduler: Access to mocked scheduler
    """
    with (
        patch("models.flux_fast_lora_hotswap_img2img.predict.FluxPipeline") as MockPipeline,
        patch("models.flux_fast_lora_hotswap_img2img.predict.FlowMatchEulerDiscreteScheduler") as MockScheduler,
        patch("models.flux_fast_lora_hotswap_img2img.predict.torch.compile") as MockCompile,
        patch("torch.cuda.is_available", return_value=True),
    ):
        # Setup mock returns
        MockPipeline.from_pretrained.return_value = mock_pipeline
        MockScheduler.from_config.return_value = mock_scheduler
        MockCompile.return_value = mock_pipeline

        # Fix Pydantic FieldInfo issues by setting default values
        from models.flux_fast_lora_hotswap_img2img import predict as predict_module
        
        defaults = {
            "width": 512,
            "height": 512,
            "steps": 20,
            "seed": 42,
            "guidance_scale": 3.5,
            "num_outputs": 1,
        }
        
        for key, value in defaults.items():
            if hasattr(predict_module, key):
                monkeypatch.setattr(predict_module, key, value)

        # Import and setup predictor
        from models.flux_fast_lora_hotswap_img2img.predict import Predictor
        pred = Predictor()
        pred.setup()

        # Attach mocks for test inspection
        pred.mock_pipeline = mock_pipeline
        pred.mock_scheduler = mock_scheduler
        
        return pred


# ---------------------------------------------------------------------
# Unsloth Mock (for reasoning models)
# ---------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def mock_unsloth_globally():
    """
    Mock unsloth library globally to enable CPU-only testing.
    
    This runs once per test session and prevents ImportErrors
    when unsloth is not installed or requires GPU.
    """
    if 'unsloth' not in sys.modules:
        mock_unsloth = MagicMock()
        
        # Mock FastLanguageModel.from_pretrained
        def mock_from_pretrained(*args, **kwargs):
            model = MagicMock()
            model.parameters.return_value = [torch.nn.Parameter(torch.zeros(1))]
            model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
            model.to.return_value = model
            model.eval.return_value = model
            model.config = MagicMock()
            
            tokenizer = MagicMock()
            tokenizer.eos_token_id = 2
            tokenizer.pad_token_id = 0
            
            def tokenizer_call(*args, **kwargs):
                return {
                    "input_ids": torch.tensor([[1, 2, 3]]),
                    "attention_mask": torch.tensor([[1, 1, 1]])
                }
            
            tokenizer.side_effect = tokenizer_call
            tokenizer.decode.return_value = "generated text output"
            
            return model, tokenizer
        
        mock_unsloth.FastLanguageModel.from_pretrained = mock_from_pretrained
        sys.modules['unsloth'] = mock_unsloth


# ---------------------------------------------------------------------
# Pytest Configuration Hooks
# ---------------------------------------------------------------------
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", 
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers",
        "gpu: marks tests that require GPU (skip on CPU-only systems)"
    )


def pytest_collection_modifyitems(config, items):
    """
    Automatically skip GPU tests if CUDA is not available.
    
    Tests marked with @pytest.mark.gpu will be skipped unless
    CUDA is available.
    """
    skip_gpu = pytest.mark.skip(reason="GPU not available")
    
    for item in items:
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_gpu)


# ---------------------------------------------------------------------
# Helper Context Managers (for advanced test scenarios)
# ---------------------------------------------------------------------
@contextmanager
def temporary_env_var(key, value):
    """
    Temporarily set an environment variable.
    
    Usage:
        with temporary_env_var("HF_TOKEN", "test"):
            # HF_TOKEN is "test" here
            pass
        # HF_TOKEN is restored here
    """
    old_value = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old_value


@pytest.fixture
def isolated_modules():
    """
    Fixture to reload modules for isolated testing.
    
    Usage:
        def test_with_isolation(isolated_modules):
            with isolated_modules("my_module"):
                # Module is fresh here
                pass
    """
    @contextmanager
    def _reload_module(module_name):
        """Reload a module and restore it after test."""
        if module_name in sys.modules:
            old_module = sys.modules[module_name]
            del sys.modules[module_name]
        else:
            old_module = None
        
        try:
            yield
        finally:
            if old_module:
                sys.modules[module_name] = old_module
            elif module_name in sys.modules:
                del sys.modules[module_name]
    
    return _reload_module