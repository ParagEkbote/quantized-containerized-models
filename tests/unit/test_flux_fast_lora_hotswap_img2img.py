from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests
import torch
from PIL import Image
from pytest import raises

from models.flux_fast_lora_hotswap_img2img.predict import (
    Predictor,
    load_image,
    login_with_env_token,
    save_image,
)


# ========================
# login_with_env_token tests
# ========================
def test_login_with_valid_token(monkeypatch):
    """Test successful login with HF_TOKEN."""
    monkeypatch.setenv("HF_TOKEN", "dummy_token")
    called = {}

    def mock_login(token):
        called["token"] = token

    monkeypatch.setattr("models.flux_fast_lora_hotswap_img2img.predict.login", mock_login)
    login_with_env_token()
    assert called["token"] == "dummy_token"


def test_login_with_missing_token(monkeypatch):
    """Test ValueError raised when HF_TOKEN is missing."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with raises(ValueError, match="HF_TOKEN not found"):
        login_with_env_token()


def test_login_with_custom_env_var(monkeypatch):
    """Test login with custom environment variable."""
    monkeypatch.setenv("CUSTOM_TOKEN", "custom_value")
    called = {}

    def mock_login(token):
        called["token"] = token

    monkeypatch.setattr("models.flux_fast_lora_hotswap_img2img.predict.login", mock_login)
    login_with_env_token(env_var="CUSTOM_TOKEN")
    assert called["token"] == "custom_value"


# ========================
# save_image tests
# ========================
def test_save_image(tmp_path):
    """Test basic image saving functionality."""
    img = Image.new("RGB", (10, 10), color="red")
    output_path = save_image(img, output_dir=tmp_path)
    
    assert output_path.exists()
    assert output_path.suffix == ".png"
    assert Image.open(output_path).size == (10, 10)


def test_save_image_creates_directory(tmp_path):
    """Test that nested directories are created automatically."""
    img = Image.new("RGB", (10, 10), color="blue")
    nested_dir = tmp_path / "nested" / "dir"
    output_path = save_image(img, output_dir=nested_dir)
    
    assert nested_dir.exists()
    assert output_path.exists()


def test_save_image_unique_filenames(tmp_path):
    """Test that each save generates a unique filename."""
    img = Image.new("RGB", (10, 10), color="green")
    path1 = save_image(img, output_dir=tmp_path)
    path2 = save_image(img, output_dir=tmp_path)
    
    assert path1 != path2


# ========================
# load_image tests
# ========================
def test_load_image_from_file(tmp_path):
    """Test loading image from local file."""
    file_path = tmp_path / "test.png"
    img = Image.new("RGB", (5, 5), color="blue")
    img.save(file_path)
    
    loaded_img = load_image(str(file_path))
    assert isinstance(loaded_img, Image.Image)
    assert loaded_img.size == (5, 5)


def test_load_image_from_url(monkeypatch):
    """Test loading image from URL."""
    dummy_img = Image.new("RGB", (8, 8), color="green")
    buf = BytesIO()
    dummy_img.save(buf, format="PNG")
    buf.seek(0)

    class MockResponse:
        content = buf.getvalue()
        def raise_for_status(self):
            pass

    monkeypatch.setattr(requests, "get", lambda url, timeout: MockResponse())
    
    loaded_img = load_image("https://example.com/fake.png")
    assert isinstance(loaded_img, Image.Image)
    assert loaded_img.size == (8, 8)


# ========================
# Predictor tests
# ========================
class MockPipeline:
    """Simplified mock pipeline for testing."""
    def __init__(self):
        self.images = [Image.new("RGB", (64, 64), color="pink")]
        self.set_adapters_calls = []
        self.call_kwargs = None

    def set_adapters(self, adapters, adapter_weights):
        self.set_adapters_calls.append({
            'adapters': adapters, 
            'weights': adapter_weights
        })

    def __call__(self, **kwargs):
        self.call_kwargs = kwargs
        return self

    def reset_tracking(self):
        self.set_adapters_calls = []
        self.call_kwargs = None


@pytest.fixture
def predictor_with_mock(monkeypatch, sample_image_path):
    """Create predictor with mocked pipeline."""
    monkeypatch.setattr(Predictor, "pipe", None, raising=False)
    
    pred = Predictor()
    mock_pipe = MockPipeline()
    monkeypatch.setattr(pred, "pipe", mock_pipe)
    
    # Mock save_image to return fake path
    monkeypatch.setattr(
        "models.flux_fast_lora_hotswap_img2img.predict.save_image",
        lambda img, output_dir=None: Path("/tmp/fake_image.png"),
    )
    
    # Set up predictor state
    pred.current_adapter = "open-image-preferences"
    pred.lora1_triggers = ["Cinematic", "Anime", "Digital art"]
    pred.lora2_triggers = ["GHIBSKY"]
    
    return pred


# ========================
# Adapter switching tests
# ========================
def test_predict_switches_to_lora1(predictor_with_mock, sample_image_path):
    """Test switching from LORA2 to LORA1."""
    predictor_with_mock.current_adapter = "flux-ghibsky"
    
    path = predictor_with_mock.predict(
        prompt="Test prompt", 
        trigger_word="Anime", 
        init_image=sample_image_path
    )
    
    assert isinstance(path, Path)
    assert predictor_with_mock.current_adapter == "open-image-preferences"
    assert len(predictor_with_mock.pipe.set_adapters_calls) == 1
    assert predictor_with_mock.pipe.set_adapters_calls[0] == {
        'adapters': ["open-image-preferences"],
        'weights': [1.0]
    }


def test_predict_switches_to_lora2(predictor_with_mock, sample_image_path):
    """Test switching from LORA1 to LORA2."""
    path = predictor_with_mock.predict(
        prompt="Ghibli style", 
        trigger_word="GHIBSKY",
        init_image=sample_image_path
    )
    
    assert predictor_with_mock.current_adapter == "flux-ghibsky"
    assert len(predictor_with_mock.pipe.set_adapters_calls) == 1
    assert predictor_with_mock.pipe.set_adapters_calls[0] == {
        'adapters': ["flux-ghibsky"],
        'weights': [0.8]
    }


def test_predict_no_switch_when_already_correct(predictor_with_mock, sample_image_path):
    """Test that redundant adapter switches are avoided."""
    predictor_with_mock.current_adapter = "open-image-preferences"
    
    predictor_with_mock.predict(
        prompt="Test", 
        trigger_word="Cinematic", 
        init_image=sample_image_path
    )
    
    assert predictor_with_mock.current_adapter == "open-image-preferences"
    assert len(predictor_with_mock.pipe.set_adapters_calls) == 0


def test_predict_unknown_trigger_no_switch(predictor_with_mock, sample_image_path):
    """Test that unknown triggers don't cause switches."""
    initial_adapter = predictor_with_mock.current_adapter
    
    predictor_with_mock.predict(
        prompt="Test", 
        trigger_word="UnknownWord", 
        init_image=sample_image_path
    )
    
    assert predictor_with_mock.current_adapter == initial_adapter
    assert len(predictor_with_mock.pipe.set_adapters_calls) == 0


# ========================
# Pipeline parameter tests
# ========================
def test_predict_passes_correct_parameters(predictor_with_mock, sample_image_path):
    """Test that pipeline receives correct generation parameters."""
    predictor_with_mock.predict(
        prompt="Test image", 
        trigger_word="Anime", 
        init_image=sample_image_path
    )

    kwargs = predictor_with_mock.pipe.call_kwargs
    assert kwargs['prompt'] == "Test image"
    assert kwargs['height'] == 1024
    assert kwargs['width'] == 1024
    assert kwargs['guidance_scale'] == 3.5
    assert kwargs['num_inference_steps'] == 28


@pytest.mark.parametrize("trigger", ["Cinematic", "Anime", "Digital art"])
def test_all_lora1_triggers_work(predictor_with_mock, trigger, sample_image_path):
    """Test that all LORA1 triggers switch correctly."""
    predictor_with_mock.current_adapter = "flux-ghibsky"
    predictor_with_mock.pipe.reset_tracking()

    predictor_with_mock.predict(
        prompt=f"Test {trigger}", 
        trigger_word=trigger, 
        init_image=sample_image_path
    )

    assert predictor_with_mock.current_adapter == "open-image-preferences"
    assert len(predictor_with_mock.pipe.set_adapters_calls) == 1
    assert predictor_with_mock.pipe.set_adapters_calls[0]['weights'] == [1.0]


# ========================
# State persistence tests
# ========================
def test_predict_state_persists_across_calls(predictor_with_mock, sample_image_path):
    """Test adapter state persistence across multiple calls."""
    # Switch to LORA2
    predictor_with_mock.predict(
        prompt="First", 
        trigger_word="GHIBSKY", 
        init_image=sample_image_path
    )
    assert predictor_with_mock.current_adapter == "flux-ghibsky"

    # Stay on LORA2
    predictor_with_mock.pipe.reset_tracking()
    predictor_with_mock.predict(
        prompt="Second", 
        trigger_word="GHIBSKY", 
        init_image=sample_image_path
    )
    assert len(predictor_with_mock.pipe.set_adapters_calls) == 0

    # Switch to LORA1
    predictor_with_mock.pipe.reset_tracking()
    predictor_with_mock.predict(
        prompt="Third", 
        trigger_word="Anime", 
        init_image=sample_image_path
    )
    assert predictor_with_mock.current_adapter == "open-image-preferences"
    assert len(predictor_with_mock.pipe.set_adapters_calls) == 1