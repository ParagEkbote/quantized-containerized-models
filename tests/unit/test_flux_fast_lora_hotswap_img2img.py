from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, call, patch

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
# HF_TOKEN login tests
# ========================
def test_login_with_valid_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "dummy_token")
    called = {}

    def mock_login(token):
        called["token"] = token

    monkeypatch.setattr("models.flux_fast_lora_hotswap_img2img.predict.login", mock_login)
    login_with_env_token()
    assert called["token"] == "dummy_token", "Login should be called with the correct token"


def test_login_with_missing_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with raises(ValueError, match="HF_TOKEN not found"):
        login_with_env_token()


def test_login_with_custom_env_var(monkeypatch):
    monkeypatch.setenv("CUSTOM_TOKEN", "custom_value")
    called = {}

    def mock_login(token):
        called["token"] = token

    monkeypatch.setattr("models.flux_fast_lora_hotswap_img2img.predict.login", mock_login)
    login_with_env_token(env_var="CUSTOM_TOKEN")
    assert called["token"] == "custom_value", "Should use custom environment variable"


# ========================
# save_image tests
# ========================
def test_save_image(tmp_path):
    img = Image.new("RGB", (10, 10), color="red")
    output_path = save_image(img, output_dir=tmp_path)
    assert output_path.exists(), "Output file should exist"
    assert output_path.suffix == ".png", "Output file should be PNG format"
    loaded_img = Image.open(output_path)
    assert loaded_img.size == (10, 10), "Loaded image should have correct dimensions"


def test_save_image_creates_directory(tmp_path):
    img = Image.new("RGB", (10, 10), color="blue")
    nested_dir = tmp_path / "nested" / "dir"
    output_path = save_image(img, output_dir=nested_dir)
    assert nested_dir.exists(), "Nested directory should be created"
    assert output_path.exists(), "Image should be saved in nested directory"


def test_save_image_unique_filenames(tmp_path):
    img = Image.new("RGB", (10, 10), color="green")
    path1 = save_image(img, output_dir=tmp_path)
    path2 = save_image(img, output_dir=tmp_path)
    assert path1 != path2, "Each saved image should have a unique filename"


# ========================
# load_image tests
# ========================
def test_load_image_from_file(tmp_path):
    file_path = tmp_path / "test.png"
    img = Image.new("RGB", (5, 5), color="blue")
    img.save(file_path)
    loaded_img = load_image(str(file_path))
    assert isinstance(loaded_img, Image.Image), "Should return PIL Image"
    assert loaded_img.size == (5, 5), "Loaded image should have correct dimensions"


def test_load_image_from_url(monkeypatch):
    dummy_img = Image.new("RGB", (8, 8), color="green")
    buf = BytesIO()
    dummy_img.save(buf, format="PNG")
    buf.seek(0)

    class DummyResponse:
        def __init__(self, content):
            self.content = content

        def raise_for_status(self):
            pass

    def mock_get(url, timeout):
        return DummyResponse(buf.getvalue())

    monkeypatch.setattr(requests, "get", mock_get)
    loaded_img = load_image("https://example.com/fake.png")
    assert isinstance(loaded_img, Image.Image), "Should return PIL Image from URL"
    assert loaded_img.size == (8, 8), "Loaded image should have correct dimensions"


# ========================
# Predictor tests
# ========================
class DummyPipeline:
    def __init__(self):
        self.text_encoder = lambda x: x
        self.text_encoder_2 = lambda x: x
        self.vae = lambda x: x
        self.images = [Image.new("RGB", (64, 64), color="pink")]
        self.set_adapters_calls = []
        self.call_kwargs = None

    def set_adapters(self, adapters, adapter_weights):
        self.set_adapters_calls.append({'adapters': adapters, 'weights': adapter_weights})
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


@pytest.fixture
def predictor(monkeypatch):
    pred = Predictor()
    dummy_pipe = DummyPipeline()
    monkeypatch.setattr(pred, "pipe", dummy_pipe)

    # Track torch.compile calls
    compile_calls = []

    def mock_compile(f, **kwargs):
        compile_calls.append({'func': f, 'kwargs': kwargs})
        return f

    monkeypatch.setattr(torch, "compile", mock_compile)
    pred._compile_calls = compile_calls

    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 1e9)  # 1 GB
    monkeypatch.setattr(
        "models.flux_fast_lora_hotswap_img2img.predict.save_image",
        lambda img, output_dir=None: Path("/tmp/fake_image.png"),
    )

    pred.current_adapter = "open-image-preferences"
    pred.lora1_triggers = [
        "Cinematic",
        "Photographic",
        "Anime",
        "Manga",
        "Digital art",
        "Pixel art",
        "Fantasy art",
        "Neonpunk",
        "3D Model",
        "Painting",
        "Animation",
        "Illustration",
    ]
    pred.lora2_triggers = ["GHIBSKY"]
    return pred


def test_predict_lora1_trigger_switches_from_lora2(predictor, capsys):
    """Test switching from flux-ghibsky to open-image-preferences"""
    # Start with a different adapter
    predictor.current_adapter = "flux-ghibsky"

    path = predictor.predict(prompt="A test prompt", trigger_word="Anime")

    assert isinstance(path, Path), "Should return a Path object"
    assert path.name == "fake_image.png", "Should return correct image path"
    assert (
        predictor.current_adapter == "open-image-preferences"
    ), "Adapter should switch to open-image-preferences"

    # Verify set_adapters was called with correct parameters
    assert len(predictor.pipe.set_adapters_calls) == 1, "set_adapters should be called once"
    assert predictor.pipe.set_adapters_calls[0]['adapters'] == [
        "open-image-preferences"
    ], "Should set correct adapter"
    assert predictor.pipe.set_adapters_calls[0]['weights'] == [
        1.0
    ], "Should use weight 1.0 for LORA1"

    captured = capsys.readouterr()
    assert "[Prompt]: A test prompt | [Trigger Word]: Anime" in captured.out
    assert "Used memory: 1.00 GB" in captured.out


def test_predict_lora2_trigger_switch(predictor, capsys):
    """Test switching from open-image-preferences to flux-ghibsky"""
    path = predictor.predict(prompt="Another prompt", trigger_word="GHIBSKY")

    assert isinstance(path, Path), "Should return a Path object"
    assert path.name == "fake_image.png", "Should return correct image path"
    assert predictor.current_adapter == "flux-ghibsky", "Adapter should switch to flux-ghibsky"

    # Verify set_adapters was called with correct parameters
    assert len(predictor.pipe.set_adapters_calls) == 1, "set_adapters should be called once"
    assert predictor.pipe.set_adapters_calls[0]['adapters'] == [
        "flux-ghibsky"
    ], "Should set correct adapter"
    assert predictor.pipe.set_adapters_calls[0]['weights'] == [
        0.8
    ], "Should use weight 0.8 for LORA2"

    captured = capsys.readouterr()
    assert "[Prompt]: Another prompt | [Trigger Word]: GHIBSKY" in captured.out
    assert "Used memory: 1.00 GB" in captured.out


def test_predict_no_adapter_switch_when_same_adapter(predictor, capsys):
    """Test that adapter doesn't switch when already on correct adapter"""
    predictor.current_adapter = "open-image-preferences"

    path = predictor.predict(prompt="Test prompt", trigger_word="Cinematic")

    assert isinstance(path, Path), "Should return a Path object"
    assert predictor.current_adapter == "open-image-preferences", "Adapter should remain unchanged"

    # Verify set_adapters was NOT called (no redundant switching)
    assert (
        len(predictor.pipe.set_adapters_calls) == 0
    ), "set_adapters should NOT be called when adapter is already correct"

    captured = capsys.readouterr()
    assert "[Prompt]: Test prompt | [Trigger Word]: Cinematic" in captured.out


def test_predict_no_adapter_switch_for_lora2_when_already_set(predictor):
    """Test that flux-ghibsky doesn't switch when already active"""
    predictor.current_adapter = "flux-ghibsky"

    path = predictor.predict(prompt="Ghibli style", trigger_word="GHIBSKY")

    assert predictor.current_adapter == "flux-ghibsky", "Adapter should remain flux-ghibsky"
    assert (
        len(predictor.pipe.set_adapters_calls) == 0
    ), "set_adapters should NOT be called when already on flux-ghibsky"


def test_predict_unknown_trigger_word_no_switch(predictor):
    """Test that unknown trigger words don't cause adapter switching"""
    predictor.current_adapter = "open-image-preferences"

    path = predictor.predict(prompt="Random prompt", trigger_word="UnknownStyle")

    assert (
        predictor.current_adapter == "open-image-preferences"
    ), "Adapter should not change for unknown trigger"
    assert (
        len(predictor.pipe.set_adapters_calls) == 0
    ), "set_adapters should NOT be called for unknown trigger"


def test_predict_pipeline_called_with_correct_kwargs(predictor):
    """Test that the pipeline is called with correct parameters"""
    predictor.predict(prompt="Test image", trigger_word="Anime")

    assert predictor.pipe.call_kwargs is not None, "Pipeline should be called"
    kwargs = predictor.pipe.call_kwargs

    assert kwargs['prompt'] == "Test image", "Prompt should be passed correctly"
    assert kwargs['height'] == 1024, "Height should be 1024"
    assert kwargs['width'] == 1024, "Width should be 1024"
    assert kwargs['guidance_scale'] == 3.5, "Guidance scale should be 3.5"
    assert kwargs['num_inference_steps'] == 28, "Num inference steps should be 28"
    assert kwargs['max_sequence_length'] == 512, "Max sequence length should be 512"


def test_predict_torch_compile_called(predictor):
    """Test that torch.compile is called for all components"""
    predictor.predict(prompt="Test", trigger_word="Anime")

    assert len(predictor._compile_calls) == 3, "torch.compile should be called 3 times"

    # Verify compile parameters
    for call_info in predictor._compile_calls:
        assert call_info['kwargs']['fullgraph'] == False, "fullgraph should be False"
        assert call_info['kwargs']['mode'] == "reduce-overhead", "mode should be reduce-overhead"


@patch('torch.no_grad')
def test_predict_uses_torch_no_grad_context(mock_no_grad, predictor):
    """Test that torch.no_grad context manager is used"""
    mock_context = MagicMock()
    mock_no_grad.return_value = mock_context

    predictor.predict(prompt="Test", trigger_word="Anime")

    mock_no_grad.assert_called_once()
    mock_context.__enter__.assert_called_once()
    mock_context.__exit__.assert_called_once()


@pytest.mark.parametrize(
    "trigger",
    [
        "Cinematic",
        "Photographic",
        "Anime",
        "Manga",
        "Digital art",
        "Pixel art",
        "Fantasy art",
        "Neonpunk",
        "3D Model",
        "Painting",
        "Animation",
        "Illustration",
    ],
)
def test_all_lora1_triggers_switch_correctly(predictor, trigger):
    """Test that all LORA1 triggers switch to open-image-preferences"""
    predictor.current_adapter = "flux-ghibsky"
    predictor.pipe.reset_tracking()

    predictor.predict(prompt=f"Test {trigger}", trigger_word=trigger)

    assert (
        predictor.current_adapter == "open-image-preferences"
    ), f"Trigger '{trigger}' should switch to open-image-preferences"
    assert (
        len(predictor.pipe.set_adapters_calls) == 1
    ), f"Trigger '{trigger}' should call set_adapters"
    assert predictor.pipe.set_adapters_calls[0]['weights'] == [
        1.0
    ], f"Trigger '{trigger}' should use weight 1.0"


def test_predict_with_empty_prompt(predictor):
    """Test prediction with empty prompt"""
    path = predictor.predict(prompt="", trigger_word="Anime")

    assert isinstance(path, Path), "Should handle empty prompt"
    assert predictor.pipe.call_kwargs['prompt'] == "", "Empty prompt should be passed to pipeline"


def test_predict_with_long_prompt(predictor):
    """Test prediction with very long prompt"""
    long_prompt = "A " * 1000  # Very long prompt
    path = predictor.predict(prompt=long_prompt, trigger_word="Cinematic")

    assert isinstance(path, Path), "Should handle long prompt"
    assert (
        predictor.pipe.call_kwargs['prompt'] == long_prompt
    ), "Long prompt should be passed to pipeline"


def test_predict_state_persistence_across_calls(predictor):
    """Test that adapter state persists correctly across multiple calls"""
    # First call - switch to flux-ghibsky
    predictor.predict(prompt="First", trigger_word="GHIBSKY")
    assert predictor.current_adapter == "flux-ghibsky"

    # Second call - stay on flux-ghibsky
    predictor.pipe.reset_tracking()
    predictor.predict(prompt="Second", trigger_word="GHIBSKY")
    assert predictor.current_adapter == "flux-ghibsky"
    assert (
        len(predictor.pipe.set_adapters_calls) == 0
    ), "Should not switch when already on correct adapter"

    # Third call - switch to open-image-preferences
    predictor.pipe.reset_tracking()
    predictor.predict(prompt="Third", trigger_word="Anime")
    assert predictor.current_adapter == "open-image-preferences"
    assert len(predictor.pipe.set_adapters_calls) == 1, "Should switch adapter"


def test_predict_case_sensitive_trigger_words(predictor):
    """Test that trigger words are case-sensitive"""
    predictor.current_adapter = "open-image-preferences"

    # Lowercase version should not trigger switch
    path = predictor.predict(prompt="Test", trigger_word="anime")
    assert (
        predictor.current_adapter == "open-image-preferences"
    ), "Lowercase 'anime' should not trigger switch"
    assert (
        len(predictor.pipe.set_adapters_calls) == 0
    ), "Case mismatch should not trigger adapter switch"


def test_predict_memory_reporting(predictor, capsys):
    """Test that memory usage is correctly reported"""
    predictor.predict(prompt="Test", trigger_word="Anime")

    captured = capsys.readouterr()
    assert "Used memory: 1.00 GB" in captured.out, "Memory usage should be reported"
