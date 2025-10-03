from io import BytesIO
from pathlib import Path

import requests
import torch
from PIL import Image

import pytest
from models.flux_fast_lora_hotswap_img2img.predict import (
    Predictor,
    load_image,
    login_with_env_token,
    save_image,
)
from pytest import raises


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
    assert called["token"] == "dummy_token"


def test_login_with_missing_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with raises(ValueError, match="HF_TOKEN not found"):
        login_with_env_token()


# ========================
# save_image tests
# ========================
def test_save_image(tmp_path):
    img = Image.new("RGB", (10, 10), color="red")
    output_path = save_image(img, output_dir=tmp_path)
    assert output_path.exists()
    assert output_path.suffix == ".png"
    loaded_img = Image.open(output_path)
    assert loaded_img.size == (10, 10)


# ========================
# load_image tests
# ========================
def test_load_image_from_file(tmp_path):
    file_path = tmp_path / "test.png"
    img = Image.new("RGB", (5, 5), color="blue")
    img.save(file_path)
    loaded_img = load_image(str(file_path))
    assert isinstance(loaded_img, Image.Image)
    assert loaded_img.size == (5, 5)


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
    assert isinstance(loaded_img, Image.Image)
    assert loaded_img.size == (8, 8)


# ========================
# Predictor tests
# ========================
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


@pytest.fixture
def predictor(monkeypatch):
    pred = Predictor()
    monkeypatch.setattr(pred, "pipe", DummyPipeline())
    monkeypatch.setattr(torch, "compile", lambda f, **kwargs: f)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 1e9)  # 1 GB
    monkeypatch.setattr(
        "models.flux_fast_lora_hotswap_img2img.predict.save_image",
        lambda img: Path("/tmp/fake_image.png"),
    )
    pred.current_adapter = "open-image-preferences"
    pred.lora1_triggers = ["Cinematic", "Anime"]
    pred.lora2_triggers = ["GHIBSKY"]
    return pred


def test_predict_lora1_trigger(predictor, capsys):
    path = predictor.predict(prompt="A test prompt", trigger_word="Anime")
    assert isinstance(path, Path)
    assert path.name == "fake_image.png"
    assert predictor.current_adapter == "open-image-preferences"
    captured = capsys.readouterr()
    assert "[Prompt]: A test prompt | [Trigger Word]: Anime" in captured.out
    assert "Used memory: 1.00 GB" in captured.out


def test_predict_lora2_trigger_switch(predictor, capsys):
    path = predictor.predict(prompt="Another prompt", trigger_word="GHIBSKY")
    assert isinstance(path, Path)
    assert path.name == "fake_image.png"
    assert predictor.current_adapter == "flux-ghibsky"
    captured = capsys.readouterr()
    assert "[Prompt]: Another prompt | [Trigger Word]: GHIBSKY" in captured.out
    assert "Used memory: 1.00 GB" in captured.out


def test_predict_no_trigger_change(predictor, capsys):
    path = predictor.predict(prompt="Random prompt", trigger_word="None")
    assert isinstance(path, Path)
    assert predictor.current_adapter == "open-image-preferences"
    captured = capsys.readouterr()
    assert "[Prompt]: Random prompt | [Trigger Word]: None" in captured.out
    assert "Used memory: 1.00 GB" in captured.out
