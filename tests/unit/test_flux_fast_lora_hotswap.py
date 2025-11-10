# tests/test_flux_predictor.py
import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch
from io import BytesIO
from PIL import Image

from models.flux_fast_lora_hotswap_img2img.predict import (
    login_with_env_token,
    save_image,
    load_image,
    Predictor,
)
import pytest

@pytest.fixture(autouse=True)
def _patch_cog_fieldinfo(monkeypatch):
    import models.flux_fast_lora_hotswap_img2img.predict as mod

    Predictor = mod.Predictor

    # Map of fields to proper test values
    FIELDS = {
        "width": 1024,
        "height": 1024,
        "steps": 28,
        "num_inference_steps": 28,
        "seed": 42,
        "guidance_scale": 7.5,
        "strength": 0.6,
        "max_sequence_length": 512,
    }

    for attr, value in FIELDS.items():
        if hasattr(Predictor, attr):
            monkeypatch.setattr(Predictor, attr, value, raising=False)

# ---------------------------------------------
# login_with_env_token tests
# ---------------------------------------------
def test_login_valid(monkeypatch):
    called = {}

    def fake_login(token):
        called["token"] = token

    monkeypatch.setenv("HF_TOKEN", "abc123")
    monkeypatch.setattr("models.flux_fast_lora_hotswap_img2img.predict.login", fake_login)

    login_with_env_token()
    assert called["token"] == "abc123"


def test_login_missing(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(ValueError):
        login_with_env_token()


# ---------------------------------------------
# save_image tests
# ---------------------------------------------
def test_save_image(tmp_path):
    img = Image.new("RGB", (10, 10), "red")
    p = save_image(img, tmp_path)
    assert p.exists() and p.suffix == ".png"


# ---------------------------------------------
# load_image tests
# ---------------------------------------------
def test_load_image_local(tmp_path):
    img_path = tmp_path / "a.png"
    Image.new("RGB", (5, 5), "blue").save(img_path)
    img = load_image(str(img_path))
    assert img.size == (5, 5)


def test_load_image_url(monkeypatch):
    img = Image.new("RGB", (8, 8), "green")
    buf = BytesIO()
    img.save(buf, "PNG")
    buf.seek(0)

    class DummyResp:
        content = buf.getvalue()
        def raise_for_status(self): pass

    monkeypatch.setattr(
        "models.flux_fast_lora_hotswap_img2img.predict.requests.get",
        lambda *a, **k: DummyResp(),
    )

    out = load_image("https://example.com/test.png")
    assert out.size == (8, 8)


# ---------------------------------------------
# Predictor.setup mock helpers
# ---------------------------------------------
def mock_flux_pipeline():
    pipe = MagicMock()
    pipe.to.return_value = pipe
    pipe.enable_lora_hotswap = MagicMock()
    pipe.load_lora_weights = MagicMock()
    pipe.set_adapters = MagicMock()

    # components to compile
    pipe.text_encoder = MagicMock()
    pipe.text_encoder_2 = MagicMock()
    pipe.vae = MagicMock()

    # return an object with `.images`
    fake_img = Image.new("RGB", (32, 32), "black")
    pipe.return_value = MagicMock(images=[fake_img])

    return pipe


# ---------------------------------------------
# Predictor.setup test
# ---------------------------------------------
@patch("torch.cuda.is_available", lambda: False)
@patch("models.flux_fast_lora_hotswap_img2img.predict.FluxImg2ImgPipeline.from_pretrained")
def test_predictor_setup(mock_from_pretrained, monkeypatch):
    pipe = mock_flux_pipeline()
    mock_from_pretrained.return_value = pipe

    # mock compile to identity
    monkeypatch.setattr("torch.compile", lambda f, **k: f)

    pred = Predictor()
    pred.setup()

    assert pred.pipe is pipe
    assert pred.current_adapter == "open-image-preferences"


# ---------------------------------------------
# Predictor.predict test
# ---------------------------------------------
@patch("torch.cuda.is_available", lambda: False)
@patch("models.flux_fast_lora_hotswap_img2img.predict.FluxImg2ImgPipeline.from_pretrained")
@patch("torch.no_grad")
def test_predict(
    mock_no_grad, mock_from_pretrained, tmp_path, monkeypatch
):
    pipe = mock_flux_pipeline()
    mock_from_pretrained.return_value = pipe

    # mock compile
    monkeypatch.setattr("torch.compile", lambda f, **k: f)

    # patch save_image to tmp_path/result.png
    monkeypatch.setattr(
        "models.flux_fast_lora_hotswap_img2img.predict.save_image",
        lambda img: tmp_path / "result.png",
    )

    # patch load_image to return a dummy unique image
    monkeypatch.setattr(
        "models.flux_fast_lora_hotswap_img2img.predict.load_image",
        lambda p: Image.new("RGB", (32, 32), "yellow"),
    )

    pred = Predictor()
    pred.setup()

    out = pred.predict(
        prompt="hi",
        trigger_word="Anime",
        init_image="x.png",
        strength=0.5,
        guidance_scale=7.5,
        num_inference_steps=8,
        seed=42,
    )

    assert out.name == "result.png"

    # pipeline should have been called
    pipe.assert_called_once()

    # adapter should be set to LORA1
    pipe.set_adapters.assert_called_once_with(
        ["open-image-preferences"], adapter_weights=[1.0]
    )
