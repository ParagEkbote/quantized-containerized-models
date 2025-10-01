from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

import pytest
from models.flux_fast_lora_hotswap_img2img.predict import (
    load_image,
    login_with_env_token,
    save_image,
)
from pytest import raises


# ------------------------
# Test login_with_env_token
# ------------------------
def test_login_with_valid_token(monkeypatch):
    # Mock environment variable
    monkeypatch.setenv("HF_TOKEN", "dummy_token")

    called = {}

    # Mock huggingface login
    def mock_login(token):
        called["token"] = token

    monkeypatch.setattr("models.flux_fast_lora_hotswap_img2img.predict.login", mock_login)

    login_with_env_token()

    assert called["token"] == "dummy_token"


def test_login_with_missing_token(monkeypatch):
    # Ensure HF_TOKEN is not set
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with raises(ValueError, match="HF_TOKEN not found"):
        login_with_env_token()


# ------------------------
# Test save_image
# ------------------------
def test_save_image(tmp_path):
    img = Image.new("RGB", (10, 10), color="red")

    # Use pytest tmp_path for temporary directory
    output_path = save_image(img, output_dir=tmp_path)

    # Check that the file exists
    assert output_path.exists()
    assert output_path.suffix == ".png"

    # Verify that the image is correctly saved
    loaded_img = Image.open(output_path)
    assert loaded_img.size == (10, 10)


# ------------------------
# Test load_image from local file
# ------------------------
def test_load_image_from_file(tmp_path):
    # Create a dummy PNG file in tmp_path
    file_path = tmp_path / "test.png"
    img = Image.new("RGB", (5, 5), color="blue")
    img.save(file_path)

    loaded_img = load_image(str(file_path))
    assert isinstance(loaded_img, Image.Image)
    assert loaded_img.size == (5, 5)


# ------------------------
# Test load_image from URL
# ------------------------
def test_load_image_from_url(monkeypatch):
    # Create an in-memory image
    dummy_img = Image.new("RGB", (8, 8), color="green")
    buf = BytesIO()
    dummy_img.save(buf, format="PNG")
    buf.seek(0)

    # Define a dummy response class
    class DummyResponse:
        def __init__(self, content):
            self.content = content

        def raise_for_status(self):
            pass  # do nothing

    # Define the mock get function
    def mock_get(url, timeout):
        return DummyResponse(buf.getvalue())

    # Patch requests.get
    monkeypatch.setattr(requests, "get", mock_get)

    # Call load_image with a fake URL
    loaded_img = load_image("https://example.com/fake.png")

    # Assertions
    assert isinstance(loaded_img, Image.Image)
    assert loaded_img.size == (8, 8)
