import pytest
from models.flux_fast_lora_hotswap_img2img.predict import login_with_env_token


def test_login_with_valid_token(monkeypatch):
    # Mock environment variable
    monkeypatch.setenv("HF_TOKEN", "dummy_token")

    called = {}

    # Mock huggingface login
    def mock_login(token):
        called["token"] = token

    monkeypatch.setattr("models.flux-fast-lora-hotswap-img2img.predict.login", mock_login)

    login_with_env_token()

    assert called["token"] == "dummy_token"


def test_login_with_missing_token(monkeypatch):
    # Ensure HF_TOKEN is not set
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with pytest.raises(ValueError, match="HF_TOKEN not found"):
        login_with_env_token()
