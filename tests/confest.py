from pathlib import Path

import pytest


@pytest.fixture
def tmp_data_dir(tmp_path) -> Path:
    """
    Provides a temporary directory for storing test data/images.
    """
    return tmp_path


# ------------------------
# Example: monkeypatch HF_TOKEN for multiple tests
# ------------------------
@pytest.fixture(autouse=True)
def ensure_hf_token(monkeypatch):
    """
    Automatically sets a dummy HF_TOKEN if not set.
    Useful for login_with_env_token tests.
    """
    if not (token := monkeypatch.getenv("HF_TOKEN")):
        monkeypatch.setenv("HF_TOKEN", "dummy_token")
