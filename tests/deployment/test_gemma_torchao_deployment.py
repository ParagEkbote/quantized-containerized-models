import os
import subprocess
import time
from pathlib import Path

import pytest
import requests

# ------------------------------------------------------
# Skip RULES (module-level)
# ------------------------------------------------------

# Skip if Modal credentials missing
if not (os.getenv("MODAL_TOKEN_ID") and os.getenv("MODAL_TOKEN_SECRET")):
    pytest.skip(
        "Modal credentials missing → skipping deployment tests",
        allow_module_level=True,
    )

# Skip if no GPU available (Gemma TorchAO expects CUDA in container)
if not (os.environ.get("NVIDIA_VISIBLE_DEVICES") or os.path.isdir("/proc/driver/nvidia")):
    pytest.skip(
        "GPU unavailable → skipping gemma-torchao deployment tests",
        allow_module_level=True,
    )


# ------------------------------------------------------
# HELPER: Wait for server to come online
# ------------------------------------------------------


def wait_for_server(url: str = "http://localhost:5000/ping", timeout: int = 60) -> bool:
    """Poll /ping until server responds 200 or timeout expires."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(url, timeout=2)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


# ------------------------------------------------------
# TEST 1 — Container builds successfully
# ------------------------------------------------------


@pytest.mark.deployment
def test_gemma_torchao_container_builds() -> None:
    """Ensure `cog build` for gemma-torchao succeeds without errors."""
    result = subprocess.run(
        ["cog", "build", "-t", "gemma-torchao-test"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 0, f"Cog build failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"


# ------------------------------------------------------
# TEST 2 — Server boots
# ------------------------------------------------------


@pytest.mark.deployment
def test_gemma_torchao_server_boots() -> None:
    """Ensure `cog serve` boots and responds to /ping."""
    proc = subprocess.Popen(
        ["cog", "serve"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        ok = wait_for_server()
        assert ok, "cog serve did not become ready within timeout"
    finally:
        proc.terminate()
        proc.wait(timeout=10)


# ------------------------------------------------------
# TEST 3 — Missing required fields produce 422
# ------------------------------------------------------


@pytest.mark.deployment
def test_gemma_torchao_missing_fields() -> None:
    """
    POST /predictions without required `prompt` must return HTTP 422.

    This validates that the running container's OpenAPI / schema contract
    matches the predict.py Input definitions.
    """
    proc = subprocess.Popen(
        ["cog", "serve"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        assert wait_for_server(), "Server did not become ready"

        # Missing prompt → 422
        resp = requests.post(
            "http://localhost:5000/predictions",
            json={},
            timeout=10,
        )
        assert resp.status_code == 422, f"Expected 422, got {resp.status_code}, body={resp.text}"
    finally:
        proc.terminate()
        proc.wait(timeout=10)


# ------------------------------------------------------
# TEST 4 — Full inference
# ------------------------------------------------------


@pytest.mark.deployment
def test_gemma_torchao_full_prediction(tmp_path: Path) -> None:
    """
    Send a real request through the container and ensure:

    - server responds with 200
    - JSON contains 'output'
    - for this deployment, output is a string (text response)
    """
    proc = subprocess.Popen(
        ["cog", "serve"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        assert wait_for_server(), "Server did not become ready"

        # Minimal but realistic payload matching gemma-torchao schema
        payload = {
            "prompt": "Explain the difference between supervised and unsupervised learning in one paragraph.",
            "image_url": None,  # optional, exercising pure text path
            "max_new_tokens": 128,
            "temperature": 0.7,
            "top_p": 0.9,
            "seed": 42,
            "use_quantization": "true",
            "use_sparsity": "false",
            "sparsity_type": "magnitude",
            "sparsity_ratio": 0.3,
        }

        resp = requests.post(
            "http://localhost:5000/predictions",
            json=payload,
            timeout=120,
        )

        assert resp.status_code == 200, f"Prediction failed: {resp.status_code} {resp.text}"

        data = resp.json()
        assert "output" in data, f"Missing 'output' key in response: {data}"

        output_text = data["output"]
        assert isinstance(output_text, str), f"Expected text output, got {type(output_text)}"
        assert len(output_text.strip()) > 0, "Model returned empty output"

        # Optionally persist output to disk under tmp_path, just to verify writability
        out_file = tmp_path / "gemma_torchao_output.txt"
        out_file.write_text(output_text, encoding="utf-8")
        assert out_file.exists()
    finally:
        proc.terminate()
        proc.wait(timeout=10)
