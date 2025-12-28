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

# Skip if no GPU available
if not (os.environ.get("NVIDIA_VISIBLE_DEVICES") or os.path.isdir("/proc/driver/nvidia")):
    pytest.skip(
        "GPU unavailable → skipping deployment tests",
        allow_module_level=True,
    )


# ------------------------------------------------------
# HELPER: Wait for server to come online
# ------------------------------------------------------


def wait_for_server(url="http://localhost:5000/ping", timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


# ------------------------------------------------------
# TEST 1 — Container builds successfully
# ------------------------------------------------------


@pytest.mark.deployment
def test_smollm3_container_builds():
    """Ensure `cog build` succeeds without errors."""

    result = subprocess.run(
        ["cog", "build", "-t", "smollm3-test"],
        stdout=subprocess.PIPE,  # ✅ Add these instead
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 0, "Cog build failed.\nSTDOUT:\n" + result.stdout + "\nSTDERR:\n" + result.stderr


# ------------------------------------------------------
# TEST 2 — Server boots
# ------------------------------------------------------


@pytest.mark.deployment
def test_smollm3_server_boots():
    """Ensure `cog serve` boots and responds to /ping."""

    proc = subprocess.Popen(
        ["cog", "serve"],
        stdout=subprocess.PIPE,  # ✅ Add these instead
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
def test_smollm3_missing_fields():
    """POST /predictions without prompt must return HTTP 422."""

    # Boot server
    proc = subprocess.Popen(["cog", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    try:
        assert wait_for_server(), "Server did not become ready"

        # Missing prompt → 422
        response = requests.post(
            "http://localhost:5000/predictions",
            json={},
            timeout=5,
        )

        assert response.status_code == 422, f"Expected 422, got {response.status_code}"

    finally:
        proc.terminate()
        proc.wait(timeout=10)


# ------------------------------------------------------
# TEST 4 — Full inference
# ------------------------------------------------------


@pytest.mark.deployment
def test_smollm3_full_prediction():
    """
    Send a real request through the container and ensure:
    - server responds
    - output_path is returned
    - file exists on disk
    """

    proc = subprocess.Popen(["cog", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    try:
        assert wait_for_server(), "Server did not become ready"

        payload = {
            "prompt": "Explain reinforcement learning briefly.",
            "max_new_tokens": 64,
            "mode": "no_think",
        }

        response = requests.post(
            "http://localhost:5000/predictions",
            json=payload,
            timeout=60,
        )

        assert response.status_code == 200, f"Prediction failed: {response.text}"

        data = response.json()
        assert "output" in data, "Missing output in response"

        output_path = data["output"]
        assert isinstance(output_path, str)
        assert Path(output_path).exists(), f"Output file not found: {output_path}"

    finally:
        proc.terminate()
        proc.wait(timeout=10)
