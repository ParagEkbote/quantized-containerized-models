import os
import subprocess
import time
from pathlib import Path

import pytest
import requests

# ------------------------------------------------------
# Skip RULES (module-level)
# ------------------------------------------------------

if not (os.getenv("MODAL_TOKEN_ID") and os.getenv("MODAL_TOKEN_SECRET")):
    pytest.skip(
        "Modal credentials missing → skipping deployment tests",
        allow_module_level=True,
    )

if not (os.environ.get("NVIDIA_VISIBLE_DEVICES") or os.path.isdir("/proc/driver/nvidia")):
    pytest.skip(
        "GPU unavailable → skipping deployment tests",
        allow_module_level=True,
    )


# ------------------------------------------------------
# Wait until server is ready
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
# TEST 1 — Build
# ------------------------------------------------------


@pytest.mark.deployment
def test_flux_fast_lora_hotswap_container_builds():
    result = subprocess.run(
        ["cog", "build", "-t", "flux-fast-lora-hotswap-test"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 0, f"Cog build failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"


# ------------------------------------------------------
# TEST 2 — Serve boots
# ------------------------------------------------------


@pytest.mark.deployment
def test_flux_fast_lora_hotswap_server_boots():
    proc = subprocess.Popen(
        ["cog", "serve"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        assert wait_for_server(), "cog serve did not become ready within timeout"
    finally:
        proc.terminate()
        proc.wait(timeout=10)


# ------------------------------------------------------
# TEST 3 — 422 schema error check
# ------------------------------------------------------


@pytest.mark.deployment
def test_flux_fast_lora_hotswap_missing_fields():
    proc = subprocess.Popen(["cog", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    try:
        assert wait_for_server(), "Server did not become ready"

        response = requests.post(
            "http://localhost:5000/predictions",
            json={},  # missing prompt/trigger_word
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
def test_flux_fast_lora_hotswap_full_prediction():
    proc = subprocess.Popen(["cog", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    try:
        assert wait_for_server(), "Server did not become ready"

        payload = {
            "prompt": "a leopard drinking milkshake",
            "trigger_word": "GHIBSKY",
        }

        response = requests.post(
            "http://localhost:5000/predictions",
            json=payload,
            timeout=60,
        )

        assert response.status_code == 200, f"Prediction failed: {response.text}"

        data = response.json()
        assert "output" in data, "Missing output field"

        out_file = Path(data["output"])
        assert out_file.exists(), f"Output file missing: {out_file}"
        assert out_file.stat().st_size > 0, "Generated output image file is empty"

    finally:
        proc.terminate()
        proc.wait(timeout=10)
