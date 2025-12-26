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
# HELPER: wait loop
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
def test_flux_fast_lora_hotswap_img2img_container_builds():
    result = subprocess.run(
        ["cog", "build", "-t", "flux-fast-lora-hotswap-img2img-test"],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, f"Cog build failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"


# ------------------------------------------------------
# TEST 2 — Serve boots
# ------------------------------------------------------


@pytest.mark.deployment
def test_flux_fast_lora_hotswap_img2img_server_boots():
    proc = subprocess.Popen(
        ["cog", "serve"],
        capture_output=True,
        check=False,
        text=True,
    )
    try:
        assert wait_for_server(), "Server did not become ready"
    finally:
        proc.terminate()
        proc.wait(timeout=10)


# ------------------------------------------------------
# TEST 3 — 422 for missing schema fields
# ------------------------------------------------------


@pytest.mark.deployment
def test_flux_fast_lora_hotswap_img2img_missing_fields():
    proc = subprocess.Popen(["cog", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    try:
        assert wait_for_server(), "Server did not become ready"

        resp = requests.post(
            "http://localhost:5000/predictions",
            json={},  # no prompt
            timeout=5,
        )

        assert resp.status_code == 422, f"Expected 422 but got {resp.status_code}"

    finally:
        proc.terminate()
        proc.wait(timeout=10)


# ------------------------------------------------------
# TEST 4 — Full inference
# ------------------------------------------------------


@pytest.mark.deployment
def test_flux_fast_lora_hotswap_img2img_full_prediction():
    proc = subprocess.Popen(["cog", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    try:
        assert wait_for_server(), "Server did not become ready"

        payload = {
            "seed": 45,
            "prompt": "A whimsical storybook illustration of a wise owl perched on a branch in an enchanted forest.",
            "strength": 0.63,
            "init_image": "https://images.pexels.com/photos/33649783/pexels-photo-33649783.jpeg",
            "trigger_word": "Painting",
            "guidance_scale": 6.7,
            "num_inference_steps": 28,
        }

        resp = requests.post(
            "http://localhost:5000/predictions",
            json=payload,
            timeout=60,
        )

        assert resp.status_code == 200, f"Prediction failed: {resp.text}"

        data = resp.json()

        assert "output" in data, "Missing output field"

        out_path = data["output"]
        out_file = Path(out_path)

        assert out_file.exists(), f"Output file missing: {out_file}"
        assert out_file.stat().st_size > 0, "Output image file is empty"

    finally:
        proc.terminate()
        proc.wait(timeout=10)
