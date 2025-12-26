import modal
import subprocess
import sys
import json
from pathlib import Path

app = modal.App("replicate-deployment")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "curl")
    .pip_install(
        "pytest",
        "requests",
        "cog",
        "torch",
        "transformers",
    )
)

@app.function(
    gpu="A100",
    timeout=60 * 60,
    image=image,
)
def deploy_and_test(model_name: str):
    """
    1. cog build
    2. pytest -m deployment
    3. cog push
    """

    model_dir = Path(f"src/models/{model_name}")

    def run(cmd, step):
        print(f"\n===== {step} =====")
        proc = subprocess.run(
            cmd,
            cwd=model_dir,
            capture_output=True,
            check=False,
            text=True,
        )

        print(proc.stdout)
        if proc.returncode != 0:
            print(proc.stderr, file=sys.stderr)
            raise RuntimeError(f"{step} failed")

    try:
        # -------------------------
        # BUILD
        # -------------------------
        run(["cog", "build"], "COG BUILD")

        # -------------------------
        # DEPLOYMENT TESTS
        # -------------------------
        run(
            ["pytest", "-m", "deployment", "tests/deployment"],
            "DEPLOYMENT TESTS",
        )

        # -------------------------
        # PUSH
        # -------------------------
        run(
            ["cog", "push", f"replicate/ParagEkbote/{model_name}"],
            "COG PUSH",
        )

        # -------------------------
        # SUCCESS SIGNAL
        # -------------------------
        print("\n✅ DEPLOYMENT SUCCESS")
        print(json.dumps({
            "status": "success",
            "model": model_name,
            "owner": "ParagEkbote",
            "registry": "replicate",
        }, indent=2))

    except Exception as e:
        print("\n❌ DEPLOYMENT FAILED")
        print(str(e), file=sys.stderr)
        sys.exit(1)
