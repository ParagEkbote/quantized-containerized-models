import subprocess
import modal

app = modal.App("replicate-deployment")


@app.function(
    gpu="A100-40GB",  # or "L40S"
    timeout=60 * 60,
    secrets=[
        # Required for `cog push`
        modal.Secret.from_name("replicate-token"),
        # Optional: only if Modal auth is needed elsewhere
        modal.Secret.from_name("modal-token"),
    ],
)
def deploy(model_name: str):
    """
    Build and push a model to Replicate using Cog.

    - Uses cog.yaml to define image + runtime
    - Does NOT run tests
    - Does NOT modify model selection logic
    """

    model_dir = f"src/models/{model_name}"

    print(f"🚀 Deploying model: {model_name}")
    print(f"📁 Model directory: {model_dir}")

    # --------------------------------------------------
    # Build image (handled fully by cog.yaml)
    # --------------------------------------------------
    subprocess.run(
        ["cog", "build"],
        cwd=model_dir,
        check=True,
    )

    # --------------------------------------------------
    # Push to Replicate
    # --------------------------------------------------
    subprocess.run(
        [
            "cog",
            "push",
            f"replicate/ParagEkbote/{model_name}",
        ],
        cwd=model_dir,
        check=True,
    )

    print("✅ Deployment to Replicate complete")


# --------------------------------------------------
# Local entrypoint (required for modal run)
# --------------------------------------------------
@app.local_entrypoint()
def main(model_name: str):
    deploy.remote(model_name)
