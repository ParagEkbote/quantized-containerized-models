import subprocess
import modal

app = modal.App("deployment-tests")


@app.function(
    gpu="A100-40GB",  # or "L40S"
    timeout=60 * 60,
    # IMPORTANT: no replicate-token secret needed
    secrets=[
        modal.Secret.from_name("modal-token"),  # only for MODAL_TOKEN_ID/SECRET
    ],
)
def run_deployment_tests(model_name: str):
    """
    Run ONLY pytest deployment tests for a given model.

    - Tests are selected from tests/deployment/
    - Uses pytest -m deployment
    - Filters tests by model_name using -k
    - Does NOT build or push to Replicate
    """

    print(f"🧪 Running deployment tests for model: {model_name}")

    cmd = [
        "pytest",
        "-m",
        "deployment",
        "tests/deployment",
        "-k",
        model_name,
        "-vv",
    ]

    subprocess.run(cmd, check=True)

    print("✅ Deployment tests passed")


# --------------------------------------------------
# Local entrypoint (required for modal run)
# --------------------------------------------------
@app.local_entrypoint()
def main(model_name: str):
    run_deployment_tests.remote(model_name)

