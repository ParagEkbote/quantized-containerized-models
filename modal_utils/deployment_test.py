# dev/modal/deployment_tests.py
import modal

app = modal.App("deployment-tests")

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
def run_deployment_tests():
    import subprocess

    subprocess.run(
        ["pytest", "-m", "deployment", "tests/deployment"],
        check=True,
    )
