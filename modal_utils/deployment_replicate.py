import modal
import subprocess
import argparse

app = modal.App("replicate-cd")

@app.function(
    gpu=modal.gpu.A100(),  # or L40S
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("replicate-token")],
)
def deploy(model_name: str):
    model_dir = f"src/models/{model_name}"

    print(f"🚀 Deploying {model_name}")

    # Cog handles image via cog.yaml
    subprocess.run(
        ["cog", "build"],
        cwd=model_dir,
        check=True,
    )

    subprocess.run(
        ["cog", "push", f"replicate/ParagEkbote/{model_name}"],
        cwd=model_dir,
        check=True,
    )

    print("✅ Deployment complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    args = parser.parse_args()

    deploy.remote(args.model_name)
