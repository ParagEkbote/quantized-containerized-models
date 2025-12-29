import os
import sys
import httpx

REPLICATE_API = "https://api.replicate.com/v1/models"

MODELS = {
    "STABLE_GEMMA_TORCHAO_MODEL_ID": "paragekbote/gemma3-torchao-quant-sparse",
    "STABLE_PHI4_MODEL_ID": "paragekbote/phi-4-reasoning-plus-unsloth",
    "STABLE_FLUX_IMG2IMG_MODEL_ID":"paragekbote/flux-fast-lora-hotswap-img2img",
    "STABLE_FLUX_MODEL_ID":"paragekbote/flux-fast-lora-hotswap",
    "STABLE_SMOLLM3_MODEL_ID":"paragekbote/smollm3-3b-smashed",
}


def resolve_latest_version(model_base: str, token: str) -> str:
    owner, name = model_base.split("/", 1)
    url = f"{REPLICATE_API}/{owner}/{name}/versions"

    headers = {
        "Authorization": f"Token {token}",
        "Accept": "application/json",
    }

    with httpx.Client(timeout=10.0) as client:
        r = client.get(url, headers=headers)
        r.raise_for_status()
        data = r.json()

    versions = data.get("results", [])
    if not versions:
        raise RuntimeError(f"No versions found for {model_base}")

    versions.sort(key=lambda v: v["created_at"], reverse=True)
    return f"{model_base}:{versions[0]['id']}"


def main():
    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        print("REPLICATE_API_TOKEN not set", file=sys.stderr)
        sys.exit(1)

    for env_var, model_base in MODELS.items():
        resolved = resolve_latest_version(model_base, token)
        print(f"{env_var}={resolved}")


if __name__ == "__main__":
    main()
