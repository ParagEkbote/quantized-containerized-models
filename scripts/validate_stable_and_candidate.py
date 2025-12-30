import os
import sys

# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODELS = {
    "GEMMA_TORCHAO": "paragekbote/gemma3-torchao-quant-sparse",
    "PHI4": "paragekbote/phi-4-reasoning-plus-unsloth",
    "FLUX_IMG2IMG": "paragekbote/flux-fast-lora-hotswap-img2img",
    "FLUX": "paragekbote/flux-fast-lora-hotswap",
    "SMOLLM3": "paragekbote/smollm3-3b-smashed",
}

# --------------------------------------------------
# Helpers
# --------------------------------------------------


def warn(msg: str) -> None:
    print(f"⚠️  {msg}", file=sys.stderr)


def fail(msg: str) -> None:
    print(f"❌ {msg}", file=sys.stderr)
    sys.exit(1)


# --------------------------------------------------
# Main
# --------------------------------------------------


def main() -> None:
    """
    Validate that:
      - Stable model IDs are explicitly provided
      - Candidate model IDs were captured by the deploy step

    This script MUST NOT resolve versions itself.
    It only validates and re-exports environment variables.
    """

    for key in MODELS:
        stable_env = f"STABLE_{key}_MODEL_ID"
        candidate_env = "CANDIDATE_MODEL_ID"

        stable = os.environ.get(stable_env)
        candidate = os.environ.get(candidate_env)

        if not stable:
            fail(f"{stable_env} is not set (explicit stable baseline required)")

        if not candidate:
            warn(
                f"{candidate_env} is not set — canary will skip for {key}. "
                "This is allowed during rollout."
            )
            continue

        print(f"{stable_env}={stable}")
        print(f"{candidate_env}={candidate}")


if __name__ == "__main__":
    main()
