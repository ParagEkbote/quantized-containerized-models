import json
import time
import logging
from datetime import datetime
from pathlib import Path

from benchmark.utils import safe_replicate_run, normalize_output

logger = logging.getLogger(__name__)


def benchmark_flux_lora_text2img(
    num_runs=3,
    prompt="A majestic dragon flying above a futuristic skyline.",
    trigger_word="Painting",
):
    """
    Benchmark Flux Fast LoRA (text-to-image).
    Schema: prompt, trigger_word only.
    """

    deployment_id = (
        "paragekbote/flux-fast-lora-hotswap:"
        "a958687317369721e1ce66e5436fa989bcff2e40a13537d9b4aa4c6af4a34539"
    )

    input_params = {"prompt": prompt, "trigger_word": trigger_word}

    logger.info("=" * 90)
    logger.info("FLUX FAST LORA — TEXT → IMAGE BENCHMARK")
    logger.info("=" * 90)
    logger.info(json.dumps(input_params, indent=2))

    results = {
        "deployment_id": deployment_id,
        "deployment_name": "Flux-Fast-LoRA-Text2Img",
        "timestamp": datetime.now().isoformat(),
        "input_params": input_params,
        "runs": [],
    }

    out_dir = Path("benchmark/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    run_times = []

    for run in range(1, num_runs + 1):
        logger.info(f"--- Run {run}/{num_runs} ---")

        try:
            start = time.time()

            output = safe_replicate_run(deployment_id, input_params)
            normalized = normalize_output(output)  # should be bytes

            elapsed = time.time() - start
            run_times.append(elapsed)

            # Save PNG
            img_path = out_dir / f"flux_text2img_run_{run}.png"
            img_path.write_bytes(normalized)

            results["runs"].append(
                {
                    "run_number": run,
                    "elapsed_time": elapsed,
                    "output_file": str(img_path),
                    "status": "success",
                }
            )

            logger.info(f"✓ Run {run} image saved ({elapsed:.2f}s)")

        except Exception as e:
            logger.error(f"✗ Run {run} failed: {e}")
            results["runs"].append({"run_number": run, "status": "failed", "error": str(e)})

    (out_dir / "flux_text2img_benchmark.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    benchmark_flux_lora_text2img(num_runs=3)
