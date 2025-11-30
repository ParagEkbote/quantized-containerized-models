import json
import time
import logging
from datetime import datetime
from pathlib import Path

from benchmark.utils import safe_replicate_run, normalize_output

logger = logging.getLogger(__name__)

def benchmark_flux_fast_lora_img2img(
    num_runs=3,
    prompt="A magical illustration of a dragon.",
    trigger_word="Painting",
    init_image="https://images.pexels.com/photos/33649783/pexels-photo-33649783.jpeg",
    seed=42,
    strength=0.6,
    guidance_scale=7.5,
    num_inference_steps=28,
):
    deployment_id = (
        "paragekbote/flux-fast-lora-hotswap-img2img:"
        "e6e00065d5aa5e5dba299ab01b5177db8fa58dc4449849aa0cb3f1edf50430cd"
    )

    input_params = {
        "prompt": prompt,
        "trigger_word": trigger_word,
        "init_image": init_image,
        "seed": seed,
        "strength": strength,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
    }

    results = {
        "deployment_id": deployment_id,
        "timestamp": datetime.now().isoformat(),
        "input_params": input_params,
        "runs": [],
    }

    run_times = []

    for run in range(1, num_runs + 1):
        logger.info(f"--- Run {run}/{num_runs} ---")

        try:
            start = time.time()
            output = safe_replicate_run(deployment_id, input_params)

            normalized = normalize_output(output)

            elapsed = time.time() - start
            run_times.append(elapsed)

            # Save file
            out_dir = Path("benchmark/results")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"flux_img2img_run_{run}.png"
            out_path.write_bytes(normalized)

            results["runs"].append(
                {
                    "run_number": run,
                    "elapsed_time": elapsed,
                    "output_path": str(out_path),
                    "status": "success",
                }
            )

        except Exception as e:
            results["runs"].append(
                {"run_number": run, "status": "failed", "error": str(e)}
            )

    # Save summary
    out_dir = Path("benchmark/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "flux_img2img_benchmark.json").write_text(
        json.dumps(results, indent=2)
    )

    return results


if __name__ == "__main__":
    benchmark_flux_fast_lora_img2img(num_runs=3)
