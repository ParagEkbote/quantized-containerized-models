import os
import time
import json
import logging
from datetime import datetime

from benchmark.utils import safe_replicate_run, normalize_output

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler("benchmark_flux_img2img.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def benchmark_flux_img2img(num_runs: int = 3):
    deployment_id = (
        "paragekbote/flux-fast-lora-hotswap-img2img:"
        "e6e00065d5aa5e5dba299ab01b5177db8fa58dc4449849aa0cb3f1edf50430cd"
    )

    out_dir = os.getenv("BENCHMARK_OUTPUT_DIR", ".")
    os.makedirs(out_dir, exist_ok=True)

    # --------------------------------------------------
    # Canonical img2img benchmark inputs (schema-aligned)
    # --------------------------------------------------
    input_params = {
        # Determinism
        "seed": 47,

        # img2img inputs
        "init_image": (
            "https://images.pexels.com/photos/4934914/"
            "pexels-photo-4934914.jpeg"
        ),
        "strength": 0.75,

        # Text + LoRA routing
        "prompt": (
            "A majestic peacock with enhanced detail, vibrant iridescent feathers, "
            "sharp eye-spots, natural lighting, and improved clarity and depth."
        ),
        "trigger_word": "Manga",

        # Diffusion controls (latency-critical)
        "guidance_scale": 6.0,
        "num_inference_steps": 20,

        # Explicitly exercise LoRA hotswapping
        "hotswap": True,
    }

    logger.info("Running Flux Img2Img benchmark")
    logger.info(json.dumps(input_params, indent=2))

    results = {
        "deployment_id": deployment_id,
        "input_params": input_params,
        "timestamp": datetime.now().astimezone().isoformat(),
        "runs": [],
    }

    # --------------------------------------------------
    # Execute runs (single source of truth)
    # --------------------------------------------------
    for i in range(1, num_runs + 1):
        logger.info("--- Run %d/%d ---", i, num_runs)

        try:
            start = time.time()

            raw_out = safe_replicate_run(deployment_id, input_params)
            output = normalize_output(raw_out, output_type="image")

            elapsed = time.time() - start

            if not isinstance(output, (bytes, bytearray)):
                raise TypeError(
                    f"normalize_output returned {type(output)}, expected bytes"
                )

            out_path = os.path.join(out_dir, f"flux_img2img_run_{i}.png")
            with open(out_path, "wb") as f:
                f.write(output)

            results["runs"].append(
                {
                    "run_number": i,
                    "elapsed_time": elapsed,
                    "output_file": out_path,
                    "status": "success",
                }
            )

            logger.info("Run %d succeeded in %.3fs", i, elapsed)

        except Exception as e:
            logger.exception("Run failed")
            results["runs"].append(
                {
                    "run_number": i,
                    "status": "failed",
                    "error": str(e),
                }
            )

    # --------------------------------------------------
    # Derive statistics FROM recorded runs
    # --------------------------------------------------
    successful_runs = [
        r for r in results["runs"] if r.get("status") == "success"
    ]

    times = [r["elapsed_time"] for r in successful_runs]

    if times:
        avg = sum(times) / len(times)
        mn = min(times)
        mx = max(times)
        std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5
        cv = (std / avg * 100) if avg else None

        cold_run = successful_runs[0]
        warm_runs = successful_runs[1:]

        warm_avg = (
            sum(r["elapsed_time"] for r in warm_runs) / len(warm_runs)
            if warm_runs else None
        )

        results["statistics"] = {
            "successful_runs": len(successful_runs),
            "failed_runs": num_runs - len(successful_runs),

            "avg_latency": avg,
            "min_latency": mn,
            "max_latency": mx,
            "std_latency": std,
            "latency_cv": cv,

            # Cold start explicitly linked to artifact
            "cold_start": {
                "run_number": cold_run["run_number"],
                "elapsed_time": cold_run["elapsed_time"],
                "output_file": cold_run["output_file"],
            },

            "warm_avg_latency": warm_avg,
            "cold_vs_warm_ratio": (
                cold_run["elapsed_time"] / warm_avg
                if warm_avg else None
            ),
        }

    out_json = os.path.join(out_dir, "flux_img2img_benchmark.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Saved → %s", out_json)
    return results


if __name__ == "__main__":
    benchmark_flux_img2img()
