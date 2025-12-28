import os
import time
import json
import logging
from datetime import datetime

from benchmark.utils import safe_replicate_run, normalize_output

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler("benchmark_flux_text2img.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def benchmark_flux_text2img(num_runs: int = 3):
    deployment_id = (
        "paragekbote/flux-fast-lora-hotswap:"
        "ca315a7351d15d3c813e6c71ba26c540893961932c3791e05101596c93502e94"
    )

    out_dir = os.getenv("BENCHMARK_OUTPUT_DIR", ".")
    os.makedirs(out_dir, exist_ok=True)

    # --------------------------------------------------
    # Canonical benchmark inputs (schema-aligned)
    # --------------------------------------------------
    input_params = {
        "prompt": "A majestic dragon soaring above a futuristic city.",
        "trigger_word": "Painting",

        # Explicit geometry (latency-critical)
        "height": 1024,
        "width": 1024,

        # Diffusion controls (latency-critical)
        "guidance_scale": 3.5,
        "num_inference_steps": 40,

        # Explicitly enable LoRA hotswapping
        # (required for adapter reuse benchmarks)
        "hotswap": True,
    }

    logger.info("Running Flux Text2Img benchmark")
    logger.info(json.dumps(input_params, indent=2))

    results = {
        "deployment_id": deployment_id,
        "input_params": input_params,
        "timestamp": datetime.now().astimezone().isoformat(),
        "runs": [],
    }

    # --------------------------------------------------
    # Execute runs
    # --------------------------------------------------
    for i in range(1, num_runs + 1):
        logger.info(f"--- Run {i}/{num_runs} ---")

        try:
            start = time.time()

            raw_out = safe_replicate_run(deployment_id, input_params)
            output = normalize_output(raw_out, output_type="image")

            elapsed = time.time() - start

            if not isinstance(output, (bytes, bytearray)):
                raise TypeError(
                    f"normalize_output returned {type(output)}, expected bytes"
                )

            out_path = os.path.join(out_dir, f"flux_text2img_run_{i}.png")
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
    # Statistics (derived)
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

    out_json = os.path.join(out_dir, "flux_text2img_benchmark.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Saved → %s", out_json)
    return results


if __name__ == "__main__":
    benchmark_flux_text2img()
