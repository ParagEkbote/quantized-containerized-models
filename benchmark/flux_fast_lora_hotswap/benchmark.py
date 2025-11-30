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


def benchmark_flux_text2img(num_runs=3):
    deployment_id = (
        "paragekbote/flux-fast-lora-hotswap:"
        "a958687317369721e1ce66e5436fa989bcff2e40a13537d9b4aa4c6af4a34539"
    )

    out_dir = os.getenv("BENCHMARK_OUTPUT_DIR", ".")
    os.makedirs(out_dir, exist_ok=True)

    input_params = {
        "prompt": "A majestic dragon soaring above a futuristic city.",
        "trigger_word": "Painting",
    }

    logger.info("Running Flux Text2Img")
    logger.info(json.dumps(input_params, indent=2))

    results = {
        "deployment_id": deployment_id,
        "input_params": input_params,
        "timestamp": datetime.now().isoformat(),
        "runs": [],
    }

    run_times = []

    for i in range(1, num_runs + 1):
        logger.info(f"--- Run {i}/{num_runs} ---")

        try:
            start = time.time()
            raw_out = safe_replicate_run(deployment_id, input_params)
            output = normalize_output(raw_out)
            elapsed = time.time() - start

            run_times.append(elapsed)

            out_path = os.path.join(out_dir, f"flux_text2img_run_{i}.png")
            with open(out_path, "wb") as f:
                f.write(output)

            results["runs"].append({
                "run_number": i,
                "elapsed_time": elapsed,
                "output_file": out_path,
                "status": "success",
            })

        except Exception as e:
            err = str(e)
            logger.error(f"Run failed: {err}")
            results["runs"].append({"run_number": i, "status": "failed", "error": err})

    if run_times:
        avg = sum(run_times) / len(run_times)
        mn = min(run_times)
        mx = max(run_times)

        std = (sum((t - avg) ** 2 for t in run_times) / len(run_times)) ** 0.5
        cv = std / avg * 100

        cold = run_times[0]
        warm = run_times[1:]
        warm_avg = sum(warm) / len(warm) if warm else None

        results["statistics"] = {
            "successful_runs": len(run_times),
            "failed_runs": num_runs - len(run_times),
            "avg_latency": avg,
            "min_latency": mn,
            "max_latency": mx,
            "std_latency": std,
            "latency_cv": cv,
            "cold_start_latency": cold,
            "warm_avg_latency": warm_avg,
            "cold_vs_warm_ratio": cold / warm_avg if warm_avg else None,
        }

    out_json = os.path.join(out_dir, "flux_text2img_benchmark.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved → {out_json}")
    return results


if __name__ == "__main__":
    benchmark_flux_text2img()
