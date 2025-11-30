import os
import time
import json
import logging
from datetime import datetime
from pathlib import Path

import replicate


# ============================================================
# Output directory (for GitHub Actions artifacts)
# ============================================================
OUTPUT_DIR = Path(os.getenv("BENCHMARK_OUTPUT_DIR", "benchmark_results_flux_text2img"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUTPUT_DIR / "benchmark_flux_lora_text2img.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ============================================================
# Benchmark Function
# ============================================================
def benchmark_flux_lora_text2img(
    num_runs=3,
    prompt="A majestic dragon soaring above a futuristic city.",
    trigger_word="Painting",
):
    """
    Benchmark Flux Fast LoRA (text → image).
    Only uses: prompt + trigger_word (schema compliant).
    """

    deployment_id = (
        "paragekbote/flux-fast-lora-hotswap:"
        "a958687317369721e1ce66e5436fa989bcff2e40a13537d9b4aa4c6af4a34539"
    )

    # Replicate client using secure API token (required)
    client = replicate.Client(api_token=os.environ["REPLICATE_API_TOKEN"])

    input_params = {
        "prompt": prompt,
        "trigger_word": trigger_word,
    }

    # Header logging
    logger.info("=" * 90)
    logger.info("FLUX FAST LORA (TEXT → IMAGE) BENCHMARK")
    logger.info("=" * 90)
    logger.info(f"Deployment: {deployment_id}")
    logger.info("Input Params:")
    logger.info(json.dumps(input_params, indent=2))
    logger.info(f"Runs: {num_runs}")
    logger.info("=" * 90)

    results = {
        "deployment_id": deployment_id,
        "deployment_name": "flux-fast-lora-text2img",
        "timestamp": datetime.now().isoformat(),
        "input_params": input_params,
        "runs": [],
    }

    run_times = []

    # ============================================================
    # Execute Runs
    # ============================================================
    for run_num in range(1, num_runs + 1):
        logger.info(f"--- Run {run_num}/{num_runs} ---")

        try:
            start = time.time()

            # Replicate returns a single File object for image output
            output_file = client.run(deployment_id, input=input_params)

            # Read image bytes
            img_bytes = output_file.read()

            elapsed = time.time() - start
            run_times.append(elapsed)

            # Save image output
            img_path = OUTPUT_DIR / f"flux_lora_text2img_run_{run_num}.png"
            img_path.write_bytes(img_bytes)

            logger.info(f"✓ Completed in {elapsed:.2f}s")
            logger.info(f"  Generated: {img_path}")
            logger.info(f"  URL: {output_file.url()}")

            results["runs"].append(
                {
                    "run_number": run_num,
                    "elapsed_time": elapsed,
                    "output_file": str(img_path),
                    "file_url": output_file.url(),
                    "status": "success",
                }
            )

        except Exception as e:
            err = str(e)
            logger.error(f"✗ Run failed: {err}")

            results["runs"].append(
                {
                    "run_number": run_num,
                    "status": "failed",
                    "error": err,
                }
            )

    # ============================================================
    # Statistics
    # ============================================================
    if run_times:
        avg_time = sum(run_times) / len(run_times)
        min_time = min(run_times)
        max_time = max(run_times)

        std_dev = (sum((t - avg_time) ** 2 for t in run_times) / len(run_times)) ** 0.5
        cv = (std_dev / avg_time * 100) if avg_time else 0

        cold_start = run_times[0]
        warm_times = run_times[1:]
        warm_avg = sum(warm_times) / len(warm_times) if warm_times else None

        results["statistics"] = {
            "successful_runs": len(run_times),
            "failed_runs": num_runs - len(run_times),
            "avg_latency": avg_time,
            "min_latency": min_time,
            "max_latency": max_time,
            "std_dev_latency": std_dev,
            "latency_cv_percent": cv,
            "cold_start_latency": cold_start,
            "warm_avg_latency": warm_avg,
            "cold_vs_warm_ratio": cold_start / warm_avg if warm_avg else None,
        }

        insights = []

        if warm_avg and cold_start > warm_avg * 1.5:
            insights.append(
                f"Cold start significantly slower ({cold_start:.2f}s vs warm {warm_avg:.2f}s)."
            )

        if cv < 20:
            insights.append(f"Stable performance (CV {cv:.1f}%).")
        elif cv > 50:
            insights.append(f"High latency variability (CV {cv:.1f}%).")

        if avg_time < 4:
            insights.append("Extremely fast image generation.")
        elif avg_time < 8:
            insights.append("Good generation speed.")
        else:
            insights.append("Slow generation — backend cold or overloaded.")

        results["insights"] = insights

        # Summary log
        logger.info("\n" + "=" * 90)
        logger.info("BENCHMARK SUMMARY")
        logger.info("=" * 90)

        for key, val in results["statistics"].items():
            logger.info(f"{key}: {val}")

        logger.info("\nInsights:")
        for insight in insights:
            logger.info(f" • {insight}")

        logger.info("=" * 90)

    # ============================================================
    # Save JSON Results
    # ============================================================
    json_path = OUTPUT_DIR / "flux_lora_text2img_benchmark.json"
    json_path.write_text(json.dumps(results, indent=2))

    logger.info(f"\n✓ JSON saved to {json_path}")
    logger.info(f"✓ Logs saved to {LOG_FILE}\n")

    return results


# ============================================================
# Execute
# ============================================================
if __name__ == "__main__":
    benchmark_flux_lora_text2img(num_runs=3)
