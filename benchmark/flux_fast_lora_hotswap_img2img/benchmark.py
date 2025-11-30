import os
import time
import json
import logging
from datetime import datetime
from pathlib import Path

import requests
import replicate


# =============================================================
# Output Directory (GitHub Actions compatible)
# =============================================================
OUTPUT_DIR = Path(os.getenv("BENCHMARK_OUTPUT_DIR", "benchmark_results_flux"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUTPUT_DIR / "benchmark_flux.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# =============================================================
# Benchmark Function
# =============================================================
def benchmark_flux_fast_lora(
    num_runs=3,
    prompt="A magical illustration of a dragon flying above the clouds.",
    trigger_word="Painting",
    init_image="https://images.pexels.com/photos/33649783/pexels-photo-33649783.jpeg",
    seed=42,
    strength=0.6,
    guidance_scale=7.5,
    num_inference_steps=28,
):
    """
    Benchmark for Flux Fast LoRA Img2Img deployment.
    """

    deployment_id = (
        "paragekbote/flux-fast-lora-hotswap-img2img:"
        "e6e00065d5aa5e5dba299ab01b5177db8fa58dc4449849aa0cb3f1edf50430cd"
    )

    client = replicate.Client(api_token=os.environ["REPLICATE_API_TOKEN"])

    # Model schema input
    input_params = {
        "prompt": prompt,
        "trigger_word": trigger_word,
        "init_image": init_image,
        "seed": seed,
        "strength": strength,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
    }

    # Logging Header
    logger.info("=" * 90)
    logger.info("FLUX FAST LORA (IMG2IMG) BENCHMARK")
    logger.info("=" * 90)
    logger.info(f"Deployment: {deployment_id}")
    logger.info(json.dumps(input_params, indent=2))
    logger.info(f"Runs: {num_runs}")
    logger.info("=" * 90)

    results = {
        "deployment_id": deployment_id,
        "deployment_name": "Flux-Fast-LoRA-Img2Img",
        "timestamp": datetime.now().isoformat(),
        "input_params": input_params,
        "runs": [],
    }

    run_times = []

    # =============================================================
    # RUN BENCHMARK
    # =============================================================
    for run_num in range(1, num_runs + 1):
        logger.info(f"--- Run {run_num}/{num_runs} ---")

        try:
            start = time.time()

            # Replicate returns a URL STRING, not a file object
            img_url = client.run(deployment_id, input=input_params)

            if not isinstance(img_url, str):
                raise ValueError(f"Unexpected response: {img_url}")

            # Download image manually
            response = requests.get(img_url, timeout=60)
            response.raise_for_status()

            img_bytes = response.content
            elapsed = time.time() - start
            run_times.append(elapsed)

            # Save output image
            out_path = OUTPUT_DIR / f"flux_output_run_{run_num}.png"
            out_path.write_bytes(img_bytes)

            logger.info(f"✓ Completed in {elapsed:.2f}s")
            logger.info(f"  Saved → {out_path}")
            logger.info(f"  URL: {img_url}")

            results["runs"].append(
                {
                    "run_number": run_num,
                    "elapsed_time": elapsed,
                    "output_file": str(out_path),
                    "file_url": img_url,
                    "status": "success",
                }
            )

        except Exception as e:
            err = str(e)
            logger.error(f"✗ Run failed: {err}")
            results["runs"].append(
                {"run_number": run_num, "status": "failed", "error": err}
            )

    # =============================================================
    # STATISTICS
    # =============================================================
    if run_times:
        avg_time = sum(run_times) / len(run_times)
        min_time = min(run_times)
        max_time = max(run_times)

        std_dev = (sum((t - avg_time) ** 2 for t in run_times) / len(run_times)) ** 0.5
        cv = (std_dev / avg_time * 100) if avg_time > 0 else 0

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

        # Cold start
        if warm_avg and cold_start > warm_avg * 1.5:
            insights.append(
                f"Cold start significantly slower: {cold_start:.2f}s vs warm {warm_avg:.2f}s"
            )

        # Stability
        if cv < 20:
            insights.append(f"Stable performance (CV {cv:.1f}%)")
        elif cv > 50:
            insights.append(f"High variability (CV {cv:.1f}%)")

        # Absolute speed
        if avg_time < 4:
            insights.append("Extremely fast LoRA generation.")
        elif avg_time < 8:
            insights.append("Good speed for img2img with LoRA.")
        else:
            insights.append("Slow inference — backend cold or overloaded.")

        results["insights"] = insights

        logger.info("\n" + "=" * 90)
        logger.info("BENCHMARK SUMMARY")
        logger.info("=" * 90)
        for k, v in results["statistics"].items():
            logger.info(f"{k}: {v}")

        logger.info("\nInsights:")
        for i in insights:
            logger.info(f" • {i}")
        logger.info("=" * 90)

    # Save final JSON
    json_path = OUTPUT_DIR / "flux_benchmark_results.json"
    json_path.write_text(json.dumps(results, indent=2))

    logger.info(f"\n✓ Saved: {json_path}")
    logger.info(f"✓ Logs: {LOG_FILE}\n")

    return results


# =============================================================
# Run when executed directly
# =============================================================
if __name__ == "__main__":
    benchmark_flux_fast_lora(num_runs=3)
