import replicate
import time
import json
import logging
from datetime import datetime

# ---------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("benchmark_flux_lora_text2img.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Benchmark Function
# ---------------------------------------------------------
def benchmark_flux_lora_text2img(
    num_runs=3,
    prompt="A majestic dragon soaring above a futuristic city.",
    trigger_word="Painting",
):
    """
    Benchmark for Flux Fast LoRA (text-to-image) variant.
    Only uses prompt + trigger_word as required by schema.
    """

    deployment_id = (
        "paragekbote/flux-fast-lora-hotswap:"  
        "a958687317369721e1ce66e5436fa989bcff2e40a13537d9b4aa4c6af4a34539"
    )

    # Schema-compliant input
    input_params = {
        "prompt": prompt,
        "trigger_word": trigger_word,
    }

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
        "deployment_name": "Flux-Fast-LoRA-Text2Img",
        "timestamp": datetime.now().isoformat(),
        "input_params": input_params,
        "runs": [],
    }

    run_times = []

    # ---------------------------------------------------------
    # Execute Runs
    # ---------------------------------------------------------
    for run_num in range(1, num_runs + 1):
        logger.info(f"--- Run {run_num}/{num_runs} ---")

        try:
            start = time.time()

            # Replicate returns a File object for images
            output_file = replicate.run(deployment_id, input=input_params)
            img_bytes = output_file.read()

            elapsed = time.time() - start
            run_times.append(elapsed)

            # Save to disk
            out_path = f"flux_lora_text2img_run_{run_num}.png"
            with open(out_path, "wb") as f:
                f.write(img_bytes)

            logger.info(f"✓ Completed in {elapsed:.2f}s")
            logger.info(f"  Generated: {out_path}")
            logger.info(f"  URL: {output_file.url()}")

            results["runs"].append(
                {
                    "run_number": run_num,
                    "elapsed_time": elapsed,
                    "output_file": out_path,
                    "file_url": output_file.url(),
                    "status": "success",
                }
            )

        except Exception as e:
            err = str(e)
            logger.error(f"✗ Run failed: {err}")

            results["runs"].append({
                "run_number": run_num,
                "status": "failed",
                "error": err
            })

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------
    if run_times:
        avg_t = sum(run_times) / len(run_times)
        min_t = min(run_times)
        max_t = max(run_times)

        std = (sum((t - avg_t) ** 2 for t in run_times) / len(run_times)) ** 0.5
        cv = (std / avg_t * 100) if avg_t else 0

        cold = run_times[0]
        warm = run_times[1:]
        warm_avg = sum(warm) / len(warm) if warm else None

        results["statistics"] = {
            "successful_runs": len(run_times),
            "failed_runs": num_runs - len(run_times),
            "avg_latency": avg_t,
            "min_latency": min_t,
            "max_latency": max_t,
            "std_dev_latency": std,
            "latency_cv_percent": cv,
            "cold_start_latency": cold,
            "warm_avg_latency": warm_avg,
            "cold_vs_warm_ratio": cold / warm_avg if warm_avg else None,
        }

        # Insights
        insights = []

        if warm_avg and cold > warm_avg * 1.5:
            insights.append(f"Cold start significantly slower ({cold:.2f}s vs warm {warm_avg:.2f}s).")

        if cv < 20:
            insights.append(f"Stable performance (CV {cv:.1f}%).")
        elif cv > 50:
            insights.append(f"High latency variability (CV {cv:.1f}%).")

        if avg_t < 4:
            insights.append("Extremely fast text-to-image generation.")
        elif avg_t < 8:
            insights.append("Good generation speed.")
        else:
            insights.append("Slow generation — possible cold backend or high load.")

        results["insights"] = insights

        # Log summary
        logger.info("\n" + "=" * 90)
        logger.info("BENCHMARK SUMMARY")
        logger.info("=" * 90)
        for k, v in results["statistics"].items():
            logger.info(f"{k}: {v}")
        logger.info("\nInsights:")
        for i in insights:
            logger.info(f" • {i}")
        logger.info("=" * 90)

    # ---------------------------------------------------------
    # Save JSON
    # ---------------------------------------------------------
    with open("flux_lora_text2img_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("\n✓ Saved: flux_lora_text2img_benchmark.json")
    logger.info("✓ Logs: benchmark_flux_lora_text2img.log\n")

    return results


# ---------------------------------------------------------
# Run the benchmark
# ---------------------------------------------------------
if __name__ == "__main__":
    benchmark_flux_lora_text2img(num_runs=3)
