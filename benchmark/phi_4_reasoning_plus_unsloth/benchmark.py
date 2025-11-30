import os
import time
import json
import logging
from datetime import datetime
from pathlib import Path

import replicate


# =============================================================
# Output Directory (GitHub Actions friendly)
# =============================================================
OUTPUT_DIR = Path(os.getenv("BENCHMARK_OUTPUT_DIR", "benchmark_results_phi4"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUTPUT_DIR / "benchmark_phi4.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


# =============================================================
# Utility: Efficiency Rating
# =============================================================
def calculate_efficiency(tokens_per_sec):
    if tokens_per_sec >= 50:
        return "Excellent"
    elif tokens_per_sec >= 20:
        return "Very Good"
    elif tokens_per_sec >= 10:
        return "Good"
    elif tokens_per_sec >= 5:
        return "Fair"
    return "Poor"


# =============================================================
# Benchmark Function
# =============================================================
def benchmark_phi4_unsloth(
    num_runs=3,
    prompt="Explain quantum entanglement.",
    max_new_tokens=1024,
    temperature=0.7,
    top_p=0.95,
    seed=42,
):
    """
    Benchmark for Phi-4 Reasoning+Unsloth deployment.
    """

    deployment_id = (
        "paragekbote/phi-4-reasoning-plus-unsloth:"
        "a6b2aa30b793e79ee4f7e30165dce1636730b20c2798d487fc548427ba6314d7"
    )

    # Replicate client
    client = replicate.Client(api_token=os.environ["REPLICATE_API_TOKEN"])

    # Model input
    input_params = {
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
    }

    # Header
    logger.info("=" * 90)
    logger.info("PHI-4 REASONING + UNSLOTH BENCHMARK")
    logger.info("=" * 90)
    logger.info(f"Deployment: {deployment_id}")
    logger.info(f"Input: {json.dumps(input_params, indent=2)}")
    logger.info(f"Runs: {num_runs}")
    logger.info("=" * 90)

    results = {
        "deployment_id": deployment_id,
        "deployment_name": "Phi-4-Reasoning-Plus-Unsloth",
        "timestamp": datetime.now().isoformat(),
        "input_params": input_params,
        "runs": [],
    }

    run_times = []
    token_counts = []


    # =============================================================
    # RUN BENCHMARK
    # =============================================================
    for run in range(1, num_runs + 1):
        logger.info(f"\n--- Run {run}/{num_runs} ---")

        try:
            start = time.time()

            # Replicate returns TEXT, not a file
            output_text = client.run(deployment_id, input=input_params)

            elapsed = time.time() - start
            words = len(output_text.split())

            run_times.append(elapsed)
            token_counts.append(words)

            # Save text output
            out_file = OUTPUT_DIR / f"phi4_output_run_{run}.txt"
            out_file.write_text(output_text, encoding="utf-8")

            results["runs"].append(
                {
                    "run_number": run,
                    "elapsed_time": elapsed,
                    "output_words": words,
                    "status": "success",
                    "output_file": str(out_file),
                }
            )

            logger.info(f"✓ Completed in {elapsed:.2f}s – {words} words")
            logger.info(f"  Throughput: {words / elapsed:.2f} tokens/sec")

        except Exception as e:
            err = str(e)
            results["runs"].append(
                {"run_number": run, "status": "failed", "error": err}
            )
            logger.error(f"✗ Run failed: {err}")


    # =============================================================
    # STATISTICS
    # =============================================================
    if run_times:
        avg_time = sum(run_times) / len(run_times)
        min_time = min(run_times)
        max_time = max(run_times)

        avg_words = sum(token_counts) / len(token_counts)
        avg_tps = avg_words / avg_time if avg_time > 0 else 0

        std_time = (sum((t - avg_time) ** 2 for t in run_times) / len(run_times)) ** 0.5
        cv_time = (std_time / avg_time * 100) if avg_time else 0

        cold_start = run_times[0]
        warm_times = run_times[1:]
        warm_avg = sum(warm_times) / len(warm_times) if warm_times else None

        word_std_dev = (
            sum((w - avg_words) ** 2 for w in token_counts) / len(token_counts)
        ) ** 0.5 if len(token_counts) > 1 else 0

        results["statistics"] = {
            "successful_runs": len(run_times),
            "failed_runs": num_runs - len(run_times),
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "std_time": std_time,
            "cv_time": cv_time,
            "avg_words": avg_words,
            "word_std_dev": word_std_dev,
            "avg_tokens_per_sec": avg_tps,
            "cold_start_time": cold_start,
            "avg_warm_time": warm_avg,
            "cold_warm_ratio": cold_start / warm_avg if warm_avg else None,
            "efficiency_rating": calculate_efficiency(avg_tps),
            "consistency_score": max(0, 100 - cv_time),
        }

        # Insights
        insights = []

        if warm_avg and cold_start > warm_avg * 1.5:
            insights.append(
                f"Cold start slow: {cold_start:.2f}s vs warm {warm_avg:.2f}s"
            )
        if cv_time < 20:
            insights.append(f"Stable performance (CV {cv_time:.1f}%)")
        elif cv_time > 50:
            insights.append(f"High variance (CV {cv_time:.1f}%)")
        if avg_tps > 20:
            insights.append(f"Excellent throughput: {avg_tps:.1f} tokens/sec")
        elif avg_tps < 5:
            insights.append(f"Low throughput: {avg_tps:.1f} tokens/sec")
        if word_std_dev > avg_words * 0.2:
            insights.append("Output length varies significantly")

        results["insights"] = insights

        # Summary logs
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
    json_path = OUTPUT_DIR / "phi4_benchmark_results.json"
    json_path.write_text(json.dumps(results, indent=2))

    logger.info(f"\n✓ Results saved: {json_path}")
    logger.info(f"✓ Logs saved: {LOG_FILE}\n")

    return results


# =============================================================
# Run
# =============================================================
if __name__ == "__main__":
    benchmark_phi4_unsloth(num_runs=3)
