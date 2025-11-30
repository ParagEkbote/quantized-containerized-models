import os
import time
import json
import logging
from datetime import datetime
from pathlib import Path

import replicate


# =============================================================
# Output directory (for GH Actions artifacts)
# =============================================================
OUTPUT_DIR = Path(os.getenv("BENCHMARK_OUTPUT_DIR", "benchmark_results_smollm3"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUTPUT_DIR / "benchmark_smollm3.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# =============================================================
# Utility
# =============================================================
def calculate_efficiency_rating(tokens_per_sec: float) -> str:
    if tokens_per_sec >= 50:
        return "Excellent"
    if tokens_per_sec >= 20:
        return "Very Good"
    if tokens_per_sec >= 10:
        return "Good"
    if tokens_per_sec >= 5:
        return "Fair"
    return "Poor"


# =============================================================
# Benchmark SmolLM3
# =============================================================
def benchmark_smollm3(
    num_runs=3,
    prompt="What are the applications of ML in healthcare?",
    mode="no_think",
    seed=18,
    max_new_tokens=1024
):
    """
    Benchmark SmolLM3-3B-Smashed using Replicate streaming output.
    """

    deployment_id = (
        "paragekbote/smollm3-3b-smashed:"
        "232b6f87dac025cb54803cfbc52135ab8366c21bbe8737e11cd1aee4bf3a2423"
    )

    client = replicate.Client(api_token=os.environ["REPLICATE_API_TOKEN"])

    input_params = {
        "prompt": prompt,
        "mode": mode,
        "seed": seed,
        "max_new_tokens": max_new_tokens,
    }

    # Header logging
    logger.info("=" * 80)
    logger.info("SMOLLM3-3B-SMASHED BENCHMARK")
    logger.info("=" * 80)
    logger.info(f"Deployment: {deployment_id}")
    logger.info(f"Runs: {num_runs}")
    logger.info("Input Schema:")
    logger.info(json.dumps(input_params, indent=2))
    logger.info("=" * 80)

    results = {
        "deployment_id": deployment_id,
        "deployment_name": "smollm3-3b-smashed",
        "timestamp": datetime.now().isoformat(),
        "input_params": input_params,
        "runs": [],
    }

    run_times = []
    token_counts = []

    # =============================================================
    # Run benchmark
    # =============================================================
    for run_num in range(1, num_runs + 1):
        logger.info(f"--- Run {run_num}/{num_runs} ---")

        try:
            start = time.time()

            # Replicate outputs streamed text chunks
            output_chunks = client.run(deployment_id, input=input_params)

            output_text = ""
            for chunk in output_chunks:
                output_text += str(chunk)

            elapsed = time.time() - start

            # Word count ≈ token proxy
            words = len(output_text.split())
            chars = len(output_text)

            run_times.append(elapsed)
            token_counts.append(words)

            results["runs"].append(
                {
                    "run_number": run_num,
                    "elapsed_time": elapsed,
                    "output_length_chars": chars,
                    "output_length_words": words,
                    "output_text": output_text,
                    "status": "success",
                }
            )

            logger.info(f"✓ Completed in {elapsed:.2f}s")
            logger.info(f"  Output: {chars} chars, ~{words} words")
            logger.info(f"  Throughput: {words/elapsed:.2f} tokens/sec")

            # Save output text
            out_path = OUTPUT_DIR / f"smollm3_output_run_{run_num}.txt"
            out_path.write_text(output_text, encoding="utf-8")

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

    # =============================================================
    # Statistics
    # =============================================================
    if run_times:
        avg_time = sum(run_times) / len(run_times)
        min_time = min(run_times)
        max_time = max(run_times)

        avg_words = sum(token_counts) / len(token_counts)
        tps = avg_words / avg_time if avg_time else 0

        std_dev = (sum((t - avg_time) ** 2 for t in run_times) / len(run_times)) ** 0.5
        cv = (std_dev / avg_time * 100) if avg_time else 0

        cold_start = run_times[0]
        warm_times = run_times[1:]
        warm_avg = sum(warm_times) / len(warm_times) if warm_times else None

        word_std = (
            (sum((w - avg_words) ** 2 for w in token_counts) / len(token_counts)) ** 0.5
            if len(token_counts) > 1 else 0
        )

        results["statistics"] = {
            "successful_runs": len(run_times),
            "failed_runs": num_runs - len(run_times),
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "time_std_dev": std_dev,
            "time_variability_cv": cv,
            "avg_words": avg_words,
            "word_std_dev": word_std,
            "avg_tokens_per_sec": tps,
            "cold_start_time": cold_start,
            "avg_warm_time": warm_avg,
            "cold_vs_warm_ratio": (cold_start / warm_avg) if warm_avg else None,
            "efficiency_rating": calculate_efficiency_rating(tps),
            "consistency_score": max(0, 100 - cv),
        }

        # Insights
        insights = []

        if warm_avg and cold_start > warm_avg * 1.5:
            insights.append(
                f"Cold start significantly slower: {cold_start:.2f}s vs {warm_avg:.2f}s"
            )

        if cv < 20:
            insights.append(f"Stable performance (CV {cv:.1f}%)")
        elif cv > 50:
            insights.append(f"High variability (CV {cv:.1f}%)")

        if tps < 5:
            insights.append(f"Low throughput: {tps:.2f} tokens/sec")
        elif tps > 20:
            insights.append(f"High throughput: {tps:.2f} tokens/sec")

        if word_std > avg_words * 0.2:
            insights.append("Output lengths vary significantly")

        results["insights"] = insights

        logger.info("\n" + "=" * 80)
        logger.info("BENCHMARK SUMMARY")
        logger.info("=" * 80)
        for k, v in results["statistics"].items():
            logger.info(f"{k}: {v}")
        logger.info("\nInsights:")
        for i in insights:
            logger.info(f" • {i}")
        logger.info("=" * 80)

    else:
        logger.warning("All runs failed. No statistics available.")
        results["statistics"] = None

    # Save JSON
    json_path = OUTPUT_DIR / "smollm3_benchmark_results.json"
    json_path.write_text(json.dumps(results, indent=2))

    logger.info(f"\n✓ Saved results: {json_path}")
    logger.info(f"✓ Logs: {LOG_FILE}\n")

    return results


# =============================================================
# Execute
# =============================================================
if __name__ == "__main__":
    benchmark_smollm3(num_runs=3)
