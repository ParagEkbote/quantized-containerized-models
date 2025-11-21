import replicate
import time
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('benchmark.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# -------------------------
# Utility
# -------------------------
def calculate_efficiency_rating(tokens_per_sec):
    """Calculate efficiency rating based on throughput."""
    if tokens_per_sec >= 50:
        return "Excellent"
    elif tokens_per_sec >= 20:
        return "Very Good"
    elif tokens_per_sec >= 10:
        return "Good"
    elif tokens_per_sec >= 5:
        return "Fair"
    else:
        return "Poor"


# -------------------------
# Benchmark Function
# -------------------------
def benchmark_smollm3(
    num_runs=3,
    prompt="What are the applications of ML in healthcare?",
    mode="no_think",
    seed=18,
    max_new_tokens=1024
):
    """
    Benchmark SmolLM3-3B-Smashed with the full schema.
    """

    deployment_id = (
        "paragekbote/smollm3-3b-smashed:"
        "232b6f87dac025cb54803cfbc52135ab8366c21bbe8737e11cd1aee4bf3a2423"
    )

    # Full input schema
    input_params = {
        "prompt": prompt,
        "mode": mode,
        "seed": seed,
        "max_new_tokens": max_new_tokens,
    }

    logger.info("=" * 80)
    logger.info("SmolLM3-3B-Smashed Benchmark")
    logger.info("=" * 80)
    logger.info(f"Deployment: {deployment_id}")
    logger.info(f"Runs: {num_runs}")
    logger.info(f"Input Schema: {json.dumps(input_params, indent=2)}")
    logger.info("=" * 80)

    results = {
        "deployment_id": deployment_id,
        "deployment_name": "SmolLM3-3B-Smashed",
        "timestamp": datetime.now().isoformat(),
        "input_params": input_params,
        "runs": []
    }

    run_times = []
    token_counts = []

    # -------------------------------------------------------
    # Run Benchmark
    # -------------------------------------------------------
    for run_num in range(1, num_runs + 1):
        logger.info(f"--- Run {run_num}/{num_runs} ---")

        try:
            start_time = time.time()

            # Replicate output is usually a list or generator of text chunks
            output = replicate.run(deployment_id, input=input_params)

            # Concatenate streamed chunks
            if isinstance(output, list):
                output_text = "".join(output)
            else:
                output_text = ""
                for chunk in output:
                    output_text += chunk

            elapsed_time = time.time() - start_time

            # Token approximation (word count)
            word_count = len(output_text.split())
            char_count = len(output_text)

            run_times.append(elapsed_time)
            token_counts.append(word_count)

            results["runs"].append({
                "run_number": run_num,
                "elapsed_time": elapsed_time,
                "output_length_chars": char_count,
                "output_length_words": word_count,
                "output_text": output_text,
                "status": "success"
            })

            logger.info(f"✓ Completed in {elapsed_time:.2f}s")
            logger.info(f"  Output: {char_count} chars, ~{word_count} words")
            logger.info(f"  Throughput: {word_count/elapsed_time:.2f} tokens/sec")

            with open(f"output_run_{run_num}.txt", "w") as f:
                f.write(output_text)

        except Exception as e:
            error_msg = str(e)
            logger.error(f"✗ Run failed: {error_msg}")

            results["runs"].append({
                "run_number": run_num,
                "status": "failed",
                "error": error_msg
            })

    # -------------------------------------------------------
    # Statistics
    # -------------------------------------------------------
    if run_times:
        avg_time = sum(run_times) / len(run_times)
        min_time = min(run_times)
        max_time = max(run_times)

        avg_words = sum(token_counts) / len(token_counts) if token_counts else 0
        avg_tokens_per_sec = avg_words / avg_time if avg_time else 0

        # Variability
        time_std_dev = (sum((t - avg_time) ** 2 for t in run_times) / len(run_times)) ** 0.5
        cv = (time_std_dev / avg_time * 100) if avg_time else 0

        # Cold start
        cold_start = run_times[0]
        warm_avg = (sum(run_times[1:]) / (len(run_times) - 1)) if len(run_times) > 1 else None

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
            "time_std_dev": time_std_dev,
            "time_variability_cv": cv,
            "avg_words": avg_words,
            "word_std_dev": word_std,
            "avg_tokens_per_sec": avg_tokens_per_sec,
            "cold_start_time": cold_start,
            "avg_warm_time": warm_avg,
            "cold_vs_warm_ratio": (cold_start / warm_avg) if warm_avg else None,
            "consistency_score": max(0, 100 - cv),
            "efficiency_rating": calculate_efficiency_rating(avg_tokens_per_sec)
        }

        # Insights
        insights = []

        if warm_avg and cold_start > warm_avg * 1.5:
            insights.append(
                f"Cold start is significantly slower: {cold_start:.2f}s vs warm {warm_avg:.2f}s"
            )

        if cv > 50:
            insights.append(f"High variance (CV {cv:.1f}%) indicates inconsistent performance")
        elif cv < 20:
            insights.append(f"Strong stability (CV {cv:.1f}%)")

        if avg_tokens_per_sec < 5:
            insights.append(f"Low throughput: {avg_tokens_per_sec:.2f} tokens/sec")
        elif avg_tokens_per_sec > 20:
            insights.append(f"High throughput: {avg_tokens_per_sec:.2f} tokens/sec")

        if word_std > avg_words * 0.2:
            insights.append(f"Output lengths vary significantly (std {word_std:.0f})")

        results["insights"] = insights

        # Logging summary
        logger.info("\n" + "=" * 80)
        logger.info("BENCHMARK SUMMARY")
        logger.info("=" * 80)
        for k, v in results["statistics"].items():
            logger.info(f"{k}: {v}")
        logger.info("\nInsights:")
        for i in insights:
            logger.info(f" • {i}")

    else:
        logger.warning("No successful runs. No statistics generated.")
        results["statistics"] = None

    # Save results
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("\n✓ Results saved to benchmark_results.json")
    logger.info("✓ Logs saved to benchmark.log\n")

    return results


# Run Benchmark
if __name__ == "__main__":
    benchmark_smollm3(num_runs=3)
