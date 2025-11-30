import os
import replicate
import time
import json
import logging
from pathlib import Path
from datetime import datetime

# ----------------------------
# Configure Logging
# ----------------------------
OUTPUT_DIR = Path(os.getenv("BENCHMARK_OUTPUT_DIR", "benchmark_results"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(OUTPUT_DIR / 'gemma3_vlm_benchmark.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# ----------------------------
# Utility: Efficiency Rating
# ----------------------------
def _calculate_efficiency_rating(tokens_per_sec):
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


# ----------------------------
# Main Benchmark
# ----------------------------
def benchmark_gemma3_vlm(num_runs=3):
    deployment_id = (
        "paragekbote/gemma3-torchao-quant-sparse:"
        "44626bdc478fcfe56ee3d8a5a846b72f1e25abac25f740b2b615c1fcb2b63cb2"
    )

    # Replicate Client
    client = replicate.Client(api_token=os.environ["REPLICATE_API_TOKEN"])

    # Input parameters
    input_params = {
        "prompt": "Describe the image briefly. What kinds of bread are visible?",
        "image_url": "https://images.pexels.com/photos/29380151/pexels-photo-29380151.jpeg",
        "use_sparsity": "true",
        "sparsity_type": "layer_norm",
        "max_new_tokens": 200,
        "temperature": 0.7,
        "top_p": 0.9,
        "seed": 42,
        "use_quantization": "true",
        "sparsity_ratio": 0.3
    }

    # Metadata storage
    results = {
        "deployment_id": deployment_id,
        "model_type": "vision-language",
        "timestamp": datetime.now().isoformat(),
        "input_params": input_params,
        "runs": []
    }

    run_times = []
    token_counts = []

    logger.info("=" * 70)
    logger.info("Gemma3 TorchAO Quantized Sparse VLM Benchmark")
    logger.info("=" * 70)

    # ----------------------------
    # Run Benchmarks
    # ----------------------------
    for run in range(1, num_runs + 1):
        logger.info(f"\n--- Run {run}/{num_runs} ---")

        try:
            start = time.time()
            output = client.run(deployment_id, input=input_params)
            elapsed = time.time() - start

            output_text = str(output)
            word_count = len(output_text.split())

            run_times.append(elapsed)
            token_counts.append(word_count)

            # Save output text
            out_path = OUTPUT_DIR / f"gemma3_vlm_output_run_{run}.txt"
            out_path.write_text(output_text)

            results["runs"].append({
                "run_number": run,
                "elapsed_time": elapsed,
                "word_count": word_count,
                "status": "success"
            })

            logger.info(f"✓ Time: {elapsed:.2f}s | Words: {word_count}")

        except Exception as e:
            err = str(e)
            results["runs"].append({
                "run_number": run,
                "status": "failed",
                "error": err
            })
            logger.error(f"✗ Run failed: {err}")

    # ----------------------------
    # Compute Statistics
    # ----------------------------
    if run_times:
        avg = sum(run_times) / len(run_times)
        tps = (sum(token_counts) / len(token_counts)) / avg

        results["statistics"] = {
            "average_time": avg,
            "min_time": min(run_times),
            "max_time": max(run_times),
            "average_words": sum(token_counts) / len(token_counts),
            "avg_tokens_per_sec": tps,
            "efficiency_rating": _calculate_efficiency_rating(tps)
        }

        logger.info("\n=== SUMMARY ===")
        logger.info(f"Avg time: {avg:.2f}s")
        logger.info(f"Throughput: {tps:.2f} tokens/sec")
        logger.info(f"Efficiency: {_calculate_efficiency_rating(tps)}")

    # ----------------------------
    # Save Final JSON
    # ----------------------------
    (OUTPUT_DIR / "gemma3_vlm_benchmark_results.json").write_text(
        json.dumps(results, indent=2)
    )

    logger.info("\n✓ Benchmark complete.")
    logger.info(f"✓ Results saved to: {OUTPUT_DIR}")

    return results


if __name__ == "__main__":
    benchmark_gemma3_vlm(num_runs=3)
