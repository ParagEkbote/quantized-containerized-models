import os
import time
import json
import logging
from datetime import datetime

from benchmark.utils import safe_replicate_run, normalize_output

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("gemma3_vlm_benchmark.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def calculate_efficiency(tokens_per_sec):
    if tokens_per_sec >= 50:
        return "Excellent"
    if tokens_per_sec >= 20:
        return "Very Good"
    if tokens_per_sec >= 10:
        return "Good"
    if tokens_per_sec >= 5:
        return "Fair"
    return "Poor"


def benchmark_gemma3_vlm(num_runs=3):
    deployment_id = (
        "paragekbote/gemma3-torchao-quant-sparse:"
        "44626bdc478fcfe56ee3d8a5a846b72f1e25abac25f740b2b615c1fcb2b63cb2"
    )

    out_dir = os.getenv("BENCHMARK_OUTPUT_DIR", ".")
    os.makedirs(out_dir, exist_ok=True)

    input_params = {
        "prompt": "Describe the image in the photo.",
        "image_url": "https://images.pexels.com/photos/29380151/pexels-photo-29380151.jpeg",
        "use_sparsity": "true",
        "sparsity_type": "layer_norm",
        "max_new_tokens": 500,
        "temperature": 0.7,
        "top_p": 0.9,
        "seed": 42,
        "use_quantization": "true",
        "sparsity_ratio": 0.3,
    }

    logger.info("Running Gemma 3 VLM Benchmark")
    logger.info(json.dumps(input_params, indent=2))

    results = {
        "deployment_id": deployment_id,
        "timestamp": datetime.now().isoformat(),
        "input_params": input_params,
        "runs": [],
    }

    run_times = []
    token_counts = []

    for i in range(1, num_runs + 1):
        logger.info(f"--- Run {i}/{num_runs} ---")

        try:
            start = time.time()
            raw_output = safe_replicate_run(deployment_id, input_params)
            output = normalize_output(raw_output)

            elapsed = time.time() - start
            run_times.append(elapsed)

            if isinstance(output, bytes):
                output_length = len(output)
                word_count = 0
            else:
                output_length = len(output)
                word_count = len(output.split())
                token_counts.append(word_count)

            results["runs"].append({
                "run_number": i,
                "elapsed_time": elapsed,
                "output_chars": output_length,
                "output_words": word_count,
                "status": "success",
            })

            out_path = os.path.join(out_dir, f"gemma3_vlm_output_run_{i}.txt")
            with open(out_path, "wb" if isinstance(output, bytes) else "w") as f:
                f.write(output)

        except Exception as e:
            err = str(e)
            logger.error(f"Run failed: {err}")
            results["runs"].append({"run_number": i, "status": "failed", "error": err})

    if run_times:
        avg = sum(run_times) / len(run_times)
        mn = min(run_times)
        mx = max(run_times)

        avg_words = sum(token_counts) / len(token_counts) if token_counts else 0
        tps = avg_words / avg if avg else 0

        std = (sum((t - avg) ** 2 for t in run_times) / len(run_times)) ** 0.5
        cv = std / avg * 100 if avg else 0

        cold = run_times[0]
        warm = run_times[1:]
        warm_avg = sum(warm) / len(warm) if warm else None

        results["statistics"] = {
            "successful_runs": len(run_times),
            "failed_runs": num_runs - len(run_times),
            "avg_time": avg,
            "min_time": mn,
            "max_time": mx,
            "time_std_dev": std,
            "time_variability_cv": cv,
            "avg_words": avg_words,
            "avg_tokens_per_sec": tps,
            "cold_start_time": cold,
            "avg_warm_time": warm_avg,
            "cold_vs_warm_ratio": cold / warm_avg if warm_avg else None,
            "efficiency_rating": calculate_efficiency(tps),
            "consistency_score": max(0, 100 - cv),
        }

    out_json = os.path.join(out_dir, "gemma3_vlm_benchmark_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved → {out_json}")
    return results


if __name__ == "__main__":
    benchmark_gemma3_vlm()
