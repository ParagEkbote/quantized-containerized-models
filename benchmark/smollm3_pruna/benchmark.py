import os
import time
import json
import logging
from datetime import datetime

from benchmark.utils import safe_replicate_run, normalize_output

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler("benchmark_smollm3.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def calculate_efficiency_rating(tps):
    if tps >= 50:
        return "Excellent"
    if tps >= 20:
        return "Very Good"
    if tps >= 10:
        return "Good"
    if tps >= 5:
        return "Fair"
    return "Poor"


def benchmark_smollm3(num_runs=3):
    deployment_id = (
        "paragekbote/smollm3-3b-smashed:"
        "232b6f87dac025cb54803cfbc52135ab8366c21bbe8737e11cd1aee4bf3a2423"
    )

    out_dir = os.getenv("BENCHMARK_OUTPUT_DIR", ".")
    os.makedirs(out_dir, exist_ok=True)

    input_params = {
        "prompt": "What are the applications of ML in healthcare?",
        "mode": "no_think",
        "seed": 18,
        "max_new_tokens": 1024,
    }

    logger.info("Running SmolLM3 Benchmark")
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
            raw_out = safe_replicate_run(deployment_id, input_params)
            output = normalize_output(raw_out)

            elapsed = time.time() - start
            run_times.append(elapsed)

            if isinstance(output, bytes):
                text = output.decode("utf-8", errors="ignore")
            else:
                text = output

            words = len(text.split())
            token_counts.append(words)

            out_path = os.path.join(out_dir, f"smollm3_output_run_{i}.txt")
            with open(out_path, "w") as f:
                f.write(text)

            results["runs"].append({
                "run_number": i,
                "elapsed_time": elapsed,
                "word_count": words,
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
        cv = std / avg * 100 if avg else 0

        avg_words = sum(token_counts) / len(token_counts)
        tps = avg_words / avg if avg else 0

        cold = run_times[0]
        warm = run_times[1:]
        warm_avg = sum(warm) / len(warm) if warm else None

        word_std = (
            (sum((w - avg_words) ** 2 for w in token_counts) / len(token_counts)) ** 0.5
            if len(token_counts) > 1 else 0
        )

        results["statistics"] = {
            "successful_runs": len(run_times),
            "failed_runs": num_runs - len(run_times),
            "avg_time": avg,
            "min_time": mn,
            "max_time": mx,
            "time_std_dev": std,
            "time_variability_cv": cv,
            "avg_words": avg_words,
            "word_std_dev": word_std,
            "avg_tokens_per_sec": tps,
            "cold_start_time": cold,
            "avg_warm_time": warm_avg,
            "cold_vs_warm_ratio": cold / warm_avg if warm_avg else None,
            "efficiency_rating": calculate_efficiency_rating(tps),
            "consistency_score": max(0, 100 - cv),
        }

    out_json = os.path.join(out_dir, "smollm3_benchmark_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved → {out_json}")
    return results


if __name__ == "__main__":
    benchmark_smollm3()
