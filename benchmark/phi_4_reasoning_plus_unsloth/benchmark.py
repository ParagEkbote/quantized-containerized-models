import os
import time
import json
import logging
from datetime import datetime

from benchmark.utils import safe_replicate_run, normalize_output

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler("benchmark_phi4.log"),
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


def benchmark_phi4_unsloth(num_runs=3):
    deployment_id = (
        "paragekbote/phi-4-reasoning-plus-unsloth:"
        "a6b2aa30b793e79ee4f7e30165dce1636730b20c2798d487fc548427ba6314d7"
    )

    out_dir = os.getenv("BENCHMARK_OUTPUT_DIR", ".")
    os.makedirs(out_dir, exist_ok=True)

    input_params = {
        "prompt": "Explain quantum entanglement.",
        "max_new_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.95,
        "seed": 42,
    }

    logger.info("Running Phi-4 Benchmark")
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

            out_path = os.path.join(out_dir, f"phi4_output_run_{i}.txt")
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

        avg_words = sum(token_counts) / len(token_counts)
        tps = avg_words / avg

        std = (sum((t - avg) ** 2 for t in run_times) / len(run_times)) ** 0.5
        cv = std / avg * 100

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

    out_json = os.path.join(out_dir, "phi4_benchmark_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved → {out_json}")
    return results


if __name__ == "__main__":
    benchmark_phi4_unsloth()
