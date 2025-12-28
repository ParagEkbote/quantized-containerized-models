import os
import time
import json
import logging
from datetime import datetime

from benchmark.utils import safe_replicate_run, normalize_output


# --------------------------------------------------
# Logging (guarded)
# --------------------------------------------------
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler("benchmark_phi4.log"),
            logging.StreamHandler(),
        ],
    )

logger = logging.getLogger(__name__)



def benchmark_phi4_unsloth(num_runs: int = 3):
    deployment_id = (
        "paragekbote/phi-4-reasoning-plus-unsloth:"
        "22438984324149ef4ecfcea3d631185641c23e46ce526ae4439b8c89c27ac086"
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
        "timestamp": datetime.now().astimezone().isoformat(),
        "input_params": input_params,
        "runs": [],
    }

    # --------------------------------------------------
    # Execute runs (canonical record)
    # --------------------------------------------------
    for i in range(1, num_runs + 1):
        logger.info(f"--- Run {i}/{num_runs} ---")

        try:
            start = time.perf_counter()
            raw_out = safe_replicate_run(deployment_id, input_params)
            output = normalize_output(raw_out)
            elapsed = time.perf_counter() - start

            if isinstance(output, bytes):
                text = output.decode("utf-8", errors="ignore")
            else:
                text = str(output)

            word_count = len(text.split())

            out_path = os.path.join(out_dir, f"phi4_output_run_{i}.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)

            results["runs"].append({
                "run_number": i,
                "elapsed_time": elapsed,
                "word_count": word_count,
                "output_file": out_path,
                "status": "success",
            })

            logger.info(f"Run {i} succeeded in {elapsed:.3f}s")

        except Exception as e:
            logger.exception("Run failed")
            results["runs"].append({
                "run_number": i,
                "status": "failed",
                "error": str(e),
            })

    # --------------------------------------------------
    # Statistics derived FROM runs
    # --------------------------------------------------
    successful_runs = [
        r for r in results["runs"]
        if r.get("status") == "success"
    ]

    if successful_runs:
        times = [r["elapsed_time"] for r in successful_runs]
        words = [r["word_count"] for r in successful_runs]

        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        std_time = (
            (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
        )
        cv = (std_time / avg_time * 100) if avg_time else None

        avg_words = sum(words) / len(words)
        words_per_sec = avg_words / avg_time if avg_time else 0

        cold_run = successful_runs[0]
        warm_runs = successful_runs[1:]
        warm_avg = (
            sum(r["elapsed_time"] for r in warm_runs) / len(warm_runs)
            if warm_runs else None
        )

        consistency = max(0, min(100, 100 - (cv or 0)))

        results["statistics"] = {
            "successful_runs": len(successful_runs),
            "failed_runs": num_runs - len(successful_runs),

            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "time_std_dev": std_time,
            "time_variability_cv": cv,

            "avg_words": avg_words,
            "avg_words_per_sec": words_per_sec,

            "cold_start": {
                "run_number": cold_run["run_number"],
                "elapsed_time": cold_run["elapsed_time"],
                "output_file": cold_run["output_file"],
            },

            "avg_warm_time": warm_avg,
            "cold_vs_warm_ratio": (
                cold_run["elapsed_time"] / warm_avg
                if warm_avg else None
            ),
            "consistency_score": consistency,
        }

    out_json = os.path.join(out_dir, "phi4_benchmark_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved → {out_json}")
    return results


if __name__ == "__main__":
    benchmark_phi4_unsloth()
