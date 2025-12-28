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



def benchmark_gemma3_vlm(num_runs: int = 3):
    deployment_id = (
        "paragekbote/gemma3-torchao-quant-sparse:"
        "396049cbfd6b79f8422fe41152aa2c0a0ddc0a602d21efb6dfd49c23799f7d74"
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
            start = time.time()
            raw_output = safe_replicate_run(deployment_id, input_params)
            output = normalize_output(raw_output)
            elapsed = time.time() - start

            if isinstance(output, bytes):
                text = None
                output_chars = len(output)
                output_words = None
            else:
                text = str(output)
                output_chars = len(text)
                output_words = len(text.split())

            out_path = os.path.join(out_dir, f"gemma3_vlm_output_run_{i}.txt")

            if text is None:
                with open(out_path, "wb") as f:
                    f.write(output)
            else:
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(text)

            results["runs"].append({
                "run_number": i,
                "elapsed_time": elapsed,
                "output_chars": output_chars,
                "output_words": output_words,
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

    times = [r["elapsed_time"] for r in successful_runs]
    word_runs = [r for r in successful_runs if r.get("output_words") is not None]

    if times:
        avg = sum(times) / len(times)
        mn = min(times)
        mx = max(times)
        std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5
        cv = (std / avg * 100) if avg else None

        avg_words = (
            sum(r["output_words"] for r in word_runs) / len(word_runs)
            if word_runs else 0
        )

        tps = avg_words / avg if avg else 0

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

            "avg_time": avg,
            "min_time": mn,
            "max_time": mx,
            "time_std_dev": std,
            "time_variability_cv": cv,

            "avg_words": avg_words,
            "avg_tokens_per_sec": tps,

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

    out_json = os.path.join(out_dir, "gemma3_vlm_benchmark_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved → {out_json}")
    return results


if __name__ == "__main__":
    benchmark_gemma3_vlm()
