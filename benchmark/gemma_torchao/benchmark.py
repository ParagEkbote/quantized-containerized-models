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

    # --------------------------------------------------
    # Multimodal inputs (image + text → text)
    # --------------------------------------------------
    input_params = {
        "prompt": "Describe the image in the photo.",
        "image_url": (
            "https://images.pexels.com/photos/29380151/"
            "pexels-photo-29380151.jpeg"
        ),

        # Model controls
        "max_new_tokens": 500,
        "temperature": 0.7,
        "top_p": 0.9,
        "seed": 42,

        # Optimization flags (typed correctly)
        "use_quantization": True,
        "use_sparsity": True,
        "sparsity_type": "layer_norm",
        "sparsity_ratio": 0.3,
    }

    logger.info("Running Gemma-3 VLM benchmark")
    logger.info(json.dumps(input_params, indent=2))

    results = {
        "deployment_id": deployment_id,
        "model_type": "vlm",
        "modality": "image+text → text",
        "timestamp": datetime.now().astimezone().isoformat(),
        "input_params": input_params,
        "runs": [],
    }

    # --------------------------------------------------
    # Execute runs
    # --------------------------------------------------
    for i in range(1, num_runs + 1):
        logger.info("--- Run %d/%d ---", i, num_runs)

        try:
            start = time.time()
            raw_output = safe_replicate_run(deployment_id, input_params)
            text = normalize_output(raw_output, output_type="vlm")
            elapsed = time.time() - start

            if not isinstance(text, str):
                raise TypeError(
                    f"normalize_output returned {type(text)}, expected str"
                )

            output_chars = len(text)
            output_words = len(text.split())

            out_path = os.path.join(
                out_dir, f"gemma3_vlm_output_run_{i}.txt"
            )
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)

            results["runs"].append(
                {
                    "run_number": i,
                    "elapsed_time": elapsed,
                    "output_chars": output_chars,
                    "output_words": output_words,
                    "output_file": out_path,
                    "status": "success",
                }
            )

            logger.info("Run %d succeeded in %.3fs", i, elapsed)

        except Exception as e:
            logger.exception("Run failed")
            results["runs"].append(
                {
                    "run_number": i,
                    "status": "failed",
                    "error": str(e),
                }
            )

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------
    successful_runs = [
        r for r in results["runs"] if r["status"] == "success"
    ]

    times = [r["elapsed_time"] for r in successful_runs]
    word_runs = [r for r in successful_runs if r["output_words"] is not None]

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

        words_per_sec = avg_words / avg if avg else 0

        cold_run = successful_runs[0]
        warm_runs = successful_runs[1:]
        warm_avg = (
            sum(r["elapsed_time"] for r in warm_runs) / len(warm_runs)
            if warm_runs else None
        )

        results["statistics"] = {
            "successful_runs": len(successful_runs),
            "failed_runs": num_runs - len(successful_runs),

            "avg_time": avg,
            "min_time": mn,
            "max_time": mx,
            "time_std_dev": std,
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
        }

    out_json = os.path.join(
        out_dir, "gemma3_vlm_benchmark_results.json"
    )
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Saved → %s", out_json)
    return results


if __name__ == "__main__":
    benchmark_gemma3_vlm()
