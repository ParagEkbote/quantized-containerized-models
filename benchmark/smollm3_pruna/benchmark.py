import json
import time
import logging
from datetime import datetime
from pathlib import Path

from benchmark.utils import safe_replicate_run, normalize_output

logger = logging.getLogger(__name__)


def benchmark_smollm3(
    num_runs=3,
    prompt="What are the applications of machine learning in healthcare?",
    mode="no_think",
    seed=18,
    max_new_tokens=1024,
):
    """
    Benchmark SmolLM3-3B Smashed deployment.
    """

    deployment_id = (
        "paragekbote/smollm3-3b-smashed:"
        "232b6f87dac025cb54803cfbc52135ab8366c21bbe8737e11cd1aee4bf3a2423"
    )

    input_params = {
        "prompt": prompt,
        "mode": mode,
        "seed": seed,
        "max_new_tokens": max_new_tokens,
    }

    logger.info("=" * 80)
    logger.info("SMOLLM3-3B SMASHED BENCHMARK")
    logger.info("=" * 80)
    logger.info(json.dumps(input_params, indent=2))

    results = {
        "deployment_id": deployment_id,
        "deployment_name": "SmolLM3-3B-Smashed",
        "timestamp": datetime.now().isoformat(),
        "input_params": input_params,
        "runs": [],
    }

    run_times = []
    token_counts = []

    out_dir = Path("benchmark/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    for run in range(1, num_runs + 1):
        logger.info(f"--- Run {run}/{num_runs} ---")

        try:
            start = time.time()

            output = safe_replicate_run(deployment_id, input_params)
            normalized = normalize_output(output)

            elapsed = time.time() - start
            run_times.append(elapsed)

            if isinstance(normalized, str):
                text = normalized
                words = len(text.split())
            else:
                text = "<binary output>"
                words = 0

            token_counts.append(words)

            # Save output text
            out_path = out_dir / f"smollm3_run_{run}.txt"
            out_path.write_text(text, encoding="utf-8")

            results["runs"].append(
                {
                    "run_number": run,
                    "elapsed_time": elapsed,
                    "output_words": words,
                    "output_file": str(out_path),
                    "status": "success",
                }
            )

            logger.info(f"✓ Run {run} completed ({words} words in {elapsed:.2f}s)")

        except Exception as e:
            logger.error(f"✗ Run {run} failed: {e}")
            results["runs"].append({"run_number": run, "status": "failed", "error": str(e)})

    # Save JSON
    (out_dir / "smollm3_benchmark.json").write_text(json.dumps(results, indent=2))

    return results


if __name__ == "__main__":
    benchmark_smollm3(num_runs=3)
