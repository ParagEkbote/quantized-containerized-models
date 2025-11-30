import json
import time
import logging
from datetime import datetime
from pathlib import Path

from benchmark.utils import safe_replicate_run, normalize_output

logger = logging.getLogger(__name__)

def benchmark_phi4_unsloth(
    num_runs=3,
    prompt="Explain quantum entanglement.",
    max_new_tokens=1024,
    temperature=0.7,
    top_p=0.95,
    seed=42,
):
    deployment_id = (
        "paragekbote/phi-4-reasoning-plus-unsloth:"
        "a6b2aa30b793e79ee4f7e30165dce1636730b20c2798d487fc548427ba6314d7"
    )

    input_params = {
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
    }

    results = {
        "deployment_id": deployment_id,
        "timestamp": datetime.now().isoformat(),
        "input_params": input_params,
        "runs": [],
    }

    run_times = []
    token_counts = []

    for run in range(1, num_runs + 1):
        logger.info(f"--- Run {run}/{num_runs} ---")

        try:
            start = time.time()
            output = safe_replicate_run(deployment_id, input_params)

            normalized = normalize_output(output)

            if isinstance(normalized, str):
                output_text = normalized
                words = len(output_text.split())
                chars = len(output_text)
            else:
                # Not text (unexpected for Phi4), treat as binary
                output_text = None
                words = 0
                chars = len(normalized)

            elapsed = time.time() - start

            run_times.append(elapsed)
            token_counts.append(words)

            # Save output
            out_dir = Path("benchmark/results")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"phi4_run_{run}.txt"

            if output_text:
                out_path.write_text(output_text, encoding="utf-8")
            else:
                out_path.write_bytes(normalized)

            results["runs"].append(
                {
                    "run_number": run,
                    "elapsed_time": elapsed,
                    "output_words": words,
                    "output_chars": chars,
                    "status": "success",
                }
            )

        except Exception as e:
            results["runs"].append(
                {"run_number": run, "status": "failed", "error": str(e)}
            )

    # Save summary
    out_dir = Path("benchmark/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phi4_benchmark_results.json").write_text(
        json.dumps(results, indent=2)
    )

    return results


if __name__ == "__main__":
    benchmark_phi4_unsloth(num_runs=3)
