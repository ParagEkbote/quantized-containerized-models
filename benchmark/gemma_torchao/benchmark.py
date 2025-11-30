import json
import time
import logging
from datetime import datetime
from pathlib import Path

from benchmark.utils import safe_replicate_run, normalize_output

logger = logging.getLogger(__name__)


def benchmark_gemma3_vlm(
    num_runs=3,
    prompt="Describe the image in detail.",
    image_url="https://images.pexels.com/photos/29380151/pexels-photo-29380151.jpeg",
    use_sparsity="true",
    sparsity_type="layer_norm",
    sparsity_ratio=0.3,
    use_quantization="true",
    max_new_tokens=500,
    temperature=0.7,
    top_p=0.9,
    seed=42,
):
    """
    Benchmark Gemma-3 TorchAO Quant-Sparse VLM deployment.
    """

    deployment_id = (
        "paragekbote/gemma3-torchao-quant-sparse:"
        "44626bdc478fcfe56ee3d8a5a846b72f1e25abac25f740b2b615c1fcb2b63cb2"
    )

    input_params = {
        "prompt": prompt,
        "image_url": image_url,
        "use_sparsity": use_sparsity,
        "sparsity_type": sparsity_type,
        "sparsity_ratio": sparsity_ratio,
        "use_quantization": use_quantization,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
    }

    logger.info("=" * 90)
    logger.info("GEMMA-3 QUANT-SPARSE VLM BENCHMARK")
    logger.info("=" * 90)
    logger.info(json.dumps(input_params, indent=2))

    results = {
        "deployment_id": deployment_id,
        "deployment_name": "Gemma3-TorchAO-Quant-Sparse-VLM",
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
                output_text = normalized
                words = len(output_text.split())
                tokens = words
            else:
                # VLM should return text — but safe fallback
                output_text = "<binary output>"
                words = 0
                tokens = 0

            token_counts.append(tokens)

            # Save output
            file_path = out_dir / f"gemma3_vlm_run_{run}.txt"
            file_path.write_text(output_text, encoding="utf-8")

            results["runs"].append(
                {
                    "run_number": run,
                    "elapsed_time": elapsed,
                    "output_words": words,
                    "output_file": str(file_path),
                    "status": "success",
                }
            )

            logger.info(f"✓ Run {run} ok — {words} words in {elapsed:.2f}s")

        except Exception as e:
            logger.error(f"✗ Run {run} failed: {e}")
            results["runs"].append({"run_number": run, "status": "failed", "error": str(e)})

    # Save summary
    (out_dir / "gemma3_vlm_benchmark.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    benchmark_gemma3_vlm(num_runs=3)
