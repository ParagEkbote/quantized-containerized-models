import replicate
import time
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('benchmark.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def _calculate_efficiency_rating(tokens_per_sec):
    """Calculate efficiency rating based on throughput."""
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

def benchmark_smollm3(num_runs=3):
    """
    Benchmark the SmolLM3-3B-Smashed deployment.
    
    Args:
        num_runs: Number of benchmark runs (default: 3)
    """
    deployment_id = "paragekbote/smollm3-3b-smashed:232b6f87dac025cb54803cfbc52135ab8366c21bbe8737e11cd1aee4bf3a2423"
    
    input_params = {
        "seed": 18,
        "prompt": "What are the applications of ML in healthcare?",
        "max_new_tokens": 1420
    }
    
    logger.info("="*70)
    logger.info("SmolLM3-3B-Smashed Benchmark")
    logger.info("="*70)
    logger.info(f"Deployment: {deployment_id}")
    logger.info(f"Number of runs: {num_runs}")
    logger.info(f"Prompt: {input_params['prompt']}")
    logger.info("="*70)
    
    results = {
        "deployment_id": deployment_id,
        "deployment_name": "SmolLM3-3B-Smashed",
        "timestamp": datetime.now().isoformat(),
        "input_params": input_params,
        "runs": []
    }
    
    run_times = []
    token_counts = []
    
    for run_num in range(1, num_runs + 1):
        logger.info(f"--- Run {run_num}/{num_runs} ---")
        
        try:
            start_time = time.time()
            
            # Run the model
            output = replicate.run(deployment_id, input=input_params)
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Read output
            output_text = output.read().decode('utf-8')
            output_length = len(output_text)
            word_count = len(output_text.split())
            
            run_times.append(elapsed_time)
            token_counts.append(word_count)
            
            # Store run details
            run_result = {
                "run_number": run_num,
                "elapsed_time": elapsed_time,
                "output_length_chars": output_length,
                "output_length_words": word_count,
                "output_text": output_text,
                "status": "success"
            }
            results["runs"].append(run_result)
            
            logger.info(f"✓ Completed in {elapsed_time:.2f}s")
            logger.info(f"  Output: {output_length} characters, ~{word_count} words")
            logger.info(f"  Tokens/sec: {word_count/elapsed_time:.2f}")
            
            # Save individual output
            with open(f"output_run_{run_num}.txt", "w") as f:
                f.write(output_text)
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"✗ Failed: {error_msg}")
            
            run_result = {
                "run_number": run_num,
                "status": "failed",
                "error": error_msg
            }
            results["runs"].append(run_result)
    
    # Calculate statistics
    if run_times:
        avg_time = sum(run_times) / len(run_times)
        min_time = min(run_times)
        max_time = max(run_times)
        avg_words = sum(token_counts) / len(token_counts) if token_counts else 0
        avg_tokens_per_sec = avg_words / avg_time if avg_time > 0 else 0
        
        # Calculate additional metrics
        time_std_dev = (sum((t - avg_time) ** 2 for t in run_times) / len(run_times)) ** 0.5
        time_variance_coefficient = (time_std_dev / avg_time * 100) if avg_time > 0 else 0
        
        # Time to first token (cold start) vs subsequent runs
        cold_start_time = run_times[0] if run_times else None
        warm_run_times = run_times[1:] if len(run_times) > 1 else []
        avg_warm_time = sum(warm_run_times) / len(warm_run_times) if warm_run_times else None
        
        # Output consistency
        word_std_dev = (sum((w - avg_words) ** 2 for w in token_counts) / len(token_counts)) ** 0.5 if len(token_counts) > 1 else 0
        
        results["statistics"] = {
            "successful_runs": len(run_times),
            "failed_runs": num_runs - len(run_times),
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "time_std_dev": time_std_dev,
            "time_variance_coefficient": time_variance_coefficient,
            "avg_words": avg_words,
            "word_std_dev": word_std_dev,
            "avg_tokens_per_sec": avg_tokens_per_sec,
            "cold_start_time": cold_start_time,
            "avg_warm_time": avg_warm_time,
            "cold_vs_warm_ratio": cold_start_time / avg_warm_time if avg_warm_time and cold_start_time else None,
            "consistency_score": max(0, 100 - time_variance_coefficient),
            "efficiency_rating": self._calculate_efficiency_rating(avg_tokens_per_sec)
        }
        
        # Performance insights
        insights = []
        
        # Cold start analysis
        if cold_start_time and avg_warm_time and cold_start_time > avg_warm_time * 1.5:
            insights.append(f"Significant cold start delay detected: {cold_start_time:.2f}s vs {avg_warm_time:.2f}s warm average")
        
        # Consistency analysis
        if time_variance_coefficient > 50:
            insights.append(f"High variability in response times (CV: {time_variance_coefficient:.1f}%) - inconsistent performance")
        elif time_variance_coefficient < 20:
            insights.append(f"Excellent consistency (CV: {time_variance_coefficient:.1f}%) - predictable performance")
        
        # Throughput analysis
        if avg_tokens_per_sec < 5:
            insights.append(f"Low throughput: {avg_tokens_per_sec:.2f} tokens/sec - may need optimization")
        elif avg_tokens_per_sec > 20:
            insights.append(f"High throughput: {avg_tokens_per_sec:.2f} tokens/sec - excellent performance")
        
        # Output consistency
        if word_std_dev > avg_words * 0.2:
            insights.append(f"Variable output lengths (±{word_std_dev:.0f} words) - may indicate non-deterministic behavior")
        
        results["insights"] = insights
        
        # Print summary
        logger.info("")
        logger.info("="*70)
        logger.info("BENCHMARK SUMMARY")
        logger.info("="*70)
        logger.info(f"Successful runs: {len(run_times)}/{num_runs}")
        logger.info(f"Average time: {results['statistics']['avg_time']:.2f}s")
        logger.info(f"Min time: {results['statistics']['min_time']:.2f}s")
        logger.info(f"Max time: {results['statistics']['max_time']:.2f}s")
        logger.info(f"Time variability (CV): {results['statistics']['time_variance_coefficient']:.1f}%")
        logger.info(f"Average output: ~{results['statistics']['avg_words']:.0f} words (±{results['statistics']['word_std_dev']:.0f})")
        logger.info(f"Average throughput: {results['statistics']['avg_tokens_per_sec']:.2f} tokens/sec")
        logger.info(f"Efficiency rating: {results['statistics']['efficiency_rating']}")
        logger.info(f"Consistency score: {results['statistics']['consistency_score']:.1f}/100")
        
        if results['statistics']['cold_start_time'] and results['statistics']['avg_warm_time']:
            logger.info(f"Cold start: {results['statistics']['cold_start_time']:.2f}s")
            logger.info(f"Warm runs avg: {results['statistics']['avg_warm_time']:.2f}s")
            logger.info(f"Cold/Warm ratio: {results['statistics']['cold_vs_warm_ratio']:.2f}x")
        
        if results.get('insights'):
            logger.info("")
            logger.info("PERFORMANCE INSIGHTS:")
            for insight in results['insights']:
                logger.info(f"  • {insight}")
        
        logger.info("="*70)
    else:
        logger.warning("All runs failed. No statistics available.")
        results["statistics"] = None
    
    # Save results to JSON
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info("")
    logger.info("✓ Detailed results saved to benchmark_results.json")
    logger.info("✓ Individual outputs saved to output_run_*.txt")
    logger.info("✓ Full log saved to benchmark.log")
    
    return results


# Run the benchmark
if __name__ == "__main__":
    # Runs 3 benchmarks
    benchmark_smollm3(num_runs=3)