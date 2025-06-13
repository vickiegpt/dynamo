#!/usr/bin/env python3

import argparse
import json
import logging
import os
import subprocess
import matplotlib.pyplot as plt

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def get_genai_perf_cmd(
    model,
    prefix_ratio,
    isl,
    osl,
    requests,
    concurrency,
    seed,
    num_prompts,
    artifact_dir,
    url="http://localhost:8888",
):
    """Build genai-perf command based on prefix ratio"""
    prefix_length = int(isl * prefix_ratio)
    synthetic_input_length = int(isl * (1 - prefix_ratio))

    return [
        "genai-perf",
        "profile",
        "--model",
        model,
        "--tokenizer",
        model,
        "--endpoint-type",
        "chat",
        "--endpoint",
        "v1/chat/completions",
        "--streaming",
        "--url",
        url,
        "--synthetic-input-tokens-mean",
        str(synthetic_input_length),
        "--synthetic-input-tokens-stddev",
        "0",
        "--output-tokens-mean",
        str(osl),
        "--output-tokens-stddev",
        "0",
        "--extra-inputs",
        f"max_tokens:{osl}",
        "--extra-inputs",
        f"min_tokens:{osl}",
        "--extra-inputs",
        "ignore_eos:true",
        "--extra-inputs",
        '{"nvext":{"ignore_eos":true}}',
        "--concurrency",
        str(concurrency),
        "--request-count",
        str(requests),
        "--num-dataset-entries",
        str(requests),
        "--random-seed",
        str(seed),
        "--prefix-prompt-length",
        str(prefix_length),
        "--num-prefix-prompts",
        str(num_prompts),
        "--artifact-dir",
        artifact_dir,
        "--",
        "-v",
        "--max-threads",
        "256",
        "-H",
        "Authorization: Bearer NOT USED",
        "-H",
        "Accept: text/event-stream",
    ]


def get_gap_result(artifact_dir: str) -> dict:
    """Parse genai-perf results from JSON file"""
    json_file_path = None
    for root, _, files in os.walk(artifact_dir):
        if "profile_export_genai_perf.json" in files:
            json_file_path = os.path.join(root, "profile_export_genai_perf.json")
            break

    if json_file_path is None:
        raise FileNotFoundError(
            f"profile_export_genai_perf.json not found in {artifact_dir}"
        )

    with open(json_file_path, "r") as f:
        return json.load(f)


def run_benchmark(
    model,
    prefix_ratio,
    isl,
    osl,
    requests,
    concurrency,
    seed,
    num_prompts,
    output_dir,
    url="http://localhost:8888",
):
    """Run genai-perf benchmark for a specific prefix ratio"""
    logger.info(f"Running benchmark with prefix_ratio={prefix_ratio}, seed={seed}")

    artifact_dir = f"{output_dir}/prefix_ratio_{prefix_ratio}_seed_{seed}"
    os.makedirs(artifact_dir, exist_ok=True)

    genai_perf_cmd = get_genai_perf_cmd(
        model,
        prefix_ratio,
        isl,
        osl,
        requests,
        concurrency,
        seed,
        num_prompts,
        artifact_dir,
        url,
    )

    logger.info(f"Running command: {' '.join(genai_perf_cmd)}")

    try:
        gap_process = subprocess.run(
            genai_perf_cmd, capture_output=True, text=True, check=True
        )

        logger.info("Genai-perf profiling completed successfully")
        logger.info(gap_process.stdout)

        gap_result = get_gap_result(artifact_dir)
        return gap_result

    except subprocess.CalledProcessError as e:
        logger.error(f"Genai-perf failed with error code: {e.returncode}")
        logger.error(f"stderr: {e.stderr}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark prefix ratios and plot results"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name",
    )
    parser.add_argument("--isl", type=int, default=7000, help="Input sequence length")
    parser.add_argument("--osl", type=int, default=100, help="Output sequence length")
    parser.add_argument("--requests", type=int, default=100, help="Number of requests")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrency level")
    parser.add_argument("--seed", type=int, default=42, help="Initial random seed")
    parser.add_argument(
        "--num-prompts", type=int, default=1, help="Number of prefix prompts"
    )
    parser.add_argument(
        "--prefix-ratios",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        help="List of prefix ratios to test",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="prefix_ratio_results_llama_main_vllm",
        help="Output directory for results",
    )
    parser.add_argument(
        "--url", type=str, default="http://localhost:8888", help="Server URL"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Store results
    prefix_ratios = []
    ttft_values = []
    throughput_values = []

    current_seed = args.seed

    # Run benchmarks for each prefix ratio
    for prefix_ratio in args.prefix_ratios:
        result = run_benchmark(
            args.model,
            prefix_ratio,
            args.isl,
            args.osl,
            args.requests,
            args.concurrency,
            current_seed,
            args.num_prompts,
            args.output_dir,
            args.url,
        )

        if result is not None:
            ttft = result["time_to_first_token"]["avg"]
            throughput = result["output_token_throughput"]["avg"]

            prefix_ratios.append(prefix_ratio)
            ttft_values.append(ttft)
            throughput_values.append(throughput)

            logger.info(
                f"Prefix ratio {prefix_ratio}: TTFT={ttft:.2f}ms, Throughput={throughput:.2f} tokens/s"
            )

        current_seed += 1

    # Create plots
    if prefix_ratios and ttft_values and throughput_values:
        # Plot TTFT vs Prefix Ratio
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(prefix_ratios, ttft_values, "bo-", linewidth=2, markersize=8)
        plt.xlabel("Prefix Ratio")
        plt.ylabel("Time to First Token (ms)")
        plt.title("TTFT vs Prefix Ratio")
        plt.grid(True, alpha=0.3)
        for i, (pr, ttft) in enumerate(zip(prefix_ratios, ttft_values)):
            plt.annotate(
                f"{ttft:.1f}ms",
                (pr, ttft),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )

        # Plot Throughput vs Prefix Ratio
        plt.subplot(1, 2, 2)
        plt.plot(prefix_ratios, throughput_values, "ro-", linewidth=2, markersize=8)
        plt.xlabel("Prefix Ratio")
        plt.ylabel("Output Token Throughput (tokens/s)")
        plt.title("Throughput vs Prefix Ratio")
        plt.grid(True, alpha=0.3)
        for i, (pr, thpt) in enumerate(zip(prefix_ratios, throughput_values)):
            plt.annotate(
                f"{thpt:.1f}",
                (pr, thpt),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )

        plt.tight_layout()

        # Save plot
        plot_path = f"{args.output_dir}/prefix_ratio_performance.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        logger.info(f"Performance plot saved to {plot_path}")
        plt.show()

        # Save results to JSON
        results_data = {
            "prefix_ratios": prefix_ratios,
            "ttft_values": ttft_values,
            "throughput_values": throughput_values,
            "config": {
                "model": args.model,
                "isl": args.isl,
                "osl": args.osl,
                "requests": args.requests,
                "concurrency": args.concurrency,
                "initial_seed": args.seed,
            },
        }

        results_path = f"{args.output_dir}/results_summary.json"
        with open(results_path, "w") as f:
            json.dump(results_data, f, indent=2)
        logger.info(f"Results summary saved to {results_path}")

    else:
        logger.error("No successful benchmark results to plot")


if __name__ == "__main__":
    main()
