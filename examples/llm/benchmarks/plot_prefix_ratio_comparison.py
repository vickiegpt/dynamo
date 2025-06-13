import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_results(file_path):
    """Load results from JSON file"""
    with open(file_path, "r") as f:
        return json.load(f)


def extract_data(results, exclude_ratio=None):
    """Extract and optionally filter data from results"""
    prefix_ratios = results["prefix_ratios"]
    ttft_values = results["ttft_values"]
    throughput_values = results["throughput_values"]

    if exclude_ratio is not None:
        filtered_indices = [
            i for i, ratio in enumerate(prefix_ratios) if ratio != exclude_ratio
        ]
        return (
            [prefix_ratios[i] for i in filtered_indices],
            [ttft_values[i] for i in filtered_indices],
            [throughput_values[i] for i in filtered_indices],
        )

    return prefix_ratios, ttft_values, throughput_values


def plot_dataset(
    ax1, ax2, prefix_ratios, ttft_values, throughput_values, label, color_marker
):
    """Helper function to plot a dataset on both TTFT and throughput axes"""
    ax1.plot(
        prefix_ratios, ttft_values, color_marker, label=label, linewidth=2, markersize=6
    )
    ax2.plot(
        prefix_ratios,
        throughput_values,
        color_marker,
        label=label,
        linewidth=2,
        markersize=6,
    )


def configure_axis(ax, xlabel, ylabel, title, xlim=(0.05, 0.95), yscale="log"):
    """Configure axis with common settings"""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(*xlim)
    ax.set_yscale(yscale)


def print_dataset_stats(datasets):
    """Print summary statistics for all datasets"""
    for name, (_, ttft_values, throughput_values, _) in datasets.items():
        print(
            f"\n{name} - TTFT range: {min(ttft_values):.1f} - {max(ttft_values):.1f} ms"
        )
        print(
            f"{name} - Throughput range: {min(throughput_values):.1f} - {max(throughput_values):.1f} tokens/s"
        )


def create_comparison_plots():
    """Create comparison plots for dynamo vs vLLM vs lowcpu results"""

    # Dataset configurations: (file_path, label, style, exclude_ratio)
    dataset_configs = [
        (
            "prefix_ratio_results_llama_main_vllm/results_summary.json",
            "Latest release vLLM",
            "r-o",
            None,
        ),
        (
            "prefix_ratio_results_llama_vllm/results_summary.json",
            "Dynamo patched vLLM",
            "c-s",
            None,
        ),
        ("prefix_ratio_results_llama/results_summary.json", "Dynamo", "b-^", None),
        (
            "prefix_ratio_results_llama_lowcpu/results_summary.json",
            "Dynamo (1 CPU)",
            "g-D",
            None,
        ),
    ]

    # Load and extract data for all datasets
    datasets = {}
    reference_results = None

    for file_path, label, style, exclude_ratio in dataset_configs:
        results = load_results(file_path)
        if reference_results is None:
            reference_results = results  # Use first dataset for config info

        data = extract_data(results, exclude_ratio)
        datasets[label] = (*data, style)

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # Plot all datasets
    for label, (
        prefix_ratios,
        ttft_values,
        throughput_values,
        style,
    ) in datasets.items():
        plot_dataset(
            ax1, ax2, prefix_ratios, ttft_values, throughput_values, label, style
        )

    # Configure both axes
    configure_axis(
        ax1, "Prefix Ratio", "TTFT (ms)", "Time to First Token vs Prefix Ratio"
    )
    configure_axis(
        ax2, "Prefix Ratio", "Throughput (tokens/s)", "Throughput vs Prefix Ratio"
    )

    # Add overall title
    fig.suptitle("Llama 3.1 8B Agg DP1 TP1 (L40 GPU)", fontsize=16, fontweight="bold")

    # Adjust layout and save
    plt.tight_layout()

    # Save plots
    for ext, dpi in [("png", 300), ("pdf", None)]:
        output_path = f"prefix_ratio_comparison_llama.{ext}"
        save_kwargs = {"bbox_inches": "tight"}
        if dpi:
            save_kwargs["dpi"] = dpi
        plt.savefig(output_path, **save_kwargs)
        print(f"Plot saved as: {output_path}")

    # Print summary statistics
    config = reference_results["config"]
    print(f"\nSummary Statistics:")
    print(f"Model: {config['model']}")
    print(f"Input Sequence Length: {config['isl']}")
    print(f"Output Sequence Length: {config['osl']}")
    print(f"Requests: {config['requests']}")
    print(f"Concurrency: {config['concurrency']}")

    print_dataset_stats(datasets)


if __name__ == "__main__":
    create_comparison_plots()
