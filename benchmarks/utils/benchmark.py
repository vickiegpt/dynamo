#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import sys
from pathlib import Path

from benchmarks.utils.workflow import run_benchmark_workflow


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Orchestrator")
    parser.add_argument("--agg", required=True, help="Path to aggregated DGD manifest")
    parser.add_argument(
        "--disagg", required=True, help="Path to disaggregated DGD manifest"
    )
    parser.add_argument("--namespace", required=True, help="Kubernetes namespace")
    parser.add_argument(
        "--isl", type=int, default=200, help="Input sequence length (default: 200)"
    )
    parser.add_argument(
        "--std",
        type=int,
        default=10,
        help="Input sequence standard deviation (default: 10)",
    )
    parser.add_argument(
        "--osl", type=int, default=200, help="Output sequence length (default: 200)"
    )
    parser.add_argument(
        "--concurrency", type=int, default=10, help="Concurrency level (default: 10)"
    )
    parser.add_argument(
        "--model",
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Model name (default: deepseek-ai/DeepSeek-R1-Distill-Llama-8B)",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory")
    args = parser.parse_args()

    for p in [args.agg, args.disagg]:
        if not Path(p).is_file():
            print(f"ERROR: Manifest not found: {p}")
            return 1

    asyncio.run(
        run_benchmark_workflow(
            namespace=args.namespace,
            agg_manifest=args.agg,
            disagg_manifest=args.disagg,
            isl=args.isl,
            std=args.std,
            osl=args.osl,
            concurrency=args.concurrency,
            model=args.model,
            output_dir=args.output_dir,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
